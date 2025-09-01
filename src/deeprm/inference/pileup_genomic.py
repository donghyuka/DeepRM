"""Utilities for mapping transcript-relative coordinates to genomic coordinates
and aggregating per-site statistics across a transcriptome.

This module provides:

* `TranscriptMapper`: A vectorized mapper that converts 0-based, half-open
  transcript coordinates into genomic coordinates for both '+' and '-' strands,
  returning `np.nan` for out-of-bounds inputs.
* `worker`: A multiprocessing worker that applies the mapper per transcript and
  aggregates metrics.
* `parse_refflat`: A helper to parse RefFlat/RefGene/GenePred annotations into a
  normalized DataFrame.
* `load_split_data`: A helper that splits grouped input data into balanced shards
  for parallel processing.
* `pileup_genomic`: A high-level function that produces a per-(chrom, strand, pos)
  pileup with derived scores.

All exon intervals are assumed to be 0-based, half-open `[start, end)`, sorted by
genomic start in ascending order.
"""

import gc
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm


class TranscriptMapper:
    """Vectorized mapper from transcript coordinates to genomic coordinates.

    This class maps 0-based, half-open transcript offsets into genomic 0-based
    positions using exon interval metadata. It supports both '+' and '-' strands
    and returns `np.nan` for any input coordinate that falls outside the valid
    transcript range `[0, total_len)` or is non-finite.

    Attributes:
        starts (numpy.ndarray): 1D array of exon start genomic coordinates (0-based, inclusive),
            sorted ascending.
        ends (numpy.ndarray): 1D array of exon end genomic coordinates (0-based, exclusive),
            same shape as `starts`.
        lengths (numpy.ndarray): Exon lengths, computed as `ends - starts`.
        total_len (int): Total transcript length, i.e., `lengths.sum()`.
        strand (str): Strand symbol, either `'+'` or `'-'`.
        cumsum (numpy.ndarray): Precomputed cumulative exon lengths. For `'+'`,
            `cumsum[i]` is the total length before exon `i`; for `'-'`, it is
            computed over reversed exons to enable reverse mapping.

    Raises:
        ValueError: If `strand` is not `'+'` or `'-'`.
    """

    def __init__(self, exon_starts: np.ndarray, exon_ends: np.ndarray, strand: str):
        """Initialize the mapper with exon intervals and strand.

        Args:
            exon_starts (numpy.ndarray): 1D array of exon starts (0-based, inclusive),
                sorted ascending by genomic coordinate.
            exon_ends (numpy.ndarray): 1D array of exon ends (0-based, exclusive),
                same shape as `exon_starts`. Each `end` must be strictly greater
                than its corresponding `start`.
            strand (str): `'+'` or `'-'`.

        Raises:
            ValueError: If `strand` is not `'+'` or `'-'`.
        """
        self.starts = exon_starts
        self.ends = exon_ends
        self.lengths = self.ends - self.starts
        self.total_len = int(self.lengths.sum())
        self.strand = strand

        # Precompute cumulative sums
        if self.strand == "+":
            self.cumsum = np.concatenate(([0], np.cumsum(self.lengths)))
        elif self.strand == "-":
            self.cumsum = np.concatenate(([0], np.cumsum(self.lengths[::-1])))
        else:
            raise ValueError("strand must be '+' or '-'.")

    def map(self, coords: np.ndarray) -> np.ndarray:
        """Map transcript offsets to genomic positions (vectorized).

        Input coordinates are treated as 0-based offsets into the concatenated
        transcript exonic sequence with a valid range `[0, total_len)`. Values
        outside this range or non-finite values (NaN/Inf) yield `np.nan` in the
        output. The result is always `float64` to accommodate NaNs.

        For `'+'` strand: genomic position = `starts[idx] + offset`.
        For `'-'` strand: genomic position = `ends[idx] - 1 - offset`.

        Args:
            coords (numpy.ndarray): Array-like of transcript offsets to map. May be any
                shape; the returned array will match this shape.

        Returns:
            numpy.ndarray: Array of genomic positions (dtype `float64`) with `np.nan`
            for out-of-bounds or non-finite inputs. Positions are 0-based.

        Notes:
            * Assumes exon intervals are half-open `[start, end)` and sorted by
              genomic start ascending. It should be validated in parse_refflat().
            * No exceptions are raised for invalid `coords`; they are marked as NaN.
        """
        out = np.full(coords.shape, np.nan, dtype=np.float64)

        # Valid coords in [0, total_len)
        # also guard against inf/NaN in input by requiring finite
        valid = np.isfinite(coords) & (coords >= 0) & (coords < self.total_len)
        if not np.any(valid):
            return out

        coords = coords[valid].astype(np.int64, copy=False)

        if self.strand == "+":
            idx = np.searchsorted(self.cumsum, coords, side="right") - 1
            offsets = coords - self.cumsum[idx]
            out[valid] = (self.starts[idx] + offsets).astype(np.float64, copy=False)
        else:
            r_idx = np.searchsorted(self.cumsum, coords, side="right") - 1
            offsets = coords - self.cumsum[r_idx]
            idx = self.starts.size - r_idx - 1
            out[valid] = (self.ends[idx] - offsets - 1).astype(np.float64, copy=False)

        return out


def worker(df_list, refflat_df, collect_list):
    """Multiprocessing worker that maps transcript to genomic positions and aggregates metrics.

    Iterates over per-transcript DataFrames, maps the `ref_pos` transcript offsets to
    genomic positions using `TranscriptMapper`, drops rows with NaN genomic positions,
    and aggregates metrics per (chrom, strand, pos). Results are appended to a shared
    `multiprocessing.Manager().list()` for collection by the parent process.

    Args:
        df_list (list): List of DataFrames, each corresponding to a single
            transcript. Each DataFrame is expected to contain:
            - `transcript_id` (str): All rows share the same ID.
            - `ref_pos` (int): Transcript-relative offsets (0-based).
            - Metric columns used for aggregation:
            `kl_div_neg`, `kl_div_pos`, `count_all`, `count_pos`, `logsum_1_p_pos`.
        refflat_df (pandas.DataFrame): Annotation DataFrame indexed by `transcript_id`.
            Must provide columns: `exonStarts` (numpy.ndarray), `exonEnds` (numpy.ndarray),
            `strand` ('+'|'-'), and `chrom` (str).
        collect_list (list): Shared list where the worker
            appends its aggregated result DataFrame.

    Returns:
        None: Results are appended to `collect_list` as a `pandas.DataFrame` with columns:
            `chrom`, `strand`, `pos`, `kl_div_neg`, `kl_div_pos`, `count_all`,
            `count_pos`, `logsum_1_p_pos`.

    Notes:
        * Transcripts missing in `refflat_df` or with invalid mapping are skipped.
        * Any unexpected exception within a transcript block is caught and skipped,
          allowing the worker to continue processing subsequent transcripts.
        * A tqdm progress bar is displayed with `leave=False`.
    """
    local_collect = []
    for transcript_df in tqdm(df_list, desc="Converting to genomic coordinates", leave=False):
        if len(transcript_df) == 0:
            continue
        transcript_id = transcript_df["transcript_id"].iloc[0]
        try:
            refflat_row = refflat_df.loc[transcript_id]
        except KeyError:
            continue
        if len(refflat_row) == 0:
            continue

        try:
            exon_starts = refflat_row["exonStarts"]
            exon_ends = refflat_row["exonEnds"]
            strand = refflat_row["strand"]
            chrom = refflat_row["chrom"]

            mapper = TranscriptMapper(exon_starts=np.array(exon_starts), exon_ends=np.array(exon_ends), strand=strand)

            transcript_df["chrom"] = chrom
            transcript_df["strand"] = strand
            transcript_df["pos"] = mapper.map(transcript_df["ref_pos"].to_numpy())

            local_collect.append(transcript_df.dropna())

        except Exception:
            continue

    if len(local_collect) > 0:
        df = pd.concat(local_collect)
        df = df.groupby(["chrom", "strand", "pos"]).agg(
            {
                "kl_div_neg": "sum",
                "kl_div_pos": "sum",
                "count_all": "sum",
                "count_pos": "sum",
                "logsum_1_p_pos": "sum",
            }
        )
        df = df.reset_index()
        collect_list.append(df)

    return None


def load_split_data(data_df, cpu):
    """Group by transcript and split into balanced shards for parallel processing.

    The input is grouped by `transcript_id` (renamed from `ref_names`), sorted by
    descending group size, then distributed across up to `cpu` shards using a
    zig-zag assignment (`0..cpu-1..0`) to balance large and small groups.

    Args:
        data_df (pandas.DataFrame): Input DataFrame containing at least:
            - `ref_names` (str): Transcript identifier per row; will be renamed to
            `transcript_id`.
            - Other columns required downstream (e.g., `ref_pos`, metrics).
        cpu (int): Maximum number of shards (typically the number of worker processes).

    Returns:
        list: A list of length `min(cpu, n_groups)` where each element
        is a list of per-transcript DataFrames to be handled by one worker.

    Notes:
        * Groups are sorted by size to improve load balancing.
        * The zig-zag distribution helps avoid piling all large groups onto early shards.
    """
    data_df = data_df.rename({"ref_names": "transcript_id"}, axis=1)
    data_df = data_df.groupby("transcript_id")
    ## sort by size
    data_df = sorted(data_df, key=lambda x: len(x[1]), reverse=True)

    df_list_split = [[] for _ in range(min(cpu, len(data_df)))]
    for idx, (gene, df) in enumerate(data_df):
        split_idx = idx % (2 * cpu)
        if split_idx >= cpu:
            split_idx = 2 * cpu - split_idx - 1
        df_list_split[split_idx].append(df)

    return df_list_split


def parse_refflat(refflat_path):
    """
    Parse RefFlat/RefGene/GenePred annotation into a normalized DataFrame.

    This function reads a tab-delimited annotation file and normalizes it into a
    common schema with the following columns:
    `transcript_id`, `chrom`, `strand`, `txStart`, `txEnd`,
    `cdsStart`, `cdsEnd`, `exonCount`, `exonStarts`, `exonEnds`.

    It accepts three formats based on column count:
    * 11 columns (RefFlat): drops the first column.
    * 15 columns (RefGene): keeps the first 10 columns.
    * 10 columns (GenePred): uses as-is.

    Exon start/end lists are expected as comma-separated strings with a trailing comma
    (UCSC style) and are converted to `numpy.ndarray` of `int`. Invalid transcripts are
    filtered out based on interval consistency and ordering. The resulting DataFrame is
    indexed by `transcript_id`.

    Args:
        refflat_path (str): Path to the annotation file.

    Returns:
        pandas.DataFrame: Normalized and validated annotation indexed by `transcript_id`.
            Columns:
            - `chrom` (str)
            - `strand` (str; '+' or '-')
            - `txStart`, `txEnd`, `cdsStart`, `cdsEnd` (int)
            - `exonCount` (int)
            - `exonStarts`, `exonEnds` (numpy.ndarray of int; 0-based, half-open)

    Raises:
        ValueError: If the file has an unexpected number of columns or if no valid
            transcripts remain after validation.

    Notes:
        * Validation ensures: `txEnd > txStart`, `cdsEnd >= cdsStart`, `exonCount > 0`,
            `len(exonStarts) == len(exonEnds) == exonCount`, strictly positive exon lengths,
            and non-decreasing starts/ends across exons.
    """
    col_list = [
        "transcript_id",
        "chrom",
        "strand",
        "txStart",
        "txEnd",
        "cdsStart",
        "cdsEnd",
        "exonCount",
        "exonStarts",
        "exonEnds",
    ]
    with open(refflat_path) as infile:
        refflat_df = pd.read_csv(infile, sep="\t", header=None)

    ## check if number of columns is correct (refflat = 11, refgene = 15, genepred = 10)
    n_cols = refflat_df.shape[1]
    if n_cols == 11:
        refflat_df = refflat_df.iloc[:, 1:]
    elif n_cols == 15:
        refflat_df = refflat_df.iloc[:, :10]
    elif n_cols == 10:
        pass
    else:
        raise ValueError(
            "Invalid annotation file format. Expected 10 (GenePred), 11 (RefFlat), or 15 (RefGene) columns."
        )

    refflat_df.columns = col_list

    refflat_df[["txStart", "txEnd", "cdsStart", "cdsEnd"]] = refflat_df[
        ["txStart", "txEnd", "cdsStart", "cdsEnd"]
    ].astype(int)

    ## Filter out invalid entries
    refflat_df = refflat_df[refflat_df["txEnd"] > refflat_df["txStart"]]
    refflat_df = refflat_df[refflat_df["cdsEnd"] >= refflat_df["cdsStart"]]
    refflat_df = refflat_df[refflat_df["exonCount"] > 0]
    refflat_df["exonStarts"] = refflat_df["exonStarts"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))
    refflat_df["exonEnds"] = refflat_df["exonEnds"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))
    refflat_df = refflat_df[refflat_df["exonStarts"].apply(len) == refflat_df["exonCount"]]
    refflat_df = refflat_df[refflat_df["exonEnds"].apply(len) == refflat_df["exonCount"]]
    refflat_df = refflat_df[refflat_df.apply(lambda x: np.all(x["exonEnds"] > x["exonStarts"]), axis=1)]
    refflat_df = refflat_df[refflat_df.apply(lambda x: np.all(x["exonStarts"][1:] >= x["exonStarts"][:-1]), axis=1)]
    refflat_df = refflat_df[refflat_df.apply(lambda x: np.all(x["exonEnds"][1:] >= x["exonEnds"][:-1]), axis=1)]
    refflat_df.drop_duplicates(subset="transcript_id", keep="first", inplace=True, ignore_index=True)

    if len(refflat_df) == 0:
        raise ValueError("No valid transcripts found in the annotation file.")

    refflat_df.set_index("transcript_id", inplace=True)
    return refflat_df


def pileup_genomic(args, input_df):
    """
    Aggregate per-genomic-position metrics using multiprocessing.

    Spawns up to `args.thread` worker processes to convert transcript-relative
    positions to genomic coordinates and aggregate metrics across all input rows.
    Requires an annotation file path at `args.annot`.

    The final output is a DataFrame aggregated by `(chrom, strand, pos)` with
    derived columns:
    * `stoichiometry` = `kl_div_pos` / (`kl_div_neg` + `kl_div_pos`)
    * `modscore` = a modification prediction score.

    Args:
        args (argparse.Namespace): Must contain:
            - `annot` (str or path-like): Path to RefFlat/RefGene/GenePred file.
            - `thread` (int): Number of worker processes to spawn.
        input_df (pandas.DataFrame): Input rows containing at least:
            - `ref_names` (str): Transcript ID; will be renamed to `transcript_id`.
            - `ref_pos` (int): Transcript-relative position (0-based).
            - `kl_div_neg`, `kl_div_pos`, `count_all`, `count_pos`, `logsum_1_p_pos`:
            Metric columns to be summed.

    Returns:
        pandas.DataFrame: Aggregated DataFrame with columns:
            `chrom`, `strand`, `pos`, `modscore`, `stoichiometry`,
            `count_all`, `count_pos`.

    Raises:
        ValueError: If no valid genomic positions are produced (e.g., due to
            mismatched or invalid annotations).

    Notes:
        * Uses a `multiprocessing.Manager().list()` to collect per-process results.
        * `load_split_data` is used to balance workload across processes.
        * The output `pos` is 0-based genomic coordinate (float in intermediate steps,
            but will be integral where valid).
    """
    man = mp.Manager()
    collect_list = man.list()
    proc_list = []
    refflat_df = parse_refflat(args.annot)
    df_list_split = load_split_data(input_df, args.thread)

    for pid, df_list in enumerate(df_list_split):
        proc = mp.Process(target=worker, args=(df_list, refflat_df, collect_list))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()

    collect_list = list(collect_list)
    man.shutdown()

    if len(collect_list) == 0:
        raise ValueError("No valid genomic position found. Please verify the annotation file.")

    df = pd.concat(collect_list)
    gc.collect()

    df = df.groupby(["chrom", "strand", "pos"]).agg(
        {
            "kl_div_neg": "sum",
            "kl_div_pos": "sum",
            "count_all": "sum",
            "logsum_1_p_pos": "sum",
            "count_pos": "sum",
        }
    )

    df["stoichiometry"] = df["kl_div_pos"] / (df["kl_div_neg"] + df["kl_div_pos"])
    df["modscore"] = -(2 - df["stoichiometry"]) * df["logsum_1_p_pos"] / df["count_all"] + (
        (1 - df["stoichiometry"]) * np.log10(np.clip(1 - df["stoichiometry"], 1e-30, 1))
        + df["stoichiometry"] * np.log10(np.clip(df["stoichiometry"], 1e-30, 1))
    ) * (df["count_pos"] / df["count_all"])

    df = df.reset_index()
    df = df[["chrom", "strand", "pos", "modscore", "stoichiometry", "count_all", "count_pos"]]
    return df
