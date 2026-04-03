"""Utilities for mapping transcript-relative coordinates to genomic coordinates."""

import gc
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm


class TranscriptMapper:
    """Vectorized mapper from transcript coordinates to genomic coordinates."""

    def __init__(self, exon_starts: np.ndarray, exon_ends: np.ndarray, strand: str):
        self.starts = exon_starts
        self.ends = exon_ends
        self.lengths = self.ends - self.starts
        self.total_len = int(self.lengths.sum())
        self.strand = strand

        if self.strand == "+":
            self.cumsum = np.concatenate(([0], np.cumsum(self.lengths)))
        elif self.strand == "-":
            self.cumsum = np.concatenate(([0], np.cumsum(self.lengths[::-1])))
        else:
            raise ValueError("strand must be '+' or '-'.")

    def map(self, coords: np.ndarray) -> np.ndarray:
        out = np.full(coords.shape, np.nan, dtype=np.float64)
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
    """Worker that maps transcript to genomic positions and aggregates metrics."""
    local_collect = []
    for transcript_df in tqdm(df_list, desc="Converting to genomic coordinates", leave=False):
        if len(transcript_df) == 0:
            continue

        transcript_id = transcript_df["transcript_id"].iloc[0]
        try:
            refflat_row = refflat_df.loc[transcript_id]
        except KeyError:
            continue

        exon_starts = np.asarray(refflat_row["exonStarts"], dtype=np.int64)
        exon_ends = np.asarray(refflat_row["exonEnds"], dtype=np.int64)
        strand = refflat_row["strand"]
        chrom = refflat_row["chrom"]

        mapper = TranscriptMapper(exon_starts=exon_starts, exon_ends=exon_ends, strand=strand)
        mapped_df = transcript_df.copy()
        mapped_df["chrom"] = chrom
        mapped_df["strand"] = strand
        mapped_df["pos"] = mapper.map(mapped_df["ref_pos"].to_numpy())
        mapped_df = mapped_df.dropna(subset=["pos"])
        if len(mapped_df) == 0:
            continue
        mapped_df["pos"] = mapped_df["pos"].astype(np.int64)
        local_collect.append(mapped_df)

    if local_collect:
        df = pd.concat(local_collect, ignore_index=True)
        df = df.groupby(["chrom", "strand", "pos"], as_index=False).agg(
            {
                "kl_div_neg": "sum",
                "kl_div_pos": "sum",
                "count_all": "sum",
                "count_pos": "sum",
                "logsum_1_p_pos": "sum",
            }
        )
        collect_list.append(df)
    return None


def load_split_data(data_df, cpu):
    """Group by transcript and split into balanced shards for parallel processing."""
    if cpu <= 0:
        raise ValueError("cpu must be positive")

    data_df = data_df.rename({"ref_names": "transcript_id"}, axis=1)
    grouped = sorted(data_df.groupby("transcript_id"), key=lambda x: len(x[1]), reverse=True)
    if len(grouped) == 0:
        return []

    n_shards = min(cpu, len(grouped))
    df_list_split = [[] for _ in range(n_shards)]
    cycle = max(1, n_shards)
    for idx, (_, df) in enumerate(grouped):
        split_idx = idx % (2 * cycle)
        if split_idx >= cycle:
            split_idx = 2 * cycle - split_idx - 1
        df_list_split[split_idx].append(df)
    return df_list_split


def _parse_exon_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    text = str(value).strip()
    if text.endswith(","):
        text = text[:-1]
    if text == "":
        return np.array([], dtype=np.int64)
    return np.fromstring(text, sep=",", dtype=np.int64)


def parse_refflat(refflat_path):
    """Parse RefFlat/RefGene/GenePred annotation into a normalized DataFrame."""
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
    refflat_df = pd.read_csv(refflat_path, sep="\t", header=None)

    n_cols = refflat_df.shape[1]
    if n_cols == 11:
        refflat_df = refflat_df.iloc[:, 1:]
    elif n_cols == 15:
        refflat_df = refflat_df.iloc[:, :10]
    elif n_cols != 10:
        raise ValueError(
            "Invalid annotation file format. Expected 10 (GenePred), 11 (RefFlat), or 15 (RefGene) columns."
        )

    refflat_df.columns = col_list
    refflat_df[["txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount"]] = refflat_df[
        ["txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount"]
    ].astype(np.int64)

    refflat_df = refflat_df[refflat_df["txEnd"] > refflat_df["txStart"]]
    refflat_df = refflat_df[refflat_df["cdsEnd"] >= refflat_df["cdsStart"]]
    refflat_df = refflat_df[refflat_df["exonCount"] > 0]
    refflat_df["exonStarts"] = refflat_df["exonStarts"].apply(_parse_exon_array)
    refflat_df["exonEnds"] = refflat_df["exonEnds"].apply(_parse_exon_array)
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
    """Aggregate per-genomic-position metrics using multiprocessing."""
    if len(input_df) == 0:
        raise ValueError("input_df is empty. No transcript pileup entries were provided.")

    n_proc = getattr(args, "thread", None)
    if n_proc is None:
        n_proc = max(1, int(0.95 * mp.cpu_count()))
    if n_proc <= 0:
        raise ValueError("args.thread must be a positive integer.")

    refflat_df = parse_refflat(args.annot)
    df_list_split = load_split_data(input_df, n_proc)
    if len(df_list_split) == 0:
        raise ValueError("No transcript groups were available for genomic aggregation.")

    man = mp.Manager()
    collect_list = man.list()
    proc_list = []

    for df_list in df_list_split:
        proc = mp.Process(target=worker, args=(df_list, refflat_df, collect_list))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Genomic pileup worker exited with code {proc.exitcode}.")

    collect_list = list(collect_list)
    man.shutdown()

    if len(collect_list) == 0:
        raise ValueError("No valid genomic position found. Please verify the annotation file.")

    df = pd.concat(collect_list, ignore_index=True)
    gc.collect()

    df = df.groupby(["chrom", "strand", "pos"], as_index=False).agg(
        {
            "kl_div_neg": "sum",
            "kl_div_pos": "sum",
            "count_all": "sum",
            "logsum_1_p_pos": "sum",
            "count_pos": "sum",
        }
    )

    epsilon = getattr(args, "epsilon", 1e-30)
    digitization = 1000
    df["stoichiometry"] = df["kl_div_pos"] / (df["kl_div_neg"] + df["kl_div_pos"] + epsilon)
    df["modscore"] = 1 - np.power(
        10, df["logsum_1_p_pos"] / df["count_all"] * (1 + np.power(10, 2 * (df["stoichiometry"] - 1)))
    )
    df["modscore"] = np.digitize(df["modscore"], np.linspace(0, 1, digitization + 1), right=True) / digitization
    df["stoichiometry"] = df["stoichiometry"] * (
        (np.log10(1 - args.threshold) * df["stoichiometry"]) > (df["logsum_1_p_pos"] / df["count_all"])
    )
    df = df[["chrom", "strand", "pos", "modscore", "stoichiometry", "count_all", "count_pos"]]
    df["pos"] = df["pos"].astype(np.int64)
    return df
