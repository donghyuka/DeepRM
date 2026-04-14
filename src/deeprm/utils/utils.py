"""
Utility functions for DeepRM.
"""

import numpy as np


def mean_phred(phred):
    """
    Calculates the mean Phred quality score.

    Args:
        phred (numpy.ndarray): Array or list of Phred quality scores.

    Returns:
        float: Mean Phred quality score.
    """
    if not isinstance(phred, np.ndarray):
        phred = np.array(phred, dtype=int)
    else:
        phred = phred.astype(int)
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))


def maybe_index_bam(bam_path: str, threads: int = 1) -> str:
    """
    Ensure a BAM is indexed. If indexing fails because the BAM is not coordinate-sorted,
    write a coordinate-sorted copy (<stem>.sorted[.N].bam), index it, and return that path.

    Returns:
        Path (as str) to the BAM that is guaranteed to be indexed (original or sorted copy).
    """
    from pathlib import Path

    import psutil
    import pysam

    bam_path = Path(bam_path)
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM not found: {bam_path}")
    if bam_path.suffix.lower() != ".bam":
        raise ValueError(f"Expected a .bam file, got: {bam_path.name}")

    def _has_usable_index(p) -> bool:
        try:
            with pysam.AlignmentFile(str(p), "rb", check_sq=False) as bam:
                # has_index(): index file discoverable; check_index(): validate it matches & is readable
                return bool(bam.has_index()) and bool(bam.check_index())
        except Exception:
            return False

    # Fast path: already indexed & usable
    if _has_usable_index(bam_path):
        return str(bam_path)

    # Before attempting to index, ensure the BAM has @SQ. Without it, indexing is not possible.
    try:
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            sq = bam.header.get("SQ", None)
            if not sq:
                raise ValueError(f"{bam_path} has no @SQ lines in the header; cannot build a BAM index.")
    except Exception as e:
        # Re-raise with a clearer message
        raise RuntimeError(f"Failed to read BAM header: {bam_path}") from e

    # Try indexing the BAM as-is. If it fails due to unsortedness, fall back to sorting.
    try:
        pysam.index(str(bam_path), nthreads=threads)
        if _has_usable_index(bam_path):
            return str(bam_path)
        # If indexing "succeeded" but index still isn't usable, treat as failure.
        raise RuntimeError("Index creation did not produce a usable index.")
    except Exception as e:
        err_msg = str(e).lower()
        print(f"Indexing failed for {bam_path}: {err_msg}")

    # Sort to a new file, then index that.
    out_base = bam_path.with_suffix("")  # strip .bam
    sorted_path = out_base.with_name(out_base.name + ".sorted").with_suffix(".bam")
    if sorted_path.exists():
        # Make a unique name: .sorted.1.bam, .sorted.2.bam, ...
        maxretcount = 100
        i = 1
        while i <= maxretcount:
            candidate = out_base.with_name(f"{out_base.name}.sorted.{i}").with_suffix(".bam")
            if not candidate.exists():
                sorted_path = candidate
                break
            i += 1

    pysam_thread_memory = max(1, int(0.5 * psutil.virtual_memory().available / (1024**2) / threads))
    pysam.sort("-@", str(threads), f"-m {pysam_thread_memory}M", "-o", str(sorted_path), str(bam_path))
    pysam.index("-@", str(threads), str(sorted_path))

    if not _has_usable_index(sorted_path):
        raise RuntimeError(f"Sorted BAM was created but index is not usable: {sorted_path}")

    return str(sorted_path)
