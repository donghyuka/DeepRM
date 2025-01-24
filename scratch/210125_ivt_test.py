import pysam
import tqdm
from utils.utils import mean_phred
import numpy as np

def main():
    min_query_length = 300
    max_query_length = 1000
    min_mapped_length = 300
    max_mapped_length = 1000
    min_mapq = 30
    min_bq = 7

    passed = 0
    failed = 0

    with pysam.AlignmentFile("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_IVT/ON0095/ON0095/result/pre_070/intermediates/dorado_output.aligned.filtered.sorted.pentamer.bam", "rb") as bam:
        with tqdm.tqdm(total=bam.mapped+bam.unmapped) as pbar:
            for read in bam:
                pbar.update(1)
                pbar.set_postfix(passed=passed, failed=failed)
                ## Filter out secondary, supplementary, unmapped, and reverse reads
                if not read.is_secondary and not read.is_supplementary and not read.is_unmapped and not read.is_reverse:
                    if read.query_length >= min_query_length and read.query_length <= max_query_length:
                        if read.mapping_quality >= min_mapq:
                            if read.query_alignment_length >= min_mapped_length and read.query_alignment_length <= max_mapped_length:
                                qscore = mean_phred(np.array(read.query_qualities, dtype=int))
                                if qscore >= min_bq:
                                    passed += 1
                                    continue
                failed += 1

    print(f"Passed: {passed:,}, Failed: {failed:,}")


if __name__ == "__main__":
    main()
