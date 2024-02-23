import pysam
import numpy as np
import tqdm
from utils.utils import mean_phred


def extract_move(bam_path,ncpu=1,bq_cutoff=0):
    ## Extract mv tag from bam and save to separate file
    id_list = []

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped) as pbar:
            for read in input_bam:

                if read.has_tag("pi"):
                    continue
                if mean_phred(np.array(read.query_qualities, dtype=int)) < bq_cutoff:
                    continue
                print(str(read.query_name))
                id_list.append(str(read.query_name))
                pbar.update(1)

                ## break at 10000
                if pbar.n == 100000:
                    break


    return id_list


if __name__ == "__main__":
    set1 = extract_move("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/old/result/alignstats/dorado_output.aligned.fltered.bam")
    set2 = extract_move("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/old/result/dorado/intermediates/dorado_output.bam")
    set1 = set(set1)
    set2 = set(set2)
    intersection = set1.intersection(set2)
    print(len(set1))
    print(len(set2))
    print(len(intersection))