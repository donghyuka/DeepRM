import argparse
import gc
import glob
import multiprocessing as mp
import os
import pickle
import sys
import numpy as np
import pandas as pd
import pod5
import pysam
import atexit
import tqdm
from utils.utils import oom_killer, printmessage
import toml

## Warning: This script has heavy parallel I/O operations and large memory usage.
## > 1TB Read / Write operations and > 100 GB RAM usage is expected (> 4GB/s disk write was observed).
## Running this on NFS may cause significant performance degradation.



def extract_move(bam_path, ncpu):
    ## Extract mv tag from bam and save to separate file
    data_dict = {}
    count = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:

                if read.has_tag("pi"):
                    read_id = str(read.get_tag("pi"))
                else:
                    read_id = str(read.query_name)

                if not read.has_tag("mv"):
                    continue

                data_dict[read_id] = np.array(read.query_qualities, dtype=int)

                pbar.update(1)
                count += 1

    printmessage(f"Valid read count: {count}", msg_type="info")

    return data_dict

def sample_block_df(block_df_path):
    df = pd.read_pickle(block_df_path)
    df["5mer"] = df["motif"].str[8:13]
    df = df[df["penalty"] == 0]
    df = df.groupby(["5mer","cb_idx"])
    df_list = []
    for name, group in df:
        df_list.append(group.sample(min(2000, len(group))))
    df = pd.concat(df_list)
    return df

def main():
    args = parse_args()
    bam_dict = extract_move(args.bam, args.ncpu)
    block_df = sample_block_df(args.block)
    block_df["bq"] = block_df.apply(lambda x: bam_dict.get(x["read_id"], None), axis=1)
    block_df = block_df.dropna()
    block_df["bq"] = block_df.apply(lambda x: x["bq"][x["start_pos"]-3:x["end_pos"]+3], axis=1)
    block_df.to_pickle(args.output)
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Extract BQ from BAM and merge with block_df")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Path to BAM file")
    parser.add_argument("--block", "-k", type=str, required=True, help="Path to block_df")
    parser.add_argument("--output", "-o", type=str, default = None, help="Path to output file")
    parser.add_argument("--ncpu", "-c", type=int, default=120, help="Number of CPUs to use")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.block+".bqextended.sampled.pkl"
    return args


if __name__ == "__main__":
    main()
