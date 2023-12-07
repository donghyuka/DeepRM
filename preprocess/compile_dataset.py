## Usage: "python compile_dataset.py --cpu {args.thread} --signal {signal_path} --block {block_df_path} --output {args.output}"

import os, glob, tqdm, gc, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd

def extract_segment_stats(signal_df_path_arr, seg_df_path, block_df):

    for signal_df_path in tqdm.tqdm(signal_df_path_arr):
        signal_df = pd.read_pickle(signal_df_path)
        block_df_proc = block_df.merge(signal_df, left_on="id", right_on="id", how="inner")
        block_df_proc.to_pickle(f"{seg_df_path}/segmented_{os.path.basename(signal_df_path)}")
        del signal_df, block_df, block_df_proc
        gc.collect()

    return None

def parse_args():
    pass

def main():
    pass

if __name__ == "__main__":
    main()