import numpy as np
from tqdm import tqdm
import RNA
import argparse
import multiprocessing as mp
import os
import pandas as pd
import glob

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input", "-i", type=str, required=True, help="Input path")
    args.add_argument("--output", "-o", type=str, required=True, help="Output path")
    args.add_argument("--threads", "-t", type=int, default=120, help="Number of threads")
    args = args.parse_args()
    return args


def seq_to_bpp(seq):
    fc = RNA.fold_compound(seq)
    fc.pf()
    bpp = np.array(fc.bpp())
    return bpp

def seq_to_mfe(seq):
    mfe = RNA.fold(seq)
    return mfe[1]

def process_df(path_list, output):
    for path in tqdm(path_list, desc="Calculating MFE"):
        data = pd.read_pickle(path)
        # data["bpp"] = data["kmer_token"].apply(seq_to_bpp)
        data["mfe"] = data["kmer_token"].apply(seq_to_mfe)
        data.to_pickle(f"{output}/{path.split('/')[-3]}/{path.split('/')[-2]}/{path.split('/')[-1]}")
    return None


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    setname = ["train", "val"]
    groupname = ["pos", "neg"]
    for s in setname:
        for g in groupname:
            os.makedirs(f"{args.output}/{s}/{g}", exist_ok=True)

    path_list = list(glob.glob(f"{args.input}/*/*/*.pkl"))
    split_path_list = np.array_split(path_list, args.threads)
    proc_list = []
    for i, path_list in enumerate(split_path_list):
        proc = mp.Process(target=process_df, args=(path_list, args.output))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return None


if __name__ == "__main__":
    main()