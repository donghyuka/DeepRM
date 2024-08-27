import os
import glob
from tqdm import tqdm
import multiprocessing as mp
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run a command on all files in a directory")
    parser.add_argument("--input", "-i", type=str, help="Input directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--threads", "-t", type=int, default=1, help="Number of threads")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    for setname in ["train", "val"]:
        os.makedirs(os.path.join(args.output, setname), exist_ok=True)
        for classname in ["pos", "neg"]:
            os.makedirs(os.path.join(args.output, setname, classname), exist_ok=True)

    files = (glob.glob(os.path.join(args.input, "*/*/*.pkl")))
    print(f"Found {len(files)} files")
    files = np.array_split(files, args.threads)
    proc_list = []
    for i in range(args.threads):
        p = mp.Process(target=process_files, args=(files[i], args.output))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()
    print("All done")

    return None


def process_files(files, output):
    for file in tqdm(files):
        df = pd.read_pickle(file)
        df.drop(columns=["bq_token"], inplace=True)
        df["segment_len_arr"] = df["segment_len_arr"].apply(lambda x: x.astype(np.uint8))
        df["signal_token"] = df["signal_token"].apply(lambda x: x.astype(np.float32))
        df["dwell_token"] = df["dwell_token"].apply(lambda x: x[2:-2].astype(np.float32))
        filename_split = file.split("/")
        setname = filename_split[-3]
        classname = filename_split[-2]
        filename = filename_split[-1]
        df.to_pickle(os.path.join(output, setname, classname, filename))
    return None


if __name__ == "__main__":
    main()
