import pod5
import multiprocessing as mp
import argparse
import os
from tqdm import tqdm
import glob
import numpy as np


def trim_one_pod5(in_path, out_dir, trim, min_len):
    out_path = str(os.path.join(out_dir, os.path.basename(in_path)))
    with pod5.Reader(in_path) as reader:
        with pod5.Writer(out_path) as writer:
            for record in reader:
                signal = record.signal
                if len(signal) < (min_len + trim):
                    continue
                read = record.to_read()
                read.signal = read.signal[trim:]
                writer.add_read(read)

    return None


def trim_pod5_worker(in_files, out_dir, trim, min_len):
    for in_path in tqdm(in_files):
        trim_one_pod5(in_path, out_dir, trim, min_len)
    return None


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cpu", "-c", type=int, default=int(mp.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--input", "-i", type=str, required=True, help="Input path")
    args.add_argument("--output", "-o", type=str, required=True, help="Output path")
    args.add_argument("--trim", "-t", type=int, default=10000, help="Trimming size")
    args.add_argument("--min_len", "-m", type=int, default=10000, help="Minimum length")
    args = args.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    file_list = glob.glob(f"{args.input}/*.pod5")
    ncpu = min(len(file_list), args.cpu)
    split_file_list = np.array_split(file_list, ncpu)
    proc_list = []
    for i in range(ncpu):
        p = mp.Process(target=trim_pod5_worker, args=(split_file_list[i], args.output, args.trim, args.min_len))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()
    return None


if __name__ == "__main__":
    main()

