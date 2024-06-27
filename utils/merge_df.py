import multiprocessing as mp
import os
import pandas as pd
import tqdm
import argparse
import glob
import numpy as np
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", "-c", type=int, default=None, help="Number of CPUs to use")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    args = parser.parse_args()
    if args.cpu is None:
        args.cpu = int (0.9 * mp.cpu_count())
    os.makedirs(args.output, exist_ok=True)
    return args


def worker(pid, file_paths, out_path):
    worker_buffer = []
    for path in tqdm.tqdm(file_paths):
        data = pd.read_csv(path, sep="\t")
        worker_buffer.append(data)
    worker_buffer = pd.concat(worker_buffer, axis=0)
    worker_buffer.to_pickle(f"{out_path}/inference_{pid}.pkl")
    del worker_buffer
    gc.collect()
    return None


def main():
    args = parse_args()
    file_paths = glob.glob(f"{args.input}/*.tsv")
    proc_list = []
    file_paths_split = np.array_split(file_paths, args.cpu)

    for pid, file_paths in enumerate(file_paths_split):
        proc = mp.Process(target=worker, args=(pid, file_paths, args.output))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    gc.collect()
    return None


if __name__ == "__main__":
    main()