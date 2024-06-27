import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm as tqdm
import argparse
import os


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input", "-i", type=str, required=True, help="Input path")
    args.add_argument("--output", "-o", type=str, required=True, help="Output path")
    args.add_argument("--threads", "-t", type=int, default=int(os.cpu_count()*0.9), help="Number of threads")
    args = args.parse_args()
    return args

def main():
    args = parse_args()
    ## split the tsv into equal parts
    temp_dir = os.path.dirname(args.input)+"/temp"
    os.makedirs(temp_dir, exist_ok=True)

    ## split the tsv into equal parts
    os.system(f"split -n l/{args.threads} -d --additional-suffix .tsv {args.input} {temp_dir}/")

    proc_list = []
    for i in range(args.threads):
        proc = mp.Process(target=process_f5c_file, args=(f"{temp_dir}/xaa{i}", f"{args.output}/xaa{i}"))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    return None


def process_f5c_file(input_path, output_path):