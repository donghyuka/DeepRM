import pandas as pd
import numpy as np
import argparse
import multiprocessing as mp
import gc
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input pileup file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output pileup file")
    parser.add_argument("--min_depth", "-m", type=int, default=5, help="Minimum depth")
    parser.add_argument("--max_depth", "-x", type=int, default=None, help="Maximum depth")
    parser.add_argument("--base", "-b", type=str, nargs="+", default=["A"], help="Base to select")
    parser.add_argument("--cpu", "-c",  type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args = parser.parse_args()
    return args


def filter_pileup(args):
    if args.input.endswith(".tsv"):
        df = pd.read_csv(args.input, sep="\t", header=None, quoting=3, engine="c")
    elif args.input.endswith(".pkl"):
        df = pd.read_pickle(args.input)
    else:
        raise ValueError("Input file must be either .tsv or .pkl")
    if args.input.endswith(".pkl"):
        df.to_pickle(args.input.replace(".tsv",".pkl"))
    ## Create columns
    df.columns = ["ref","pos","base","depth","align","qual"]
    # df["depth"] = df["align"].str.count("\\.")
    df = df[["ref","pos","base","depth"]]
    print(df)
    df_list_split = np.array_split(df, args.cpu)
    del df
    gc.collect()

    proc_list = []
    man = mp.Manager()
    collect_dict = man.dict()

    for pid, df_proc in enumerate(df_list_split):
        proc = mp.Process(target=filter_pileup_proc, args=(args, df_proc, collect_dict, pid))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    processed_df_list = []
    process_later_df_list = []
    for pid in collect_dict.keys():
        df, process_later_df = collect_dict[pid]
        processed_df_list.append(df)
        process_later_df_list.append(process_later_df)

    process_later_df = pd.concat(process_later_df_list, ignore_index=True)
    df_list = [group[1] for group in process_later_df.groupby("ref", sort=False)]
    df_list = [group for group in df_list if group["depth"].max() >= args.min_depth]
    local_list = []

    for df in df_list:
        df = process_df(df, args)
        if df is None:
            continue
        local_list.append(df)
        gc.collect()

    process_later_df = pd.concat(local_list, ignore_index=True)
    processed_df_list.append(process_later_df)
    df = pd.concat(processed_df_list, ignore_index=True)
    df.to_pickle(args.output)
    print(df)
    return None


def filter_pileup_proc(args, df_proc, collect_dict, pid):
    local_list = []

    df_list = [group[1] for group in df_proc.groupby("ref", sort=False)]
    first_df = df_list[0]
    last_df = df_list[-1]
    df_list = df_list[1:-1]
    process_later_df = pd.concat([first_df,last_df], ignore_index=True)
    del df_proc
    gc.collect()
    ## Filter out if depth of all bases is less than min_depth
    df_list = [group for group in df_list if group["depth"].max() >= args.min_depth]

    for df in df_list:
        df = process_df(df, args)
        if df is None:
            continue
        local_list.append(df)
        gc.collect()
    df = pd.concat(local_list, ignore_index=True)
    collect_dict[pid] = (df, process_later_df)
    return None


def process_df(df, args):
    df = df.reset_index(drop=True)
    if len(df) < 5:
        return None
    ## Create 5-mer motif
    for shift in [-2,-1,1,2]:
        df[f"base_{shift}"] = df["base"].shift(-shift, fill_value="N")
    df = df.iloc[2:-2]
    df["5mer"] = df["base_-2"] + df["base_-1"] + df["base"] + df["base_1"] + df["base_2"]
    df.drop(["base_-2","base_-1","base_1","base_2"], axis=1, inplace=True)
    if len(args.base) == 1:
        df = df[df["base"] == args.base[0]]
    else:
        df = df[df["base"].isin(args.base)]
    if args.max_depth is not None:
        df = df[(df["depth"] >= args.min_depth) & (df["depth"] <= args.max_depth)].copy()
    else:
        df = df[df["depth"] >= args.min_depth].copy()
    return df


def main():
    args = parse_args()
    filter_pileup(args)
    return None

if __name__ == "__main__":
    main()

