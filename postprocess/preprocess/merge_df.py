import argparse
import os
import glob
import pandas as pd
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_files = glob.glob(f"{args.input}/*.pkl")
    df = pd.concat([pd.read_pickle(f) for f in input_files])
    print(df)
    cols_list = ["motif", "mapq", "flag", "pi", "read_bq", "block_bq", "base_bq", "error", "query_pos", "query_len", "left_soft_clip"]
    for col in cols_list:
        df[col] = df["metadata"].apply(lambda x: x[col])
    df.drop(columns=["metadata"], inplace=True)
    df = df[df["label"] >= -1]

    print(df)
    print(df["left_soft_clip"].describe())

    df = df.groupby("label_id")
    ## convert to dict
    df_dict = {}
    for label_id, group in df:
        if len(group) > 20 or len(group) < 5:
            continue
        group = group.sort_values("pred", ascending=False)
        df_dict[label_id] = group
    with open(args.output, "wb") as f:
        pickle.dump(df_dict, f)

    return None

if __name__ == "__main__":
    main()