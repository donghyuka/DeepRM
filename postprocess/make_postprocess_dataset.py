import argparse
import pandas as pd
import numpy as np
import os
import tqdm
import pickle
import glob
import gc


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input", "-i", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v5_genome_drach", help="Source path")
    args.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v14", help="Output path")
    args.add_argument("--min_depth", "-m", type=int, default=5, help="Minimum depth")
    args.add_argument("--max_depth", "-x", type=int, default=20, help="Maximum depth")
    args = args.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args

def main():
    args = parse_args()
    input_files = glob.glob(f"{args.input}/*.pkl")
    source_df = pd.concat([pd.read_pickle(f) for f in input_files])
    print(source_df)
    cols_list = ["motif", "mapq", "flag", "pi", "read_bq", "block_bq", "base_bq", "error", "query_pos",
                 "query_len", "left_soft_clip"]
    for col in cols_list:
        source_df[col] = source_df["metadata"].apply(lambda x: x[col])
    source_df.drop(columns=["metadata"], inplace=True)
    source_df = source_df[source_df["label"] >= -1]
    source_df = source_df[source_df["label"] != 0].copy()

    print(source_df)
    print(source_df["left_soft_clip"].describe())

    source_df["flag"] = source_df["flag"] > 0
    source_df["flag"] = source_df["flag"].astype(int)
    source_df["mapq"] = np.clip(source_df["mapq"] / 60, 0, 1)
    source_df[["read_bq","block_bq","base_bq"]] = np.clip(source_df[["read_bq","block_bq","base_bq"]] / 40, 0, 1)
    source_df[["query_pos","query_len"]] = np.clip(source_df[["query_pos","query_len"]] / 4000, 0, 1)
    source_df["left_soft_clip"] = np.clip(source_df["left_soft_clip"] / 100, 0, 1)
    source_df = source_df.sort_values("pred", ascending=False)

    source_df.to_pickle(f"{args.output}/source.pkl")

    source_df = source_df.groupby("label_id")
    label_id_unique = source_df["label_id"].unique()
    label_id_unique = np.random.permutation(label_id_unique)

    train_df_dict = {}
    val_df_dict = {}
    test_df_dict = {}

    idx = 0

    for label_id in tqdm.tqdm(label_id_unique):
        try:
            df = source_df.get_group(label_id)
        except:
            continue

        print(df)

        undersample_flag = False

        label = df["label"].iloc[0]


        depth = len(df)

        if depth < args.min_depth:
            continue

        m6a_level = df["dom"].iloc[0]

        if depth > args.max_depth:
            df = df.sample(args.max_depth)
            df = df.sort_values("pred", ascending=False)
            depth = args.max_depth
            undersample_flag = True

        df = df.copy()

        df["label"] = label
        df["m6a_level"] = m6a_level
        df["depth"] = depth / args.max_depth

        dom_pred = df["pred"][(df["pred"] <= 0.2) | (df["pred"] >= 0.6)]
        dom_pred = dom_pred > 0.5
        dom_pred = dom_pred.mean()

        df["dom_pred"] = dom_pred

        if idx % 10 <= 0:
            val_df_dict[label_id] = df
        elif idx % 10 <= 6:
            if (not undersample_flag) and (label == 1):
                test_df_dict[label_id] = df
            else:
                train_df_dict[label_id] = df
        else:
            train_df_dict[label_id] = df

        idx += 1


    print(f"Train: {len(train_df_dict)}")
    print(f"Val: {len(val_df_dict)}")
    print(f"Test: {len(test_df_dict)}")

    with open(f"{args.output}/train.pkl", "wb") as f:
        pickle.dump(train_df_dict, f)
    with open(f"{args.output}/val.pkl", "wb") as f:
        pickle.dump(val_df_dict, f)
    with open(f"{args.output}/test.pkl", "wb") as f:
        pickle.dump(test_df_dict, f)

    return None


if __name__ == "__main__":
    main()