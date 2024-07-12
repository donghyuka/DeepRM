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
    args.add_argument("--input", "-i", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240626-093708-12-215000-token_normalise_drach_d30_normalise_metadata", help="Source path")
    args.add_argument("--realign", "-r", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/realign_meta/output", help="Source path")
    args.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v14", help="Output path")
    args.add_argument("--min_depth", "-m", type=int, default=5, help="Minimum depth")
    args.add_argument("--max_depth", "-x", type=int, default=20, help="Maximum depth")
    args.add_argument("--neg", "-n", type=float, default=1.0, help="Negative-to-positive ratio")
    args = args.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args

def main():
    args = parse_args()
    #
    # input_files = glob.glob(f"{args.input}/*.pkl")
    # source_df = pd.concat([pd.read_pickle(f) for f in input_files])
    # source_df["read_id"] = source_df["block_id"].apply(lambda x: x.split(":")[0])
    # source_df["block_label_id"] = source_df["read_id"] + ":" + source_df["label_id"]
    #
    # realign_files = glob.glob(f"{args.realign}/*.pkl")
    # realign_df = pd.concat([pd.read_pickle(f) for f in realign_files])
    #
    # print(source_df)
    # print(realign_df)
    # realign_df = realign_df.rename(columns={"error": "realigned_error"})
    # source_df = source_df.merge(realign_df, on = "block_label_id", how = "inner")
    # source_df.drop(columns=["block_label_id"], inplace=True)
    #
    # print(source_df)
    #
    # cols_list = ["motif", "mapq", "flag", "pi", "read_bq", "block_bq", "base_bq", "error", "query_pos",
    #              "query_len", "left_soft_clip"]
    # for col in cols_list:
    #     source_df[col] = source_df["metadata"].apply(lambda x: x[col])
    # source_df.drop(columns=["metadata"], inplace=True)
    # source_df = source_df[source_df["label"] >= -1]
    #
    # print(source_df)
    # print(source_df["left_soft_clip"].describe())
    #
    # source_df["flag"] = source_df["flag"] > 0
    # source_df["flag"] = source_df["flag"].astype(int)
    # source_df["mapq"] = np.clip(source_df["mapq"] / 60, 0, 1)
    # source_df[["read_bq","block_bq","base_bq"]] = np.clip(source_df[["read_bq","block_bq","base_bq"]] / 40, 0, 1)
    # source_df[["query_pos","query_len"]] = np.clip(source_df[["query_pos","query_len"]] / 5000, 0, 1)
    # source_df["left_soft_clip"] = np.clip(source_df["left_soft_clip"] / 100, 0, 1)
    # source_df = source_df.sort_values("pred", ascending=False)
    #
    # source_df.to_pickle(f"{args.output}/source.pkl")

    source_df = pd.read_pickle(f"{args.output}/source.pkl")

    label_id_unique = source_df["label_id"].unique()
    label_id_unique = np.random.permutation(label_id_unique)
    source_df = source_df.groupby("label_id")

    main_df_dict = {}
    extra_df_dict = {}
    neg_df_dict = {}

    for label_id in tqdm.tqdm(label_id_unique):
        undersample_flag = False
        df = source_df.get_group(label_id)
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

        df = df.dropna(axis=0).reset_index(drop=True).copy()
        if len(df) == 0:
            continue

        if label == 0:
            neg_df_dict[label_id] = df

        else:
            if label == 1 and not undersample_flag:
                main_df_dict[label_id] = df
            else:
                extra_df_dict[label_id] = df

    print(f"Pos_Main: {len(main_df_dict)}")
    print(f"Pos_Extra: {len(extra_df_dict)}")
    print(f"Neg: {len(neg_df_dict)}")

    ## split train, val, test
    ## 50% of main goes to test, 10% of main goes to val, 40% of main goes to train
    ## 100% of extra goes to train

    train_df_dict = {}
    val_df_dict = {}
    test_df_dict = {}

    main_label_id_unique = list(main_df_dict.keys())
    main_label_id_unique = np.random.permutation(main_label_id_unique)

    for i, label_id in enumerate(main_label_id_unique):
        df = main_df_dict[label_id]
        if i < len(main_label_id_unique) * 0.1:
            val_df_dict[label_id] = df
        elif i < len(main_label_id_unique) * 0.5:
            train_df_dict[label_id] = df
        else:
            test_df_dict[label_id] = df

    for label_id in extra_df_dict.keys():
        train_df_dict[label_id] = extra_df_dict[label_id]

    ## sample negative
    neg_label_id_unique = list(neg_df_dict.keys())
    neg_label_id_unique = np.random.permutation(neg_label_id_unique)[:int(len(train_df_dict) * args.neg)]
    for label_id in neg_label_id_unique:
        train_df_dict[label_id] = neg_df_dict[label_id]

    with open(f"{args.output}/train.pkl", "wb") as f:
        pickle.dump(train_df_dict, f)
    with open(f"{args.output}/val.pkl", "wb") as f:
        pickle.dump(val_df_dict, f)
    with open(f"{args.output}/test.pkl", "wb") as f:
        pickle.dump(test_df_dict, f)

    print(f"Train: {len(train_df_dict)}")
    print(f"Val: {len(val_df_dict)}")
    print(f"Test: {len(test_df_dict)}")

    return None


if __name__ == "__main__":
    main()