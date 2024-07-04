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
    args.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV4.GP3.depth5_None.twm6astrict.notsampled.drach.tsv", help="Source path")
    args.add_argument("--source", "-s", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v5_genome_drach", help="Source path")
    args.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v11", help="Output path")
    args.add_argument("--min_depth", "-m", type=int, default=5, help="Minimum depth")
    args.add_argument("--max_depth", "-x", type=int, default=20, help="Maximum depth")
    args = args.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args

def main():
    args = parse_args()
    # source_df = []
    # for path in tqdm.tqdm(glob.glob(f"{args.source}/*.pkl")):
    #     x = pd.read_pickle(path)
    #     x.drop(columns=["genome_id", "block_id"], inplace=True)
    #     source_df.append(x)
    # source_df = pd.concat(source_df, axis=0)
    # del x
    # gc.collect()
    #
    # def explode_dict(dict):
    #     return_list = [dict["read_bq"], dict["block_bq"], dict["base_bq"], dict["error"], dict["pi"], dict["flag"], dict["mapq"]]
    #     return return_list
    #
    # source_df[['read_bq', 'block_bq', 'base_bq', 'error', 'pi', 'flag', 'mapq']] = pd.DataFrame(source_df['metadata'].apply(explode_dict).tolist(), index=source_df.index)
    # source_df.drop(columns=["metadata"], inplace=True)
    #
    # source_df["flag"] = source_df["flag"] > 0
    # source_df["flag"] = source_df["flag"].astype(int)
    # source_df["mapq"] = np.clip(source_df["mapq"] / 60, 0, 1)
    # source_df[["read_bq","block_bq","base_bq"]] = np.clip(source_df[["read_bq","block_bq","base_bq"]] / 40, 0, 1)
    #
    # source_df = source_df.sort_values("pred", ascending=False)
    # source_df.to_pickle(f"{args.output}/source.pkl")

    label_df = pd.read_csv(args.label, sep='\t')
    label_df.set_index("id", inplace=True)

    source_df = pd.read_pickle(f"/extdata4/baeklab/Hyeonseo/m6A/postprocess/old/dataset_v3/source.pkl")
    label_id_unique = source_df["label_id"].unique()
    source_df = source_df.groupby("label_id")
    label_id_unique = np.random.permutation(label_id_unique)

    train_pos_df_dict = {}
    val_pos_df_dict = {}
    test_pos_df_dict = {}
    train_neg_df_dict = {}
    val_neg_df_dict = {}

    idx = 0

    max_num_positives = label_df[label_df["label"] != 0].shape[0]

    stop_negative_flag = False

    for label_id in tqdm.tqdm(label_id_unique):
        try:
            df = source_df.get_group(label_id)
        except:
            continue

        undersample_flag = False

        depth = len(df)

        if depth < args.min_depth:
            continue

        try:
            label_row = label_df.loc[label_id]
        except:
            continue

        m6a_level = label_row["m6A_level"]
        label = label_row["label"]

        if label == 0:
            if stop_negative_flag:
                continue
            else:
                if len(train_neg_df_dict) > max_num_positives:
                    stop_negative_flag = True

        if depth > args.max_depth:
            df = df.sample(args.max_depth)
            df = df.sort_values("pred", ascending=False)
            depth = args.max_depth
            undersample_flag = True


        df["label"] = label
        df["m6a_level"] = m6a_level
        df["depth"] = depth / args.max_depth

        dom_pred = df["pred"][(df["pred"] <= 0.3) | (df["pred"] >= 0.7)]
        dom_pred = dom_pred > 0.5
        dom_pred = dom_pred.mean()

        df["dom_pred"] = dom_pred

        if label == 0:
            if idx % 10 <= 0:
                val_neg_df_dict[label_id] = df
            else:
                train_neg_df_dict[label_id] = df

        else:
            if idx % 10 <= 0:
                val_pos_df_dict[label_id] = df
            elif idx % 10 <= 6:
                if (not undersample_flag) and label == 1:
                    test_pos_df_dict[label_id] = df
                else:
                    train_pos_df_dict[label_id] = df
            else:
                train_pos_df_dict[label_id] = df

        idx += 1

    ## match positive and negative samples
    train_neg_df_dict = dict(list(train_neg_df_dict.items())[:len(train_pos_df_dict)])
    val_neg_df_dict = dict(list(val_neg_df_dict.items())[:len(val_pos_df_dict)])
    train_pos_df_dict.update(train_neg_df_dict)
    val_pos_df_dict.update(val_neg_df_dict)

    print(f"Train: {len(train_pos_df_dict)}")
    print(f"Val: {len(val_pos_df_dict)}")
    print(f"Test: {len(test_pos_df_dict)}")

    with open(f"{args.output}/train.pkl", "wb") as f:
        pickle.dump(train_pos_df_dict, f)
    with open(f"{args.output}/val.pkl", "wb") as f:
        pickle.dump(val_pos_df_dict, f)
    with open(f"{args.output}/test.pkl", "wb") as f:
        pickle.dump(test_pos_df_dict, f)

    return None


if __name__ == "__main__":
    main()