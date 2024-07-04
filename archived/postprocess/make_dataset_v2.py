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
    args.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV3.GP3.depth5_20.twm6astrict.notsampled.drach.tsv", help="Source path")
    args.add_argument("--source", "-s", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v5_genome_drach", help="Source path")
    args.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v10", help="Output path")
    args.add_argument("--min_depth", "-m", type=int, default=5, help="Minimum depth")
    args.add_argument("--max_depth", "-x", type=int, default=20, help="Maximum depth")
    args = args.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args

def get_pred_geo(data_df):
    threshold_neg = 0.20
    threshold_pos = 0.90
    epsilon = 1e-6
    data_df_bin = data_df[(data_df["pred"] <= threshold_neg) | ( data_df["pred"] >= threshold_pos)].copy()
    if len(data_df_bin) < 5:
        return 0, len(data_df_bin)
    else:
        pred_geo = np.log10(1 - np.clip(data_df_bin["pred"].to_numpy(), 0.0, 1 - epsilon))
        pred_geo = 1 - (10**(pred_geo.mean()))
        return pred_geo, len(data_df_bin)

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
    source_df["error"] = source_df["error"].apply(lambda x: (x>0).astype(bool))
    source_df = source_df.groupby("label_id")

    train_df_dict = {}
    val_df_dict = {}
    test_df_dict = {}

    idx = 0

    for label_id, df in tqdm.tqdm(source_df, total=len(source_df)):

        depth = len(df)

        if depth < args.min_depth:
            continue

        pred, depth = get_pred_geo(df)

        if depth < args.min_depth:
            continue

        elif depth > args.max_depth:
            continue

        # if pred < 0.50:
        #     continue

        df["pred_geo"] = pred

        try:
            label_row = label_df.loc[label_id]
        except:
            continue

        m6a_level = label_row["m6A_level"]
        label = label_row["label"]

        if label < 0:
            continue

        if label == 1 and m6a_level == 0:
            flag = True

        else:
            idx += 1
            flag = False

        df["label"] = label
        df["m6a_level"] = m6a_level
        df["depth"] = depth / args.max_depth

        dom_pred = df["pred"][(df["pred"] <= 0.3) | (df["pred"] >= 0.7)]
        dom_pred = dom_pred > 0.5
        dom_pred = dom_pred.mean()

        df["dom_pred"] = dom_pred

        # p_pred = df["pred"][(df["pred"] <= 0.02) | (df["pred"] >= 0.98)]
        # p_pred = np.clip(p_pred, 0.0, 1 - 1e-6)
        # p_pred = np.mean(np.log(1-p_pred))
        # p_pred = 1 - np.exp(p_pred)
        # df["p_pred"] = p_pred

        if not flag:
            if idx % 10 <= 0:
                val_df_dict[label_id] = df
            elif idx % 10 <= 5:
                test_df_dict[label_id] = df
            else:
                train_df_dict[label_id] = df
        else:
            test_df_dict[label_id] = df

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