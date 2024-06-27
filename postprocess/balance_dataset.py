import argparse
import pandas as pd
import numpy as np
import os
import tqdm
import pickle


def get_pred_geo(data_df):
    threshold_neg = 0.20
    threshold_pos = 0.90
    epsilon = 1e-6
    data_df_bin = data_df[(data_df["pred"] <= threshold_neg) | ( data_df["pred"] >= threshold_pos)].copy()
    if len(data_df_bin) < 5:
        return 0
    else:
        pred_geo = np.log10(1 - np.clip(data_df_bin["pred"].to_numpy(), 0.0, 1 - epsilon))
        pred_geo = 1 - (10**(pred_geo.mean()))
        return pred_geo

def main():
    train_path = "/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v9/train.pkl"
    val_path =   "/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v9/val.pkl"
    test_path =  "/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v9/test.pkl"
    train_data = pickle.load(open(train_path, "rb"))
    val_data = pickle.load(open(val_path, "rb"))
    test_data = pickle.load(open(test_path, "rb"))

    train_dict = {}
    val_dict = {}
    test_dict = {}

    for key, df in tqdm.tqdm(train_data.items(), total=len(train_data)):

        pred = get_pred_geo(df)
        df["pred_geo"] = pred
        if pred >= 0.50:
            train_dict[key] = df

    for key, df in tqdm.tqdm(val_data.items(), total=len(val_data)):

        pred = get_pred_geo(df)
        df["pred_geo"] = pred
        if pred >= 0.50:
            val_dict[key] = df

    for key, df in tqdm.tqdm(test_data.items(), total=len(test_data)):

        pred = get_pred_geo(df)
        df["pred_geo"] = pred
        if pred >= 0.50:
            test_dict[key] = df

    os.makedirs("/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v9", exist_ok=True)
    with open("/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v9/train_pred_positive.pkl", "wb") as f:
        pickle.dump(train_dict, f)
    with open("/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v9/val_pred_positive.pkl", "wb") as f:
        pickle.dump(val_dict, f)
    with open("/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v9/test_pred_positive.pkl", "wb") as f:
        pickle.dump(test_dict, f)

    return None


if __name__ == "__main__":
    main()