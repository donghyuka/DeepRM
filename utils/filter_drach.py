import os
import pandas as pd
import argparse
from utils.utils import printmessage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dorado", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado_m6a/dorado_m6a_basecalled.pileup.bed", help="Dorado path")
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.GP3.depth5_None.twm6astrict.drach.tsv", help="Label path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/plot", help="Output path")
    parser.add_argument("--min_depth", "-md", type=int, default=20, help="Minimum depth")
    parser.add_argument("--max_depth", "-xd", type=int, default=0, help="Maximum depth")
    args = parser.parse_args()

    return args



def process_dorado_inferece(data_path):

    data_df = pd.read_csv(data_path, quoting = 3, sep = "\t", header = None, dtype=str)
    ## Keep column 0, 1, 4, 9
    data_df = data_df[[0, 1, 4, 9]]
    data_df.columns = ["nmid", "pos", "depth", "pred_dorado"]
    data_df["depth"] = data_df["depth"].astype(int)
    data_df["pred_dorado"] = data_df["pred_dorado"].str.split(" ").str[1].astype(float) / 100
    data_df["label_id"] = data_df["nmid"].str.split(".").str[0] + ":" + data_df["pos"]
    data_df = data_df[["label_id", "pred_dorado","depth"]].copy()
    data_df.rename(columns = {"depth": "dorado_count"}, inplace = True)
    data_df.to_pickle(data_path.replace(".bed", ".pkl"))

    return data_df


def process_label(label_path):

    label_df = pd.read_csv(label_path, sep = "\t")
    label_df = label_df[["id", "depth", "label", "m6A_level", "5mer", "drach"]]
    label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    # label_df = label_df[label_df["label"] >= 0].copy()

    return label_df


def save_dorado_drach():
    args = parse_args()

    dorado_pred2 = process_dorado_inferece(args.dorado)
    label_df = process_label(args.label)
    printmessage(f"Label count: {len(label_df)}")

    dorado_pred2 = dorado_pred2.merge(label_df, on = "label_id", how = "inner")
    dorado_pred2.fillna(0, inplace = True)
    dorado_pred2 = dorado_pred2[dorado_pred2["drach"]].copy()
    dorado_pred2 = dorado_pred2[["label_id", "pred_dorado", "dorado_count"]].copy()
    print(dorado_pred2)

    dorado_pred2.to_pickle(args.output)

    return None


if __name__ == "__main__":
    save_dorado_drach()