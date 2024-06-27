import argparse, os
import pandas as pd
import numpy as np
import multiprocessing as mp
from utils.utils import is_drach


## TODO: Refactor to remove these fixed paths.
GLORI_PATH = "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/glori_reformatted.tsv"
DEPTH_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado/dorado_output.sorted.pileup.filtered.v2.tsv"



def main():
    depth_cutoff = 20

    glori_df = pd.read_csv(GLORI_PATH, sep='\t')
    glori_df["m6A_level"] = glori_df[["m6A_level_rep1","m6A_level_rep2"]].mean(axis=1)
    glori_df["id"] = glori_df["NMID"] + ":" + glori_df["transcript_coordinate"].astype(str)
    glori_df = glori_df[["id","m6A_level"]]
    print(glori_df)

    depth_df = pd.read_csv(DEPTH_PATH, sep='\t', header = 0)
    depth_df.rename({"ref":"nmid"}, axis=1, inplace=True)
    depth_df["nmid"] = depth_df["nmid"].str.split(".").str[0]
    depth_df = depth_df[depth_df["nmid"] == "NM_001101"]
    depth_df["pos"] = depth_df["pos"] - 1
    depth_df["id"] = depth_df["nmid"] + ":" + depth_df["pos"].astype(str)
    depth_df = depth_df[["id", "depth", "nmid", "pos", "5mer"]]
    depth_df = depth_df[depth_df["depth"] > depth_cutoff].copy()
    print(depth_df)

    depth_df = depth_df.merge(glori_df, how="left", on="id")
    depth_df["m6A_level"] = depth_df["m6A_level"].fillna(0)
    depth_df["label"] = depth_df["m6A_level"].apply(lambda x: 0 if x < 0.1 else 1)

    depth_df = depth_df.dropna()
    depth_df.to_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/label/actb_glori.tsv", sep='\t', index=False)
    return None


if __name__ == "__main__":
    main()

