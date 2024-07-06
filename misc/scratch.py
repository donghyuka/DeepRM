import os
import glob
import pandas as pd
path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0100/ON0100/ON0100/result/intermediates/block_flush_20240702173206/df_99_*.pkl"


df_list = []
for file in glob.glob(path):
    print(file)
    df = pd.read_pickle(file)
    df_list.append(df)

block_df = pd.concat(df_list, axis=0).reset_index(drop=True)
block_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0100/ON0100/ON0100/result/intermediates/block_flush_20240702173206/df_99.pkl")