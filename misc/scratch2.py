import pandas as pd
import numpy as np
from tqdm import tqdm

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.GP3.depth5_None.twm6astrict.tsv"
df = pd.read_csv(path, sep='\t')


print(df)
def remove_adjacent_sites(depth_df):
    depth_df_groupby = depth_df.groupby("nmid")
    depth_df_list = []
    for nmid, group in tqdm(depth_df_groupby):
        group = group.sort_values("pos")
        ## calculate distance from nearest m6A site
        union_m6a_pos_list = group[np.abs(group["label"]) == 1]["pos"].values
        # glori_m6a_pos_list = group[group["m6A_level"] >= 0.1]["pos"].values

        if len(union_m6a_pos_list) == 0:
            group["dist_from_union_m6a"] = [np.array([])] * len(group)

        else:
            group["dist_from_union_m6a"] = group["pos"].apply(lambda x: np.abs(union_m6a_pos_list - x))

        # if len(glori_m6a_pos_list) == 0:
        #     group["dist_from_glori_m6a"] = [np.array([])] * len(group)
        # else:
        #     group["dist_from_glori_m6a"] = group["pos"].apply(lambda x: np.abs(glori_m6a_pos_list - x))

        depth_df_list.append(group)

    depth_df = pd.concat(depth_df_list).reset_index(drop=True)

    return depth_df

df = remove_adjacent_sites(df)

print(df)

drach_df = df[df["drach"]]
drach_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.GP3.depth5_None.twm6astrict.adjacency.drach.pkl")

df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/Baeklab.070.GP3.depth5_None.twm6astrict.adjacency.pkl")
