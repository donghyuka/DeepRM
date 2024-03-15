import argparse, os
import pandas as pd
import numpy as np
import multiprocessing as mp
from utils.utils import is_drach


## TODO: Refactor to remove these fixed paths.
UNION_PATHS = ["/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/miclip2_pc_reformatted.tsv",
               "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/miclip_reformatted.tsv",
               "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/glori_reformatted.tsv",
               "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/sac_seq.reformatted.tsv",
               "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/m6ace_reformatted.tsv"]

INTERSECT_PATHS = ["/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/miclip2_pc_reformatted.tsv",
                   "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/m6ace_reformatted.tsv",
                   "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/glori_reformatted.tsv",]

GLORI_PATH = "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/glori_reformatted.tsv"
DEPTH_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado/dorado_output.sorted.pileup.filtered.v2.tsv"



def get_label_df():
    label_df_list = []
    for label_path in UNION_PATHS:
        label_df  = pd.read_csv(label_path, sep='\t')
        label_df["id"] = label_df[["NMID","transcript_coordinate"]].astype(str).agg(":".join, axis=1)
        label_df = label_df[["id","NMID","transcript_coordinate"]]
        label_df_list.append(label_df)

    ## Get union by id
    left = label_df_list[0]
    for right in label_df_list[1:]:
        left = left.merge(right, how="outer", on="id")
        ## if NMID_x is not null, then NMID_x, else NMID_y
        left["NMID"] = np.where(left["NMID_x"].isnull(), left["NMID_y"], left["NMID_x"])
        left["transcript_coordinate"] = np.where(left["transcript_coordinate_x"].isnull(), left["transcript_coordinate_y"], left["transcript_coordinate_x"])
        left.drop(["NMID_x","NMID_y","transcript_coordinate_x","transcript_coordinate_y"], axis=1, inplace=True)
    union_df = left.copy().reset_index(drop=True)


    label_df_list = []
    for label_path in INTERSECT_PATHS:
        label_df  = pd.read_csv(label_path, sep='\t')
        label_df["id"] = label_df[["NMID","transcript_coordinate"]].astype(str).agg(":".join, axis=1)
        label_df = label_df[["id","NMID","transcript_coordinate"]]
        label_df_list.append(label_df)

    ## Get intersection by id
    left = label_df_list[0]
    for right in label_df_list[1:]:
        left = left.merge(right, how="inner", on="id")
        left = left[['id','NMID_x','transcript_coordinate_x']]
        left.rename({"NMID_x":"NMID","transcript_coordinate_x":"transcript_coordinate"}, axis=1, inplace=True)

    intersect_df = left.copy().reset_index(drop=True)

    ## Remove intersection from union
    union_df = union_df[~union_df["id"].isin(intersect_df["id"])]

    return union_df, intersect_df



def make_label_df(return_list, union_df, intersect_df, glori_df, depth_df):
    depth_df["label"] = depth_df["id"].isin(union_df["id"]) * -1
    depth_df["label"] = depth_df["label"] + depth_df["id"].isin(intersect_df["id"])
    depth_df = depth_df.merge(glori_df, how="left", on="id")
    depth_df["m6A_level"] = depth_df["m6A_level"].fillna(0)
    return_list.append(depth_df)
    return None


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args = args.parse_args()
    depth_cutoff = 20
    union_df, intersect_df = get_label_df()
    union_df["union"] = 1
    intersect_df["intersect"] = 1
    union_df = union_df[["id","union"]]
    intersect_df = intersect_df[["id","intersect"]]
    print(union_df)
    print(intersect_df)

    glori_df = pd.read_csv(GLORI_PATH, sep='\t')
    glori_df["m6A_level"] = glori_df[["m6A_level_rep1","m6A_level_rep2"]].mean(axis=1)
    glori_df["id"] = glori_df["NMID"] + ":" + glori_df["transcript_coordinate"].astype(str)
    glori_df = glori_df[["id","m6A_level"]]
    print(glori_df)

    depth_df = pd.read_csv(DEPTH_PATH, sep='\t', header = 0)
    depth_df.rename({"ref":"nmid"}, axis=1, inplace=True)
    depth_df["nmid"] = depth_df["nmid"].str.split(".").str[0]
    depth_df["pos"] = depth_df["pos"] - 1
    depth_df["id"] = depth_df["nmid"] + ":" + depth_df["pos"].astype(str)
    depth_df = depth_df[["id", "depth", "nmid", "pos", "5mer"]]
    depth_df = depth_df[depth_df["depth"] > depth_cutoff].copy()
    print(depth_df)
    depth_df_split = np.array_split(depth_df, args.cpu)

    proc_list = []
    man = mp.Manager()
    return_list = man.list()

    for depth_df_proc in depth_df_split:
        proc = mp.Process(target=make_label_df, args=(return_list, union_df, intersect_df, glori_df, depth_df_proc))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    return_list = list(return_list)
    datid_df = pd.concat(return_list)
    datid_df = datid_df.dropna()
    datid_df.to_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/miclip2_glori_m6ace.v2.tsv", sep='\t', index=False)
    datid_df_s = sample_eval_data(datid_df, drach = False)
    datid_df_s.to_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/miclip2_glori_m6ace.v2.sampled.tsv", sep='\t', index=False)
    datid_df_s = sample_eval_data(datid_df, drach = True)
    datid_df_s.to_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/miclip2_glori_m6ace.v2.sampled.drach.tsv", sep='\t', index=False)
    return None


def sample_eval_data(datid_df, seed = 42, ratio = 10, drach = False):
    if drach:
        datid_df["drach"] = datid_df["5mer"].apply(is_drach)
        datid_df = datid_df[datid_df["drach"]]
        datid_df.drop("drach", axis=1, inplace=True)

    datid_df_pos = datid_df[datid_df["label"] == 1]
    datid_df_neg = datid_df[datid_df["label"] == 0]
    datid_df_neg = datid_df_neg.sample(n=int(len(datid_df_pos)*ratio), random_state=seed)
    datid_df = pd.concat([datid_df_pos, datid_df_neg])
    return datid_df


if __name__ == "__main__":
    main()

