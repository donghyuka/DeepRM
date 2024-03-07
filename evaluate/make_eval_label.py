import argparse, os


## TODO: Refactor to remove these fixed paths.
LABEL_PATHS = ["/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/miclip2_pc_reformatted.tsv",
               "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/glori_reformatted.tsv",
               # "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/sac_seq.reformatted.tsv",
               "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/m6ace_reformatted.tsv"]
GLORI_PATH = "/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/glori_reformatted.tsv"
IMAGE_PATH = "/extdata4/baeklab/Hyeonseo/exp_MRNA/{EXP}/index/image_11mer_messy_{crit}"
OUT_PATH = "/extdata3/baeklab/Jungmin/RNAmod/exp_MRNA/{EXP}/{EXP}/save_path/image_label_comp_messy_{crit}.tsv"
DEPTH_PATH = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado2/dorado_output.sorted.pileup.filtered.tsv"


import pandas as pd
import numpy as np
import multiprocessing as mp

def get_image_id(nmid,pos):
    ## The format for image_id is "XX:AAAAAAAAA:NNNNNN"
    ## XX is NMID prefix
    ## AAAAAAAAA is NMID suffix
    ## NNNNNN is position
    nmid_prefix = nmid[:2]
    nmid_suffix = nmid.split(".")[0][3:]
    nmid_suffix = "0" * (9 - len(nmid_suffix)) + nmid_suffix
    image_id = f"{nmid_prefix}:{nmid_suffix}:{pos:06d}"
    return image_id


def plot_depth_hist(datid_df,exp):
    ## mark 90th, 95th, percentile
    ## Draw Histogram
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    plt.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title("Depth Histogram")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Count")
    ## Label 90th, 95th percentile
    ax.text(np.percentile(datid_df["depth"], 90), 0.9 * ax.get_ylim()[1], "90th percentile", color="red")
    ax.text(np.percentile(datid_df["depth"], 95), 0.8 * ax.get_ylim()[1], "95th percentile", color="red")
    ax.axvline(np.percentile(datid_df["depth"], 90), color="red", linestyle="--")
    ax.axvline(np.percentile(datid_df["depth"], 95), color="red", linestyle="--")
    ax.text(np.percentile(datid_df["depth"], 90), 0.7 * ax.get_ylim()[1], f"{np.percentile(datid_df['depth'], 90):.0f}", color="red")
    ax.text(np.percentile(datid_df["depth"], 95), 0.6 * ax.get_ylim()[1], f"{np.percentile(datid_df['depth'], 95):.0f}", color="red")

    ## Remove outliers at 99th percentile
    datid_df_depth_list = datid_df[datid_df["depth"] < np.percentile(datid_df["depth"], 99)]["depth"].to_numpy()

    sns.histplot(datid_df_depth_list, ax=ax, kde=True, stat="density", bins=100)

    fig.savefig(OUT_PATH.format(EXP=exp).replace(".tsv","_depth_hist.png"), dpi=300)
    plt.close(fig)


    return None

def get_label_df():
    label_df_list = []
    for label_path in LABEL_PATHS:
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

    ## Get union by id
    left = label_df_list[0]
    for right in label_df_list[1:]:
        left = left.merge(right, how="outer", on="id")
        ## if NMID_x is not null, then NMID_x, else NMID_y
        left["NMID"] = np.where(left["NMID_x"].isnull(), left["NMID_y"], left["NMID_x"])
        left["transcript_coordinate"] = np.where(left["transcript_coordinate_x"].isnull(), left["transcript_coordinate_y"], left["transcript_coordinate_x"])
        left.drop(["NMID_x","NMID_y","transcript_coordinate_x","transcript_coordinate_y"], axis=1, inplace=True)
    union_df = left.copy().reset_index(drop=True)

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

    depth_df = pd.read_csv(DEPTH_PATH, sep='\t', header=None)
    depth_df.columns = ["nmid", "pos", "nuc", "depth"]
    depth_df["nmid"] = depth_df["nmid"].str.split(".").str[0]
    depth_df["pos"] = depth_df["pos"] - 1
    depth_df["id"] = depth_df["nmid"] + ":" + depth_df["pos"].astype(str)
    depth_df = depth_df[["id", "depth", "nmid", "pos"]]
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
    datid_df.to_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/rna004_label.tsv", sep='\t', index=False)
    datid_df = sample_eval_data(datid_df)
    datid_df.to_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/rna004_label_sampled.tsv", sep='\t', index=False)
    return None


def main2():
    datid_df = pd.read_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/rna004_label.tsv", sep='\t')
    datid_df = sample_eval_data(datid_df)
    datid_df.to_csv("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/rna004_label_sampled.tsv", sep='\t', index=False)
    return None

def sample_eval_data(datid_df, depth_cutoff = 20, seed = 42, ratio = 10):
    datid_df = datid_df[datid_df["depth"] > depth_cutoff]
    datid_df_pos = datid_df[datid_df["label"] == 1]
    datid_df_neg = datid_df[datid_df["label"] == 0]
    datid_df_neg = datid_df_neg.sample(n=int(len(datid_df_pos)*ratio), random_state=seed)
    datid_df = pd.concat([datid_df_pos, datid_df_neg])
    return datid_df

if __name__ == "__main__":
    main2()

