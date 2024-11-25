import argparse, os
import pandas as pd
import numpy as np
import multiprocessing as mp
from utils.utils import is_drach
import tqdm
import gc



def get_data_df(gp_cutoff,data_path ):
    data_df = pd.read_csv(data_path, sep='\t')
    data_df["id"] = data_df["NMID"] + ":" + data_df["transcript_coordinate"].astype(str)
    data_df["intersect"] = data_df["validated"] >= gp_cutoff
    data_df["intersect"] = data_df["intersect"].astype(int)
    data_df["label"] = 2 * data_df["intersect"] - 1
    data_df = data_df[['id', 'NMID', 'DoM', 'label']]
    data_df.rename({"DoM":"m6A_level"}, axis=1, inplace=True)
    return data_df

def make_label_df(return_list, data_df, depth_df,
                  filter_adjacent, adjacent_distance , adjacent_strict,
                  filter_no_m6a , no_m6a_strict ):

    if len(depth_df) == 0:
        return None
    if len(data_df) == 0:
        data_df = [pd.DataFrame(columns=["id","m6A_level","NMID","label"])]
    data_df = pd.concat(data_df).reset_index(drop=True)
    data_df.drop("NMID", axis=1, inplace=True)
    depth_df = pd.concat(depth_df).reset_index(drop=True)
    depth_df = depth_df.merge(data_df, how="left", on="id")
    depth_df.fillna(0, inplace=True)

    del data_df
    gc.collect()

    if len(depth_df) == 0:
        return None

    depth_df["label"] = depth_df["label"].astype(int)

    if filter_adjacent:
        depth_df = remove_adjacent_sites(depth_df, strict_site_only = adjacent_strict, distance = adjacent_distance)
        if depth_df is None:
            return None

    depth_df["label"] = depth_df["label"].astype(int)

    if filter_no_m6a:
        depth_df = remove_no_m6a_transripts(depth_df, strict_site_only = no_m6a_strict)
        if depth_df is None:
            return None

    depth_df["label"] = depth_df["label"].astype(int)

    depth_df = depth_df.dropna()
    return_list.append(depth_df)
    return None


def remove_adjacent_sites(depth_df, strict_site_only = False, distance = 10):
    depth_df_groupby = depth_df.groupby("nmid")
    depth_df_list = []
    for nmid, group in depth_df_groupby:
        group = group.sort_values("pos")
        ## calculate distance from nearest m6A site
        if strict_site_only:
            m6a_pos_list = group[group["label"] == 1]["pos"].values
        else:
            m6a_pos_list = group[np.abs(group["label"]) == 1]["pos"].values

        if len(m6a_pos_list) == 0:
            depth_df_list.append(group)

        else:
            group["min_dist_from_m6a"] = group["pos"].apply(lambda x: np.min(np.abs(m6a_pos_list - x)))
            group["label"] = group.apply(lambda row: -2 if (row["min_dist_from_m6a"] <= distance and row["label"] == 0) else row["label"], axis=1)
            depth_df_list.append(group)

    if len(depth_df_list) == 0:
        depth_df = None
    else:
        depth_df = pd.concat(depth_df_list).reset_index(drop=True)

    if "min_dist_from_m6a" in depth_df.columns:
        depth_df.drop("min_dist_from_m6a", axis=1, inplace=True)

    return depth_df


def remove_no_m6a_transripts(depth_df, strict_site_only = False):
    depth_df_groupby = depth_df.groupby("nmid")
    depth_df_list = []
    for nmid, group in depth_df_groupby:
        if strict_site_only:
            if 1 not in group["label"].values:
                group["label"] = -3
        else:
            if (1 not in np.abs(group["label"]).values):
                group["label"] = -3
        depth_df_list.append(group)

    if len(depth_df_list) == 0:
        depth_df = None
    else:
        depth_df = pd.concat(depth_df_list).reset_index(drop=True)

    return depth_df


def sample_eval_data(datid_df, seed = None, ratio = False, drach = False, non_drach = False, downsample = None):
    if drach:
        datid_df["drach"] = datid_df["5mer"].apply(is_drach)
        datid_df = datid_df[datid_df["drach"]]
        datid_df = datid_df.copy()
        datid_df.drop("drach", axis=1, inplace=True)

    elif non_drach:
        datid_df["drach"] = datid_df["5mer"].apply(is_drach)
        datid_df = datid_df[datid_df["drach"] == False]
        datid_df = datid_df.copy()
        datid_df.drop("drach", axis=1, inplace=True)

    if ratio is not False:
        datid_df_pos = datid_df[datid_df["label"] == 1]
        datid_df_neg = datid_df[datid_df["label"] == 0]
        if len(datid_df_pos) < len(datid_df_neg) // ratio:
            datid_df_neg = datid_df_neg.sample(n=int(len(datid_df_pos)*ratio), random_state=seed)
        else:
            datid_df_pos = datid_df_pos.sample(n=len(datid_df_neg)//ratio, random_state=seed)
        if downsample is not None:
            datid_df_pos = datid_df_pos.sample(frac=downsample, random_state=seed)
            datid_df_neg = datid_df_neg.sample(frac=downsample, random_state=seed)
        datid_df = pd.concat([datid_df_pos, datid_df_neg], ignore_index=True)
    return datid_df


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess RNA-seq data for training')
    parser.add_argument('--depth', '-d', type=str, help='Depth file', required=True)
    parser.add_argument('--out', '-o', type=str, help='Output directory', required=True)
    parser.add_argument('--data', '-a', type=str, help='Data file', default="/extdata3/baeklab/Hyeonseo/m6A/res/m6asites/GLORI_unconv5nt.tsv")
    parser.add_argument('--cpu', '-c', type=int, default=None, help='Number of threads')
    parser.add_argument('--gp', type=int, default=3, help='GP cutoff')
    parser.add_argument('--adj', type=bool, default=False, help='Filter adjacent sites')
    parser.add_argument('--adj_strict', type=bool, default=False, help='Filter adjacent sites strictly')
    parser.add_argument('--adj_distance', type=int, default=10, help='Distance to adjacent sites')
    parser.add_argument('--nom6a', type=bool, default=True, help='Filter no m6A genes')
    parser.add_argument('--nom6a_strict', type=bool, default=True, help='Filter no m6A genes strictly')
    parser.add_argument('--min_depth', type=int, default=5, help='Minimum depth')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth')

    args = parser.parse_args()
    if args.cpu is None:
        args.cpu = int(os.cpu_count() * 0.9)

    os.makedirs(args.out, exist_ok=True)
    return args


def main():

    args = parse_args()

    gp_cutoff = args.gp
    filter_adjacent = args.adj
    adjacent_strict = args.adj_strict
    filter_no_m6a = args.nom6a
    no_m6a_strict = args.nom6a_strict
    min_depth = args.min_depth
    max_depth = args.max_depth
    adjacent_distance = args.adj_distance

    data_df = get_data_df(gp_cutoff, args.data)
    print(data_df)

    depth_df = pd.read_pickle(args.depth)
    depth_df.rename({"ref":"nmid"}, axis=1, inplace=True)
    depth_df["nmid"] = depth_df["nmid"].str.split(".").str[0]
    depth_df["pos"] = depth_df["pos"] - 1
    depth_df["id"] = depth_df["nmid"] + ":" + depth_df["pos"].astype(str)
    depth_df = depth_df[["id", "depth", "nmid", "pos", "5mer"]]
    if min_depth is not None:
        depth_df = depth_df[depth_df["depth"] > min_depth].copy()
    if max_depth is not None:
        depth_df = depth_df[depth_df["depth"] < max_depth].copy()
    print(depth_df)

    depth_df_groupby = depth_df.groupby("nmid")
    data_df_groupby = data_df.groupby("NMID")

    ## split into ncpu chunks, respecting the nmid groupby
    depth_df_split = [[] for i in range(args.cpu)]
    data_df_split = [[] for i in range(args.cpu)]

    ## sort groups according to length
    groups = [group for name, group in depth_df_groupby]
    len_groups =len(groups)
    groups = sorted(groups, key = lambda x: len(x), reverse = True)

    for idx, group in tqdm.tqdm(enumerate(groups), desc="Splitting depth_df", total=len_groups):
        nmid = group["nmid"].values[0]
        split_idx = idx % (2*args.cpu)
        if split_idx >= args.cpu:
            split_idx = 2*args.cpu - split_idx - 1
        depth_df_split[split_idx].append(group)
        try:
            data_df_group = data_df_groupby.get_group(nmid)
            data_df_split[split_idx].append(data_df_group)
        except KeyError:
            pass

    del depth_df, depth_df_groupby, data_df, data_df_groupby
    gc.collect()

    print("Splitting complete")

    proc_list = []
    man = mp.Manager()
    return_list = man.list()

    for depth_df, data_df in zip(depth_df_split, data_df_split):
        proc = mp.Process(target=make_label_df, args=(return_list, data_df, depth_df,
                                                      filter_adjacent, adjacent_distance, adjacent_strict,
                                                      filter_no_m6a, no_m6a_strict))
        proc_list.append(proc)
        proc.start()

    del depth_df_split, data_df_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    return_list = list(return_list)
    datid_df = pd.concat(return_list)
    datid_df = datid_df.dropna()
    man.shutdown()
    del return_list
    gc.collect()

    label_name = f"GP{gp_cutoff}.depth{min_depth}_{max_depth}"
    if filter_adjacent:
        label_name += f".adj{adjacent_distance}"
        if adjacent_strict:
            label_name += "strict"
    if filter_no_m6a:
        label_name += ".twm6a"
        if no_m6a_strict:
            label_name += "strict"

    datid_df["drach"] = datid_df["5mer"].apply(is_drach)
    datid_df.to_csv(f"{args.out}.{label_name}.tsv", sep='\t', index=False)

    drach_df = datid_df[datid_df["drach"]]
    drach_df.to_csv(f"{args.out}.{label_name}.drach.tsv", sep='\t', index=False)

    return None


if __name__ == "__main__":
    main()
