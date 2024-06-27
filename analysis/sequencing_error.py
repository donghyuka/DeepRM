import gc
import os
import numpy as np
import pandas as pd
import argparse
import glob
import pysam
import tqdm
import pickle
import re
import multiprocessing as mp

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v2_gp3_depth20_drach", nargs="+", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/analyses/error_lowdepth", help="Output path")
    parser.add_argument("--id", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/id/baeklab_v2_depth5_drach_block_id", help="Baseline path")
    parser.add_argument("--label", "-l", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/label/BaeklabV2_GP3.depth20.twm6astrict.notsampled.drach.tsv", help="Label path")
    parser.add_argument("--bam", "-b", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado/dorado_output.sorted.filtered.bam", help="Bam path")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok = True)


    return args


def process_inferece(data_path, updated_label_path, outdir, id_path):

    data_paths = sorted(glob.glob(f"{data_path}/*.tsv"))
    data_df = pd.concat([pd.read_csv(data_path, sep = "\t") for data_path in data_paths]).reset_index(drop = True)
    id_paths = sorted(glob.glob(f"{id_path}/*.tsv"))
    # id_df = pd.concat([pd.read_csv(id_path, sep = "\t") for id_path in id_paths]).reset_index(drop = True)
    # id_df["read_id"] = id_df["block_id"].str.split(":").str[0]
    # id_df["position"] = id_df["block_id"].str.split(":").str[1].astype(int)
    # id_df.drop("block_id", axis = 1, inplace = True)
    # print(data_df)
    # print(id_df)
    #
    # data_df = pd.concat([data_df, id_df], axis = 1)

    data_df["read_id"] = data_df["block_id"].str.split(":").str[0]
    data_df["position"] = data_df["block_id"].str.split(":").str[1].astype(int)
    data_df.drop("block_id", axis = 1, inplace = True)

    data_df.dropna(subset = ["pred"], inplace = True) ## TODO: Check NA values
    data_df.drop(["label"], axis = 1, inplace = True)

    updated_label_df = pd.read_csv(updated_label_path, sep = "\t")
    updated_label_df = updated_label_df[["id", "label", "m6A_level"]]
    updated_label_df.rename(columns = {"id": "label_id","m6A_level": "dom_label"}, inplace = True)
    data_df = data_df.merge(updated_label_df, on = "label_id", how = "left")
    data_df.dropna(subset = ["label"], inplace = True)
    data_df = data_df[data_df["label"] >= 0].copy()

    data_df.to_pickle(f"{outdir}/data_df.pkl")

    print(data_df)

    # data_df_tp = data_df[(data_df["dom_label"] > 0.95) & (data_df["pred"] > 0.95)]
    # data_df_fp = data_df[(data_df["label"] == 0) & (data_df["pred"] > 0.95)]
    # data_df_fn = data_df[(data_df["dom_label"] > 0.95) & (data_df["pred"] < 0.05)]
    # data_df_tn = data_df[(data_df["label"] == 0) & (data_df["pred"] < 0.05)]
    #
    # data_df_tn["group"] = "TN"
    # data_df_fp["group"] = "FP"
    # data_df_fn["group"] = "FN"
    # data_df_tp["group"] = "TP"
    #
    # print(f"TP: {data_df_tp.shape[0]}, FP: {data_df_fp.shape[0]}, FN: {data_df_fn.shape[0]}, TN: {data_df_tn.shape[0]}")
    #
    # data_df = pd.concat([data_df_tp, data_df_fp, data_df_fn, data_df_tn])
    # data_df.to_pickle(f"{outdir}/data_df_grouped.pkl")

    # data_df_tp = data_df[data_df["group"] == "TP"]
    # data_df_fp = data_df[data_df["group"] == "FP"]
    # data_df_fn = data_df[data_df["group"] == "FN"]
    # data_df_tn = data_df[data_df["group"] == "TN"]
    #
    # ## Match count
    # min_count = 100000
    # data_df_tp = data_df_tp.sample(min(min_count,len(data_df_tp)))
    # data_df_fp = data_df_fp.sample(min(min_count,len(data_df_fp)))
    # data_df_fn = data_df_fn.sample(min(min_count,len(data_df_fn)))
    # data_df_tn = data_df_tn.sample(min(min_count,len(data_df_tn)))
    #
    # data_df = pd.concat([data_df_tp, data_df_fp, data_df_fn, data_df_tn]).reset_index(drop = True)
    # data_df.to_pickle(f"{outdir}/data_df_grouped_matched.pkl")
    #
    # del data_df_tp, data_df_fp, data_df_fn, data_df_tn

    return data_df


def get_alignment_data(bam_path):
    alignment_dict = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in tqdm.tqdm(bam, total = bam.mapped):
            qname = read.query_name
            cigar = read.cigarstring
            query_length = read.query_length
            md = read.get_tag("MD")
            alignment_dict[qname] = (cigar, md, query_length)
    return alignment_dict


def md_to_mismatch_arr(md):
    mis_arr = []
    digit_buffer = ""
    del_flag = False
    del_count = 0
    for char in md:
        if char.isdigit():
            if del_flag:
                del_flag = False
                if del_count > 0:
                    mis_arr += [0] * del_count
                    del_count = 0
            digit_buffer += char
        else:
            if del_flag:
                del_count += 1
                continue
            if len(digit_buffer) > 0:
                digit_buffer = int(digit_buffer)
                if digit_buffer > 0:
                    mis_arr += [0] * digit_buffer
                digit_buffer = ""
            if char == "^":
                del_flag = True
            else:
                mis_arr.append(1)

    if del_flag:
        if del_count > 0:
            mis_arr += [0] * del_count
    if len(digit_buffer) > 0:
        digit_buffer = int(digit_buffer)
        if digit_buffer > 0:
            mis_arr += [0] * digit_buffer

    mis_arr = np.array(mis_arr, dtype=int)
    return mis_arr


def cigar_to_error_arr(cigar, md, query_length):
    ## 3 channels: Mismatch, Insertion, Deletion
    error_arr = np.zeros((query_length,3),dtype=int)
    query_idx= 0
    ref_idx = 0
    cigartuples = re.findall(r"(\d+)([A-Z])", cigar)
    mis_arr = md_to_mismatch_arr(md)
    for length, op in cigartuples:
        length = int(length)
        if op == "M":
            mis_arr_slice = mis_arr[ref_idx:ref_idx+length]
            error_arr[query_idx:query_idx+length, 0] = mis_arr_slice
            query_idx += length
            ref_idx += length
        elif op == "I":
            error_arr[query_idx:query_idx+length, 1] = 1
            query_idx += length
        elif op == "D":
            error_arr[query_idx, 2] = length
            ref_idx += length
        elif op == "S":
            query_idx += length
        else:
            raise RuntimeWarning("Unrecognized CIGAR operation: ", op)

    error_arr = np.array(error_arr)
    return error_arr


def assign_error_worker(alignment_dict, collect_list):
    local_dict = {}
    for qname, (cigar, md, query_length) in tqdm.tqdm(alignment_dict.items(), total = len(alignment_dict)):
        error_arr = cigar_to_error_arr(cigar, md, query_length)
        local_dict[qname] = error_arr
    collect_list.append(local_dict)
    return None


def assign_error(alignment_dict, ncpu = 120):
    man = mp.Manager()
    collect_list = man.list()
    error_dict_keys_split = np.array_split(list(alignment_dict.keys()), ncpu)
    error_dict_split = [{key: alignment_dict[key] for key in keys} for keys in error_dict_keys_split]
    proc_list = []
    for i, split in enumerate(error_dict_split):
        proc = mp.Process(target = assign_error_worker, args = (split, collect_list))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    collect_list = list(collect_list)
    man.shutdown()
    error_dict = {}
    for local_dict in collect_list:
        error_dict.update(local_dict)
    return error_dict


def plot_pr(data_df, out_dir):
    plt.rcParams.update({'font.size': 24, 'axes.linewidth': 2.0})
    col_list = ["mis_c17", "ins_c17", "del_c17", "err_c17", "mis_c1", "ins_c1", "del_c1", "err_c1", "mis_c3", "ins_c3", "del_c3", "err_c3", "mis_c5", "ins_c5", "del_c5", "err_c5", "mis_c7", "ins_c7", "del_c7", "err_c7", "mis_c9", "ins_c9", "del_c9", "err_c9"]
    fig, ax = plt.subplots(1,1, figsize = (20, 20))
    threshold_neg = 0.20
    threshold_pos = 0.90
    min_depth = 5
    epsilon = 1e-12
    data_df = data_df[(data_df["pred"] <= threshold_neg) | ( data_df["pred"] >= threshold_pos)].copy()
    data_df["count"] = 1
    data_df["pred"] = np.clip(data_df["pred"].to_numpy(), 0.0, 1 - epsilon)
    data_df["pred"] = np.log10(1 - data_df["pred"].to_numpy())

    cmap = plt.get_cmap("gist_rainbow")

    for idx, col in enumerate(col_list):
        data_df_selected = data_df[data_df[col] == 0]
        data_df_selected = data_df_selected.groupby("label_id").agg({"pred": "mean","count": "sum", "label": "first"}).reset_index()
        data_df_selected["pred"] = np.clip(data_df_selected["pred"], None, 0)
        data_df_selected["pred"] = (1 - 10**data_df_selected["pred"].to_numpy())
        data_df_selected = data_df_selected[data_df_selected["count"] >= min_depth].copy()

        precision, recall, _ = precision_recall_curve(data_df_selected["label"], data_df_selected["pred"])
        auc_score = auc(recall, precision)
        ax.plot(recall, precision, label = f"{col} AUC: {auc_score:.3f}", linewidth = 3, color = cmap(idx/len(col_list)))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.savefig(f"{out_dir}/pr_curve.png")
    return None


def stat_worker(data_df, collect_list):

    data_df["mis_c17"] = data_df["error_arr"].apply(lambda x: np.sum(x[:,0]))
    data_df["ins_c17"] = data_df["error_arr"].apply(lambda x: np.sum(x[:,1]))
    data_df["del_c17"] = data_df["error_arr"].apply(lambda x: np.sum(x[:,2]))
    data_df["err_c17"] = data_df["error_arr"].apply(lambda x: np.sum(x))

    data_df["mis_c1"] = data_df["error_arr"].apply(lambda x: np.sum(x[8,0]))
    data_df["ins_c1"] = data_df["error_arr"].apply(lambda x: np.sum(x[8,1]))
    data_df["del_c1"] = data_df["error_arr"].apply(lambda x: np.sum(x[8,2]))
    data_df["err_c1"] = data_df["error_arr"].apply(lambda x: np.sum(x[8]))

    data_df["mis_c3"] = data_df["error_arr"].apply(lambda x: np.sum(x[7:10,0]))
    data_df["ins_c3"] = data_df["error_arr"].apply(lambda x: np.sum(x[7:10,1]))
    data_df["del_c3"] = data_df["error_arr"].apply(lambda x: np.sum(x[7:10,2]))
    data_df["err_c3"] = data_df["error_arr"].apply(lambda x: np.sum(x[7:10]))

    data_df["mis_c5"] = data_df["error_arr"].apply(lambda x: np.sum(x[6:11,0]))
    data_df["ins_c5"] = data_df["error_arr"].apply(lambda x: np.sum(x[6:11,1]))
    data_df["del_c5"] = data_df["error_arr"].apply(lambda x: np.sum(x[6:11,2]))
    data_df["err_c5"] = data_df["error_arr"].apply(lambda x: np.sum(x[6:11]))

    data_df["mis_c7"] = data_df["error_arr"].apply(lambda x: np.sum(x[5:12,0]))
    data_df["ins_c7"] = data_df["error_arr"].apply(lambda x: np.sum(x[5:12,1]))
    data_df["del_c7"] = data_df["error_arr"].apply(lambda x: np.sum(x[5:12,2]))
    data_df["err_c7"] = data_df["error_arr"].apply(lambda x: np.sum(x[5:12]))

    data_df["mis_c9"] = data_df["error_arr"].apply(lambda x: np.sum(x[4:13,0]))
    data_df["ins_c9"] = data_df["error_arr"].apply(lambda x: np.sum(x[4:13,1]))
    data_df["del_c9"] = data_df["error_arr"].apply(lambda x: np.sum(x[4:13,2]))
    data_df["err_c9"] = data_df["error_arr"].apply(lambda x: np.sum(x[4:13]))

    data_df.drop("error_arr", axis = 1, inplace = True)
    collect_list.append(data_df)
    return None



def main():
    args = parse_args()
    # alignment_dict = get_alignment_data(args.bam)
    # with open(f"{args.output}/alignment_dict.pkl", "wb") as outfile:
    #     pickle.dump(alignment_dict, outfile)
    # data_df = process_inferece(args.input, args.label, args.output, args.id)
    # alignment_dict = pickle.load(open(f"{args.output}/alignment_dict.pkl", "rb"))
    # read_id_arr = data_df["read_id"].unique()
    # alignment_dict = {key: alignment_dict[key] for key in read_id_arr}
    # error_dict = assign_error(alignment_dict, ncpu = 120)
    #
    # with open(f"{args.output}/error_dict.pkl", "wb") as outfile:
    #     pickle.dump(error_dict, outfile)
    #
    # del alignment_dict
    # gc.collect()
    #
    # data_df["error_arr"] = data_df["read_id"].map(error_dict)
    #
    # del error_dict
    # gc.collect()
    #
    # data_df["error_arr"] = data_df.apply(lambda x: x["error_arr"][x["position"]-8: x["position"]+9], axis = 1)
    # data_df.to_pickle(f"{args.output}/data_df_error.pkl")
    # data_df = pd.read_pickle(f"{args.output}/data_df_error.pkl")
    #
    # man = mp.Manager()
    # collect_list = man.list()
    # proc_list = []
    #
    # data_df_split = np.array_split(data_df, 100)
    #
    # del data_df
    # gc.collect()
    #
    # for i, split in enumerate(data_df_split):
    #     proc = mp.Process(target = stat_worker, args = (split, collect_list))
    #     proc_list.append(proc)
    #     proc.start()
    #
    # for proc in proc_list:
    #     proc.join()
    #
    # collect_list = list(collect_list)
    # data_df = pd.concat(collect_list)
    # man.shutdown()
    # del collect_list
    # gc.collect()
    #
    # data_df.to_pickle(f"{args.output}/data_df_error_processed.pkl")
    # print(data_df)
    data_df = pd.read_pickle(f"{args.output}/data_df_error_processed.pkl")
    plot_pr(data_df, args.output)

    return None


if __name__ == "__main__":
    main()



