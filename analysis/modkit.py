import torch
import pysam
import os
import argparse
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import pickle
import multiprocessing as mp
import gc
from utils.utils import mean_phred


def parse_args():
    parser = argparse.ArgumentParser(description='Inference to BAM')
    parser.add_argument('--inference', type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240404-092850-26-221000-baeklab_v2_depth20_drach", help='Inference file path')
    parser.add_argument('--id', type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/inference/block_id", help='Inference file path')
    parser.add_argument('--bam', type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado/dorado_output.sorted.filtered.bam", help='BAM file path')
    parser.add_argument('--mod_type', type=str, default="A+a", help='Modification type')
    parser.add_argument('--score_cutoff', type=float, default=0.0, help='Score cutoff')
    parser.add_argument('--threads', type=int, default=120, help='Number of threads')
    return parser.parse_args()



def convert_inference_df_to_mod_dict_worker(group_list, collect_list, mod_type = "A+a"):
    mod_dict_local = {}
    for group in tqdm(group_list):
        read_id = group["read_id"].iloc[0]
        group = group[["position", "pred"]]
        group = group.dropna()
        group = group.sort_values("position", ascending = True)
        print(group)
        mod_dict_query = group.set_index("position").to_dict()["pred"]
        print(mod_dict_query)
        mm_string, ml_string = convert_mod_dict_to_sam_tags(mod_dict_query, mod_type = mod_type)
        mod_dict_local[read_id] = (mm_string, ml_string)
    collect_list.append(mod_dict_local)
    return None


def convert_inference_df_to_mod_dict(inference_path, id_path, mod_type = "A+a", score_cutoff = 0.0, threads = 120):
    ## 1. Group df by "read_id"
    ## 2. For each group, convert "position" and "mod_prob" to "mod_dict"
    ## 3. Return "mod_dict" as a dictionary of read_id -> mod_dict. So it is a Dict(str, Dict(int, float)).

    # id_df = pd.concat([pd.read_csv(file, sep = "\t", header=0) for file in sorted(glob.glob(f"{id_path}/*.tsv"))]).reset_index(drop=True)
    # id_df["read_id"] = id_df["block_id"].str.split(":").str[0]
    # id_df["position"] = id_df["block_id"].str.split(":").str[1].astype(int)
    # id_df.drop("block_id", axis = 1, inplace = True)
    #
    # inference_df = pd.concat([pd.read_csv(file, sep = "\t") for file in sorted(glob.glob(f"{inference_path}/*.tsv"))]).reset_index(drop=True)
    # inference_df = pd.concat([inference_df, id_df], axis = 1)
    # inference_df.drop(["label", "label_id"], axis = 1, inplace = True)
    # print(inference_df)
    #
    # del id_df
    # gc.collect()
    # inference_df.to_pickle(f"{inference_path}/inference.pkl")

    inference_df = pd.read_pickle(f"{inference_path}/inference.pkl")

    group_dict = inference_df.groupby("read_id")
    group_key_split = np.array_split(list(group_dict.groups.keys()), threads)
    group_dict_split = [[group_dict.get_group(group_key) for group_key in group_key_split[i]] for i in range(threads)]

    del inference_df, group_dict, group_key_split
    gc.collect()

    man = mp.Manager()
    collect_list = man.list()

    proc_list = []
    for i in range(threads):
        group_list = group_dict_split[i]
        proc = mp.Process(target = convert_inference_df_to_mod_dict_worker, args = (group_list, collect_list, mod_type))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    mod_dict = {}
    for mod_dict_local in collect_list:
        mod_dict.update(mod_dict_local)

    del collect_list
    man.shutdown()
    gc.collect()

    with open(inference_path + "/mod.pkl", "wb") as f:
        pickle.dump(mod_dict, f)

    return mod_dict


def convert_mod_dict_to_sam_tags(mod_dict, mod_type = "A+a"):

    ## Convert mod_dict to SAM tags (MM and ML)
    ## Only include positions with mod_prob > score_cutoff.
    pos_list = [-1] + list(mod_dict.keys())
    prob_list = list(mod_dict.values())


    ## Generate ML tags.
    prob_list = [0 if np.isnan(x) else x for x in prob_list]
    ml_list = [int(x*255) for x in prob_list]

    return pos_list, ml_list


def write_mod_tags_to_bam(inference_path, bam_path, mod_dict, min_phred = 7, mod_type = "A+a"):
    ## Write mod_dict to BAM file as SAM tags.
    ## Use pysam to write SAM tags to BAM file.
    ## Use convert_mod_dict_to_sam_tags to convert mod_dict to SAM tags
    ## Use pysam.AlignmentFile to read and write BAM files.

    print("Reading BAM file")
    bamfile = pysam.AlignmentFile(bam_path, "rb", threads=16)
    header = bamfile.header
    out_bam_path = f"{inference_path}/modtag.bam"
    out_bamfile = pysam.AlignmentFile(out_bam_path, "wb", header = header, threads = 16)

    for read in tqdm(bamfile, total = bamfile.mapped, colour = "magenta"):

        # phred = mean_phred(np.array(read.query_qualities, dtype=int))
        #
        # if phred < min_phred:
        #     continue

        read.set_tag("mv", None)

        try:
            pos_list, ml_list = mod_dict[read.query_name]

        except KeyError:
            continue

        if len(ml_list) == 0:
            continue

        ## Generate MM tags.
        mm_list = []
        seq = read.query_sequence

        for pos_1, pos_2 in zip(pos_list[:-1], pos_list[1:]):
            mm_list.append(seq[pos_1+1:pos_2].count("A"))

        mm_string = f"{mod_type}?,{','.join([str(x) for x in mm_list])};"

        read.set_tag("MM", mm_string, "Z")
        read.set_tag("ML", ml_list)

        out_bamfile.write(read)

    bamfile.close()
    out_bamfile.close()

    return None


def main():
    args = parse_args()
    mod_dict = convert_inference_df_to_mod_dict(args.inference, args.id,
                                                mod_type = args.mod_type, score_cutoff = args.score_cutoff,
                                                threads = args.threads)
    write_mod_tags_to_bam(args.inference, args.bam, mod_dict)
    return None


if __name__ == "__main__":
    main()