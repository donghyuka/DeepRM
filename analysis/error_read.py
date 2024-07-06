import pandas as pd
import numpy as np
import glob
import os
import pysam
from utils.utils import parse_refflat
from evaluate.reformat_m6A_site_df import match_site_to_gene, chromosomal_to_transcript_coordinate
from tqdm import tqdm


## FP: label == 0, count_dom_070 >= 10, count_dom_043 >= 10, pm6a_070 - pm6a_043 >= 0.3
## FN: label == 1, count_dom_070 >= 10, count_dom_043 >= 10, pm6a_070 - pm6a_043 <= -0.3

def convert_id(label_df):
    refflat_df = parse_refflat()

    label_df["chr"]=label_df["genome_id"].str.split(":").str[0]
    label_df["strand"]=label_df["genome_id"].str.split(":").str[1]
    label_df["pos"]=label_df["genome_id"].str.split(":").str[2].astype(int)
    label_df["chrstrand"]=label_df["chr"]+label_df["strand"]
    label_df["NMID"]=label_df.apply(lambda x: match_site_to_gene(x["chrstrand"],x["pos"],refflat_df),axis=1)

    ## Filter out sites that do not match to a gene
    label_df = label_df[label_df["NMID"].apply(lambda x: len(x)>0)]

    ## Explode the NMID column
    label_df = label_df.explode("NMID")

    ## Second, convert chromosome coordinate to transcript coordinate
    label_df["transcript_coordinate"]=label_df.apply(lambda x: chromosomal_to_transcript_coordinate(
        x["NMID"],x["pos"], x["chrstrand"], x["strand"],refflat_df),axis=1)

    label_df["label_id"] = label_df["NMID"] + ":" + label_df["transcript_coordinate"].astype(str)

    label_df = label_df[["genome_id", "label_id"]].dropna().copy()

    return label_df

def main():
    fp_path = "/extdata4/baeklab/Hyeonseo/m6A/inference/plot/BERMUDA-Proto-v19-20240626-093708-12-215000-genomic/fp.tsv"
    fp_df = pd.read_csv(fp_path, sep = "\t")
    fp_df.reset_index(inplace = True, drop = False)
    fp_label = fp_df[["genome_id"]].copy()
    fp_label = convert_id(fp_label)

    print(fp_label)

    transcript_label = fp_label["label_id"].to_numpy()

    path_043 = "/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240404-092850-26-221000-token_normalise_drach_v2"
    path_070 = "/extdata4/baeklab/Hyeonseo/m6A/inference/inference/BERMUDA-Proto-v19-20240629-113654-2-341000-token_normalise_drach_v2"

    file_paths_043 = glob.glob(f"{path_043}/*.pkl")
    df_list_043 = []
    for path in file_paths_043:
        df = pd.read_pickle(path)
        df = df[df["label_id"].isin(transcript_label)]
        df_list_043.append(df)
    df_043 = pd.concat(df_list_043, axis = 0)

    file_paths_070 = glob.glob(f"{path_070}/*.pkl")
    df_list_070 = []
    for path in file_paths_070:
        df = pd.read_pickle(path)
        df = df[df["label_id"].isin(transcript_label)]
        df_list_070.append(df)
    df_070 = pd.concat(df_list_070, axis = 0)

    fp_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/fp.pkl")
    df_043.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/043.pkl")
    df_070.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/070.pkl")

    print(fp_df)
    print(df_043)
    print(df_070)

    df_043["read_id"] = df_043["block_id"].str.split(":").str[0]
    df_070["read_id"] = df_070["block_id"].str.split(":").str[0]

    df_043 = df_043[["label_id", "read_id", "pred"]]
    df_070 = df_070[["label_id", "read_id", "pred"]]

    df_043.rename(columns = {"pred": "pred_043"}, inplace = True)
    df_070.rename(columns = {"pred": "pred_070"}, inplace = True)

    df = df_043.merge(df_070, on = ["label_id", "read_id"], how = "outer")

    df = df.merge(fp_label, on = "label_id", how = "left")
    print(df)

    df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/merged.pkl")

    return None


def main2():
    path = "/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/merged.pkl"
    df = pd.read_pickle(path)
    excl_df = df[df["pred_043"].apply(np.isnan)]
    incl_df = df[df["pred_070"].apply(np.isnan)]
    common_df = df[~df["pred_043"].apply(np.isnan) & ~df["pred_070"].apply(np.isnan)]
    print(excl_df)
    print(incl_df)
    print(common_df)

    read_id_to_find = df["read_id"].to_numpy()

    bam_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.bam"
    bq_dict = {}

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in tqdm(bam, total = bam.mapped + bam.unmapped):
            try:
                bq_dict[read.query_name] = np.array(read.query_qualities, dtype = int)
            except:
                continue

    bq_dict_select = {}
    for key in read_id_to_find:
        try:
            bq_dict_select[key] = bq_dict[key]
        except:
            continue

    bq_df = pd.DataFrame(bq_dict_select.items(), columns = ["read_id", "bq"])
    bq_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/bq.pkl")

    # path = "/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/bq.pkl"
    # bq_df = pd.read_pickle(path)

    excl_df = excl_df.merge(bq_df, on = "read_id", how = "inner")
    incl_df = incl_df.merge(bq_df, on = "read_id", how = "inner")
    common_df = common_df.merge(bq_df, on = "read_id", how = "inner")

    excl_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/excl.pkl")
    incl_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/incl.pkl")
    common_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/analyses/fp_070/common.pkl")


    print(excl_df)
    print(incl_df)
    print(common_df)



    return None


if __name__ == "__main__":
    main2()










