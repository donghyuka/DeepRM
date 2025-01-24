import os

import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
from utils.utils import parse_refflat_v2
import glob
import time
import shutil
import tqdm


def transcript_pos(pos,strand,exon_starts,exon_ends):
    in_exon = (pos >= exon_starts) & (pos < exon_ends)
    ## which exon the position is in
    if np.any(in_exon):
        in_exon = np.where(in_exon)[0][0]
    else:
        return None
    ## convert the position to the transcript position
    if strand=="+":
        transcript_pos = (exon_ends-exon_starts)[:in_exon].sum() + pos - exon_starts[in_exon] + 1
    elif strand=="-":
        transcript_pos = (exon_ends-exon_starts)[in_exon+1:].sum() + exon_ends[in_exon] - pos
    else:
        raise ValueError("Invalid strand")
    return transcript_pos


def match_site_to_gene(chrstrand, strand, pos, refflat_df):
    refflat_df_chrstrand = refflat_df.get_group(chrstrand)
    refflat_df_gene = refflat_df_chrstrand[(refflat_df_chrstrand["txStart"]<=pos) & (refflat_df_chrstrand["txEnd"]>pos)].copy()
    if len(refflat_df_gene)==0:
        return [], []
    refflat_df_gene["transcript_pos"]=refflat_df_gene.apply(lambda x: transcript_pos(pos,strand,x["exonStarts"],x["exonEnds"]),axis=1)
    return list(refflat_df_gene["transcript_id"]), list(refflat_df_gene["transcript_pos"])


def reformat_m6A_df_worker(m6a_df, refflat_df, outpath):
    m6a_df[['transcript_id', 'transcript_pos']] = m6a_df.apply(lambda x: match_site_to_gene(x["chrstrand"], x["str"], x["pos"], refflat_df), axis=1, result_type="expand")
    m6a_df = m6a_df.explode(["transcript_id", "transcript_pos"])
    m6a_df = m6a_df.dropna(subset=["transcript_id", "transcript_pos"])
    m6a_df.to_pickle(outpath)
    return None


def reformat_m6A_df(m6a_df, outdir, ncpu=120):

    refflat_df = parse_refflat_v2().groupby("chrstrand")
    print(refflat_df)

    m6a_df_split = np.array_split(m6a_df,ncpu)
    proc_list = []

    flush_path = os.path.join(outdir, f"temp_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(flush_path, exist_ok = True)

    for i, m6a_df in enumerate(m6a_df_split):
        proc = mp.Process(target=reformat_m6A_df_worker, args=(m6a_df, refflat_df, os.path.join(flush_path, f"{i}.pkl")))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    m6a_df = pd.concat([pd.read_pickle(f) for f in glob.glob(os.path.join(flush_path, "*.pkl"))])

    shutil.rmtree(flush_path)
    return m6a_df


def mark_cluster(m6a_df, outdir, ncpu=120):
    m6a_df_groupby = m6a_df.groupby("transcript_id")
    m6a_df_size = m6a_df_groupby.size().sort_values(ascending=False)
    m6a_df_split = {i: [] for i in range(ncpu)}
    for i, (transcript_id, size) in tqdm.tqdm(enumerate(m6a_df_size.items()), total=len(m6a_df_size), desc="Splitting"):
        m6a_df_split[i % ncpu].append(m6a_df_groupby.get_group(transcript_id).copy())

    proc_list = []
    flush_path = os.path.join(outdir, f"temp_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(flush_path, exist_ok = True)

    for i, m6a_df in m6a_df_split.items():
        proc = mp.Process(target=mark_cluster_worker, args=(m6a_df, os.path.join(flush_path, f"{i}.pkl")))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    m6a_df = pd.concat([pd.read_pickle(f) for f in glob.glob(os.path.join(flush_path, "*.pkl"))])

    ## groupby genome_id and collapse
    agg_dict = {k:"first" for k in ["chr", "str", "pos",]}
    agg_dict["nearest"] = "min"
    m6a_df = m6a_df.groupby("genome_id").agg(agg_dict).reset_index()

    shutil.rmtree(flush_path)
    return m6a_df


def mark_cluster_worker(m6a_df, outpath):
    results = []
    for transcript_df in tqdm.tqdm(m6a_df, desc="Marking cluster"):
        if len(transcript_df) > 1:
            transcript_df = transcript_df.sort_values("transcript_pos")
            nearest = np.diff(transcript_df["transcript_pos"].values)
            nearest = np.min(np.stack([np.concatenate([[np.inf], nearest]), np.concatenate([nearest, [np.inf]])]), axis=0)
            transcript_df["nearest"] = nearest
        else:
            transcript_df["nearest"] = np.inf
        results.append(transcript_df)
    results = pd.concat(results)
    results.to_pickle(outpath)
    return None


def read_m6a_df():
    exp_list = ["glori","m6ace","miclip2","sac"]
    exp_dict = {}
    exp_path = "/extdata3/baeklab/Jungmin/RNAmod/valid.new/cmp/{exp}.txt"
    for exp in exp_list:
        df = pd.read_csv(exp_path.format(exp=exp), sep="\t")
        df["genome_id"] = df["chr"] + ":" + df["str"].astype(str) + ":" + df["pos"].astype(str)
        if "qval_neg" in df.columns:
            df["m6a"] = (df["qval"] < 1E-7) & ~(df["qval_neg"] < 0.1)
        else:
            df["m6a"] = df["qval"] < 1E-7
        exp_dict[exp] = df
    for exp in ["m6ace","miclip2","sac"]:
        df = exp_dict[exp]
        df = df.rename(columns={"m6a": f"label_{exp}", "depth": f"depth_{exp}", "qval": f"q_{exp}"})
        df = df[["genome_id", f"label_{exp}", f"depth_{exp}", f"q_{exp}"]].copy()
        exp_dict[exp] = df
    for exp in ["glori"]:
        df = exp_dict[exp]
        df = df.rename(columns={"m6a": f"label_{exp}", "depth": f"depth_{exp}", "dom": f"dom_{exp}", "qval": f"q_{exp}"})
        df = df[["genome_id", f"label_{exp}", f"depth_{exp}", f"dom_{exp}", f"q_{exp}"]].copy()
        exp_dict[exp] = df
    for exp in exp_list:
        df = exp_dict[exp]
        df = df.set_index("genome_id", drop=True)
        exp_dict[exp] = df

    label_df = pd.concat(exp_dict.values(), axis=1).reset_index()
    label_df["label_glori_alt"] = (label_df["dom_glori"] >= 0.1) * label_df["label_glori"]
    label_df["positive"] = label_df[["label_glori_alt", "label_m6ace", "label_miclip2", "label_sac"]].fillna(0).astype(int).sum(axis=1) >= 3

    label_df = label_df[label_df["positive"]].copy()
    label_df["chr"]=label_df["genome_id"].str.split(":").str[0]
    label_df["str"]=label_df["genome_id"].str.split(":").str[1]
    label_df["pos"]=label_df["genome_id"].str.split(":").str[2].astype(int)
    label_df["chrstrand"]=label_df["chr"]+label_df["str"]

    label_df = label_df[["chr", "str", "pos", "chrstrand", "genome_id"]].fillna(0).copy()

    return label_df


def main():
    outpath = "/extdata4/baeklab/Hyeonseo/m6A/analyses/cluster_4exp"
    threads = 120

    glori_df = read_m6a_df()
    print(glori_df)
    glori_df = reformat_m6A_df(glori_df, outpath, threads)
    print(glori_df)
    glori_df = mark_cluster(glori_df, outpath, threads)
    print(glori_df)
    glori_df.to_pickle(os.path.join(outpath, "glori_df.clustered.pkl"))

    return None


if __name__=="__main__":
    main()
