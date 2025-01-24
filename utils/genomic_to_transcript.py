import os

import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
from utils.utils import parse_refflat_v2

def check_in_exon(pos,exon_starts,exon_ends):
    exon_zip = zip(exon_starts,exon_ends)
    in_exon = [(pos >= exon_start) & (pos < exon_end) for exon_start,exon_end in exon_zip]
    in_exon = np.array(np.any(in_exon))

    return in_exon

def match_site_to_gene(chrstrand, pos, refflat_df):
    refflat_df_chrstrand = refflat_df.get_group(chrstrand)
    refflat_df_gene = refflat_df_chrstrand[(refflat_df_chrstrand["txStart"]<=pos) & (refflat_df_chrstrand["txEnd"]>pos)]

    if len(refflat_df_gene)==0:
        return np.array([])

    # ## Check if the coordinate is in the exon.
    # refflat_df_gene["in_exon"]=refflat_df_gene.apply(lambda x: check_in_exon(pos,x["exonStarts"],x["exonEnds"]),axis=1)
    # refflat_df_gene = refflat_df_gene[refflat_df_gene["in_exon"]==True]

    return refflat_df_gene["gene_id"].unique()


def mp_worker(m6a_df, refflat_df, m6a_df_list):

    m6a_df["gene_id"]=m6a_df.apply(lambda x: match_site_to_gene(x["chrstrand"],x["pos"],refflat_df),axis=1)
    m6a_df_list.append(m6a_df)

    return None


def reformat_m6A_df(infilename="/extdata4/baeklab/Hyeonseo/m6A/poster/merged_label_df.pkl",
                    outfilename="/extdata4/baeklab/Hyeonseo/m6A/poster/merged_label_df.geneid2.pkl", ncpu=120):

    m6a_df = pd.read_pickle(infilename)
    m6a_df = m6a_df.reset_index()
    m6a_df["chr"]=m6a_df["genome_id"].str.split(":").str[0]
    m6a_df["strand"]=m6a_df["genome_id"].str.split(":").str[1]
    m6a_df["pos"]=m6a_df["genome_id"].str.split(":").str[2].astype(int)
    m6a_df["chrstrand"]=m6a_df["chr"]+m6a_df["strand"]
    m6a_df_split = np.array_split(m6a_df,ncpu)
    print(m6a_df)

    refflat_df = parse_refflat_v2().groupby("chrstrand")
    print(refflat_df)

    man = mp.Manager()
    m6a_df_list =  man.list()
    proc_list = []

    for m6a_df in m6a_df_split:
        proc = mp.Process(target=mp_worker, args=(m6a_df, refflat_df, m6a_df_list))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    m6a_df_list = list(m6a_df_list)
    m6a_df = pd.concat(m6a_df_list,ignore_index=True)

    print(m6a_df)

    print(f"Writing output file to {outfilename}")
    m6a_df.to_pickle(outfilename)

    return None



def main():
    reformat_m6A_df()
    return None


if __name__=="__main__":
    main()