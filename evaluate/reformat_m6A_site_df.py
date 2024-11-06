import pandas as pd
import numpy as np
import multiprocessing as mp


REFFLAT_PATH="/extdata4/baeklab/Hyeonseo/m6A/res/ref/GRCh38_latest_genomic.gtf.refflat.txt"
WDIR="/extdata3/baeklab/Hyeonseo/m6A/res/m6asites"

def chromosomal_to_transcript_coordinate(nmid, chr_coordinate, chrstrand, strand, refflat_df):
    ## Purpose: convert chromosome coordinate to mrna coordinate, based on exon starts and ends
    ## Expects and returns zero-based coordinates

    ## Get the exon starts and ends
    refflat_df_chrs = refflat_df[refflat_df["chrstrand"]==chrstrand]
    exon_starts = refflat_df_chrs.loc[nmid,"exonStarts"]
    exon_ends = refflat_df_chrs.loc[nmid,"exonEnds"]


    exons=np.stack((exon_starts,exon_ends),axis=1)
    exon_cumsum=np.concatenate(([0],np.cumsum(exons[:,1]-exons[:,0])))
    try:
        exon_index = np.searchsorted(exons[:,0],chr_coordinate,side="right")-1
    except:
        print(nmid,chr_coordinate,chrstrand,strand)
        raise ValueError
    mrna_coordinate=exon_cumsum[exon_index]+(chr_coordinate-exons[exon_index,0])
    mrna_length=exon_cumsum[-1]

    if strand=="-":
        mrna_coordinate=mrna_length-mrna_coordinate-1

    return mrna_coordinate

def ncid_to_chr(ncid):
    ncid_int=int(ncid.split(".")[0][3:])
    if ncid_int<=22:
        chr=f"chr{ncid_int}"
    elif ncid_int==23:
        chr="chrX"
    elif ncid_int==24:
        chr="chrY"
    else:
        chr="chrUnk"
    return chr


def parse_refflat(refflat_path=REFFLAT_PATH):
    col_list=["NMID","NCID","strand","txStart","txEnd","cdsStart","cdsEnd","exonCount","exonStarts","exonEnds"]
    with open(refflat_path,"r") as infile:
        refflat_df=pd.read_csv(infile,sep="\t",header=None,names=col_list)

    refflat_df["chr"]=refflat_df["NCID"].apply(lambda x: ncid_to_chr(x))
    refflat_df["NMID"]=refflat_df["NMID"].str.split(".").str[0]
    refflat_df = refflat_df[refflat_df["chr"]!="chrUnk"]
    refflat_df=refflat_df[refflat_df["cdsEnd"]>=refflat_df["cdsStart"]]
    refflat_df["chrstrand"]=refflat_df["chr"].astype(str)+refflat_df["strand"]
    refflat_df[["txStart","txEnd","cdsStart","cdsEnd"]]=refflat_df[["txStart","txEnd","cdsStart","cdsEnd"]].astype(int)
    refflat_df["exonStarts"]=refflat_df["exonStarts"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))
    refflat_df["exonEnds"]=refflat_df["exonEnds"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))


    ## Drop duplicates
    chr_list = [f"chr{i}" for i in list(range(1,23))+["X","Y","M"]]
    refflat_df_list=[]
    for chr in chr_list:
        refflat_df_chrs = refflat_df[refflat_df["chr"]==chr].copy()
        refflat_df_chrs.drop_duplicates(subset='NMID',keep="first",inplace=True,ignore_index=True)
        refflat_df_list.append(refflat_df_chrs)

    refflat_df = pd.concat(refflat_df_list,ignore_index=True)
    refflat_df.set_index("NMID",inplace=True)

    return refflat_df


def check_in_exon(pos,exon_starts,exon_ends):
    exon_zip = zip(exon_starts,exon_ends)
    in_exon = [(pos >= exon_start) & (pos < exon_end) for exon_start,exon_end in exon_zip]
    in_exon = np.array(np.any(in_exon))

    return in_exon


def match_site_to_gene(chrstrand, pos, refflat_df):
    refflat_df_chrstrand = refflat_df[refflat_df["chrstrand"]==chrstrand]
    refflat_df_gene = refflat_df_chrstrand[(refflat_df_chrstrand["txStart"]<=pos) & (refflat_df_chrstrand["txEnd"]>pos)]

    if len(refflat_df_gene)==0:
        return []

    ## Check if the coordinate is in the exon.
    refflat_df_gene["in_exon"]=refflat_df_gene.apply(lambda x: check_in_exon(pos,x["exonStarts"],x["exonEnds"]),axis=1)
    refflat_df_gene = refflat_df_gene[refflat_df_gene["in_exon"]==True]

    return refflat_df_gene.index.to_numpy()


def mp_worker(m6a_df, refflat_df, pos_col, strand_col, m6a_df_list):

    m6a_df["NMID"]=m6a_df.apply(lambda x: match_site_to_gene(x["chrstrand"],x[pos_col],refflat_df),axis=1)

    ## Filter out sites that do not match to a gene
    m6a_df = m6a_df[m6a_df["NMID"].apply(lambda x: len(x)>0)]

    ## Explode the NMID column
    m6a_df = m6a_df.explode("NMID")

    ## Second, convert chromosome coordinate to transcript coordinate
    m6a_df["transcript_coordinate"]=m6a_df.apply(lambda x: chromosomal_to_transcript_coordinate(
        x["NMID"],x[pos_col], x["chrstrand"], x[strand_col],refflat_df),axis=1)

    m6a_df_list.append(m6a_df)

    return None


def reformat_m6A_df(chr_col="chr",pos_col="pos",strand_col="str",pos_offset=0,neg_offset=0,
                    infilename="m6A_Jungmin_110823.txt",outfilename="m6A_Jungmin_110823.tsv", ncpu=120):

    m6a_df = pd.read_csv(f"{WDIR}/{infilename}",sep="\t")

    m6a_df["DoM"] = (m6a_df["DoM_1"] + m6a_df["DoM_2"])/2
    m6a_df[["SAC","GLORI","MICLIP2","M6ACE"]] = m6a_df[["SAC","GLORI","MICLIP2","M6ACE"]].apply(lambda x: x=="yes")
    m6a_df["validated"] = m6a_df[["SAC","GLORI","MICLIP2","M6ACE"]].astype(int).sum(axis=1)
    m6a_df.loc[m6a_df["GLORI"]==False,"DoM"]=0

    print(m6a_df)
    add_cols=["m6A_level_rep1","m6A_level_rep2","DoM","SAC","GLORI","MICLIP2","M6ACE","validated"]
    add_cols=["m6A_level_rep1","m6A_level_rep2","DoM","SAC","GLORI","MICLIP2","M6ACE","validated"]
    add_cols = [col for col in add_cols if col in m6a_df.columns]

    m6a_df=m6a_df[[chr_col,strand_col,pos_col]+add_cols]

    ## Apply positive offset to positive strand sites
    pos_idx = m6a_df[strand_col]=="+"
    m6a_df.loc[pos_idx,pos_col] = m6a_df.loc[pos_idx,pos_col] + pos_offset

    ## Apply negative offset to negative strand sites
    neg_idx = m6a_df[strand_col]=="-"
    m6a_df.loc[neg_idx,pos_col] = m6a_df.loc[neg_idx,pos_col] + neg_offset

    ## load refflat file
    refflat_df = parse_refflat()

    print(refflat_df.head())
    ## First, match each site to a gene
    m6a_df["chrstrand"]=m6a_df[chr_col].astype(str)+m6a_df[strand_col]

    m6a_df_split = np.array_split(m6a_df,ncpu)
    man = mp.Manager()
    m6a_df_list =  man.list()
    proc_list = []

    for m6a_df in m6a_df_split:
        proc = mp.Process(target=mp_worker, args=(m6a_df, refflat_df, pos_col, strand_col, m6a_df_list))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    m6a_df_list = list(m6a_df_list)
    m6a_df = pd.concat(m6a_df_list,ignore_index=True)

    print(m6a_df.head())

    print(f"Writing output file to {WDIR}/{outfilename}")
    m6a_df.to_csv(f"{WDIR}/{outfilename}",sep="\t",index=False)

    return None


if __name__=="__main__":
    reformat_m6A_df()