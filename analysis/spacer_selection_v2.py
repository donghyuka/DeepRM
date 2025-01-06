import numpy as np
import multiprocessing as mp
import itertools as it
import math
import re

import pysam
from tqdm import tqdm
from RNA import fold, cofold
import polyleven as pl
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
import pickle
from matplotlib import pyplot as plt
from utils.utils import revcomp_RNA
import os

wdir = "/extdata2/baeklab/Hyeonseo/res/spacer_design_55bb_2"
os.makedirs(wdir, exist_ok=True)


def generate_candidates(ncpu = 120, n_blocks = 3, kmer = 10):
    ## order : A, C, G, U
    nucs = "ACGU"
    front_ligation = "CGAC"
    back_ligation = "AGUC"
    assert n_blocks >= 2, "n_blocks should be greater than 1"
    assert kmer >= 4, "kmer should be greater than 3"
    front_occurence = [front_ligation.count(x) for x in nucs]
    back_occurence = [back_ligation.count(x) for x in nucs]

    existing = [front_occurence] + ([[0,0,0,0]]*(n_blocks-2)) + [back_occurence]
    print(existing)
    partitions = [kmer//4]*4
    partitions[:kmer%4] = [x+1 for x in partitions[:kmer%4]]
    print(partitions)
    partitions = list(set(it.permutations(partitions)))
    existing = np.array(existing)
    partitions = np.array(partitions)
    partition_combinations = list(it.product(range(partitions.shape[0]),repeat=n_blocks))
    print(f"Number of partitions: {partitions.shape[0]}")
    print(f"Number of combinations: {len(partition_combinations)}")
    part_comb_split = np.array_split(partition_combinations,ncpu)
    proc_list = []
    man = mp.Manager()
    return_list = man.list()
    for i in range(ncpu):
        p = mp.Process(target=generate_candidates_proc,args=(part_comb_split[i],partitions,nucs,n_blocks,kmer,existing,return_list))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()
    sequence_list = list(return_list)
    man.shutdown()
    sequence_list = np.concatenate(sequence_list)
    sequence_df = pd.DataFrame(sequence_list, columns=[f"block_{i}" for i in range(n_blocks)])
    sequence_df["block_0"] = front_ligation+sequence_df["block_0"]
    sequence_df[f"block_{n_blocks-1}"] = sequence_df[f"block_{n_blocks-1}"]+back_ligation
    return sequence_df


def sample_from_set(x, n):
    x = list(x)
    idx = np.random.choice(len(x), n, replace=False)
    return [x[i] for i in idx]

def generate_candidates_proc(part_comb_split, partitions, nucs, n_parts,kmer,existing,return_list):
    max_size = 1000
    sequence_list = []

    if len(part_comb_split) == 0:
        return None

    for part_comb in tqdm(part_comb_split):
        part_to_add = np.vstack([partitions[part_comb[i]] for i in range(n_parts)])
        composition = np.sum(part_to_add,axis=0)
        if np.max(composition) - np.min(composition) <= 1:
            part_to_add -= existing
            if np.all(part_to_add >= 0):
                block_sequence_list = []
                for block in range(n_parts):
                    possible_seqs = "".join([nucs[i] * part_to_add[block,i] for i in range(len(nucs))])
                    possible_seqs = set(it.permutations(possible_seqs))
                    if len(possible_seqs) > max_size:
                        possible_seqs = sample_from_set(possible_seqs,max_size)
                    block_sequence_list.append(possible_seqs)

                block_sequence_list = list(it.product(*block_sequence_list))
                block_sequence_list = [["".join(x) for x in y] for y in block_sequence_list]
                sequence_list.extend(block_sequence_list)

    if len(sequence_list) > 0:
        return_list.append(sequence_list)

    return None


def leven_filtering_proc(sequence_df,min_ed,return_list,n_blocks,kmer=6):
    leven_pass_list = []
    sequence_df_values = sequence_df[[f"block_{x}" for x in range(n_blocks)]].values
    for sequence_row_idx in tqdm(range(sequence_df_values.shape[0])):
        sequence_row = sequence_df_values[sequence_row_idx]
        ligate = sequence_row[n_blocks-1]+sequence_row[0]
        ligate_6mers = [ligate[i:i+6] for i in range(1, len(ligate)-6)]
        leven_pass = True
        for i in range(len(sequence_row)-1):
            for j in range(i+1,len(sequence_row)):
                if pl.levenshtein(sequence_row[i],sequence_row[j], min_ed - 1) < min_ed:
                    leven_pass = False
                    break
            if not leven_pass:
                break
        # if leven_pass:
        #     for i in range(len(sequence_row)-1):
        #         for j in range(len(ligate_6mers)):
        #             if pl.levenshtein(sequence_row[i],ligate_6mers[j], min_ed - 1) < min_ed:
        #                 leven_pass = False
        #                 break
        #         if not leven_pass:
        #             break
        leven_pass_list.append(leven_pass)

    sequence_df["leven_pass"] = leven_pass_list
    return_list.append(sequence_df)
    return None


def leven_filtering(sequence_df, min_ed = 4, ncpu = 120, n_blocks = 3):
    sequence_df_split = np.array_split(sequence_df,ncpu)
    proc_list = []
    man = mp.Manager()
    return_list = man.list()

    for i in range(ncpu):
        p = mp.Process(target=leven_filtering_proc,args=(sequence_df_split[i],min_ed,return_list,n_blocks))
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()

    return_list = list(return_list)
    sequence_df = pd.concat(return_list)
    sequence_df = sequence_df[sequence_df["leven_pass"]]
    sequence_df = sequence_df.drop(columns=["leven_pass"])
    sequence_df = sequence_df.reset_index(drop=True)

    return sequence_df



def extract_error_dict(kmer=5,ncpu=120):
    fasta_path = "/extdata2/baeklab/Hyeonseo/res/ref/hg38_rna_nrnm.fasta"
    align_path = "/extdata2/baeklab/Hyeonseo/res/aligned/dorado_output.sorted.filtered.bam"
    fasta_dict = SeqIO.parse(fasta_path, "fasta")
    fasta_dict = SeqIO.to_dict(fasta_dict)

    bam_dict = {"RNAME":[],"POS":[],"CIGAR":[]}

    with pysam.AlignmentFile(align_path, "rb") as bam:
        for read in tqdm(bam, total=bam.mapped):
            bam_dict["RNAME"].append(read.reference_name)
            bam_dict["POS"].append(read.reference_start)
            bam_dict["CIGAR"].append(read.cigarstring)

    new_df_split = np.array_split(pd.DataFrame(bam_dict),ncpu)

    man =   mp.Manager()
    return_list = man.list()
    proc_list = []
    possible_kmers = set(it.product("ACGU",repeat=kmer))
    possible_kmers= ["".join(x) for x in possible_kmers]

    for i in range(ncpu):
        proc = mp.Process(target=extract_error_dict_proc, args = (new_df_split[i], kmer, fasta_dict, return_list))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    return_list = list(return_list)
    return_dict = {x:[] for x in possible_kmers}
    for error_dict in return_list:
        for seq in error_dict:
            if seq in return_dict:
                return_dict[seq].extend(error_dict[seq])

    for seq in return_dict:
        error_list = return_dict[seq]
        ## cutoff at 99.9% percentile
        if len(error_list) > 100 :
            error_list = np.array(error_list)
            error_list = error_list[error_list < np.percentile(error_list,99.9)]
            return_dict[seq] = np.mean(error_list) / 6
        else:
            return_dict[seq] = None

    return return_dict


def extract_error_dict_proc(new_df, kmer, fasta_dict, return_list):

    error_dict = defaultdict(list)

    for ref_name, ref_pos, cigar in tqdm(zip(new_df["RNAME"], new_df["POS"], new_df["CIGAR"]), total=new_df.shape[0]):
        ref_seq = str(fasta_dict[ref_name].seq).replace("T","U")
        cigar_list = re.findall(r'(\d+)([A-Z,=])', cigar)
        error_array = []
        query_pos = 0

        for length, match in cigar_list:
            if match in ["M","=","X"]:
                ## Consume both the reference and query
                error_array.extend([0]*int(length))
                query_pos += int(length)
            elif match in ["I"]:
                ## Consume only the query
                if len(error_array) > 0:
                    error_array[-1] += int(length)
                query_pos += int(length)
            elif match in ["D","N"]:
                ## Consume only the reference
                error_array.extend([1]*int(length))

        ref_seq = ref_seq[ref_pos:ref_pos+len(error_array)]
        for i in range(len(ref_seq)-kmer):
            seq = ref_seq[i:i+kmer]
            error_dict[seq].append(sum(error_array[i:i+kmer]))

    return_list.append(error_dict)

    return None


def get_random_sequence(length=60,count=1000):
    nucs = "ACGU"
    template = nucs * (length//4)
    template = np.array(list(template))
    random_shuffle = [np.random.permutation(template) for _ in range(count)]
    random_shuffle = ["".join(x) for x in random_shuffle]
    return random_shuffle


def apply_random_sequence(template_list,random_sequence_list,cb_pad,boi="A"):
    return_list = []
    n_blocks = len(template_list)
    for random_sequence in random_sequence_list:
        block_seq_total = ""
        for block in range(n_blocks - 1):
            block_seq = template_list.iloc[block] + random_sequence[block * 2 * cb_pad:(block * 2 + 1) * cb_pad] + boi + random_sequence[(block * 2 + 1) * cb_pad:(block * 2 + 2) * cb_pad]
            block_seq_total += block_seq
        block_seq_total += template_list.iloc[-1]
        return_list.append(block_seq_total)
    return return_list


def get_mfe(sequence_df, random_count, cb_pad, n_blocks, ncpu = 120):
    split_df = np.array_split(sequence_df,ncpu)
    proc_list = []
    man = mp.Manager()
    return_list = man.list()
    random_sequence = get_random_sequence(length=cb_pad*(n_blocks-1)*2,count=random_count)
    for i in range(ncpu):
        proc = mp.Process(target=get_mfe_proc, args = (split_df[i], random_sequence, cb_pad, n_blocks, return_list))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    return_list = list(return_list)
    return_df = pd.concat(return_list)
    return return_df


def get_mfe_proc(sequence_df, random_sequence, cb_pad, n_blocks, return_list):

    block_list = [f"block_{i}" for i in range(n_blocks)]
    sequence_df["random_sequence"] = sequence_df.apply(lambda row: apply_random_sequence(row[block_list],random_sequence,cb_pad),axis=1)
    # sequence_df["mfe"] = sequence_df["random_sequence"].apply(lambda x: get_mfe_list(x))
    mfe_master_list = []
    for sequences in tqdm(sequence_df["random_sequence"]):
        mfe_list = get_mfe_list(sequences)
        mfe_master_list.append(mfe_list)
    sequence_df["mfe"] = mfe_master_list
    sequence_df.drop(columns=["random_sequence"], inplace=True)
    return_list.append(sequence_df)
    return None

def get_mfe_list(sequence_list):
    mfe_list = []
    for sequence in sequence_list:
        mfe = fold(sequence)[1]
        mfe_list.append(mfe)
    return mfe_list


def get_dimer_mfe(sequence_df, random_count, cb_pad, n_blocks, ncpu=120):
    split_df = np.array_split(sequence_df,ncpu)
    proc_list = []
    man = mp.Manager()
    return_list = man.list()

    random_seq_set1 = get_random_sequence(length=cb_pad*(n_blocks-1)*2,count=random_count)
    random_seq_set2 = get_random_sequence(length=cb_pad*(n_blocks-1)*2,count=random_count)

    for i in range(ncpu):
        proc = mp.Process(target=get_dimer_mfe_proc, args = (split_df[i], random_seq_set1, random_seq_set2, cb_pad, n_blocks, return_list))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    return_list = list(return_list)
    return_df = pd.concat(return_list)
    return return_df

def get_dimer_mfe_proc(sequence_df,  random_seq_set1, random_seq_set2, cb_pad, n_blocks, return_list):
    block_list = [f"block_{i}" for i in range(n_blocks)]
    sequence_df["random_sequence_1"] = sequence_df.apply(lambda row: apply_random_sequence(row[block_list],random_seq_set1,cb_pad),axis=1)
    sequence_df["random_sequence_2"] = sequence_df.apply(lambda row: apply_random_sequence(row[block_list],random_seq_set2,cb_pad),axis=1)

    mfe_master_list = []
    for sequences1, sequences2 in tqdm(zip(sequence_df["random_sequence_1"],sequence_df["random_sequence_2"]), total=len(sequence_df)):
        mfe_list = get_dimer_mfe_list(sequences1, sequences2)
        mfe_master_list.append(mfe_list)
    sequence_df["dimer_mfe"] = mfe_master_list
    sequence_df.drop(columns=["random_sequence_1","random_sequence_2"], inplace=True)
    return_list.append(sequence_df)
    return None


def get_dimer_mfe_list(sequence_list1, sequence_list2):
    mfe_list = []
    for sequence1, sequence2 in zip(sequence_list1, sequence_list2):
        mfe = cofold(sequence1,sequence2, )[1]
        mfe_list.append(mfe)
    return mfe_list


def is_reverse_complement(seq1,seq2, kmer=4):
    seq2_rc = revcomp_RNA(seq2)
    kmers_seq1 = set([seq1[i:i+kmer] for i in range(len(seq1)-kmer+1)])
    kmers_seq2_rc = set([seq2_rc[i:i+kmer] for i in range(len(seq2_rc)-kmer+1)])
    if len(kmers_seq1.intersection(kmers_seq2_rc)) > 0:
        return True
    else:
        return False

def is_any_revcomp(seq_list, kmer=4):
    seq_ligate = seq_list.iloc[-1] + seq_list.iloc[0]
    seq_ligate_kmer = list(set([seq_ligate[i:i+kmer] for i in range(len(seq_ligate)-kmer+1)]))
    seq_list = list(seq_list.iloc[1:-1]) + seq_ligate_kmer
    for i in range(len(seq_list)-1):
        for j in range(i,len(seq_list)):
            if is_reverse_complement(seq_list[i],seq_list[j],kmer):
                return True
    return False


def mark_revcomp_proc(sequence_df, kmer, return_list, n_block):
    sequence_df["is_revcomp"] = sequence_df.apply(lambda row: is_any_revcomp(row[[f"block_{i}" for i in range(n_block)]], kmer),axis=1)
    return_list.append(sequence_df)

def mark_revcomp(sequence_df, kmer=4, ncpu=120, n_block=4):
    sequence_df_split = np.array_split(sequence_df,ncpu)
    proc_list = []
    man = mp.Manager()
    return_list = man.list()
    for pid in range(ncpu):
        proc = mp.Process(target=mark_revcomp_proc, args=(sequence_df_split[pid], kmer, return_list, n_block))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return_list = list(return_list)
    sequence_df = pd.concat(return_list)
    return sequence_df


def calculate_error(seq_list, error_dict, kmer=5):
    error_list = []
    for seq in seq_list:
        seq_kmers = [seq[i:i+kmer] for i in range(len(seq)-kmer+1)]
        error = np.mean([error_dict[x] for x in seq_kmers if x in error_dict])
        error_list.append(error)
    return error_list


def main(spacer_size = 15, n_blocks=2, cb_pad = 12, ncpu = 100):
    block_list = [f"block_{i}" for i in range(n_blocks)]


    sequence_df = generate_candidates(ncpu=ncpu, n_blocks=n_blocks, kmer=spacer_size)
    print("1. Candidates generated")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.pkl")

    sequence_df = leven_filtering(sequence_df, min_ed=spacer_size//2, ncpu=ncpu, n_blocks=n_blocks)
    print("2. Levenshtein filtered")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates.leven_filtered_{spacer_size}x{n_blocks}.pkl")

    sequence_df = mark_revcomp(sequence_df, kmer=5, ncpu=ncpu, n_block=2)
    sequence_df = sequence_df[~sequence_df["is_revcomp"]]
    sequence_df.drop(columns=["is_revcomp"], inplace=True)

    print("3. Revcomp filtered")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.pkl")
    error_dict = extract_error_dict(kmer=5,ncpu=ncpu)
    with open(f"{wdir}/error_dict.pkl","wb") as f:
        pickle.dump(error_dict,f)
    with open(f"{wdir}/error_dict.pkl","rb") as f:
        error_dict = pickle.load(f)

    sequence_df = pd.read_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.pkl")
    sequence_df["error_rate"] = sequence_df.apply(lambda row: calculate_error(row[block_list],error_dict,kmer=5), axis=1)
    sequence_df = sequence_df[sequence_df["error_rate"].apply(lambda x: None not in x)]
    sequence_df["mean_error_rate"] = sequence_df["error_rate"].apply(lambda x: np.mean(x))
    sequence_df["max_error_rate"] = sequence_df["error_rate"].apply(lambda x: np.max(x))
    print("4. Error rate calculated")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate.pkl")

    sequence_df = sequence_df[sequence_df["mean_error_rate"] < 0.05]
    sequence_df.sort_values(by="mean_error_rate", inplace=True)
    print("5. Error rate filtered")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate_filtered.pkl")
    sequence_df = pd.read_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate_filtered.pkl")
    sequence_df.sort_values(by="mean_error_rate", inplace=True, ascending=True)
    ## sort by mean_error_rate and select top 5%
    sequence_df = sequence_df.head(n=math.ceil(sequence_df.shape[0]*0.05)).reset_index(drop=True)
    sequence_df = get_mfe(sequence_df, random_count=1000, cb_pad=cb_pad, n_blocks=n_blocks, ncpu=ncpu)

    print("6. MFE calculated")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate_filtered.mfe.pkl")

    ## plot kde of mfe
    mfe_array = np.concatenate(sequence_df["mfe"].values)
    mfe_array = mfe_array[mfe_array < 0]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(mfe_array, bins=100)
    plt.savefig(f"{wdir}/mfe_hist.png")
    plt.close()
    print("7. MFE histogram plotted")

    sequence_df["mfe_mean"] = sequence_df["mfe"].apply(lambda x: np.mean(x))
    sequence_df["mfe_min"] = sequence_df["mfe"].apply(lambda x: np.min(x))
    ## sort by mfe_mean and select top 5%
    sequence_df.sort_values(by="mfe_mean", inplace=True, ascending=False)
    sequence_df = sequence_df.head(n=math.ceil(sequence_df.shape[0]*0.05)).reset_index(drop=True)
    print("8. MFE filtered")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate_filtered.mfe.mfe_filtered.pkl")
    sequence_df = get_dimer_mfe(sequence_df, random_count=1000, cb_pad=cb_pad, n_blocks=n_blocks, ncpu=ncpu)
    print("9. Dimer MFE calculated")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate_filtered.mfe.mfe_filtered.dimer_mfe.pkl")

    # ## plot kde of dimer mfe
    mfe_array = np.concatenate(sequence_df["dimer_mfe"].values)
    mfe_array = mfe_array[mfe_array < 0]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(mfe_array, bins=100)
    plt.savefig(f"{wdir}/dimer_mfe_hist.png")
    plt.close()
    print("10. Dimer MFE histogram plotted")

    sequence_df["dimer_mfe_mean"] = sequence_df["dimer_mfe"].apply(lambda x: np.mean(x))
    sequence_df["dimer_mfe_min"] = sequence_df["dimer_mfe"].apply(lambda x: np.min(x))
    ## sort by dimer_mfe_mean and select top 5%
    sequence_df.sort_values(by="dimer_mfe_mean", inplace=True, ascending=False)
    sequence_df = sequence_df.head(n=math.ceil(sequence_df.shape[0]*0.05))
    print("11. Dimer MFE filtered")
    print(sequence_df)
    sequence_df.to_pickle(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate_filtered.mfe.mfe_filtered.dimer_mfe.dimer_mfe_filtered.pkl")

    sequence_df = sequence_df.sort_values(by="mean_error_rate", ascending=True)
    sequence_df = sequence_df.head(n=1000)
    sequence_df.to_csv(f"{wdir}/spacer_candidates_{spacer_size}x{n_blocks}.leven_filtered.revcomp_filtered.error_rate_filtered.mfe.mfe_filtered.dimer_mfe.dimer_mfe_filtered.csv", index=False)
    print("12. Top 100 selected")
    print(sequence_df)
    return None

if __name__ == "__main__":
    main()