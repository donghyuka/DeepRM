import pandas as pd
import glob
import tqdm

orig_path = "/extdata4/baeklab/Hyeonseo/m6A/inference/embedding/AIRNA-v3B-20240726-105403-22-379000-token_normalise_drach_d100-ON0090"
ref_path = "/extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta"
import Bio.SeqIO as SeqIO
ref_id_list = []
for record in SeqIO.parse(ref_path, "fasta"):
    ref_id_list.append(record.id)

convert_dict = {x.split(".")[0]:x for x in ref_id_list}

paths = glob.glob(f"{orig_path}/*.pkl")
df_list = []

for path in tqdm.tqdm(paths):
    df = pd.read_pickle(path)
    df["NMID"] = df["label_id"].str.split(":").str[0]
    df["NMID"] = df["NMID"].map(convert_dict)
    df["pos"] = df["label_id"].str.split(":").str[1]
    df["label_id"] = df["NMID"] + ":" + df["pos"]
    df.drop(["NMID", "pos"], axis = 1, inplace = True)
    df_list.append(df)

df = pd.concat(df_list)
df.to_pickle(f"{orig_path}/merged.pkl")


