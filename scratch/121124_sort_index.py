import pandas as pd
import numpy as np

def sort_label(path):
    label_df = pd.read_pickle(path)
    label_df = label_df[["ref", "pos"]].copy()
    # label_df = label_df.sort_values("pos")
    label_df.rename({"ref":"nmid"}, axis=1, inplace=True)
    label_df["nmid"] = label_df["nmid"].str.split(".").str[0]
    label_df["pos"] = label_df["pos"] - 1
    # label_df.reset_index(inplace=True)
    label_df["index"] = label_df.index.astype(np.int32)
    label_df["label_id"] = label_df["nmid"] + ":" + label_df["pos"].astype(str)
    label_df.to_pickle(path+".clean.pkl")
    print(label_df)
    print("label_df saved to", path+".clean.pkl")

# def sort_label(path):
#     label_df = pd.read_pickle(path)
#     label_df = label_df[["ref", "pos"]].copy()
#     label_df = label_df.sort_values("pos")
#     label_df.rename({"ref":"nmid"}, axis=1, inplace=True)
#     label_df["nmid"] = label_df["nmid"].str.split(".").str[0]
#     label_df["pos"] = label_df["pos"] - 1
#     label_df.reset_index(inplace=True)
#     label_df["index"] = label_df.index.astype(np.int32)
#     label_df["label_id"] = label_df["nmid"] + ":" + label_df["pos"].astype(str)
#     label_df.to_pickle(path+".sorted.pkl")
#     print(label_df)
#     print("label_df saved to", path+".sorted.pkl")


# sort_label("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.pileup.filtered.pkl")
# sort_label("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.pileup.filtered.pkl")
# sort_label("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.pileup.filtered.pkl")

sort_label("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.pileup.T.pkl")
sort_label("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.pileup.C.pkl")



#
# label_df = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.pileup.filtered.pkl"
# label_df = pd.read_pickle(label_df)
# label_df = label_df[["ref", "pos"]].copy()
# label_df = label_df.sort_values("pos")
# label_df.rename({"ref":"nmid"}, axis=1, inplace=True)
# label_df["nmid"] = label_df["nmid"].str.split(".").str[0]
# label_df["pos"] = label_df["pos"] - 1
# label_df.reset_index(inplace=True)
# label_df["index"] = label_df.index.astype(np.int32)
# label_df["label_id"] = label_df["nmid"] + ":" + label_df["pos"].astype(str)
#
# final_df = "/extdata4/baeklab/Hyeonseo/m6A/inference/pileup/AIRNA-DW-v4-m6A-20241112-111125-0-15000-token_normalise_ver161024_ON0090_m6Afinal/pileup.npz"
# with np.load(final_df) as f:
#     final_df = {k: v for k, v in f.items()}
# final_df = pd.DataFrame(final_df)
# label_df["label_id"] = label_df["nmid"] + ":" + label_df["pos"].astype(str)
# index_id_dict = dict(zip(label_df["index"], label_df["label_id"]))
# final_df["label_id"] = final_df["label_id"].map(index_id_dict)
#
# path = f"/extdata4/baeklab/Hyeonseo/m6A/inference/pileup/AIRNA-DW-v4-m6A-20241112-111125-0-15000-token_normalise_ver161024_ON0090_m6Afinal/pileup_strid_fixed.npz"
# np.savez_compressed(path, **{key: final_df[key].values for key in final_df.columns})