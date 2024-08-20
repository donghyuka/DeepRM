import os
import pickle
path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver081424_pretrain/val"
filelist = os.listdir(path)
with open("/extdata4/baeklab/Hyeonseo/m6A/dataset/ver081424_pretrain/val.pkl", "wb") as f:
    pickle.dump(filelist, f)

path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver081424_pretrain/train"
filelist = os.listdir(path)
with open("/extdata4/baeklab/Hyeonseo/m6A/dataset/ver081424_pretrain/train.pkl", "wb") as f:
    pickle.dump(filelist, f)