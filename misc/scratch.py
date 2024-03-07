import glob
import os, shutil

path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block/block"
filelist = glob.glob(f"{path}/*.pkl_0")

for file in filelist:
    ## rename to .pkl
    new_name = file.replace(".pkl_0", ".pkl")
    os.rename(file, new_name)

