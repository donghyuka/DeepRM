import os
import numpy as np
import time
import argparse

args = argparse.ArgumentParser()
args.add_argument("--gpu", type=int, default=0, help="GPU device")
args.add_argument("--job", type=int, default=8, help="Number of jobs")
args = args.parse_args()


template = f"python -m postprocess.hpo10 --gpu {args.gpu}"

for i in range(args.job):
    time.sleep(1)
    ## run command in background
    os.system(f"{template} &")



