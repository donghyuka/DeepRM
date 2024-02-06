
import pandas as pd
import numpy as np
import multiprocessing as mp
import os, argparse, tqdm, gc, glob
from collections import defaultdict

def worker(df_path_list, return_dict):
    for df_path in df_path_list:
        df = pd.read_pickle(df_path)
        bq_mean = df["bq"].values()
        bq_mean = np.mean(bq_mean, axis=0)
        count = len(df)
        df["5mer"] = df["motif"].apply(lambda x: x[6:11])
        cnt_5mer = df["5mer"].value_counts()
        return_dict[df_path] = {"bq_mean": bq_mean, "count": count, "cnt_5mer": cnt_5mer}
    return None

def main(cpu=120):
    path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/intern"
    file_list = glob.glob(f"{path}/*.pkl")
    file_list = np.array_split(file_list, cpu)
    manager = mp.Manager()
    return_dict = manager.dict()
    proc_list = []
    for files in file_list:
        proc = mp.Process(target=worker, args=(files, return_dict))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return_dict = dict(return_dict)
    manager.shutdown()

    bq_mean = np.zeros(17)
    total_count = 0
    cnt_5mer = defaultdict(int)
    for key in return_dict:
        bq_mean += return_dict[key]["bq_mean"]*return_dict[key]["count"]
        total_count += return_dict[key]["count"]
        for k in return_dict[key]["cnt_5mer"].keys():
            cnt_5mer[k] += return_dict[key]["cnt_5mer"][k]
    bq_mean = bq_mean/total_count
    cnt_5mer_df = pd.DataFrame(cnt_5mer.items(), columns=["5mer", "count"])
    cnt_5mer_df = cnt_5mer_df.sort_values("count", ascending=False)
    print(bq_mean)
    cnt_5mer_df.to_csv(f"{path}/5mer_count.csv", index=False)
    print(cnt_5mer_df)

    return None

main()


