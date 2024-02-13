import pandas as pd
import multiprocessing as mp
import os, argparse, tqdm, gc, glob
import numpy as np


def worker(out_path, path_list, chunk = 10000):
    for path in tqdm.tqdm(path_list):
        df = pd.read_pickle(path)
        for chunk_idx in range(0, len(df) // chunk + 1):
            signal_df = df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(df))].copy()
            save_path = f"{out_path}/{path.split('/')[-1].split('.')[0]}-{chunk_idx}.pkl"
            signal_df.to_pickle(save_path)
        del df, signal_df
        gc.collect()
    return None

def main():
    path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/intermediates/normalized_segment_signal/block"
    path_list = glob.glob(f"{path}/*.pkl")
    path_list = np.array_split(path_list, 10)
    out_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/intermediates/normalized_segment_signal/block_chunk"
    os.makedirs(out_path, exist_ok=True)
    proc_list = []
    for pid in range(10):
        proc = mp.Process(target=worker, args=(out_path, path_list[pid]))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return None

if __name__ == "__main__":
    main()