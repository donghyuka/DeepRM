import pandas as pd
import multiprocessing as mp
import numpy as np

def main(threads=120):
    df = pd.read_pickle()
    df = np.array_split(df, threads)
    procs = []
    man = mp.Manager()
    collect_list = man.list()
    for i in range(threads):
        proc = mp.Process(target=worker, args=(df[i], collect_list))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    result = pd.concat(collect_list)
    return result

