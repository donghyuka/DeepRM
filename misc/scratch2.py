import os
import glob
import math
import tqdm
import pod5
import multiprocessing as mp
import numpy as np


def split_pod5(pod5_dir, max_size_mb, ncpu = 8):
    ## Split pod5 files
    pod5_path_list = glob.glob(pod5_dir + "/*.pod5")
    oversized_list = []
    for pod5_path in pod5_path_list:
        pod5_size = os.path.getsize(pod5_path) / (1024 ** 2)
        if pod5_size > max_size_mb:
            oversized_list.append((pod5_path, pod5_size))
    if len(oversized_list) == 0:
        return None
    else:
        n_proc = min(ncpu, len(oversized_list))
        oversized_list_split = [oversized_list[i::n_proc] for i in range(n_proc)]
        proc_list = []
        for i in range(n_proc):
            proc = mp.Process(target = split_pod5_proc, args = (oversized_list_split[i], max_size_mb))
            proc_list.append(proc)
            proc.start()
        for proc in proc_list:
            proc.join()

    return None


def split_pod5_proc(pod5_list, max_size_mb):
    for pod5_path, pod5_size in pod5_list:
        with pod5.Reader(pod5_path) as reader:
            batch_count = reader.batch_count
            writer_count = math.ceil(pod5_size / max_size_mb)
            writer_list = [ pod5.Writer(f"{pod5_path[:-5]}_{x}.pod5") for x in range(writer_count)]
            with tqdm.tqdm(total=batch_count) as pbar:
                for batch_idx, batch in enumerate(reader.read_batches()):
                    writer_list[batch_idx%writer_count].add_reads([x.to_read() for x in batch.reads()])
                    pbar.update(1)
            for writer in writer_list:
                writer.close()
    return None


split_pod5("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0107/ON0107/raw/pod5", 4000)
