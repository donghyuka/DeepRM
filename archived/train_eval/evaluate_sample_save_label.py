import glob
import argparse
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import tqdm


## 1. Load Eval Data and Model
## 2. Run Inference.
## 3. Create Site-level Predictions. There is no need to use PILEUP when evaluating on a sampled dataset.
## 4. Evaluate against ground truth labels and save evaluation results.
## 5. Plot evaluation results.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block/block", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/id/baeklab_v2_depth5_drach_block_id", help="Output path")
    parser.add_argument("--batch", "-b", type=int, default=4000, help="Batch size")
    parser.add_argument("--shard", "-s", type=int, default=10000, help="Shard size")
    parser.add_argument("--gpu", "-g", type=int, default=4, help="GPU device")
    parser.add_argument("--nfile", "-n", type=int, default=10, help="Number of files to load")
    parser.add_argument("--prefetch", "-p", type=int, default=100, help="Number of files to load")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    proc_list = []
    for rank in range(args.gpu):
        proc = mp.Process(target = worker, args = (rank, args.data, args.output, args.gpu))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    return None


def worker(rank, file_path, out_path, num_replicas):
    file_paths = glob.glob(f"{file_path}/*.pkl")[rank::num_replicas]
    id_list = []
    for path in tqdm.tqdm(file_paths):
        data = pd.read_pickle(path)
        id_arr = data["block_id"].values
        id_list.append(id_arr)
    id_arr = np.concatenate(id_list)
    df = pd.DataFrame({"block_id": id_arr,})
    df.to_csv(f"{out_path}/block_id_{rank}.tsv", sep = "\t", index = False)
    return None


if __name__ == "__main__":
    main()



