import os
import glob
import argparse
import numpy as np
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Negative data path")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--val_ratio", "-v", type=float, default=0.1, help="Validation ratio")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_list = glob.glob(f"{args.input}/*.pkl")
    rng = np.random.default_rng(args.seed)

    ## Shuffle and split
    rng.shuffle(data_list)
    val_size = int(len(data_list) * args.val_ratio)
    train_path_list = data_list[val_size:]
    val_path_list = data_list[:val_size]

    ## Create symbolic links
    os.makedirs(args.output, exist_ok = True)
    os.makedirs(f"{args.output}/train", exist_ok = True)
    os.makedirs(f"{args.output}/val", exist_ok = True)

    for train_path in tqdm.tqdm(train_path_list):
        os.symlink(train_path, f"{args.output}/train/{os.path.basename(train_path)}")
    for val_path in tqdm.tqdm(val_path_list):
        os.symlink(val_path, f"{args.output}/val/{os.path.basename(val_path)}")

    return None


if __name__ == "__main__":
    main()