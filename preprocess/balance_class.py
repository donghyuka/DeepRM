import os
import glob
import argparse
import numpy as np
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input data path")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--ratio", "-r", type=int, default=1, help="Neg/Pos ratio")
    parser.add_argument("--count", "-c", type=int, default=None, help="Number of positive samples")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    train_pos_list = glob.glob(f"{args.input}/train/pos/*.pkl")
    train_neg_list = glob.glob(f"{args.input}/train/neg/*.pkl")
    val_pos_list = glob.glob(f"{args.input}/val/pos/*.pkl")
    val_neg_list = glob.glob(f"{args.input}/val/neg/*.pkl")

    rng = np.random.default_rng(args.seed)

    ## sample to match the ratio
    if args.count is None:
        if len(train_pos_list) * args.ratio < len(train_neg_list):
            train_neg_list = rng.choice(train_neg_list, int(len(train_pos_list) * args.ratio), replace = False)
        else:
            train_pos_list = rng.choice(train_pos_list, int(len(train_neg_list) / args.ratio), replace = False)

        if len(val_pos_list) * args.ratio < len(val_neg_list):
            val_neg_list = rng.choice(val_neg_list, int(len(val_pos_list) * args.ratio), replace = False)
        else:
            val_pos_list = rng.choice(val_pos_list, int(len(val_neg_list) / args.ratio), replace = False)

    else:
        if len(train_pos_list) < args.count:
            raise ValueError(f"Number of positive samples is less than {args.count}")
        if len(train_neg_list) < args.count * args.ratio:
            raise ValueError(f"Number of negative samples is less than {args.count * args.ratio}")
        train_pos_list = rng.choice(train_pos_list, args.count, replace = False)
        train_neg_list = rng.choice(train_neg_list, args.count * args.ratio, replace = False)

    ## Create symbolic links
    os.makedirs(f"{args.output}/train/pos", exist_ok = True)
    os.makedirs(f"{args.output}/train/neg", exist_ok = True)
    os.makedirs(f"{args.output}/val/pos", exist_ok = True)
    os.makedirs(f"{args.output}/val/neg", exist_ok = True)

    for train_pos in tqdm.tqdm(train_pos_list):
        os.symlink(train_pos, f"{args.output}/train/pos/{os.path.basename(train_pos)}")
    for train_neg in tqdm.tqdm(train_neg_list):
        os.symlink(train_neg, f"{args.output}/train/neg/{os.path.basename(train_neg)}")
    for val_pos in tqdm.tqdm(val_pos_list):
        os.symlink(val_pos, f"{args.output}/val/pos/{os.path.basename(val_pos)}")
    for val_neg in tqdm.tqdm(val_neg_list):
        os.symlink(val_neg, f"{args.output}/val/neg/{os.path.basename(val_neg)}")

    return None


if __name__ == "__main__":
    main()