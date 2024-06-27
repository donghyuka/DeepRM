import argparse
import pandas as pd
import numpy as np

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, required=True, help="Data path")
    args = args.parse_args()
    data = pd.read_pickle(args.data)
    data["m6a"] = data["pred_pm6a"] >= 0.75
    data = data.groupby("gene_symbol").agg({"m6a": "sum"})
    total_count = len(data)
    no_m6a_count = len(data[data["m6a"] == 0])
    print(f"Total count: {total_count}")
    print(f"No m6a count: {no_m6a_count}")
    print(f"Percentage: {no_m6a_count / total_count * 100:.2f}%")
    return None

if __name__ == "__main__":
    main()