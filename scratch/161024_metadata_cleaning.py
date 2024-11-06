import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

def main(cpu=120):
    meta_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/postprocess/metadata/metadata.pkl")
    meta_df = meta_df.groupby("read_index")
    meta_df = meta_df.drop(columns=["temp_index"])
    meta_df.rename(columns={"dom": "dom_label"}, inplace=True)
    meta_df.to_pickle("/extdata4/baeklab/Hyeonseo/m6A/postprocess/metadata/metadata_cleaned.pkl")
    return None


if __name__ == "__main__":
    main()