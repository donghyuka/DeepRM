import os
import numpy as np
import shutil
import tqdm

def main():
    subdirs = ["train/pos", "train/neg", "val/pos", "val/neg"]
    source = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver112524_exc9398/score-1.0/"
    destination = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver012325_m6A_intern"
    for subdir in subdirs:
        os.makedirs(os.path.join(destination, subdir), exist_ok=True)
    for subdir in subdirs:
        if "train" in subdir:
            n_samples = 2500
        elif "val" in subdir:
            n_samples = 250
        else:
            raise ValueError("Invalid subdir")
        source_files = os.listdir(os.path.join(source, subdir))
        random_choice = np.random.choice(source_files, n_samples, replace=False)
        source_subdir = os.path.join(source, subdir)
        destination_subdir = os.path.join(destination, subdir)
        for file in tqdm.tqdm(random_choice, desc=f"Copying {subdir}"):
            shutil.copy(os.path.join(source_subdir, file), os.path.join(destination_subdir, file))
    return None

if __name__ == "__main__":
    main()