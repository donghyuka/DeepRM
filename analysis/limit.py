import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm

def random_binomial(n,p,flip=0.0):
    rand = np.random.binomial(n,p)
    if flip > 0:
        flip_pos = np.random.binomial(n-rand,flip)
        flip_neg = np.random.binomial(rand,flip)
        rand = rand + flip_pos - flip_neg
    rand /= n
    return rand


def main():
    path = "/extdata4/baeklab/Hyeonseo/m6A/inference/plot/BERMUDA-Basecaller-Compound-v4-20240429-113430-21-237000-baeklab_v4_genome_drach-updatedlabel-dorado/BaeklabV3.Genome.GP3.depth1_None.twm6astrict.notsampled.drach_no_filter_depth5_20/BERMUDA-Basecaller-Compound-v4-20240429-113430-21-237000-baeklab_v4_genome_drach.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df[df["dom_label"] > 0.0]
    df["count"] = df["count"].astype(int)

    r2_dict = {x:[] for x in [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]}
    rho2_dict = {x:[] for x in [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]}

    for x in tqdm(range(100)):
        random_000 = []
        random_001 = []
        random_005 = []
        random_010 = []
        random_020 = []
        random_030 = []

        for n, p in zip(df["count"], df["dom_label"]):
            random_000.append(random_binomial(n,p,0.00))
            random_001.append(random_binomial(n,p,0.01))
            random_005.append(random_binomial(n,p,0.05))
            random_010.append(random_binomial(n,p,0.10))
            random_020.append(random_binomial(n,p,0.20))
            random_030.append(random_binomial(n,p,0.30))

        dom_label = df["dom_label"]
        r2_dict[0.00].append(stats.pearsonr(dom_label, random_000)[0]**2)
        rho2_dict[0.00].append(stats.spearmanr(dom_label, random_000)[0]**2)
        r2_dict[0.01].append(stats.pearsonr(dom_label, random_001)[0]**2)
        rho2_dict[0.01].append(stats.spearmanr(dom_label, random_001)[0]**2)
        r2_dict[0.05].append(stats.pearsonr(dom_label, random_005)[0]**2)
        rho2_dict[0.05].append(stats.spearmanr(dom_label, random_005)[0]**2)
        r2_dict[0.10].append(stats.pearsonr(dom_label, random_010)[0]**2)
        rho2_dict[0.10].append(stats.spearmanr(dom_label, random_010)[0]**2)
        r2_dict[0.20].append(stats.pearsonr(dom_label, random_020)[0]**2)
        rho2_dict[0.20].append(stats.spearmanr(dom_label, random_020)[0]**2)
        r2_dict[0.30].append(stats.pearsonr(dom_label, random_030)[0]**2)
        rho2_dict[0.30].append(stats.spearmanr(dom_label, random_030)[0]**2)



    print(f"ERR=0.00 R2={np.mean(r2_dict[0.00]):.4f} Rho2={np.mean(rho2_dict[0.00]):.4f}")
    print(f"ERR=0.01 R2={np.mean(r2_dict[0.01]):.4f} Rho2={np.mean(rho2_dict[0.01]):.4f}")
    print(f"ERR=0.05 R2={np.mean(r2_dict[0.05]):.4f} Rho2={np.mean(rho2_dict[0.05]):.4f}")
    print(f"ERR=0.10 R2={np.mean(r2_dict[0.10]):.4f} Rho2={np.mean(rho2_dict[0.10]):.4f}")
    print(f"ERR=0.20 R2={np.mean(r2_dict[0.20]):.4f} Rho2={np.mean(rho2_dict[0.20]):.4f}")
    print(f"ERR=0.30 R2={np.mean(r2_dict[0.30]):.4f} Rho2={np.mean(rho2_dict[0.30]):.4f}")

    return None


if __name__ == "__main__":
    main()

