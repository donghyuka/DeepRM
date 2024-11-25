import multiprocessing as mp
import pandas as pd
import sys
sys.path.append("/extdata4/baeklab/Hyeonseo/m6A/modformer")
import glob
import tqdm
import gc
import numpy as np

def mean_phred(phred):
    if not isinstance(phred, np.ndarray):
        phred = np.array(phred, dtype=int)
    else:
        phred = phred.astype(int)
    ## When averaging PHRED scores, note that the PHRED score is logarithmically scaled.
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))


def worker(sub_path_list, collect_list):
    out_df = []
    drop_cols = ["signal_median", "signal_std", "signal_len", "signal_amp", "signal_rms"]
    for temp_file in tqdm.tqdm(sub_path_list):
        temp_df = pd.read_pickle(temp_file).drop(columns=drop_cols)
        temp_df["bq_centre"] = temp_df["bq"].apply(lambda x: x[10])
        temp_df["signal_centre"] = temp_df["signal_mean"].apply(lambda x: x[13])
        temp_df["bq_avg"] = temp_df["bq"].apply(lambda x: mean_phred(x))
        temp_df["signal_avg"] = temp_df["signal_mean"].apply(lambda x: np.mean(x[3:-3]))
        # temp_df["signal_3mer"] = temp_df["signal_mean"].apply(lambda x: np.mean(x[12:-12]))
        # temp_df["signal_5mer"] = temp_df["signal_mean"].apply(lambda x: np.mean(x[11:-11]))
        # temp_df["signal_7mer"] = temp_df["signal_mean"].apply(lambda x: np.mean(x[10:-10]))
        # temp_df["signal_9mer"] = temp_df["signal_mean"].apply(lambda x: np.mean(x[9:-9]))
        # temp_df["signal_11mer"] = temp_df["signal_mean"].apply(lambda x: np.mean(x[8:-8]))

        temp_df.drop(columns=["bq", "signal_mean"], inplace=True)
        out_df.append(temp_df)
    out_df = pd.concat(out_df).reset_index(drop=True)
    collect_list.append(out_df)
    gc.collect()
    return None

def main(threads = 120):
    ON0092_df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0092/ON0092/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/temp"
    ON0093_df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/temp"
    ON0096_df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0096/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/temp"
    ON0098_df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0098/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/temp"
    ON0099_df_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0099/result/dorado-070/intermediates/segmented_tokenized/signal_analysis_unnorm/temp"

    # paths = [ON0092_df_path, ON0093_df_path, ON0096_df_path, ON0098_df_path, ON0099_df_path]

    paths = [ON0093_df_path,ON0099_df_path]

    for df_path in paths:
        print(df_path)
        man = mp.Manager()
        collect_list = man.list()
        proc_list = []
        path_list = glob.glob(df_path+"/*.pkl")
        path_list_split = np.array_split(path_list, threads)
        for sub_path_list in path_list_split:
            proc = mp.Process(target=worker, args=(sub_path_list, collect_list))
            proc.start()
            proc_list.append(proc)
        for proc in proc_list:
            proc.join()
        collect_list = list(collect_list)
        out_df = pd.concat(collect_list).reset_index(drop=True)
        man.shutdown()
        out_df.to_pickle(df_path+".merged.light.pkl")
        del out_df, collect_list
        gc.collect()

    return None


if __name__ == "__main__":
    main()


