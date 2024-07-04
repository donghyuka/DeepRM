import torch
import glob
import os
import pandas as pd
import argparse
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/model", help="Path to checkpoint directory")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/checkpoint_summary", help="Path to output directory")
    parser.add_argument("--keyword", "-k", type=str, nargs="+", default=["Postprocess-ResNet"], help="Keyword for filtering")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok = True)
    return args

def main():
    args = parse_args()
    path = args.path
    subpaths = glob.glob(f"{path}/*")
    keyword = args.keyword
    checkpoint_list = []
    subpaths = [subpath for subpath in subpaths if any([key in subpath for key in keyword])]
    for subpath in subpaths:
        checkpoints = glob.glob(f"{subpath}/*.pt")
        if len(checkpoints) > 0:
            checkpoint_dict = {int(x.split(".")[0].split("-")[-2]):x for x in checkpoints}
            max_epoch = max(checkpoint_dict.keys())
            checkpoint = checkpoint_dict[max_epoch]
            checkpoint_list.append(checkpoint)

    df_dict = {"path":[],  "name": [], "data": [], "best_epoch": [], "best_step": [],  "batch_size": [],
               "num_err_layers": [], "num_output_layers": [], "num_meta_layers": [], "num_pred_layers": [],
               "weight_decay": [], "dropout_rate": [], "loss": [],
               "lr": [], "hidden_dim": [], "kernel_size":[],
               "val_RMSE": [], "val_loss": []}

    for checkpoint in tqdm.tqdm(checkpoint_list):
        record_dict = torch.load(checkpoint, map_location="cpu")
        model_config = record_dict["model_config"]
        for key in ["name", "data", "batch_size", "num_err_layers", "num_output_layers", "num_meta_layers",
                    "num_pred_layers", "weight_decay", "dropout_rate", "lr", "hidden_dim", "kernel_size", "loss"]:
            if key in model_config:
                if key == "data":
                    data= model_config[key]
                    if data.endswith("/"):
                        data = data[:-1]
                    data = "/".join(data.split("/")[-2:])
                    df_dict[key].append(data)

                else:
                    df_dict[key].append(model_config[key])
            else:
                df_dict[key].append(None)
        val_loss = record_dict["val_loss"].numpy()
        df_dict["val_loss"].append(val_loss)
        metric_dict = record_dict["metric_dict"]
        df_dict["val_RMSE"].append(metric_dict["RMSE"].numpy())
        epoch = int(checkpoint.split(".")[0].split("-")[-2])
        df_dict["best_epoch"].append(epoch)
        step = int(checkpoint.split(".")[0].split("-")[-1])
        df_dict["best_step"].append(step)
        df_dict["path"].append(checkpoint)

    df = pd.DataFrame(df_dict)
    df.sort_values(by = ["val_RMSE"], inplace = True, ascending = True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    df.to_csv(f"{args.output}/{'-'.join(keyword)}-{timestamp}.tsv", sep = "\t", index = False)
    print(df[['name', 'data', 'best_epoch', 'best_step', 'val_RMSE', 'val_loss']])
    return None

if __name__ == "__main__":
    main()

