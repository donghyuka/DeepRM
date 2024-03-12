import torch
import glob
import os
import pandas as pd

def main():
    path = "/extdata4/baeklab/Hyeonseo/m6A/model"
    subpaths = glob.glob(f"{path}/*")
    checkpoint_list = []
    keyword = ["v11","v12"]
    subpaths = [subpath for subpath in subpaths if any([key in subpath for key in keyword])]
    for subpath in subpaths:
        checkpoints = glob.glob(f"{subpath}/*.pt")
        if len(checkpoints) > 0:
            checkpoint_dict = {int(x.split(".")[0].split("-")[-1]):x for x in checkpoints}
            max_epoch = max(checkpoint_dict.keys())
            checkpoint = checkpoint_dict[max_epoch]
            checkpoint_list.append(checkpoint)

    print(checkpoint_list)

    df_dict = {"name": [], "data": [], "best_epoch": [], "enc_dim": [], "lin_dim": [], "head": [], "enc_layer": [],
               "lin_layer": [], "enc_dropout": [], "lin_dropout": [], "lr": [], "batch_size": [], "weight_decay": [],
               "val_loss": [], "val_auroc": [], "val_ap": [], "val_f1": [], "val_acc": []}

    for checkpoint in checkpoint_list:
        record_dict = torch.load(checkpoint, map_location="cpu")
        model_config = record_dict["model_config"]
        for key in ["name", "data", "enc_dim", "lin_dim", "head", "enc_layer", "lin_layer", "enc_dropout",
                    "lin_dropout", "lr", "batch_size", "weight_decay"]:
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
        df_dict["val_acc"].append(metric_dict["acc"].numpy())
        df_dict["val_auroc"].append(metric_dict["auroc"].numpy())
        df_dict["val_f1"].append(metric_dict["f-1"].numpy())
        df_dict["val_ap"].append(metric_dict["ap"].numpy())
        epoch = int(checkpoint.split(".")[0].split("-")[-1])
        df_dict["best_epoch"].append(epoch)

    df = pd.DataFrame(df_dict)
    print(df)
    df.to_csv(f"{path}/checkpoint_summary.tsv", sep = "\t", index = False)
    return None

if __name__ == "__main__":
    main()

