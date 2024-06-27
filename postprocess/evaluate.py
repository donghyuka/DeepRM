import gc

import torch
import os, glob
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from postprocess.dataloader_inference_site import load_dataset
from utils.utils import printmessage
import torch.multiprocessing as mp
import tqdm
import importlib

## 1. Load Eval Data and Model
## 2. Run Inference.
## 3. Create Site-level Predictions. There is no need to use PILEUP when evaluating on a sampled dataset.
## 4. Evaluate against ground truth labels and save evaluation results.
## 5. Plot evaluation results.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, nargs="+", help="Model path")
    parser.add_argument("--data", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block/block", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/inference/", help="Output path")
    parser.add_argument("--batch", "-b", type=int, default=512, help="Batch size")
    parser.add_argument("--gpu", "-g", type=int, default=4, help="GPU device")
    parser.add_argument("--bag_size", "-s", type=int, default=200, help="Bag size")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ## Subdirectories for results: inference, pileup, evaluation, plot
    inference_path = f"{args.output}/inference"
    plot_path = f"{args.output}/plot"
    os.makedirs(inference_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    run_inference(args)
    return None


def setup_ddp(rank,world_size):
    if world_size == 0:
        ## use CPU.
        return None

    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    return None

def run_inference(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    args_dict = vars(args)
    printmessage("Inference Program Started.")
    if args_dict["gpu"] > 0:
        printmessage(f"Using {args.gpu} GPUs.")
    else:
        printmessage("Using CPU.")

    model_list = []
    for model_path in args_dict["model"]:
        if not os.path.exists(model_path):
            model_path = "/extdata4/baeklab/Hyeonseo/m6A/model/" + model_path
            if not os.path.exists(model_path):
                raise ValueError("Invalid model path. It should be a .pt file or a directory containing .pt files.")
        if model_path.endswith(".pt"):
            model_list.append(model_path)
        elif os.path.isdir(model_path):
            model_list += [x for x in glob.glob(f"{model_path}/*.pt")]
        else:
            raise ValueError("Invalid model path. It should be a .pt file or a directory containing .pt files.")
    args_dict["model"] = model_list
    if args_dict["data"].endswith("/"):
        args_dict["data"] = args_dict["data"][:-1]
    mp.spawn(inference_worker, nprocs=max(1,args.gpu), args=(args_dict,))
    return None


def inference_worker(rank, args_dict):
    setup_ddp(rank, args_dict["gpu"])

    seed = 0
    world_size = args_dict["gpu"]
    data_path = args_dict["data"]
    eval_batch_size = args_dict["batch"]
    bag_size = args_dict["bag_size"]
    num_workers = 1
    pin_memory = False
    prefetch_factor = 10

    data_loader = load_dataset(seed, rank, world_size, data_path, eval_batch_size, num_workers, pin_memory,
                              False, prefetch_factor, bag_size, shuffle=False)

    for model_path in args_dict["model"]:
        if rank == 0:
            printmessage(f"Running inference: {model_path}")
        save_dict = torch.load(model_path, map_location=f'cuda:{rank}')
        model_config = save_dict["model_config"]
        CNNModel = importlib.import_module(f"postprocess.{model_config['model']}").CNNModel
        model = CNNModel(input_height=model_config["input_height"],
                         input_width=model_config["input_width"],
                         hidden_dim=model_config["hidden_dim"],
                         output_dim=model_config["output_dim"],
                         num_err_layers=model_config["num_err_layers"],
                         num_meta_layers=model_config["num_meta_layers"],
                         num_pred_layers=model_config["num_pred_layers"],
                         num_output_layers=model_config["num_output_layers"],
                         dropout_rate=model_config["dropout_rate"],
                         kernel_size=model_config["kernel_size"])
        if rank == 0:
            total_params = 0
            for name, parameter in model.named_parameters():
                params = parameter.numel()
                total_params += params
            printmessage(f"Total Params: {total_params:,}")
        if args_dict["gpu"] > 0:
            model.to(rank)
        model.load_state_dict(state_dict=save_dict["model_state_dict"])
        save_dict.clear()
        if args_dict["gpu"] > 0:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        model.eval()

        label_list = []
        pred_list = []

        for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), smoothing = 0):
            source, target, id = data
            error, metadata, pred, mask = source
            error = error.to(rank)
            metadata = metadata.to(rank)
            pred = pred.to(rank)
            mask = mask.to(rank)
            target = target.detach().numpy()
            output = model(error, metadata, pred, mask).cpu().detach().numpy()

            if len(output.shape) > 1:
                output = output[:, 0]

            if len(target.shape) > 1:
                target = target[:, 0]

            pred_list.append(output)
            label_list.append(id)

        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        data_df = pd.DataFrame({"label_id": label_list, "pred": pred_list})
        out_path = f"{args_dict['output']}/inference/{model_path.split('/')[-1].split('.')[0]}-{args_dict['data'].split('/')[-1]}/inference_{rank}_last.tsv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data_df.to_csv(out_path, sep='\t', index=False)

        del data_df
        gc.collect()

        dist.barrier()

    if args_dict["gpu"] > 0:
        dist.destroy_process_group()

    return None


if __name__ == "__main__":
    main()



