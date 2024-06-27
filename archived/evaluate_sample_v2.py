import gc

import torch
import os, glob
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from archived.inference_dataloader_v2 import load_dataset
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
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/", help="Output path")
    parser.add_argument("--batch", "-b", type=int, default=4000, help="Batch size")
    parser.add_argument("--shard", "-s", type=int, default=10000, help="Shard size")
    parser.add_argument("--gpu", "-g", type=int, default=4, help="GPU device")
    parser.add_argument("--nfile", "-n", type=int, default=10, help="Number of files to load")
    parser.add_argument("--prefetch", "-p", type=int, default=100, help="Number of files to load")
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
        if model_path.endswith(".pt"):
            model_list.append(model_path)
        elif os.path.isdir(model_path):
            model_list += [x for x in glob.glob(f"{model_path}/*.pt")]
        else:
            raise ValueError("Invalid model path. It should be a .pt file or a directory containing .pt files.")

    if args_dict["data"].endswith("/"):
        args_dict["data"] = args_dict["data"][:-1]
    if not os.path.isdir(args_dict["data"]):
        raise ValueError("Invalid data path. It should be a directory containing data files.")
    for model in model_list:
        printmessage(f"Running inference: {model}")
        args_dict_model = args_dict.copy()
        args_dict_model["model"] = model
        mp.spawn(inference_worker, nprocs=max(1,args.gpu), args=(args_dict_model,))
    return None


def inference_worker(rank, args_dict, flush_interval = 100):
    setup_ddp(rank, args_dict["gpu"])
    if args_dict["gpu"] > 0:
        save_dict = torch.load(args_dict["model"], map_location={'cuda:0': f'cuda:{rank}'})
    else:
        save_dict = torch.load(args_dict["model"], map_location='cpu')
    model_config = save_dict["model_config"]
    TransformerModel = importlib.import_module(f"model.{model_config['model']}").TransformerModel
    model = TransformerModel(d_model = model_config["enc_dim"], n_heads = model_config["head"], d_ff = model_config["lin_dim"],
                             n_layers = model_config["enc_layer"], lin_depth = model_config["lin_layer"],
                             t_act = model_config["t_act"], lin_act = model_config["lin_act"],
                             encoder_dropout = model_config["enc_dropout"], lin_dropout = model_config["lin_dropout"],
                             kmer_size = 5, signal_size = 25, spectrogram_size = 21, block_len = 17, seq_len=200)
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
    data_loader = load_dataset(args_dict["data"], args_dict["batch"], args_dict["shard"], rank, max(1,args_dict["gpu"]),
                               num_files_read_once = args_dict["nfile"], prefetch_factor = args_dict["prefetch"])

    id_list = []
    label_list = []
    pred_list = []
    block_id_list = []
    metadata_list = []
    genome_id_list=[]
    if "score_feature" in model_config:
        if model_config["score_feature"]:
            score_feature = True
        else:
            score_feature = False
    else:
        score_feature = False

    for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), smoothing = 0):
        data = data[0]
        if args_dict["gpu"] > 0:
            src_kmer = data["kmer_token"].to(rank)
            # src_signal = data["signal_token"]
            src_signal = data["signal_token"].to(rank)
            src_bq = data["bq_token"].to(rank)
            src_move = data["move_token"].to(rank)
            src_pad_mask = (src_kmer == 0)
            src_target_mask = data["target_mask"].to(rank)
            if score_feature:
                    ## Fill with one, shape is (batch, )
                    src_score = torch.ones(src_kmer.size(0)).to(rank)

        else:
            src_kmer = data["kmer_token"]
            # src_signal = data["signal_token"]
            src_signal = data["signal_token"]
            src_bq = data["bq_token"]
            src_move = data["move_token"]
            src_pad_mask = (src_kmer == 0)
            src_target_mask = data["target_mask"]
            if score_feature:
                src_score = torch.ones(src_kmer.size(0))

        with torch.no_grad():
            if score_feature:
                pred = model(src_kmer, src_signal, src_bq, src_move, src_pad_mask, src_target_mask, src_score)
            else:
                pred = model(src_kmer, src_signal, src_bq, src_move, src_pad_mask, src_target_mask)

        ## if pred has additional dimension, remove it.
        if len(pred.shape) > 1:
            target_mask_sum = src_target_mask.sum(dim = 1)
            pred = pred * src_target_mask
            pred = pred.sum(dim = 1)
            pred = pred / target_mask_sum

        if args_dict["gpu"] > 0:
            pred_list.append(pred.cpu().detach().numpy())
        else:
            pred_list.append(pred.detach().numpy())
        id_list.append(np.array(data["label_id"]))
        block_id_list.append(np.array(data["block_id"]))
        genome_id_list.append(np.array(data["genome_id"]))
        label_list.append(np.array(data["label"]))
        metadata_list += data["metadata"]

        if idx % flush_interval == 0 and idx > 0:
            id_list = np.concatenate(id_list)
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            block_id_list = np.concatenate(block_id_list)
            genome_id_list = np.concatenate(genome_id_list)

            data_df = pd.DataFrame({"label_id": id_list, "label": label_list, "pred": pred_list, "block_id": block_id_list, "genome_id": genome_id_list, "metadata": metadata_list})
            out_path = f"{args_dict['output']}/inference/{args_dict['model'].split('/')[-1].split('.')[0]}-{args_dict['data'].split('/')[-1]}/inference_{rank}_{idx}.pkl"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            data_df.to_pickle(out_path)

            id_list = []
            label_list = []
            pred_list = []
            block_id_list = []
            metadata_list = []
            genome_id_list = []

            del data_df
            gc.collect()

    id_list = np.concatenate(id_list)
    label_list = np.concatenate(label_list)
    pred_list = np.concatenate(pred_list)
    block_id_list = np.concatenate(block_id_list)
    genome_id_list = np.concatenate(genome_id_list)

    data_df = pd.DataFrame({"label_id": id_list, "label": label_list, "pred": pred_list, "block_id": block_id_list, "genome_id": genome_id_list, "metadata": metadata_list})
    out_path = f"{args_dict['output']}/inference/{args_dict['model'].split('/')[-1].split('.')[0]}-{args_dict['data'].split('/')[-1]}/inference_{rank}_last.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data_df.to_pickle(out_path)

    del data_df
    gc.collect()

    if args_dict["gpu"] > 0:
        dist.destroy_process_group()

    return None



if __name__ == "__main__":
    main()



