import time
import torch
import os, glob
import argparse
import numpy as np
from inference.inference_dataloader_npz import load_dataset
from utils.utils import printmessage
import torch.multiprocessing as mp
import tqdm
import importlib
from collections import deque
from torch.amp import autocast
from concurrent.futures import ProcessPoolExecutor

## 1. Load Eval Data and Model
## 2. Run Inference.
## 3. Create Site-level Predictions. There is no need to use PILEUP when evaluating on a sampled dataset.
## 4. Evaluate against ground truth labels and save evaluation results.
## 5. Plot evaluation results.


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, nargs="+", help="Model path")
    parser.add_argument("--model_type", "-t", type=str, default=None, help="Model type")
    parser.add_argument("--data", "-d", type=str, required=True, help="Data path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--batch", "-b", type=int, default=10000, help="Batch size")
    parser.add_argument("--shard", "-s", type=int, default=10000, help="Shard size")
    parser.add_argument("--gpu", "-g", type=int, default=4, help="GPU device", dest="num_gpu")
    parser.add_argument("--prefetch", "-p", type=int, default=16, help="Number of files to load")
    parser.add_argument("--worker", "-w", type=int, default=8, help="Number of workers per GPU")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output directory")
    parser.add_argument("--flush", "-f", type=int, default=100, help="Flush interval for intermediate results.")
    parser.add_argument("--resume", action="store_true", help="Resume terminated inference.")
    parser.add_argument("--gpu_pool", "-gp", type=int, nargs="+", help="GPU pool")
    parser.add_argument("--output_id", "-i", type=int, default=None, help="Output ID for Multi-output models.")
    args = parser.parse_args()
    if args.num_gpu is None:
        if args.gpu_pool is None:
            args.num_gpu = torch.cuda.device_count()
        else:
            args.num_gpu = len(args.gpu_pool)
    if args.gpu_pool is None:
        args.gpu_pool = list(range(args.num_gpu))
    return args


def main():
    """
    Main function to run the evaluation pipeline.

    Steps:
        1. Parse command-line arguments.
        2. Create necessary directories.
        3. Run inference.
    """
    args = parse_args()
    inference_path = f"{args.output}/inference"
    os.makedirs(inference_path, exist_ok=True)
    run_inference(args)
    return None


def run_inference(args):
    """
    Runs the inference process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
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
        ## make tensorboard directory
        profile_dir = os.path.join(args.profile_path,os.path.basename(model)+args_dict['postfix']+".profile"+f".{time.strftime('%Y%m%d%H%M%S')}")
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir, exist_ok=True)
        printmessage(f"Running inference: {model}")
        args_dict_model = args_dict.copy()
        args_dict_model["model"] = model
        args_dict_model["profile_dir"] = profile_dir
        out_dir = f"{args_dict['output']}/inference/{model.split('/')[-1][:-3]}-{args_dict['data'].split('/')[-1]}"
        if len(args_dict["postfix"]) > 0:
            out_dir = f"{out_dir}-{args_dict['postfix']}"
        print(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        args_dict_model["out_dir"] = out_dir
        mp.spawn(inference_worker, nprocs=max(1,args.gpu), args=(args_dict_model,))
    return None


def inference_worker(rank, args_dict):
    """
    Worker function for running inference on a single GPU.

    Args:
        rank (int): Rank of the current process.
        args_dict (dict): Dictionary of command-line arguments.

    Returns:
        None
    """
    gpu_id = args_dict["gpu_pool"][rank]
    if args_dict["gpu"] > 0:
        save_dict = torch.load(args_dict["model"], map_location={'cuda:0': f'cuda:{gpu_id}'}, weights_only=False)
    else:
        save_dict = torch.load(args_dict["model"], map_location='cpu', weights_only=False)
    model_config = save_dict["model_config"]

    if "model" not in model_config:
        model_config["model"] = model_config["model_type"]

    if "spectrogram_size" not in model_config:
        model_config["spectrogram_size"] = 21

    dwell_bq_dim = 3

    TransformerModel = importlib.import_module(f"model.{model_config['model']}").TransformerModel
    model = TransformerModel(d_model = model_config["enc_dim"], n_heads = model_config["head"], d_ff = model_config["lin_dim"],
                             n_layers = model_config["enc_layer"], lin_depth = model_config["lin_layer"],
                             t_act = model_config["t_act"], lin_act = model_config["lin_act"],
                             encoder_dropout = model_config["enc_dropout"], lin_dropout = model_config["lin_dropout"],
                             kmer_size = model_config["kmer_size"], signal_size = model_config["signal_size"],
                             spectrogram_size = model_config["spectrogram_size"], block_len = model_config["block_len"],
                             seq_len = model_config["seq_len"], signal_stride = model_config["signal_stride"],
                             dwell_bq_dim = dwell_bq_dim)

    if rank == 0:
        total_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            total_params += params
        printmessage(f"Total Params: {total_params:,}")
    if args_dict["gpu"] > 0:
        model.to(gpu_id)
    model.load_state_dict(state_dict=save_dict["model_state_dict"])
    save_dict.clear()
    model.eval()

    if args_dict["resume"]:
        saved = glob.glob(f"{args_dict['out_dir']}/inference_{rank}_*.pkl")
        if len(saved) > 0:
            saved = [int(x.split("/")[-1].split("_")[-1].split(".")[0]) for x in saved]
            saved = max(saved)
        else:
            saved = 0
    else:
        saved = 0

    data_loader = load_dataset(args_dict["data"], args_dict["batch"], args_dict["shard"], gpu_id, max(1,args_dict["gpu"]),
                               prefetch_factor = args_dict["prefetch"],
                               worker = args_dict["worker"],
                               cb_len = model_config["block_len"] + model_config["kmer_size"] - 1,
                               kmer_len = model_config["kmer_size"],
                               sampling = int(model_config["signal_size"] / model_config["kmer_size"]),
                               sig_window = model_config["kmer_size"],
                               resume_from = saved)

    inference_loop(args_dict, rank, gpu_id, model, data_loader)

    return None


def to_gpu(data, rank):
    src_signal = data["signal_token"].to(rank)
    src_seg_len = data["segment_len"].to(rank)
    src_kmer = data["kmer_token"].to(rank)
    src_dwell_bq = data["dwell_bq_token"].to(rank)
    return [src_kmer, src_signal, src_seg_len, src_dwell_bq]

def pred_step(data, model):
    pred = model(*data)
    return pred

def inference_loop(args_dict, rank, gpu_id, model, data_loader):
    with autocast(enabled=True, cache_enabled=True, device_type="cuda"):
        with torch.no_grad():
            executor = ProcessPoolExecutor()

            pred_buffer = deque(maxlen=args_dict["flush"])
            id_buffer = deque(maxlen=args_dict["flush"])
            flush_idx = -1
            idx = -1

            for next_data in tqdm.tqdm(data_loader, total=len(data_loader), smoothing = 0):

                if idx == -1:
                    ## First batch
                    this_data = to_gpu(next_data, gpu_id)
                    this_label = next_data["label_id"]
                    idx+=1
                    flush_idx += 1
                    continue

                next_data_proc = executor.submit(to_gpu, next_data, rank)
                this_pred = model(*this_data)

                if args_dict["output_id"] is not None:
                    pred = pred[args_dict["output_id"]]

                id_buffer.append(this_label)
                pred_buffer.append(this_pred)

                if flush_idx == args_dict["flush"]-1:
                    preds = torch.cat(list(pred_buffer), dim=0).detach().cpu().numpy()
                    ids = np.concatenate(list(id_buffer), axis=0)
                    id_buffer.clear()
                    pred_buffer.clear()
                    out_path = f"{args_dict['out_dir']}/inference_{rank}_{idx}.npz"
                    np.savez_compressed(out_path, label_id = ids, pred = preds)
                    flush_idx = 0
                else:
                    flush_idx += 1

                this_data = next_data_proc.result()
                this_label = next_data["label_id"]
                idx+=1

            ## Last batch
            this_pred = model(*this_data)

            if args_dict["output_id"] is not None:
                this_pred = pred[args_dict["output_id"]]

            id_buffer.append(this_label)
            pred_buffer.append(this_pred)

            preds = torch.cat(list(pred_buffer), dim=0).detach().cpu().numpy()
            ids = np.concatenate(list(id_buffer), axis=0)
            id_buffer.clear()
            pred_buffer.clear()
            out_path = f"{args_dict['out_dir']}/inference_{rank}_{idx}.npz"
            np.savez_compressed(out_path, label_id = ids, pred = preds)

    return None


if __name__ == "__main__":
    main()



