"""
DeespRM Inference Module

This program handles the inference process for DeepRM models, including loading the model,
processing input data, and saving the output predictions.
"""

import argparse
import glob
import importlib
import os
import pathlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm

from deeprm.inference.inference_dataloader import load_dataset
from deeprm.inference.pileup_deeprm import main as pileup_main
from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger

log = get_logger(__name__)
check_deps.check_torch_available()

import torch
import torch.multiprocessing as mp
from torch.amp import autocast


def add_arguments(parser: argparse.ArgumentParser):
    """Adds command-line arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.

    Returns:
        None
    """
    parser.add_argument("--input", "-i", dest="data", type=str, required=True, help="Data path")
    parser.add_argument("--bam", "-b", type=str, required=True, help="BAM file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model path")
    parser.add_argument("--model-type", "-y", type=str, default="deeprm_model", help="Model type")
    parser.add_argument("--batch", "-s", type=int, default=16000, help="Batch size")
    parser.add_argument("--gpu", "-g", type=int, default=None, help="Num. of GPU devices", dest="num_gpu")
    parser.add_argument("--prefetch", "-p", type=int, default=4, help="Number of files to load")
    parser.add_argument("--worker", "-w", type=int, default=4, help="Number of workers per GPU")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output directory")
    parser.add_argument("--flush", "-f", type=int, default=100, help="Flush interval for intermediate results.")
    parser.add_argument("--resume", action="store_true", help="Resume terminated inference.")
    parser.add_argument("--gpu-pool", "-gp", type=int, nargs="+", help="GPU pool")
    parser.add_argument("--output-id", "-id", type=int, default=None, help="Output ID for Multi-output models.")
    parser.add_argument("--thread", "-t", type=int, default=None, help="Number of threads to use for pileup")
    parser.add_argument("--threshold", "-th", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--epsilon", "-ep", type=float, default=1e-30, help="Epsilon value")
    parser.add_argument("--slice", "-sl", type=int, default=None, help="Slice index (for 2D predictions)")
    parser.add_argument("--flip", "-fl", action="store_true", help="Flip label")
    parser.add_argument(
        "--label_div", "-d", type=int, default=10**9, help="Divisor for label_id to separate transcript and position"
    )
    parser.add_argument("--annot", "-a", type=str, default=None, help="Annotation file (e.g., refFlat.txt)")

    return None


def main(args: argparse.Namespace):
    """Main function to run the evaluation pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None

    Notes:
        1. Parse command-line arguments.
        2. Create necessary directories.
        3. Run inference.
    """
    if args.model is None:
        ## Get directory of the current file
        deeprm_root = pathlib.Path(__file__).parent.parent.resolve()
        args.model = os.path.join(deeprm_root, "weight", "deeprm_weights.pt")
    if not args.model.endswith(".pt"):
        raise ValueError("Invalid model path. It should be a .pt file.")
    if args.data.endswith("/"):
        args.data = args.data[:-1]
    if not os.path.isdir(args.data):
        raise ValueError("Invalid data path. It should be a directory containing data files.")
    if args.num_gpu is None:
        if args.gpu_pool is None:
            args.num_gpu = torch.cuda.device_count()
        else:
            args.num_gpu = len(args.gpu_pool)
    if args.gpu_pool is None:
        args.gpu_pool = list(range(args.num_gpu))

    output = f"{args.output}/{os.path.basename(args.data)}"
    if len(args.postfix) > 0:
        output = f"{output}-{args.postfix}"
    args.output = output

    inference_output = os.path.join(args.output, "molecule-level")
    pileup_output = os.path.join(args.output, "site-level")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(inference_output, exist_ok=True)
    os.makedirs(pileup_output, exist_ok=True)

    args.output = inference_output
    run_inference(args)
    log.info("Inference Program Finished.")

    args.input = inference_output
    args.output = pileup_output
    pileup_main(args)
    log.info("Pileup Program Finished.")
    return None


def run_inference(args):
    """Runs the inference process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    torch.multiprocessing.set_sharing_strategy("file_system")
    log.info("Inference Program Started.")
    if args.num_gpu > 0:
        log.info(f"Using {args.num_gpu} GPUs.")
    else:
        log.info("Using CPU.")

    ## make tensorboard directory
    log.info(f"Model path: {args.model}")
    log.info(f"Output directory: {args.output}")
    mp.spawn(inference_worker, nprocs=max(1, args.num_gpu), args=(vars(args),), join=True)
    return None


def inference_worker(rank, args_dict):
    """Worker function for running inference on a single GPU.

    Args:
        rank (int): Rank of the current process.
        args_dict (dict): Dictionary of command-line arguments.

    Returns:
        None
    """
    gpu_id = args_dict["gpu_pool"][rank]
    if args_dict["num_gpu"] > 0:
        save_dict = torch.load(args_dict["model"], map_location={"cuda:0": f"cuda:{gpu_id}"}, weights_only=False)
    else:
        save_dict = torch.load(args_dict["model"], map_location="cpu", weights_only=False)
    model_config = save_dict["model_config"]

    if args_dict["model_type"] is not None:
        model_config["model"] = args_dict["model_type"]

    dwell_bq_dim = 3
    TransformerModel = importlib.import_module(f"deeprm.model.{model_config['model']}").TransformerModel

    model = TransformerModel(
        d_model=model_config["enc_dim"],
        n_heads=model_config["head"],
        d_ff=model_config["lin_dim"],
        n_layers=model_config["enc_layer"],
        lin_depth=model_config["lin_layer"],
        t_act=model_config["t_act"],
        lin_act=model_config["lin_act"],
        encoder_dropout=model_config["enc_dropout"],
        lin_dropout=model_config["lin_dropout"],
        kmer_size=model_config["kmer_size"],
        signal_size=model_config["signal_size"],
        block_len=model_config["block_len"],
        seq_len=model_config["seq_len"],
        signal_stride=model_config["signal_stride"],
        dwell_bq_dim=dwell_bq_dim,
    )

    if rank == 0:
        total_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            total_params += params
    if args_dict["num_gpu"] > 0:
        model.to(gpu_id)
    model.load_state_dict(state_dict=save_dict["model_state_dict"], strict=False)
    save_dict.clear()
    model.eval()

    if args_dict["resume"]:
        saved = glob.glob(os.path.join(args_dict["output"], f"inference_{rank}_*.pkl"))
        if len(saved) > 0:
            saved = [int(x.split("/")[-1].split("_")[-1].split(".")[0]) for x in saved]
            saved = max(saved)
        else:
            saved = 0
    else:
        saved = 0

    data_loader = load_dataset(
        args_dict["data"],
        gpu_id,
        max(1, args_dict["num_gpu"]),
        prefetch_factor=args_dict["prefetch"],
        worker=args_dict["worker"],
        cb_len=model_config["block_len"] + model_config["kmer_size"] - 1,
        kmer_len=model_config["kmer_size"],
        sampling=int(model_config["signal_size"] / model_config["kmer_size"]),
        sig_window=model_config["kmer_size"],
        resume_from=saved,
    )

    inference_loop(args_dict, rank, gpu_id, model, data_loader)

    return None


def to_gpu(data, device, stream):
    """Transfers data to the specified GPU device using a non-blocking stream.

    Args:
        data (dict): Dictionary containing the data to be transferred.
        device (torch.device): The target GPU device.
        stream (torch.cuda.Stream): The CUDA stream for non-blocking transfer.

    Returns:
        tuple: A tuple containing the transferred data tensors (src_kmer, src_signal, src_seg_len, src_dwell_bq).
    """
    with torch.cuda.stream(stream):
        src_signal = data["signal_token"].to(device, non_blocking=True)
        src_seg_len = data["segment_len"].to(device, non_blocking=True)
        src_kmer = data["kmer_token"].to(device, non_blocking=True)
        src_dwell_bq = data["dwell_bq_token"].to(device, non_blocking=True)
    return (src_kmer, src_signal, src_seg_len, src_dwell_bq)


def inference_loop(args_dict, rank, gpu_id, model, data_loader):
    """Runs the inference loop for the given model and data loader.

    Args:
        args_dict (dict): Dictionary of command-line arguments.
        rank (int): Rank of the current process.
        gpu_id (int): ID of the GPU to use.
        model (torch.nn.Module): The model to run inference on.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.

    Returns:
        None
    """
    with autocast(enabled=True, cache_enabled=True, device_type="cuda"):
        with torch.no_grad():
            executor = ThreadPoolExecutor(max_workers=1)
            copy_stream = torch.cuda.Stream(device=gpu_id)

            pred_buffer = deque(maxlen=args_dict["flush"])
            label_id_buffer = deque(maxlen=args_dict["flush"])
            read_id_buffer = deque(maxlen=args_dict["flush"])
            processed_batches = -1

            batch_data_buffer = None

            for chunk_data_cpu in tqdm.tqdm(data_loader, total=len(data_loader), smoothing=0):

                if batch_data_buffer is not None:
                    chunk_data_cpu = {k: torch.cat((batch_data_buffer[k], v), dim=0) for k, v in chunk_data_cpu.items()}
                    batch_data_buffer = None
                batch_data_split_cpu = {k: torch.split(v, args_dict["batch"]) for k, v in chunk_data_cpu.items()}

                for bidx in range(len(batch_data_split_cpu["label_id"])):
                    batch_data_cpu = {k: v[bidx] for k, v in batch_data_split_cpu.items()}

                    if len(batch_data_cpu["label_id"]) < args_dict["batch"]:
                        # Buffer the smaller batch for next read
                        if batch_data_buffer is not None:
                            log.warning("Error in data batching - more than one smaller batch occured in a read.")
                            batch_data_buffer = {
                                k: torch.cat((batch_data_buffer[k], v), dim=0) for k, v in batch_data_cpu.items()
                            }
                        else:
                            batch_data_buffer = batch_data_cpu
                        continue

                    if processed_batches == -1:
                        ## First batch
                        batch_data_gpu = to_gpu(batch_data_cpu, gpu_id, copy_stream)
                        label_id = batch_data_cpu["label_id"]
                        read_id = batch_data_cpu["read_id"]
                        processed_batches += 1
                        continue

                    to_gpu_proc = executor.submit(to_gpu, batch_data_cpu, gpu_id, copy_stream)
                    pred = model(*batch_data_gpu)

                    if args_dict["output_id"] is not None:
                        pred = pred[args_dict["output_id"]]

                    label_id_buffer.append(label_id)
                    read_id_buffer.append(read_id)
                    pred_buffer.append(pred)

                    processed_batches += 1

                    if processed_batches % args_dict["flush"] == 0 and processed_batches > 0:
                        preds = torch.cat(list(pred_buffer), dim=0).detach().cpu().numpy()
                        label_ids = torch.cat(list(label_id_buffer), axis=0).numpy()
                        read_ids = torch.cat(list(read_id_buffer), axis=0).numpy()
                        pred_buffer.clear()
                        label_id_buffer.clear()
                        read_id_buffer.clear()
                        out_path = f"{args_dict['output']}/inference_{rank}_{processed_batches}.npz"
                        np.savez_compressed(out_path, label_id=label_ids, read_id=read_ids, pred=preds)

                    torch.cuda.current_stream(gpu_id).wait_stream(copy_stream)
                    batch_data_gpu = to_gpu_proc.result()
                    label_id = batch_data_cpu["label_id"]
                    read_id = batch_data_cpu["read_id"]

            ## Last batch
            pred = model(*batch_data_gpu)

            if args_dict["output_id"] is not None:
                pred = pred[args_dict["output_id"]]

            label_id_buffer.append(label_id)
            read_id_buffer.append(read_id)
            pred_buffer.append(pred)
            processed_batches += 1

            preds = torch.cat(list(pred_buffer), dim=0).detach().cpu().numpy()
            ids = torch.cat(list(label_id_buffer), axis=0).numpy()
            label_id_buffer.clear()
            read_ids = torch.cat(list(read_id_buffer), axis=0).numpy()
            read_id_buffer.clear()
            pred_buffer.clear()
            out_path = f"{args_dict['output']}/inference_{rank}_{processed_batches}.npz"
            np.savez_compressed(out_path, label_id=ids, read_id=read_ids, pred=preds)

            executor.shutdown(wait=True)

    return None
