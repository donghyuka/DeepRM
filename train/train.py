import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import load_dataset
from model.transformer_prototype import TransformerModel
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.ClassificationMetric as cm
import GPUtil
import transformers
import argparse
import os
import time
import tqdm
import math

from utils.utils import printmessage


## To Include:
# 1. Distributed Data Parallel
# 2. Learning Rate Scheduler (transformers.get_cosine_with_hard_restarts_schedule_with_warmup)
## Default optimizer is AdamW. (Use gradient clipping)


def parse_args():
    parser = argparse.ArgumentParser("Train Transformer Model")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--val_path", type=str, default="val")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--tb_path", type=str, default="tb")
    parser.add_argument("--es_delta", type=str, default="es_delta")
    parser.add_argument("--es_patience", type=str, default="es_patience")
    parser.add_argument("--es_start", type=str, default="es_start")
    return parser.parse_args()

def setup_device(args):
    if args.gpu > 0:
        device = torch.device('cuda')
        GPUtil.showUtilization()
    else:
        device = torch.device('cpu')
    return device


def setup_ddp(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return None


def train_step(model, train_loader, optimizer, scheduler, criterion, device, tb_writer,
               epoch_num, metric_func_dict, grad_clip = 0.5, log_interval = 10):

    model.train()
    total_loss = 0.
    start_time = time.time()

    with tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for batch in train_loader:
            optimizer.zero_grad()
            src_kmer, src_signal, src_spectrogram, src_bq, target = batch
            src_kmer = src_kmer.to(device)
            src_signal = src_signal.to(device)
            src_spectrogram = src_spectrogram.to(device)
            src_bq = src_bq.to(device)
            target = target.to(device)
            output = model(src_kmer, src_signal, src_spectrogram, src_bq)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                batch_time = (time.time() - start_time)  / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                total_loss = 0
                start_time = time.time()
                pbar.set_description(f"Epoch {epoch_num}")
                pbar.update(log_interval)
                pbar.set_postfix_str(f"Batch {batch} | Loss {cur_loss:.2f} | PPL {ppl:.2f} | LR {lr:.2f} | {batch_time:.2f}s")


    tb_writer.add_scalar("Loss", total_loss/len(train_loader))
    tb_writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'])
    tb_writer.flush()

    return None


def eval_step(model, val_loader, criterion, device, tb_writer, epoch_num, metric_func_dict):
    model.eval()
    total_loss = 0.
    outputs = []
    targets = []
    with torch.no_grad():
        for batch in val_loader:
            src_kmer, src_signal, src_spectrogram, src_bq, target = batch
            src_kmer = src_kmer.to(device)
            src_signal = src_signal.to(device)
            src_spectrogram = src_spectrogram.to(device)
            src_bq = src_bq.to(device)
            target = target.to(device)
            output = model(src_kmer, src_signal, src_spectrogram, src_bq)
            loss = criterion(output, target)
            total_loss += loss.item()
            outputs.append(output)
            targets.append(target)

    total_loss /= len(val_loader)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    metric_dict = {}

    for metric_name, metric_func in metric_func_dict.items():
        metric_dict[metric_name] = metric_func(outputs, targets)
        tb_writer.add_scalar(f"Val_{metric_name}", metric_dict[metric_name])
    tb_writer.add_scalar("Val_Loss", total_loss)
    tb_writer.flush()

    evaltext = f"Epoch {epoch_num} | Val Loss {total_loss:.2f} | "
    evaltext += " | ".join([f"{k} {v:.2f}" for k,v in metric_dict.items()])
    printmessage(evaltext)

    return total_loss, metric_dict


def main():
    args = parse_args()
    device = setup_device(args)
    setup_ddp(0, args.gpu)
    train_loader = load_dataset(args.train_path, args.batch_size)
    val_loader = load_dataset(args.val_path, args.batch_size)
    model = TransformerModel(d_model = 512, n_heads = 8, d_ff = 2048, d_kmer_embedding = 128, d_signal_embedding = 128,
                                d_spectrogram_embedding = 128, d_bq_embedding = 128, d_pos_encoding = 128, n_layers = 6,
                                encoder_dropout = 0.1, lin_dropout = 0.1, kmer_size = 5, signal_size = 5, spectrogram_size = 20,
                                t_act = 'gelu', lin_act = 'relu', lin_depth = 1)
    model = model.to(device)
    model = DDP(model, device_ids = [0], output_device = 0)
    optimizer = transformers.AdamW(model.parameters(), lr = args.lr)
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = 1000, num_training_steps = 10000)
    criterion = torch.nn.MSELoss()
    tb_writer = SummaryWriter(args.tb_path)
    metric_func_dict = {"acc": cm.Accuracy(), "auc": cm.AUROC(), "f1": cm.F1()}
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    epoch = 0
    wall_clock = time.time()

    for epoch in range(args.epochs):
        train_step(model, train_loader, optimizer, scheduler, criterion, device, tb_writer, epoch, metric_func_dict)
        val_loss, metric_dict = eval_step(model, val_loader, criterion, device, tb_writer, epoch, metric_func_dict)
        if val_loss < best_val_loss:
            ## Save Model if Improved
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 'val_loss': val_loss, 'metric_dict': metric_dict,
                        }, f"{args.output}/best_model.pt")
        else:
            if epoch > args.es_start and epoch - best_val_loss_epoch > args.es_patience:
                print(f"Early Stopping at Epoch {epoch}")
                break

    print("\n==========================================================")
    printmessage(f"Training Complete.")
    wall_clock = time.time() - wall_clock
    print(f"Total Time: {time.strftime('%H:%M:%S', time.gmtime(wall_clock))}")
    print(f"Total Epochs: {epoch+1}")
    print(f"Best Val Loss: {best_val_loss:.2f} at Epoch {best_val_loss_epoch}")
    print(f"Best Model Saved at: {args.output}/best_model.pt")
    print("==========================================================\n")

    return None


if __name__ == "__main__":
    main()

