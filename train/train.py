import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train.dataloader import load_dataset, NanoporeDataset, NanoporeDataLoader
from model.transformer_prototype_v2 import TransformerModel
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torchmetrics.classification as cm
import transformers
import argparse
import os
import time
import tqdm
import math
import numpy as np
from utils.utils import printmessage


def parse_args():
    parser = argparse.ArgumentParser("Train Transformer Model")
    parser.add_argument("--gpu", type=int, default = 4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/dataset/ver021324/main/")
    parser.add_argument("--output", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/model")
    parser.add_argument("--tb", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/tensorboard")
    parser.add_argument("--es_delta", type=float, default=1e-5)
    parser.add_argument("--es_patience", type=int, default=30)
    parser.add_argument("--es_start", type=int, default=30)
    parser.add_argument("--disk_shard_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    strfttime = time.strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--name", type=str, default=f"BERMUDA-Proto-v2-{strfttime}")
    return parser.parse_args()


class Trainer:
    def __init__(
            self,
            gpu_id: int,
            model: torch.nn.Module,
            train_loader: NanoporeDataLoader,
            val_loader: NanoporeDataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_func: torch.nn.Module,
            grad_clip: float,
            metric_func_dict: dict,
            log_interval: int,
            checkpoint_path: str,
            tb_path: str,
            es_start: int,
            es_patience: int,
            es_delta: float,
            model_name: str,
            num_gpu: int,
    ) -> None:

        self.gpu_id = gpu_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.tb_writer = SummaryWriter(tb_path)
        self.metric_func_dict = metric_func_dict
        self.checkpoint_path = checkpoint_path
        self.es_start = es_start
        self.es_patience = es_patience
        self.es_delta = es_delta
        self.continue_training = 1
        self.best_val_loss = np.inf
        self.best_val_loss_epoch = 0
        self.current_val_loss = 0
        self.current_val_metric_dict = {}
        self.current_epoch = 0
        self.current_batch = 0
        self.current_log_interval_loss = 0
        self.current_batch_loss = 0
        self.pbar = None
        self.model_name = model_name
        self.num_gpu = num_gpu


        ## END of __init__


    def _feed_model(self, source, target):
        src_kmer = source["kmer_token"]
        src_signal = source["signal_token"]
        src_spectrogram = source["spectrogram_token"]
        src_bq = source["bq_token"]
        src_pad_mask = (src_kmer == 0)
        src_target_mask = source["target_mask"]
        src_move = source["move_token"]

        src_kmer = src_kmer.to(self.gpu_id)
        src_signal = src_signal.to(self.gpu_id)
        src_spectrogram = src_spectrogram.to(self.gpu_id)
        src_bq = src_bq.to(self.gpu_id)
        src_pad_mask = src_pad_mask.to(self.gpu_id)
        src_target_mask = src_target_mask.to(self.gpu_id)
        src_move = src_move.to(self.gpu_id)
        target = target.to(torch.float32)
        target = target.to(self.gpu_id)
        output = self.model(src_kmer, src_signal, src_spectrogram, src_bq, src_move, src_pad_mask, src_target_mask)
        return output, target


    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output, target = self._feed_model(source, target)
        loss = self.loss_func(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.current_log_interval_loss += loss.item()
        self.current_batch_loss = loss.item()

        if self.current_batch % self.log_interval == 0:
            lr = self.scheduler.get_last_lr()[0]
            cur_loss = self.current_log_interval_loss / self.log_interval
            self.current_log_interval_loss = 0
            self.pbar.update(self.log_interval)
            self.pbar.set_postfix_str(f"Loss {cur_loss:.3E} | LR {lr:.3E}")

        self.current_batch += 1
        return None


    def _run_epoch(self):
        self.train_loader.set_epoch(self.current_epoch)
        self.val_loader.set_epoch(self.current_epoch)
        self.current_batch_loss = 0
        self.current_log_interval_loss = 0
        self.current_batch = 0
        self.model.train()
        colour_choice = ["red", "green", "blue", "yellow", "magenta", "cyan", "white", "black"]
        with tqdm.tqdm(total=len(self.train_loader) // self.num_gpu, desc=f"[GPU {self.gpu_id}] Epoch {self.current_epoch}",
                       position=self.gpu_id, colour=colour_choice[self.gpu_id%len(colour_choice)]) as self.pbar:
            for source, targets in self.train_loader:
                self._run_batch(source, targets)
        self.tb_writer.add_scalar("Loss", self.current_batch_loss/len(self.train_loader))
        self.tb_writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'])
        self.scheduler.step()
        return None


    def _run_eval(self):
        self.model.eval()
        total_loss = 0.
        outputs = []
        targets = []
        with torch.no_grad():
            for source, target in self.val_loader:
                output, target = self._feed_model(source, target)
                loss = self.loss_func(output, target)
                total_loss += loss.item()
                outputs.append(output)
                targets.append(target)

        total_loss /= len(self.val_loader)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        metric_dict = {}

        for metric_name, metric_func in self.metric_func_dict.items():
            metric_dict[metric_name] = metric_func(outputs, targets)
            self.tb_writer.add_scalar(f"Val_{metric_name}", metric_dict[metric_name])
        self.tb_writer.add_scalar("Val_Loss", total_loss)

        evaltext = f"Epoch {self.current_epoch} | Val Loss {total_loss:.2E} | "
        evaltext += " | ".join([f"{k} {v:.2E}" for k,v in metric_dict.items()])
        if self.gpu_id == 0:
            printmessage(evaltext)

        self.current_val_loss = total_loss
        self.current_val_metric_dict = metric_dict

        return None


    def _save_checkpoint(self):
        if self.best_val_loss - self.current_val_loss > self.es_delta:
            ## Save Model if Improved
            self.best_val_loss = self.current_val_loss
            self.best_val_loss_epoch = self.current_epoch
            torch.save({'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': self.current_val_loss,
                        'metric_dict': self.current_val_metric_dict,
                        }, f"{self.checkpoint_path}/{self.model_name}-{self.current_epoch}.pt")
            printmessage(f"Model Saved at Epoch {self.current_epoch}")
            self.continue_training = 1

        elif self.current_epoch > self.es_start and self.current_epoch - self.best_val_loss_epoch > self.es_patience:
            printmessage(f"Early Stopping at Epoch {self.current_epoch}")
            self.continue_training = 0

        else:
            printmessage(f"Skipping Model Save at Epoch {self.current_epoch}")
            self.continue_training = 1

        return None


    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            self._run_epoch()
            self._run_eval()
            if self.gpu_id == 0:
                self._save_checkpoint()
            if self.continue_training == 0:
                break
        self.tb_writer.flush()
        return None

    ## END of Class NanoporeTrainer


def setup_ddp(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return None


def prepare_dataloader(data_path, batch_size, disk_shard_size, rank, num_replicas, seed):

    batch_size = batch_size
    train_pos_data_path = f"{data_path}/train/pos"
    train_neg_data_path = f"{data_path}/train/neg"
    val_pos_data_path = f"{data_path}/val/pos"
    val_neg_data_path = f"{data_path}/val/neg"

    train_loader = load_dataset(train_pos_data_path, train_neg_data_path, batch_size,
                                disk_shard_size, rank, num_replicas, seed = seed, shuffle = True, drop_last = True)
    val_loader = load_dataset(val_pos_data_path, val_neg_data_path, batch_size,
                              disk_shard_size, rank, num_replicas, seed = seed, shuffle = False, drop_last = True)

    return train_loader, val_loader


def main_worker(rank, args_dict):
    printmessage(f"[GPU {rank}] Worker Process Started.")
    setup_ddp(rank, args_dict["gpu"])

    model = TransformerModel(d_model = 128*6, n_heads = 16, d_ff = 2048, n_layers = 12,
                             t_act = 'gelu', lin_act = 'relu', lin_depth = 7, encoder_dropout = 0.1, lin_dropout = 0.2,
                             kmer_size = 5, signal_size = 25, spectrogram_size = 21, block_len = 17, seq_len=200)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args_dict["lr"], weight_decay = 0.1)
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = 10,
                                                                                num_training_steps = args_dict["epochs"],
                                                                                num_cycles = 5, last_epoch = -1)
    loss_func = torch.nn.MSELoss()
    metric_func_dict = {"acc": cm.BinaryAccuracy().to(rank),
                        "auc": cm.BinaryAUROC().to(rank),
                        "f1": cm.BinaryF1Score().to(rank),}
    train_loader, val_loader = prepare_dataloader(args_dict["data"], args_dict["batch_size"], args_dict["disk_shard_size"],
                                                  rank, args_dict["gpu"], args_dict["seed"])
    trainer = Trainer(rank, model, train_loader, val_loader, optimizer, scheduler, loss_func, 1.0, metric_func_dict,
                        10, args_dict["output"], args_dict["tb"], args_dict["es_start"], args_dict["es_patience"],
                        args_dict["es_delta"], args_dict["name"], args_dict["gpu"])
    printmessage(f"[GPU {rank}] Trainer Setup Complete.")
    trainer.train(args_dict["epochs"])
    printmessage(f"[GPU {rank}] Training Loop Complete.")
    dist.destroy_process_group()
    return None


def main_master():
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.makedirs(os.path.join(args.output, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.tb, args.name), exist_ok=True)
    args.output = os.path.join(args.output, args.name)
    args.tb = os.path.join(args.tb, args.name)
    if args.seed is None:
        args.seed = np.random.randint(0, 10000000)
    printmessage("Training Program Started.")
    printmessage(f"Seed: {args.seed}")
    printmessage(f"Using {args.gpu} GPUs.")
    args_dict = vars(args)
    mp.spawn(main_worker, nprocs=args.gpu, args=(args_dict,))
    printmessage(f"Training Program Complete.")
    return None


if __name__ == "__main__":
    main_master()

