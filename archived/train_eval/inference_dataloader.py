import functools

import torch
import math
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data import DataLoader
import pandas as pd
import glob
from utils.utils import printmessage

## Based on https://discuss.pytorch.org/t/an-iterabledataset-implementation-for-chunked-data/124437 by Majid Hajiheidari
## Load Nanopore Dataset from Pickled Pandas DataFrame
## DO NOT SHUFFLE BECAUSE THIS LOADER IS FOR INFERENCE ONLY

class NanoporeDatasetIterator:
    def __init__(self, file_paths, num_files_read_once = 1000):

        self.file_paths = file_paths
        self.current_index = -1
        self.current_iterator = None
        self.num_files_read_once = num_files_read_once

    def __iter__(self):
        return self

    def _read_df(self):
        self.current_index += 1
        read_start = self.current_index*self.num_files_read_once
        read_end = min(len(self.file_paths), (self.current_index+1)*self.num_files_read_once)
        df = [pd.read_pickle(file_path) for file_path in self.file_paths[read_start:read_end]]
        if len(df)> 0:
            df = pd.concat(df)
        else:
            raise StopIteration
        if "label_id" not in df.columns:
            df["label_id"] = ""

        df = df[["kmer_token", "bq_token", "position_token", "signal_token", "move_token", "target_mask",
                 "label_id", "label", "block_id"]]
        self.current_iterator = df.itertuples(index=False)
        return None

    def __next__(self):

        if self.current_index == -1:
            try:
                self._read_df()
            except StopIteration:
                raise StopIteration

        try:
            result = next(self.current_iterator)

        except StopIteration:
            if self.current_index == len(self.file_paths) // self.num_files_read_once:
                raise StopIteration
            else:
                try:
                    self._read_df()
                except StopIteration:
                    raise StopIteration
                try:
                    result = next(self.current_iterator)
                except StopIteration:
                    raise StopIteration

        source, target = self.nanopore_row_to_tensor(result)

        return source, target

    def nanopore_row_to_tensor(self, row):
        kmer_token = torch.tensor(row[0], dtype=torch.long)
        bq_token = torch.tensor(row[1], dtype=torch.long)
        position_token = torch.tensor(row[2], dtype=torch.long)
        signal_token = torch.tensor(row[3], dtype=torch.float)
        move_token = torch.tensor(row[4], dtype=torch.long)
        target_mask = torch.tensor(row[5], dtype=torch.float)
        label_id = row[6]
        label =row[7]
        block_id = row[8]

        return_dict = {"kmer_token": kmer_token, "bq_token": bq_token, "position_token": position_token,
                       "signal_token": signal_token, "move_token": move_token,
                       "target_mask": target_mask, "label_id": label_id, "label": label, "block_id": block_id}

        label = torch.tensor(label, dtype=torch.long)

        return return_dict, label


    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, batch_size, disk_shard_size, rank, num_replicas,
                 seed = 0, num_files_read_once = 1000):
        super(NanoporeDataset).__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.disk_shard_size = disk_shard_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.file_paths = glob.glob(f"{self.data_path}/*.pkl")
        if len(self.file_paths) == 0:
            pos_paths = glob.glob(f"{self.data_path}/pos/*.pkl")
            neg_paths = glob.glob(f"{self.data_path}/neg/*.pkl")
            self.file_paths = pos_paths + neg_paths

        self.epoch = 0
        self.seed = seed

        self.num_shard = math.ceil(len(self.file_paths)/num_replicas)
        self.total_num_shard = self.num_shard * num_replicas
        self.dataset_size = self.num_shard * disk_shard_size
        self.num_files_read_once = num_files_read_once


    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        self.file_paths = self.file_paths[self.rank::self.num_replicas]
        return NanoporeDatasetIterator(self.file_paths, num_files_read_once = self.num_files_read_once)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        return None

class NanoporeDataLoader(DataLoader):
    def __init__(self, dataset:NanoporeDataset, batch_size, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
        shuffle = False
        sampler = None
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         drop_last=drop_last, shuffle=shuffle, sampler=sampler, collate_fn=collate_fn,
                         prefetch_factor=prefetch_factor)

    ## END of NanoporeDataLoader


def load_dataset(data_path, batch_size, disk_shard_size, rank, num_replicas,
                 pad_to = 200, bq_clip = 40, num_files_read_once = 1000, prefetch_factor = 100000):
    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, bq_clip = bq_clip)
    ## Use DataLoader to load the dataset
    dataset = NanoporeDataset(data_path, batch_size, disk_shard_size, rank, num_replicas, num_files_read_once = num_files_read_once)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=False, drop_last=False,
                                    collate_fn = pad_collate_func, prefetch_factor=prefetch_factor)
    return dataloader


def pad_collate(batch, pad_to, bq_clip):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First

    label = [item[1] for item in batch]
    target = torch.stack(label, dim=0)

    token_name_list = ["kmer_token", "bq_token", "position_token", "signal_token",
                       "move_token", "target_mask"]

    source = {}

    ## Zero pad the followings: kmer_token, bq_token, position_token, signal_token, spectrogram_token, move_token
    for token_name in token_name_list:
        token = [item[0][token_name] for item in batch]
        token = torch.nn.utils.rnn.pad_sequence(token, batch_first=True, padding_value=0)
        if pad_to is not None:
            if token.shape[1] < pad_to:
                add_shape = list(token.shape)
                add_shape[1] = pad_to - token.shape[1]
                token = torch.cat((token, torch.zeros(add_shape, dtype=token.dtype)), dim=1)
            else:
                token = token[:, :pad_to]

        source[token_name] = token

    ## clip bq
    source["bq_token"] = torch.clamp(source["bq_token"], 0, bq_clip)


    source["label_id"] = [item[0]["label_id"] for item in batch]
    source["label"] = [item[0]["label"] for item in batch]
    source["block_id"] = [item[0]["block_id"] for item in batch]

    return source, target
