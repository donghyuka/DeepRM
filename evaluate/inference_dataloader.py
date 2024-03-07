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
    def __init__(self, file_paths):

        self.file_paths = file_paths
        self.current_index = -1
        self.current_iterator = None

    def __iter__(self):
        return self

    def __next__(self):

        if self.current_index == -1:
            if self.current_index == len(self.file_paths) - 1:
                raise StopIteration
            self.current_index += 1
            df = pd.read_pickle(self.file_paths[self.current_index])
            self.current_iterator = df.iterrows()

        try:
            result = next(self.current_iterator)[1]
        except StopIteration:
            if self.current_index == len(self.file_paths) - 1:
                raise StopIteration
            else:
                self.current_index += 1
                df = pd.read_pickle(self.file_paths[self.current_index])
                self.current_iterator = df.iterrows()
                result = next(self.current_iterator)[1]

        source, target = self.nanopore_row_to_tensor(result)

        return source, target

    def nanopore_row_to_tensor(self, row):
        ## Columns: "block_id", "label_id", "label", "motif", "block_score", "kmer_token", "bq_token", "position_token", "signal_token",
        ##          "spectrogram_token", "move_token", "target_mask"
        kmer_token = torch.tensor(row["kmer_token"], dtype=torch.long)
        bq_token = torch.tensor(row["bq_token"], dtype=torch.long)
        position_token = torch.tensor(row["position_token"], dtype=torch.long)
        signal_token = torch.tensor(row["signal_token"], dtype=torch.float)
        spectrogram_token = torch.tensor(row["spectrogram_token"], dtype=torch.float)
        move_token = torch.tensor(row["move_token"], dtype=torch.long)
        target_mask = torch.tensor(row["target_mask"], dtype=torch.float)
        block_id = row["block_id"]
        label_id = row["label_id"]
        label =row["label"]

        return_dict = {"kmer_token": kmer_token, "bq_token": bq_token, "position_token": position_token,
                       "signal_token": signal_token, "spectrogram_token": spectrogram_token, "move_token": move_token,
                       "target_mask": target_mask, "block_id": block_id, "label_id": label_id, "label": label}

        label = torch.tensor(label, dtype=torch.long)

        return return_dict, label


    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, batch_size, disk_shard_size, rank, num_replicas,
                 seed = 0):
        super(NanoporeDataset).__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.disk_shard_size = disk_shard_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.file_paths = glob.glob(f"{self.data_path}/*.pkl")
        self.epoch = 0
        self.seed = seed

        self.num_shard = math.ceil(len(self.file_paths)/num_replicas)
        self.total_num_shard = self.num_shard * num_replicas
        self.dataset_size = self.num_shard * disk_shard_size

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        self.file_paths = self.file_paths[self.rank::self.num_replicas]
        return NanoporeDatasetIterator(self.file_paths)

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
                 pad_to = 200, bq_clip = 40):
    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, bq_clip = bq_clip)
    ## Use DataLoader to load the dataset
    dataset = NanoporeDataset(data_path, batch_size, disk_shard_size, rank, num_replicas)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=False, drop_last=False,
                                    collate_fn = pad_collate_func, prefetch_factor=16)
    return dataloader


def pad_collate(batch, pad_to, bq_clip):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First

    label = [item[1] for item in batch]
    target = torch.stack(label, dim=0)

    token_name_list = ["kmer_token", "bq_token", "position_token", "signal_token", "spectrogram_token",
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

    source["block_id"] = [item[0]["block_id"] for item in batch]
    source["label_id"] = [item[0]["label_id"] for item in batch]
    source["label"] = [item[0]["label"] for item in batch]

    return source, target
