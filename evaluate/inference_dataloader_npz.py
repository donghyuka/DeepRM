import functools
import numpy as np
import torch
import math
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data import DataLoader
import glob
from utils.utils import printmessage

## Based on https://discuss.pytorch.org/t/an-iterabledataset-implementation-for-chunked-data/124437 by Majid Hajiheidari
## Load Nanopore Dataset from Pickled Pandas DataFrame
## DO NOT SHUFFLE BECAUSE THIS LOADER IS FOR INFERENCE ONLY



class NanoporeDatasetIterator:
    def __init__(self, file_paths, num_files_read_once = 1000,
                 cb_len = 21, kmer_len = 5, sampling = 6, sig_window = 5):

        self.file_paths = file_paths
        self.current_index = -1
        self.current_iterator = None
        self.num_files_read_once = num_files_read_once
        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.cb_lr_pad = (cb_len-kmer_len)//2
        self.trim = kmer_len//2
        self.keys = ["label_id", "segment_len_arr", "signal_token", "kmer_token", "dwell_token"]


    def __iter__(self):
        return self

    def _read_df(self):
        self.current_index += 1
        read_start = self.current_index*self.num_files_read_once
        read_end = min(len(self.file_paths), (self.current_index+1)*self.num_files_read_once)
        paths = self.file_paths[read_start:read_end]
        if len(paths) == 0:
            raise StopIteration
        current_buffer = [[] for _ in self.keys]
        for path in paths:
            with np.load(path, allow_pickle=True) as npz:
                for key_idx, key in enumerate(self.keys):
                    current_buffer[key_idx].append(npz[key])

        current_buffer = [np.concatenate(x, axis=0) for x in current_buffer]

        self.current_iterator = zip(*current_buffer)
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

        return result


    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, batch_size, disk_shard_size, rank, num_replicas,
                 seed = 0, num_files_read_once = 1000, cb_len = 21, kmer_len = 5, sampling = 6, sig_window = 5):
        super(NanoporeDataset).__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.disk_shard_size = disk_shard_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.file_paths = glob.glob(f"{self.data_path}/*.npz")
        if len(self.file_paths) == 0:
            pos_paths = glob.glob(f"{self.data_path}/pos/*.npz")
            neg_paths = glob.glob(f"{self.data_path}/neg/*.npz")
            self.file_paths = pos_paths + neg_paths

        self.epoch = 0
        self.seed = seed

        self.num_shard = math.ceil(len(self.file_paths)/num_replicas)
        self.total_num_shard = self.num_shard * num_replicas
        self.dataset_size = self.num_shard * disk_shard_size
        self.num_files_read_once = num_files_read_once

        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window


    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.file_paths = self.file_paths[self.rank::self.num_replicas]
        else:
            id = worker_info.id + self.rank * worker_info.num_workers
            nw = worker_info.num_workers * self.num_replicas
            self.file_paths = self.file_paths[id::nw]
        return NanoporeDatasetIterator(self.file_paths, num_files_read_once = self.num_files_read_once,
                                       cb_len=self.cb_len, kmer_len=self.kmer_len, sampling=self.sampling, sig_window=self.sig_window)

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
                 pad_to = 200, bq_clip = 40, num_files_read_once = 1, prefetch_factor = 100000, worker = 16,
                cb_len = 21, kmer_len = 5, sampling = 6, sig_window = 5):

    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, signal_stride = sampling, kmer_size = kmer_len)
    ## Use DataLoader to load the dataset
    dataset = NanoporeDataset(data_path, batch_size, disk_shard_size, rank, num_replicas,
                              num_files_read_once = num_files_read_once, cb_len = cb_len, kmer_len = kmer_len,
                              sampling = sampling, sig_window = sig_window)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=worker, pin_memory=True, drop_last=False,
                                    collate_fn = pad_collate_func, prefetch_factor=prefetch_factor)
    return dataloader


def pad_collate(batch, pad_to, signal_stride, kmer_size, trim = 2):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First
    ## ORDER: ["segment_len_arr", "signal_token", "kmer_token", "dwell_token"]

    label_list = []
    kmer_token_list = []
    signal_token_list = []
    dwell_token_list = []
    segment_len_list = []

    for source in batch:
        label_list.append(source[0])
        segment_len_list.append(source[1])
        signal_token_list.append(source[2])
        kmer_token_list.append(source[3])
        dwell_token_list.append(source[4])

    src_kmer = torch.tensor(np.stack(kmer_token_list), dtype=torch.int32)
    src_seg_len = torch.tensor(np.stack(segment_len_list), dtype=torch.int32)
    src_dwell = torch.tensor(np.stack(dwell_token_list), dtype=torch.float32)
    src_signal = torch.tensor(np.stack(signal_token_list), dtype=torch.float32)

    source = {}
    source["label_id"] = label_list
    source["kmer_token"] = src_kmer
    source["segment_len"] = src_seg_len
    source["signal_token"] = src_signal
    source["dwell_token"] = src_dwell

    return source
