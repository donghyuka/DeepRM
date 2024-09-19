import functools
import numpy as np
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



def sequence_to_kmer_token(seq, kmer):
    ## 1. change string to array of int - 0, 1, 2, 3
    seq = seq.upper()
    seq = seq.replace('A', '0')
    seq = seq.replace('C', '1')
    seq = seq.replace('G', '2')
    seq = seq.replace('T', '3')
    seq = seq.replace('U', '3')
    seq = np.array(list(seq), dtype=int)

    ## 2. convert to kmer token
    seq = [seq[i:kmer+i] for i in range(len(seq)-kmer+1)]
    seq = np.stack(seq, axis=1)
    quaternary = 4**np.arange(kmer).reshape(-1,1)
    seq = np.sum(seq * quaternary, axis=0) + 1 ## 0 is reserved for padding
    seq = seq.astype(np.int16)
    return seq


def create_segment_len_arr(segment_arr, sampling):
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int)
    segment_len_arr = segment_len_arr // sampling
    return segment_len_arr

def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window):
    try:
        kmer_pad = (kmer-1)//2
        lr_pad = (sig_window-1)//2
        l_skip = (np.sum(segment_len_arr[:kmer_pad])-lr_pad)*sampling
        r_skip = (np.sum(segment_len_arr[-kmer_pad:])-lr_pad)*sampling
        assert l_skip >= 0, f"Left skip is negative: {l_skip}, segment_len_arr: {segment_len_arr}"
        assert r_skip >= 0, f"Right skip is negative: {r_skip}, segment_len_arr: {segment_len_arr}"
        signal_segmented = np.concatenate(signal_segmented)
        if len(signal_segmented) % sampling != 0:
            return None
        if r_skip > 0:
            signal_segmented = signal_segmented[l_skip:-r_skip]
        else:
            signal_segmented = signal_segmented[l_skip:]
        # signal_segmented = np.lib.stride_tricks.sliding_window_view(signal_segmented, sig_window * sampling)[::sampling]
    except:
        return None
    return signal_segmented


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

        df["kmer_token"] = df.apply(lambda x: sequence_to_kmer_token(x["motif"], self.kmer_len), axis=1)
        df["segment_len_arr"] = df["signal"].apply(lambda x: create_segment_len_arr(x, self.sampling))
        df["signal_token"] = df.apply(lambda x: segmented_signal_to_block(x["signal"], x["segment_len_arr"], self.kmer_len, self.sampling, self.sig_window), axis=1)
        df["segment_len_arr"] = df["segment_len_arr"].apply(lambda x: x[self.trim:-self.trim])
        df = df[["kmer_token", "signal_token", "segment_len_arr","label_id","block_id"]][df["signal_token"].notnull()].copy()

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

        source = self.nanopore_row_to_tensor(result)

        return source

    def nanopore_row_to_tensor(self, row):
        kmer_token = torch.tensor(row[0], dtype=torch.long)
        signal_token = torch.tensor(row[1], dtype=torch.float)
        segment_len_arr = torch.tensor(row[2], dtype=torch.int32)
        label_id = row[3]
        block_id = row[4]

        return_dict = {"kmer_token": kmer_token,
                       "signal_token": signal_token,
                       "segment_len_arr": segment_len_arr,
                       "label_id": label_id,
                       "block_id": block_id}

        return return_dict



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


def pad_collate(batch, pad_to, signal_stride, kmer_size):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First


    source = {}

    ## Zero pad the followings: kmer_token, bq_token, signal_token, spectrogram_token, move_token

    token = [item["signal_token"] for item in batch]
    token = torch.nn.utils.rnn.pad_sequence(token, batch_first=True, padding_value=0)

    if pad_to is not None:
        signal_pad_to = (pad_to+kmer_size-1) * signal_stride
        if token.shape[1] < signal_pad_to:
            token = torch.cat((token, torch.zeros(token.shape[0], signal_pad_to - token.shape[1])), dim=1)
        else:
            token = token[:, :signal_pad_to]

    source["signal_token"] = token

    token = torch.stack([item["kmer_token"] for item in batch], dim=0)
    source["kmer_token"] = token

    token = torch.stack([item["segment_len_arr"] for item in batch], dim=0)
    source["segment_len"] = token

    source["label_id"] = [item["label_id"] for item in batch]
    source["block_id"] = [item["block_id"] for item in batch]

    return source
