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
    except:
        return None
    return torch.tensor(signal_segmented, dtype=torch.float)


class NanoporeDatasetIterator:
    def __init__(self, file_paths, num_files_read_once = 1000,
                 cb_len = 21, kmer_size = 5, sampling = 6, sig_window = 5, shuffle = True):

        self.file_paths = file_paths
        self.current_index = -1
        self.current_iterator = None
        self.num_files_read_once = num_files_read_once
        self.cb_len = cb_len
        self.kmer_size = kmer_size
        self.sampling = sampling
        self.sig_window = sig_window
        self.cb_lr_pad = (cb_len-kmer_size)//2
        self.trim = kmer_size//2
        self.shuffle = shuffle


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

        if self.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        df["segment_len_arr"] = df["signal"].apply(lambda x: create_segment_len_arr(x, self.sampling))
        df["signal_token"] = df.apply(lambda x: segmented_signal_to_block(x["signal"], x["segment_len_arr"], self.kmer_size, self.sampling, self.sig_window), axis=1)
        df["segment_len_arr"] = df["segment_len_arr"].apply(lambda x: x[self.trim:-self.trim])
        df["kmer_token"] = df["motif"].apply(lambda x: np.array(list(x)))
        df["bq_token"] = df["bq"].apply(lambda x: x[self.trim:-self.trim])
        df = df[["kmer_token", "signal_token", "bq_token", "segment_len_arr","label_id","block_id"]][df["signal_token"].notnull()].copy()

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

        return result


    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, batch_size, disk_shard_size, rank, num_replicas,
                 seed = 0, num_files_read_once = 1000, cb_len = 21, kmer_size = 5, sampling = 6,
                 sig_window = 5, shuffle = True, drop_last = True):
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

        if drop_last:
            self.num_shard = math.floor(len(self.file_paths)/num_replicas)
        else:
            self.num_shard = math.ceil(len(self.file_paths)/num_replicas)

        self.total_num_shard = self.num_shard * num_replicas
        self.dataset_size = self.num_shard * disk_shard_size
        self.num_files_read_once = num_files_read_once

        self.cb_len = cb_len
        self.kmer_size = kmer_size
        self.sampling = sampling
        self.sig_window = sig_window
        self.shuffle = shuffle
        self.drop_last = drop_last


    def reinit(self):
        ## After replicating the dataset, reinitialize the dataset using worker_info
        ## Required for DDP Compatibility

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.rank = worker_info.id + self.rank * worker_info.num_workers
            self.num_replicas = worker_info.num_workers * self.num_replicas

        if self.drop_last:
            self.num_shard = math.floor(len(self.file_paths)/self.num_replicas)
        else:
            self.num_shard = math.ceil(len(self.file_paths)/self.num_replicas)

        self.total_num_shard = self.num_shard * self.num_replicas
        self.dataset_size = self.num_shard * self.disk_shard_size
        return None


    def __len__(self):
        return self.dataset_size


    def __iter__(self):
        self.reinit()
        worker_info = torch.utils.data.get_worker_info()

        if self.shuffle:
            file_paths = self._deterministic_shuffle_and_sample(self.file_paths, self.num_shard, self.total_num_shard)
        else:
            file_paths = self.file_paths

        if worker_info is None:
            file_paths = file_paths[self.rank::self.num_replicas]
        else:
            id = worker_info.id + self.rank * worker_info.num_workers
            nw = worker_info.num_workers * self.num_replicas
            file_paths = file_paths[id::nw]
        return NanoporeDatasetIterator(file_paths, num_files_read_once = self.num_files_read_once,
                                       cb_len=self.cb_len, kmer_size=self.kmer_size, sampling=self.sampling,
                                       sig_window=self.sig_window, shuffle=self.shuffle)


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


    def _deterministic_shuffle_and_sample(self, data_path_list, num_shard, total_num_shard):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(data_path_list), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(data_path_list)))  # type: ignore[arg-type]

        if len(indices) < total_num_shard:
            # add extra samples to make it evenly divisible
            padding_size = total_num_shard - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        elif len(indices) > total_num_shard:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_num_shard]

        assert len(indices) == total_num_shard, f"{len(indices)} != {total_num_shard}"

        indices = indices[self.rank:total_num_shard:self.num_replicas][:num_shard]
        subsampled = [data_path_list[i] for i in indices]

        return subsampled


class NanoporeDataLoader(DataLoader):
    def __init__(self, dataset:NanoporeDataset, batch_size, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
        shuffle = False
        sampler = None
        self.dataset = dataset
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         drop_last=drop_last, shuffle=shuffle, sampler=sampler, collate_fn=collate_fn,
                         prefetch_factor=prefetch_factor, persistent_workers=True)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.dataset.set_epoch(epoch)
        return None

    ## END of NanoporeDataLoader

def load_dataset(data_path, batch_size, disk_shard_size, rank, num_replicas,
                 pad_to = 200, bq_clip = 40, num_files_read_once = 1, prefetch_factor = 100000, worker = 16,
                cb_len = 21, kmer_size = 5, signal_stride = 6, sig_window = 5, num_workers = 8, pin_memory = True,
                 drop_last = False, shuffle = True, seed = 0):

    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, signal_stride = signal_stride, kmer_size = kmer_size)
    ## Use DataLoader to load the dataset
    dataset = NanoporeDataset(data_path, batch_size, disk_shard_size, rank, num_replicas,
                              num_files_read_once = num_files_read_once, cb_len = cb_len, kmer_size = kmer_size,
                              sampling = signal_stride, sig_window = sig_window, shuffle = shuffle, drop_last = drop_last, seed = seed)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
                                    collate_fn = pad_collate_func, prefetch_factor=prefetch_factor)
    return dataloader


def pad_collate(batch, pad_to, signal_stride, kmer_size):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First
    ## "kmer_token", "signal_token", "bq_token", "segment_len_arr","label_id","block_id"

    label_id_list = []
    block_id_list = []
    kmer_token_list = []
    signal_token_list = []
    bq_token_list = []
    segment_len_list = []


    for source in batch:
        kmer_token_list.append(source[0])
        signal_token_list.append(source[1])
        bq_token_list.append(source[2])
        segment_len_list.append(source[3])
        label_id_list.append(source[4])
        block_id_list.append(source[5])

    src_kmer = torch.tensor(np.stack(kmer_token_list).view(np.int32), dtype=torch.int32)
    src_kmer = (src_kmer - 65).clip(None,8)%5 + 1
    src_seg_len = torch.tensor(np.stack(segment_len_list), dtype=torch.int32)
    src_bq = torch.tensor(np.stack(bq_token_list), dtype=torch.int32)
    src_signal = torch.nn.utils.rnn.pad_sequence(signal_token_list, batch_first=True, padding_value=0)
    if pad_to is not None:
        signal_pad_to = (pad_to+kmer_size-1) * signal_stride
        if src_signal.shape[1] < signal_pad_to:
            src_signal = torch.cat((src_signal, torch.zeros(src_signal.shape[0], signal_pad_to - src_signal.shape[1])), dim=1)
        else:
            src_signal = src_signal[:, :signal_pad_to]

    source = {}
    source["kmer_token"] = src_kmer
    source["segment_len"] = src_seg_len
    source["signal_token"] = src_signal
    source["bq_token"] = src_bq
    source["label_id"] = label_id_list
    source["block_id"] = block_id_list

    return source