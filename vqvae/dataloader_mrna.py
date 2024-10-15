import functools

import torch
import math
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data import DataLoader
import pandas as pd
import glob
from utils.utils import printmessage
import numpy as np
import gc

## Load Nanopore Dataset from Pickled Pandas DataFrame



class TranscriptDatasetIterator:
    def __init__(self, file_paths, disk_shard_size, shuffle_buffer_size,
                 shuffle = True, class_ratio = 0.5, soft_label = False, yield_period = None):

        self.paths = file_paths
        self.paths_len = len(file_paths)
        self.shuffle = shuffle
        self.disk_shard_size = disk_shard_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.current_df_index = -1
        self.current_iterator = None
        self.len_iterator = 0
        self.buffer = []
        self.yield_period = yield_period 

        assert self.yield_period <= self.shuffle_buffer_size, "Shuffle period should be less than or equal to shuffle buffer size"

    def __iter__(self):
        return self


    def __next__(self):
        return self._next()

    def _read_shuffle_data(self, first_read = False):

        if first_read:
            assert self.buffer == [], "Buffer should be empty when first read"
            fill_to = self.shuffle_buffer_size

        else:
            fill_to = self.yield_period

        len_read = 0

        while (len_read < fill_to) and (self.current_df_index < self.paths_len - 1):
            self.current_df_index += 1
            df = pd.read_pickle(self.paths[self.current_df_index])[["segment_len_arr", "signal_token"]]
            df["signal_token"] = df["signal_token"].apply(lambda x: torch.tensor(x, dtype=torch.float))
            self.buffer.append(df)
            len_read += len(df)

        del df

        ## Concat and shuffle the buffer to yield.
        if self.shuffle:
            self.buffer = [pd.concat(self.buffer, ignore_index=True).sample(frac=1)]
        else:
            self.buffer = [pd.concat(self.buffer, ignore_index=True)]

        if len(self.buffer[0]) > self.yield_period:
            self._give_df_to_iterator(self.buffer[0][:self.yield_period])
            self.buffer[0] = self.buffer[0][self.yield_period:]
        else:
            self._give_df_to_iterator(self.buffer[0])
            self.buffer = []
        gc.collect()
        return None


    def _exhaust_buffer(self):
        assert len(self.buffer) == 1, f"Buffer to be exhausted should have only one dataframe, but has {len(self.buffer)}: {self.buffer}"
        self._give_df_to_iterator(self.buffer[0])
        self.buffer = []
        return None


    def _give_df_to_iterator(self, df):
        ## ORDER: segment_len_arr, signal_token, bq_token, kmer_token
        self.current_iterator = df.itertuples(index=False)
        self.len_iterator = len(df)
        gc.collect()
        return None


    def _next(self):

        if self.current_df_index == -1: ## First time reading data
            self._read_shuffle_data(first_read = True)

        elif self.len_iterator == 0: ## Current iterator ran out of data
            if self.current_df_index < self.paths_len - 1: ## Still have data to read from disk
                self._read_shuffle_data(first_read = False)
            elif len(self.buffer) > 0: ## No data to read from disk, but buffer has data
                self._exhaust_buffer()
            else: ## No data to read from disk, and buffer is empty.
                pass

        else: ## Current iterator has data to process
            pass

        try: ## Check if the current iterator has data to process
            result = next(self.current_iterator)
            self.len_iterator -= 1

        except StopIteration: ## Current iterator ran out of data, and no more data to read.
            raise StopIteration

        return result

    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, batch_size, disk_shard_size, rank, num_replicas, shuffle_buffer_size,
                 yield_period = None, seed = 0, shuffle = True, drop_last = True, class_ratio = 1, soft_label = False):
        super(NanoporeDataset).__init__()

        self.file_paths = data_path
        self.batch_size = batch_size
        self.disk_shard_size = disk_shard_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.rank_gpu = rank
        self.num_gpu = num_replicas
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.yield_period = yield_period

        if self.drop_last:
            self.num_shard = math.floor(len(self.file_paths)/num_replicas)
        else:
            self.num_shard = math.ceil(len(self.file_paths)/num_replicas)

        self.total_num_shard = self.num_shard * num_replicas
        self.dataset_size = self.total_num_shard * disk_shard_size


    def reinit(self):
        ## After replicating the dataset, reinitialize the dataset using worker_info
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            self.rank = worker_info.id + self.rank_gpu * worker_info.num_workers
            self.num_replicas = worker_info.num_workers * self.num_gpu

        if self.drop_last:
            self.num_shard = math.floor(len(self.file_paths)/self.num_replicas)
        else:
            self.num_shard = math.ceil(len(self.file_paths)/self.num_replicas)

        self.total_num_shard = self.num_shard * self.num_replicas
        self.dataset_size = self.total_num_shard * self.disk_shard_size


    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        self.reinit()
        file_paths = self._deterministic_shuffle_and_sample(self.file_paths, self.num_shard, self.total_num_shard)
        return TranscriptDatasetIterator(file_paths,
                                          disk_shard_size = self.disk_shard_size, shuffle_buffer_size = self.shuffle_buffer_size,
                                          shuffle = self.shuffle, yield_period = self.yield_period)

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

    ## END of NanoporeDataset


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


def load_dataset(data_path, batch_size,
                 disk_shard_size, rank, num_replicas, shuffle_buffer_size, yield_period, seed = 0, shuffle = True, drop_last = True,
                 pad_to = 200, bq_clip = 40, class_ratio = 1, prefetch_factor = 512, pin_memory = True, soft_label = False,
                 num_workers = 4, signal_stride = 6, kmer_size = 5):
    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, signal_stride = signal_stride, kmer_size = kmer_size)
    ## Use DataLoader to load the dataset
    data_paths = glob.glob(f"{data_path}/*.pkl")

    if yield_period is None:
        yield_period = shuffle_buffer_size // 4

    data_len = len(data_paths)
    if data_len < num_replicas:
        raise printmessage(f"The number of datapoints is smaller than the number of GPUs: {data_len} < {num_replicas}", msg_type="error", error=ValueError)
    if data_len < num_workers * num_replicas:
        new_max_workers = data_len // num_replicas
        printmessage(f"The number of datapoints is smaller than the number of workers: {data_len} < {num_workers * num_replicas}", msg_type="warning")
        printmessage(f"Setting number of workers to {new_max_workers * num_replicas}", msg_type="warning")
        num_workers = new_max_workers

    dataset = NanoporeDataset(data_paths, batch_size, disk_shard_size, rank, num_replicas, shuffle_buffer_size,
                              yield_period, seed, shuffle, drop_last, class_ratio = class_ratio, soft_label = soft_label)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
                                    collate_fn = pad_collate_func, prefetch_factor=prefetch_factor)

    return dataloader


def pad_collate(batch, pad_to, signal_stride, kmer_size):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First
    ## ORDER: segment_len_arr, signal_token, bq_token, kmer_token

    signal_token_list = []
    segment_len_list = []


    for source in batch:
        segment_len_list.append(source[0])
        signal_token_list.append(source[1])

    src_seg_len = torch.tensor(np.stack(segment_len_list), dtype=torch.int32)
    src_signal = torch.nn.utils.rnn.pad_sequence(signal_token_list, batch_first=True, padding_value=0)

    if pad_to is not None:
        signal_pad_to = (pad_to+kmer_size-1) * signal_stride
        if src_signal.shape[1] < signal_pad_to:
            src_signal = torch.cat((src_signal, torch.zeros(src_signal.shape[0], signal_pad_to - src_signal.shape[1])), dim=1)
        else:
            src_signal = src_signal[:, :signal_pad_to]

    source = {}
    source["seg_len"] = src_seg_len
    source["signal_token"] = src_signal

    return source
