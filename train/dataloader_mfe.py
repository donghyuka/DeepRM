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
import RNA

## Load Nanopore Dataset from Pickled Pandas DataFrame



class BinaryClassDatasetIterator:
    def __init__(self, pos_file_paths, neg_file_paths, disk_shard_size, shuffle_buffer_size,
                 shuffle = True, class_ratio = 0.5, soft_label = False, yield_period = None):

        self.paths = [neg_file_paths, pos_file_paths]
        self.paths_len = [len(neg_file_paths), len(pos_file_paths)]
        self.shuffle = shuffle
        self.disk_shard_size = disk_shard_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.current_df_index = [-1,-1]
        self.current_iterator = [None, None]
        self.current_class = 0
        self.avail_class = [0,1]
        self.len_iterator = [0,0]
        self.class_ratio = class_ratio
        self.buffer = [[], []]
        self.yield_period = yield_period
        self.soft_label = soft_label

        assert self.yield_period <= self.shuffle_buffer_size, "Shuffle period should be less than or equal to shuffle buffer size"

    def __iter__(self):
        return self


    def __next__(self):
        return self._next() , self.current_class

    def _read_shuffle_data(self, first_read = False):
        
        if first_read:
            assert self.buffer[self.current_class] == [], "Buffer should be empty when first read"
            fill_to = self.shuffle_buffer_size

        else:
            fill_to = self.yield_period

        len_read = 0

        while (len_read < fill_to) and (self.current_df_index[self.current_class] < self.paths_len[self.current_class] - 1):
            self.current_df_index[self.current_class] += 1
            df = pd.read_pickle(self.paths[self.current_class][self.current_df_index[self.current_class]])
            df["mfe"] = df["kmer_token"].apply(lambda x: RNA.fold(x)[1])
            df["kmer_token"] = df["kmer_token"].apply(lambda x: np.array(list(x)))
            df["signal_token"] = df["signal_token"].apply(lambda x: torch.tensor(x, dtype=torch.float))
            self.buffer[self.current_class].append(df)
            len_read += len(df)

        del df

        ## Concat and shuffle the buffer to yield.
        if self.shuffle:
            self.buffer[self.current_class] = [pd.concat(self.buffer[self.current_class], ignore_index=True).sample(frac=1)]
        else:
            self.buffer[self.current_class] = [pd.concat(self.buffer[self.current_class], ignore_index=True)]

        if len(self.buffer[self.current_class][0]) > self.yield_period:
            self._give_df_to_iterator(self.buffer[self.current_class][0][:self.yield_period])
            self.buffer[self.current_class][0] = self.buffer[self.current_class][0][self.yield_period:]
        else:
            self._give_df_to_iterator(self.buffer[self.current_class][0])
            self.buffer[self.current_class] = []
        gc.collect()
        return None


    def _exhaust_buffer(self):
        assert len(self.buffer[self.current_class]) == 1, f"Buffer to be exhausted should have only one dataframe, but has {len(self.buffer[self.current_class])}: {self.buffer[self.current_class]}"
        self._give_df_to_iterator(self.buffer[self.current_class][0])
        self.buffer[self.current_class] = []
        return None


    def _give_df_to_iterator(self, df):
        ## ORDER: segment_len_arr, signal_token, bq_token, kmer_token, mfe
        self.current_iterator[self.current_class] = df.itertuples(index=False)
        self.len_iterator[self.current_class] = len(df)
        gc.collect()
        return None


    def _get_rand_class(self):
        rand = torch.rand(1).item()
        if rand < self.class_ratio:
            return 0
        else:
            return 1

    def _next(self):
        ## Randomly decide between positive and negative data
        if len(self.avail_class) == 0: ## No more data to read in both classes
            raise StopIteration

        elif len(self.avail_class) == 1: ## Only one class available
            self.current_class = self.avail_class[0]

        else:
            self.current_class = self._get_rand_class()

        if self.current_df_index[self.current_class] == -1: ## First time reading data
            self._read_shuffle_data(first_read = True)

        elif self.len_iterator[self.current_class] == 0: ## Current iterator ran out of data
            if self.current_df_index[self.current_class] < self.paths_len[self.current_class] - 1: ## Still have data to read from disk
                self._read_shuffle_data(first_read = False)
            elif len(self.buffer[self.current_class]) > 0: ## No data to read from disk, but buffer has data
                self._exhaust_buffer()
            else: ## No data to read from disk, and buffer is empty.
                pass

        else: ## Current iterator has data to process
            pass

        try: ## Check if the current iterator has data to process
            result = next(self.current_iterator[self.current_class])
            self.len_iterator[self.current_class] -= 1

        except StopIteration: ## Current iterator ran out of data, and no more data to read.
            self.avail_class.remove(self.current_class)
            result = self._next()

        return result

    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, pos_data_path, neg_data_path, batch_size, disk_shard_size, rank, num_replicas, shuffle_buffer_size,
                 yield_period = None, seed = 0, shuffle = True, drop_last = True, class_ratio = 1, soft_label = False):
        super(NanoporeDataset).__init__()

        self.pos_file_paths = pos_data_path
        self.neg_file_paths = neg_data_path
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
        self.class_ratio = class_ratio
        self.soft_label = soft_label
        self.yield_period = yield_period

        if self.drop_last:

            self.pos_num_shard = math.floor(len(self.pos_file_paths)/num_replicas)
            self.neg_num_shard = math.floor(len(self.neg_file_paths)/num_replicas)
            self.pos_num_shard = min(self.pos_num_shard, int(self.neg_num_shard / self.class_ratio))
            self.neg_num_shard = int(self.pos_num_shard * self.class_ratio)
            self.pos_total_num_shard = self.pos_num_shard * num_replicas
            self.pos_dataset_size = self.pos_total_num_shard * disk_shard_size
            self.neg_total_num_shard = self.neg_num_shard * num_replicas
            self.neg_dataset_size = self.neg_total_num_shard * disk_shard_size

        else:
            self.pos_num_shard = math.ceil(len(self.pos_file_paths)/num_replicas)
            self.neg_num_shard = math.ceil(len(self.neg_file_paths)/num_replicas)
            self.pos_num_shard = min(self.pos_num_shard, int(self.neg_num_shard / self.class_ratio))
            self.neg_num_shard = int(self.pos_num_shard * self.class_ratio)
            self.pos_total_num_shard = self.pos_num_shard * num_replicas
            self.pos_dataset_size = self.pos_total_num_shard * disk_shard_size
            self.neg_total_num_shard = self.neg_num_shard * num_replicas
            self.neg_dataset_size = self.neg_total_num_shard * disk_shard_size

        self.dataset_size = self.pos_dataset_size + self.neg_dataset_size


    def reinit(self):
        ## After replicating the dataset, reinitialize the dataset using worker_info
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            self.rank = worker_info.id + self.rank_gpu * worker_info.num_workers
            self.num_replicas = worker_info.num_workers * self.num_gpu

        if self.drop_last:

            self.pos_num_shard = math.floor(len(self.pos_file_paths)/self.num_replicas)
            self.neg_num_shard = math.floor(len(self.neg_file_paths)/self.num_replicas)
            self.pos_num_shard = min(self.pos_num_shard, int(self.neg_num_shard / self.class_ratio))
            self.neg_num_shard = int(self.pos_num_shard * self.class_ratio)
            self.pos_total_num_shard = self.pos_num_shard * self.num_replicas
            self.pos_dataset_size = self.pos_total_num_shard * self.disk_shard_size
            self.neg_total_num_shard = self.neg_num_shard * self.num_replicas
            self.neg_dataset_size = self.neg_total_num_shard * self.disk_shard_size

        else:
            self.pos_num_shard = math.ceil(len(self.pos_file_paths)/self.num_replicas)
            self.neg_num_shard = math.ceil(len(self.neg_file_paths)/self.num_replicas)
            self.pos_num_shard = min(self.pos_num_shard, int(self.neg_num_shard / self.class_ratio))
            self.neg_num_shard = int(self.pos_num_shard * self.class_ratio)
            self.pos_total_num_shard = self.pos_num_shard * self.num_replicas
            self.pos_dataset_size = self.pos_total_num_shard * self.disk_shard_size
            self.neg_total_num_shard = self.neg_num_shard * self.num_replicas
            self.neg_dataset_size = self.neg_total_num_shard * self.disk_shard_size

        self.dataset_size = self.pos_dataset_size + self.neg_dataset_size

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        self.reinit()
        pos_file_paths = self._deterministic_shuffle_and_sample(self.pos_file_paths, self.pos_num_shard, self.pos_total_num_shard)
        neg_file_paths = self._deterministic_shuffle_and_sample(self.neg_file_paths, self.neg_num_shard, self.neg_total_num_shard)
        class_ratio_iterator = self.class_ratio / (1 + self.class_ratio)
        return BinaryClassDatasetIterator(pos_file_paths, neg_file_paths,
                                          disk_shard_size = self.disk_shard_size, shuffle_buffer_size = self.shuffle_buffer_size,
                                          shuffle = self.shuffle, class_ratio = class_ratio_iterator,
                                          soft_label = self.soft_label, yield_period = self.yield_period)

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


def load_dataset(pos_data_path, neg_data_path, batch_size,
                 disk_shard_size, rank, num_replicas, shuffle_buffer_size, yield_period, seed = 0, shuffle = True, drop_last = True,
                 pad_to = 200, bq_clip = 40, class_ratio = 1, prefetch_factor = 512, pin_memory = True, soft_label = False,
                 num_workers = 4, signal_stride = 6, kmer_size = 5):
    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, signal_stride = signal_stride, kmer_size = kmer_size)
    ## Use DataLoader to load the dataset
    pos_data_paths = glob.glob(f"{pos_data_path}/*.pkl")
    neg_data_paths = glob.glob(f"{neg_data_path}/*.pkl")

    if class_ratio is None:
        class_ratio = len(neg_data_paths) / len(pos_data_paths)
        if rank == 0:
            printmessage(f"Neg:Pos = {class_ratio:.3f}:1", msg_type="info")

    if yield_period is None:
        yield_period = shuffle_buffer_size // 4

    min_len = min(len(pos_data_paths), len(neg_data_paths))
    if min_len < num_replicas:
        raise printmessage(f"The number of datapoints is smaller than the number of GPUs: {min_len} < {num_replicas}", msg_type="error", error=ValueError)
    if min_len < num_workers * num_replicas:
        new_max_workers = min_len // num_replicas
        printmessage(f"The number of datapoints is smaller than the number of workers: {min_len} < {num_workers * num_replicas}", msg_type="warning")
        printmessage(f"Setting number of workers to {new_max_workers * num_replicas}", msg_type="warning")
        num_workers = new_max_workers

    dataset = NanoporeDataset(pos_data_paths, neg_data_paths, batch_size, disk_shard_size, rank, num_replicas, shuffle_buffer_size,
                              yield_period, seed, shuffle, drop_last, class_ratio = class_ratio, soft_label = soft_label)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
                                    collate_fn = pad_collate_func, prefetch_factor=prefetch_factor)

    return dataloader


def pad_collate(batch, pad_to, signal_stride, kmer_size):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First
    ## ORDER: segment_len_arr, signal_token, bq_token, kmer_token, mfe

    label_list = []
    kmer_token_list = []
    signal_token_list = []
    bq_token_list = []
    segment_len_list = []
    mfe_list = []


    for source, target in batch:
        label_list.append(target)
        segment_len_list.append(source[0])
        signal_token_list.append(source[1])
        bq_token_list.append(source[2])
        kmer_token_list.append(source[3])
        mfe_list.append(source[4])

    target = torch.tensor(label_list, dtype=torch.float)
    src_kmer = torch.tensor(np.stack(kmer_token_list).view(np.int32), dtype=torch.int32)
    src_seg_len = torch.tensor(np.stack(segment_len_list), dtype=torch.int32)
    src_bq = torch.tensor(np.stack(bq_token_list), dtype=torch.int32)
    src_signal = torch.nn.utils.rnn.pad_sequence(signal_token_list, batch_first=True, padding_value=0)
    src_mfe = torch.tensor(mfe_list, dtype=torch.float) * -0.1

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
    source["mfe"] = src_mfe

    return source, target
