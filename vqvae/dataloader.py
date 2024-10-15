import functools
import torch
import math
import glob
import gc
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from utils.utils import printmessage

## Load Nanopore Dataset from NumPy .npz files


class BinaryClassDatasetIterator:
    def __init__(self, pos_file_paths, neg_file_paths, disk_shard_size, shuffle_buffer_size,
                 shuffle = True, class_ratio = 0.5, soft_label = False, yield_period = 1, batch_size = 1):

        self.paths = [neg_file_paths, pos_file_paths]
        self.paths_len = [len(neg_file_paths), len(pos_file_paths)]
        self.shuffle = shuffle
        self.disk_shard_size = disk_shard_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.current_class = 0
        self.avail_class = [0,1]
        self.total_class = len(self.avail_class)
        self.len_iterator = [0 for class_name in range(self.total_class)]
        self.current_df_index = [-1 for class_name in range(self.total_class)]
        self.current_iterator = [[] for class_name in range(self.total_class)]
        self.class_ratio = class_ratio
        self.yield_period = yield_period
        self.batch_size = batch_size
        self.soft_label = soft_label
        self.keys = ["signal_token"]
        self.buffer = [[[] for key in self.keys] for class_name in range(self.total_class)]

        assert self.yield_period <= self.shuffle_buffer_size, "Shuffle period should be less than or equal to shuffle buffer size"

    def __iter__(self):
        return self


    def __next__(self):
        return self._next() , self.current_class

    def _read_shuffle_data(self, first_read = False):

        current_buffer = self.buffer[self.current_class]
        len_buffer = len(current_buffer[0])

        if first_read:
            assert len_buffer == 0, "Buffer should be empty when first read"

        else:
            assert len_buffer <= 1, f"Buffer should have at most one item, but has {len_buffer}"
            if len_buffer == 1:
                len_buffer = len(current_buffer[0][0])

        while len_buffer < self.shuffle_buffer_size and (self.current_df_index[self.current_class] < self.paths_len[self.current_class] - 1):
            self.current_df_index[self.current_class] += 1
            with np.load(self.paths[self.current_class][self.current_df_index[self.current_class]]) as npz:
                for key_idx, key in enumerate(self.keys):
                    current_buffer[key_idx].append(npz[key])
                len_buffer += len(npz[self.keys[0]])

        ## Concat and shuffle the data
        if self.shuffle:
            shuffle_idx = np.random.permutation(len_buffer)

        data_to_yield = []

        for key_idx, key in enumerate(self.keys):
            data = np.concatenate(current_buffer[key_idx])
            assert len_buffer == len(data), f"Buffer length {len_buffer} != data length {len(data)}"

            if self.shuffle:
                data = data[shuffle_idx]

            if len_buffer > self.yield_period:
                data_to_yield.append(data[:self.yield_period])
                current_buffer[key_idx] = [data[self.yield_period:]]
            else:
                data_to_yield.append(data)
                current_buffer[key_idx] = []

        self._set_iterator(data_to_yield)

        gc.collect()
        return None


    def _exhaust_buffer(self):
        current_buffer = self.buffer[self.current_class]
        assert len(current_buffer[0]) == 1, f"Buffer to be exhausted should have exactly one item, but has {len(current_buffer[0])}"
        data_to_yield = [current_buffer[key_idx][0] for key_idx in range(len(current_buffer))]
        self._set_iterator(data_to_yield)
        for key_idx in range(len(current_buffer)):
            current_buffer[key_idx] = []
        gc.collect()
        return None


    def _set_iterator(self, data_to_yield):
        lengths = [len(data) for data in data_to_yield]
        assert len(set(lengths)) == 1, f"Data lengths are not equal: {lengths}"
        iterator = zip(*data_to_yield)
        self.current_iterator[self.current_class] = iterator
        self.len_iterator[self.current_class] = lengths[0]
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
            elif len(self.buffer[self.current_class][0]) > 0: ## No data to read from disk, but buffer has data
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


class NanoporeDataset(IterableDataset):
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
                                          soft_label = self.soft_label, yield_period = self.yield_period, batch_size = self.batch_size)

    def set_epoch(self, epoch: int) -> None:
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
        self.dataset.set_epoch(epoch)
        return None

    ## END of NanoporeDataLoader


def load_dataset(pos_data_path, neg_data_path, batch_size,
                 disk_shard_size, rank, num_replicas, shuffle_buffer_size, yield_period, seed = 0, shuffle = True, drop_last = True,
                 pad_to = 200, bq_clip = 40, class_ratio = 1, prefetch_factor = 512, pin_memory = True, soft_label = False,
                 num_workers = 4, signal_stride = 6, kmer_size = 5):
    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, signal_stride = signal_stride, kmer_size = kmer_size)
    ## Use DataLoader to load the dataset
    pos_data_paths = glob.glob(f"{pos_data_path}/*.npz")
    neg_data_paths = glob.glob(f"{neg_data_path}/*.npz")

    if class_ratio is None:
        class_ratio = len(neg_data_paths) / len(pos_data_paths)
        if rank == 0:
            printmessage(f"Neg:Pos = {class_ratio:.3f}:1", msg_type="info")

    if yield_period is None:
        yield_period = batch_size * 25

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


def pad_collate(batch, pad_to, signal_stride, kmer_size, trim = 2):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First
    ## ORDER: ["segment_len_arr", "signal_token", "kmer_token", "dwell_motor_token", "dwell_pore_token", "bq_token"]

    signal_token_list = []

    for source, target in batch:
        signal_token_list.append(source[0])

    source = {}
    source["signal_token"] = torch.tensor(np.stack(signal_token_list), dtype=torch.float32).unsqueeze(1)

    return source
