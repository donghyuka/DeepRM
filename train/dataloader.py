import torch
import math
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import glob

## Based on https://discuss.pytorch.org/t/an-iterabledataset-implementation-for-chunked-data/124437 by Majid Hajiheidari
## Load Nanopore Dataset from Pickled Pandas DataFrame

class BinaryClassDatasetIterator:
    def __init__(self, pos_file_paths, neg_file_paths, shuffle = True):

        self.pos_paths = pos_file_paths
        self.neg_paths = neg_file_paths
        self.shuffle = shuffle
        self.current_pos_df_index = -1
        self.current_neg_df_index = -1
        self.current_pos_iterator = None
        self.current_neg_iterator = None
        self.current_pos_or_neg = 1

    def __iter__(self):
        return self

    def __next__(self):
        ## Always alternate between positive and negative
        if self.current_pos_or_neg == 1:
            ## Return Positive Data
            self.current_pos_or_neg = 1
            if self.current_pos_df_index == -1:
                if self.current_pos_df_index == len(self.pos_paths) - 1:
                    raise StopIteration
                self.current_pos_df_index += 1
                if self.shuffle:
                    self.current_pos_iterator = pd.read_pickle(self.pos_paths[self.current_pos_df_index]).sample(frac=1).iterrows()
                else:
                    self.current_pos_iterator = pd.read_pickle(self.pos_paths[self.current_pos_df_index]).iterrows()

            try:
                result = next(self.current_pos_iterator)[1]
            except StopIteration:
                if self.current_pos_df_index == len(self.pos_paths) - 1:
                    raise StopIteration
                else:
                    self.current_pos_df_index += 1
                    if self.shuffle:
                        self.current_pos_iterator = pd.read_pickle(self.pos_paths[self.current_pos_df_index]).sample(frac=1).iterrows()
                    else:
                        self.current_pos_iterator = pd.read_pickle(self.pos_paths[self.current_pos_df_index]).iterrows()
                    result = next(self.current_pos_iterator)[1]
            self.current_pos_or_neg = 0

        else:
            ## Return Negative Data
            self.current_pos_or_neg = 0
            if self.current_neg_df_index == -1:
                if self.current_neg_df_index == len(self.neg_paths) - 1:
                    raise StopIteration
                self.current_neg_df_index += 1
                if self.shuffle:
                    self.current_neg_iterator = pd.read_pickle(self.neg_paths[self.current_neg_df_index]).sample(frac=1).iterrows()
                else:
                    self.current_neg_iterator = pd.read_pickle(self.neg_paths[self.current_neg_df_index]).iterrows()

            try:
                result = next(self.current_neg_iterator)[1]
            except StopIteration:
                if self.current_neg_df_index == len(self.neg_paths) - 1:
                    raise StopIteration
                else:
                    self.current_neg_df_index += 1
                    if self.shuffle:
                        self.current_neg_iterator = pd.read_pickle(self.neg_paths[self.current_neg_df_index]).sample(frac=1).iterrows()
                    else:
                        self.current_neg_iterator = pd.read_pickle(self.neg_paths[self.current_neg_df_index]).iterrows()
                    result = next(self.current_neg_iterator)[1]
            self.current_pos_or_neg = 1

        source, target = self.nanopore_row_to_tensor(result,self.current_pos_or_neg)

        return source, target

    def nanopore_row_to_tensor(self, row, pos_or_neg):
        ## Columns: "block_id", "motif", "block_score", "kmer_token", "bq_token", "position_token", "signal_token",
        ##          "spectrogram_token", "move_token", "target_mask"
        kmer_token = torch.tensor(row["kmer_token"], dtype=torch.long)
        bq_token = torch.tensor(row["bq_token"], dtype=torch.long)
        position_token = torch.tensor(row["position_token"], dtype=torch.long)
        signal_token = torch.tensor(row["signal_token"], dtype=torch.float)
        spectrogram_token = torch.tensor(row["spectrogram_token"], dtype=torch.float)
        move_token = torch.tensor(row["move_token"], dtype=torch.long)
        target_mask = torch.tensor(row["target_mask"], dtype=torch.float)

        return_dict = {"kmer_token": kmer_token, "bq_token": bq_token, "position_token": position_token,
                       "signal_token": signal_token, "spectrogram_token": spectrogram_token, "move_token": move_token,
                       "target_mask": target_mask}

        label = torch.tensor(pos_or_neg, dtype=torch.long)

        return return_dict, label


    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, pos_data_path, neg_data_path, batch_size, disk_shard_size, rank, num_replicas,
                 seed = 0, shuffle = True, drop_last = True):
        super(NanoporeDataset).__init__()

        self.pos_data_path = pos_data_path
        self.neg_data_path = neg_data_path
        self.batch_size = batch_size
        self.disk_shard_size = disk_shard_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.pos_file_paths = glob.glob(f"{self.pos_data_path}/*.pkl")
        self.neg_file_paths = glob.glob(f"{self.neg_data_path}/*.pkl")
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed

        if self.drop_last:
            self.pos_num_shard = math.floor(len(self.pos_file_paths)/num_replicas)
            self.pos_total_num_shard = self.pos_num_shard * num_replicas
            self.pos_dataset_size = self.pos_num_shard * disk_shard_size
            self.neg_num_shard = math.floor(len(self.neg_file_paths)/num_replicas)
            self.neg_total_num_shard = self.neg_num_shard * num_replicas
            self.neg_dataset_size = self.neg_num_shard * disk_shard_size
        else:
            self.pos_num_shard = math.ceil(len(self.pos_file_paths)/num_replicas)
            self.pos_total_num_shard = self.pos_num_shard * num_replicas
            self.pos_dataset_size = self.pos_num_shard * disk_shard_size
            self.neg_num_shard = math.ceil(len(self.neg_file_paths)/num_replicas)
            self.neg_total_num_shard = self.neg_total_num_shard * num_replicas
            self.neg_dataset_size = self.neg_total_num_shard * disk_shard_size

        self.dataset_size = self.pos_dataset_size + self.neg_dataset_size


    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        pos_file_paths = self._deterministic_shuffle_and_sample(self.pos_file_paths, self.pos_num_shard)
        neg_file_paths = self._deterministic_shuffle_and_sample(self.neg_file_paths, self.neg_num_shard)
        return BinaryClassDatasetIterator(pos_file_paths, neg_file_paths, self.shuffle)

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

    def _deterministic_shuffle_and_sample(self, data_path_list, num_shard):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(data_path_list), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(data_path_list)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = num_shard - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:num_shard]

        assert len(indices) == num_shard, f"{len(indices)} != {num_shard}"

        indices = indices[self.rank:num_shard:self.num_replicas]
        subsampled = [data_path_list[i] for i in indices]

        return subsampled


    ## END of NanoporeDataset


class NanoporeDataLoader(DataLoader):
    def __init__(self, dataset:NanoporeDataset, batch_size, num_workers, pin_memory, drop_last, shuffle, sampler):
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                           drop_last=drop_last, shuffle=shuffle, sampler=sampler)

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
                 disk_shard_size, rank, num_replicas, seed = 0, shuffle = True, drop_last = True):
    ## Use DataLoader to load the dataset
    dataset = NanoporeDataset(pos_data_path, neg_data_path, batch_size,
                                disk_shard_size, rank, num_replicas, seed, shuffle, drop_last)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True,
                            shuffle = False, sampler = None)
    return dataloader


