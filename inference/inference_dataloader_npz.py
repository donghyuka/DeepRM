import numpy as np
import torch
import math
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import glob
from utils.utils import printmessage

## Partially based on https://discuss.pytorch.org/t/an-iterabledataset-implementation-for-chunked-data/124437 by Majid Hajiheidari

class NanoporeDatasetIterator:
    """
    Iterator for loading Nanopore dataset from NPZ files.

    Args:
        file_paths (list): List of file paths to NPZ files.
        num_files_read_once (int): Number of files to read at once.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
    """
    def __init__(self, file_paths, num_files_read_once = 1000,
                 cb_len = 21, kmer_len = 5, sampling = 6, sig_window = 5):

        self.file_paths = file_paths
        self.current_path = None
        self.num_files_read_once = num_files_read_once
        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.cb_lr_pad = (cb_len-kmer_len)//2
        self.trim = kmer_len//2

    def __iter__(self):
        """
        Returns the iterator object itself.

        Returns:
            NanoporeDatasetIterator: The iterator object.
        """
        return self

    def _read_df(self, path):
        """
        Reads data from an NPZ file.

        Args:
            path (str): Path to the NPZ file.

        Returns:
            dict: Dictionary containing the data from the NPZ file.
        """
        data = {}
        try:
            with np.load(path, allow_pickle=True) as npz:
                data["label_id"] = npz["label_id"]
                data["segment_len"] = torch.tensor(npz["segment_len_arr"], dtype=torch.int32)
                data["signal_token"] = torch.tensor(npz["signal_token"], dtype=torch.float32)
                data["kmer_token"] = torch.tensor(npz["kmer_token"], dtype=torch.int32)
                data["dwell_motor_token"] = torch.tensor(npz["dwell_motor_token"], dtype=torch.float32)
                data["dwell_pore_token"] = torch.tensor(npz["dwell_pore_token"], dtype=torch.float32)
                data["bq_token"] = torch.tensor(npz["bq_token"], dtype=torch.float32)
        except:
            printmessage(f"Error loading {path}")
            return None
        return data

    def __next__(self):
        """
        Returns the next data from the iterator.

        Returns:
            dict: Dictionary containing the data from the next NPZ file.

        Raises:
            StopIteration: If there are no more files to read.
        """
        try:
            self.current_path = self.file_paths.pop(0)
        except IndexError:
            raise StopIteration
        return self._read_df(self.current_path)

    ## END of BinaryClassDatasetIterator

class NanoporeDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset for loading Nanopore data from NPZ files.

    Args:
        data_path (str): Path to the directory containing NPZ files.
        batch_size (int): Batch size for loading data.
        disk_shard_size (int): Size of the disk shard.
        rank (int): Rank of the current process.
        num_replicas (int): Number of replicas.
        seed (int): Random seed.
        num_files_read_once (int): Number of files to read at once.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        num_workers (int): Number of worker processes.
        resume_from (int): Number of files to skip from the start.
    """
    def __init__(self, data_path, batch_size, disk_shard_size, rank, num_replicas,
                 seed = 0, num_files_read_once = 1000, cb_len = 21, kmer_len = 5, sampling = 6, sig_window = 5,
                 num_workers = 1, resume_from = 0):
        super(NanoporeDataset).__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.disk_shard_size = disk_shard_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.file_paths = glob.glob(f"{self.data_path}/*.npz")
        self.epoch = 0
        self.seed = seed

        self.num_shard = math.ceil(len(self.file_paths)/num_replicas)
        self.num_files_read_once = num_files_read_once

        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.resume_from = resume_from

        if resume_from > 0:
            if resume_from % num_workers == 0:
                self.skip = resume_from // num_workers
            else:
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:
                    self.skip = resume_from // num_workers
                else:
                    if worker_info.id < resume_from % num_workers:
                        self.skip = resume_from // num_workers + 1
                    else:
                        self.skip = resume_from // num_workers
        else:
            self.skip = 0

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of shards in the dataset.
        """
        return self.num_shard - self.resume_from

    def __iter__(self):
        """
        Returns an iterator for the dataset.

        Returns:
            NanoporeDatasetIterator: Iterator for the dataset.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.file_paths = self.file_paths[self.rank::self.num_replicas]
        else:
            id = worker_info.id + self.rank * worker_info.num_workers
            nw = worker_info.num_workers * self.num_replicas
            self.file_paths = self.file_paths[id::nw]

        if self.skip > 0:
            self.file_paths = self.file_paths[self.skip:]

        return NanoporeDatasetIterator(self.file_paths, num_files_read_once = self.num_files_read_once,
                                       cb_len=self.cb_len, kmer_len=self.kmer_len, sampling=self.sampling, sig_window=self.sig_window)

class NanoporeDataLoader(DataLoader):
    """
    DataLoader for loading Nanopore data.

    Args:
        dataset (NanoporeDataset): The dataset to load data from.
        batch_size (int): Batch size for loading data.
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to pin memory.
        drop_last (bool): Whether to drop the last incomplete batch.
        collate_fn (callable): Function to collate data into batches.
        prefetch_factor (int): Number of batches to prefetch.
    """
    def __init__(self, dataset:NanoporeDataset, batch_size, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
        shuffle = False
        sampler = None
        batch_size = None
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         drop_last=drop_last, shuffle=shuffle, sampler=sampler, collate_fn=collate_fn,
                         prefetch_factor=prefetch_factor)

    ## END of NanoporeDataLoader

def load_dataset(data_path, batch_size, disk_shard_size, rank, num_replicas,
                 pad_to = 200, bq_clip = 40, num_files_read_once = 1, prefetch_factor = 100000, worker = 16,
                 cb_len = 21, kmer_len = 5, sampling = 6, sig_window = 5, resume_from = 0):
    """
    Loads the Nanopore dataset using DataLoader.

    Args:
        data_path (str): Path to the directory containing NPZ files.
        batch_size (int): Batch size for loading data.
        disk_shard_size (int): Size of the disk shard.
        rank (int): Rank of the current process.
        num_replicas (int): Number of replicas.
        pad_to (int): Padding length for sequences.
        bq_clip (int): Base quality clipping value.
        num_files_read_once (int): Number of files to read at once.
        prefetch_factor (int): Number of batches to prefetch.
        worker (int): Number of worker processes.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        resume_from (int): Number of files to skip from the start.

    Returns:
        NanoporeDataLoader: DataLoader for loading the dataset.
    """
    ## Use DataLoader to load the dataset
    batch_size = None
    dataset = NanoporeDataset(data_path, batch_size, disk_shard_size, rank, num_replicas,
                              num_files_read_once = num_files_read_once, cb_len = cb_len, kmer_len = kmer_len,
                              sampling = sampling, sig_window = sig_window, num_workers = worker, resume_from = resume_from)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=worker, pin_memory=True, drop_last=False,
                                    collate_fn = None, prefetch_factor=prefetch_factor)
    return dataloader