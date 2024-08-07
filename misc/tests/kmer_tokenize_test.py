import numpy as np
import torch
import pandas as pd
import glob
import time
from torch import Tensor, nn
import math
from tqdm import tqdm


def sequence_to_kmer_token(seq, kmer):
    ## 1. change string to array of int - 0, 1, 2, 3
    seq = seq.upper()
    seq = seq.replace('A', '0')
    seq = seq.replace('C', '2')
    seq = seq.replace('G', '1')
    seq = seq.replace('T', '3')
    seq = seq.replace('U', '3')
    seq = np.array(list(seq), dtype=np.int16)

    ## 2. convert to kmer token
    seq = [seq[i:kmer+i] for i in range(len(seq)-kmer+1)]
    seq = np.stack(seq, axis=1)
    quaternary = 4**np.arange(kmer).reshape(-1,1)
    seq = np.sum(seq * quaternary, axis=0) + 1 ## 0 is reserved for padding
    return seq

def method1(df):
    df["src_kmer"] = df["kmer_token"].apply(lambda x: sequence_to_kmer_token(x, 5))
    src_kmer_tensor = torch.tensor(np.stack(df["src_kmer"].values))
    return src_kmer_tensor

def sequence_to_token(seq):
    seq = np.array(list(seq))
    return seq


def nuc_to_kmer(seq: torch.Tensor, kmer: int):
    seq = (seq - 65).clip(None,8)%5 ## This is kind of a witchcraft. It converts ACGTU to 01233.
    seq = seq.unfold(1, kmer, 1)
    seq = (seq * (4**torch.arange(kmer, device = seq.device)).unsqueeze(0).unsqueeze(0)).sum(dim = -1)
    seq = seq + 1
    return seq


def method2(df):
    df["src_kmer"] = df["kmer_token"].apply(sequence_to_token)
    src_kmer_tensor = torch.tensor(np.stack(df["src_kmer"].values).view(np.int32))
    src_kmer_tensor = nuc_to_kmer(src_kmer_tensor, 5)
    return src_kmer_tensor


def main():
    dir = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver062124/score-perfect/train/pos/*.pkl"
    df = pd.read_pickle(np.random.choice(glob.glob(dir)))
    start = time.time()
    src_kmer_tensor1 = method1(df.copy())
    print(src_kmer_tensor1)
    print(time.time()-start)
    start = time.time()
    src_kmer_tensor2 = method2(df.copy())
    print(src_kmer_tensor2)
    print(time.time()-start)
    assert torch.allclose(src_kmer_tensor1, src_kmer_tensor2)
    return None

class RelativePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, src_seg_len:Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        src_seg_len_flat = torch.cat([src_seg_len, self.seq_len - src_seg_len.sum(dim = 1, keepdims=True)], dim = 1)
        assert torch.all(src_seg_len_flat.sum(dim = 1) == self.seq_len)
        assert torch.all(src_seg_len_flat[:, -1] >= 0)
        src_seg_len_flat = torch.flatten(src_seg_len_flat)
        position = torch.arange(src_seg_len.shape[1]+1,device=src_seg_len.device).repeat(src_seg_len.shape[0]).repeat_interleave(src_seg_len_flat).reshape(src_seg_len.shape[0], self.seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)).unsqueeze(0).unsqueeze(0)
        pe = torch.zeros(src_seg_len.shape[0], self.seq_len, self.d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    ## END OF PositionalEncoding


def main2():
    dir = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver062124/score-perfect/train/pos/*.pkl"
    for file in tqdm(glob.glob(dir)):
        df = pd.read_pickle(file)
        src_seg_len = torch.tensor(np.stack(df["segment_len_arr"].values))
        rpe = RelativePositionalEncoding(512, 200)
        src_seg_len = rpe(src_seg_len)

    return None


if __name__ == "__main__":
    main2()