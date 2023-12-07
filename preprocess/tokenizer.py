import numpy as np

def sequence_to_kmer_token(seq, kmer = 5):
    ## 1. change string to array of int - 0, 1, 2, 3
    seq = seq.upper()
    seq = seq.replace('A', '0')
    seq = seq.replace('C', '1')
    seq = seq.replace('G', '2')
    seq = seq.replace('T', '3')
    seq = seq.replace('U', '3')
    seq = np.array(list(seq), dtype=int)

    ## 2. convert to kmer token
    seq = [seq[i:-kmer+i+1] for i in range(kmer)]
    seq = np.stack(seq, axis=1)
    seq = np.sum(seq * (4**np.arange(kmer)), axis=1)
    return seq

def expand_token_to_segment(token, segment_arr, sampling = 5):
    segment_len_arr = np.apply_over_axes(len, segment_arr, axes=1).flatten()
    segment_len_arr /= sampling
    token = np.repeat(token, segment_len_arr)
    return token

def segmented_signal_to_block(signal_segmented, sampling = 5):
    pass


