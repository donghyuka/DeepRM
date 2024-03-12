import torch
import math

d_model = 16
seq_len = 10

position = torch.arange(seq_len).unsqueeze(1)
print(position)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
print(div_term)
pe = torch.zeros(1, seq_len, d_model)
pe[:, :, 0::2] = torch.sin(position * div_term)
pe[:, :, 1::2] = torch.cos(position * div_term)
print(pe)




