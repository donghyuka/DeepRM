import numpy as np

set_idx = np.cumsum([0] + [int(np.floor(100 * split_ratio)) for split_ratio in [0.8,0.1,0.1]])

print(set_idx)