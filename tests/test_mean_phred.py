import numpy as np

from deeprm.utils.utils import mean_phred


def test_mean_phred_basic():
    # Phred 10 corresponds to 0.1 error prob. Mean over two 10s is still 10.
    arr = np.array([10, 10])
    m = mean_phred(arr)
    assert np.isfinite(m)
    assert abs(m - 10.0) < 1e-6


def test_mean_phred_mixed():
    # Values 20 (0.01) and 30 (0.001) should give around ~23.0103
    arr = np.array([20, 30])
    m = mean_phred(arr)
    assert 22.9 < m < 23.1
