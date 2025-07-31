"""
Utility functions for DeepRM.
"""

import numpy as np


def mean_phred(phred):
    """
    Calculates the mean Phred quality score.

    Args:
        phred (numpy.ndarray): Array or list of Phred quality scores.

    Returns:
        float: Mean Phred quality score.
    """
    if not isinstance(phred, np.ndarray):
        phred = np.array(phred, dtype=int)
    else:
        phred = phred.astype(int)
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))
