## Modified from: https://github.com/AlexanderVNikitin/tsgm/blob/main/tsgm/models/augmentations.py
## Original code by: Alexander Nikitin
## Modified for PyTorch.
## Drift augmentation modified from: https://github.com/arundo/tsaug/blob/master/src/tsaug/_augmenter/drift.py
## Trned augmentation modified from: https://github.com/timeseriesAI/tsai/blob/main/tsai/data/transforms.py


import math
import numpy as np
import numpy.typing as npt
import random
import scipy.interpolate
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from torch import Tensor
import torch

AugmentationOutput = Union[Tensor, Tuple[Tensor, Tensor]]


logger = logging.getLogger("augmentations")
logger.setLevel(logging.DEBUG)


class BaseAugmenter:
    def __init__(
            self,
            per_feature: bool,
    ) -> None:
        self.per_channel = per_feature

    def _get_seeds(self, total_num: int, n_seeds: int) -> Tensor:
        seeds_idx = torch.randint(0, total_num, (n_seeds,))
        return seeds_idx

    def generate(
            self, X: Tensor, y: Optional[Tensor] = None, n_samples: int = 1
    ) -> AugmentationOutput:
        raise NotImplementedError


class BaseCompose:
    def __init__(
            self,
            augmentations: List[BaseAugmenter],
    ) -> None:
        if isinstance(augmentations, (BaseCompose, BaseAugmenter)):
            augmentations = [augmentations]

        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.augmentations)

    def __call__(self, *args, **data) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> BaseAugmenter:
        return self.augmentations[item]


class GaussianNoise(BaseAugmenter):
    """Apply noise to the input time series.
    Args:
        variance ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_feature (bool): if set to True, noise will be sampled for each feature independently.
            Otherwise, the noise will be sampled once for all features. Default: True
    """

    def __init__(
            self,
            per_feature: bool = True,
    ) -> None:
        super(GaussianNoise, self).__init__(per_feature)

    def generate(self, X: Tensor, mean: float = 0, variance: float = 1.0,) -> AugmentationOutput:
        """
        Generate synthetic data with Gaussian noise.

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: Tensor

        :param mean: The mean of the noise. Default is 0.
        :type mean: float

        :param variance: The variance of the noise. Default is 1.0.
        :type variance: float

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[Tensor, Tuple[Tensor, Tensor]]
        """
        seeds_idx = self._get_seeds(total_num=X.shape[0])

        sigma = variance**0.5
        has_labels = y is not None
        if self.per_channel:
            gauss = torch.normal(mean, sigma, (n_samples, X.shape[1], X.shape[2]))
        else:
            gauss = torch.normal(mean, sigma, (n_samples, X.shape[1], 1))
        synthetic_X = X[seeds_idx] + gauss
        if has_labels:
            synthetic_y = y[seeds_idx]
            return synthetic_X, synthetic_y
        else:
            return synthetic_X


class MagnitudeWarping(BaseAugmenter):
    """
    Magnitude warping changes the magnitude of each
    sample by convolving the data window with a smooth curve varying around one
    https://dl.acm.org/doi/pdf/10.1145/3136755.3136817
    """

    def __init__(self) -> None:
        super(MagnitudeWarping, self).__init__(per_feature=False)

    def generate(self, X: Tensor, y: Optional[Tensor] = None, n_samples: int = 1, sigma: float = 0.2, n_knots: int = 4) -> AugmentationOutput:
        """
        Generates augmented samples via MagnitudeWarping for (X, y)

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: Tensor

        :param y: Optional labels tensor. If provided, labels will also be returned
        :type y: Optional[Tensor]

        :param n_samples: Number of augmented samples to generate. Default is 1.
        :type n_samples: int

        :param sigma: Standard deviation for the random warping. Default is 0.2.
        :type sigma: float

        :param n_knots: Number of knots used for warping curve. Default is 4.
        :type n_knots: int

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[Tensor, Tuple[Tensor, Tensor]]
        """
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

        orig_steps = np.arange(n_timesteps)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(n_samples, n_knots + 2, n_features))
        warp_steps = (np.ones( (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=n_knots + 2))).T

        result = torch.zeros((n_samples, n_timesteps, n_features))
        has_labels = y is not None

        if has_labels:
            result_y = torch.zeros((n_samples, 1))
        else:
            result_y = None

        for i in range(n_samples):
            random_sample_id = random.randint(0, n_data - 1)
            warper = np.array([scipy.interpolate.CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                    for dim in range(n_features)]).T
            warper = torch.tensor(warper).float()
            result[i] = X[random_sample_id] * warper
            if has_labels:
                result_y[i] = y[random_sample_id]

        if has_labels:
            return result, result_y
        else:
            return result


class WindowWarping(BaseAugmenter):
    """
    https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """

    def __init__(self) -> None:
        super(WindowWarping, self).__init__(per_feature=False)

    def generate(self, X: Tensor, y: Optional[Tensor] = None, window_ratio: float = 0.2, scales: Tuple = (0.25, 1.0), n_samples: int = 1) -> AugmentationOutput:
        """
        Generates augmented samples via MagnitudeWarping for (X, y)

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: Tensor

        :param y: Optional labels tensor. If provided, labels will also be returned
        :type y: Optional[Tensor]

        :param window_ratio: The ratio of the window size relative to the total number of timesteps.
            Default is 0.2.
        :type window_ratio: float

        :param scale: A tuple specifying the scale range for warping.
            Default is (0.25, 1.0).
        :type scale: tuple

        :param n_samples: Number of augmented samples to generate. Default is 1.
        :type n_samples: int

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[Tensor, Tuple[Tensor, Tensor]]
        """
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

        scales_per_sample = np.random.choice(scales, n_samples)
        warp_size = max(np.round(window_ratio * n_timesteps).astype(np.int64), 1)

        result = torch.zeros((n_samples, n_timesteps, n_features))
        result_y = torch.zeros((n_samples, 1))
        has_labels = y is not None
        for i in range(n_samples):
            window_starts = np.random.randint(
                low=0, high=n_timesteps - warp_size,
                size=(n_samples))
            window_ends = window_starts + warp_size
            random_sample_id = random.randint(0, n_data - 1)
            random_sample = X[random_sample_id]

            for dim in range(n_features):
                start_seg = random_sample[:window_starts[i], dim]
                warp_ts_size = max(round(warp_size * scales_per_sample[i]), 1)
                window_seg = np.interp(
                    x=np.linspace(0, warp_size - 1, num=warp_ts_size),
                    xp=np.arange(warp_size),
                    fp=random_sample[window_starts[i] : window_ends[i], dim],
                )
                end_seg = random_sample[window_ends[i] :, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                result[i, :, dim] = np.interp(
                    np.arange(n_timesteps),
                    np.linspace(0, n_timesteps - 1.0, num=warped.size),
                    warped,
                ).T
                if has_labels:
                    result_y[i] = y[random_sample_id]

        if has_labels:
            return result, result_y
        else:
            return result




class Drift(_Augmenter):
    """
    Drift the value of time series.

    The augmenter drifts the value of time series from its original values
    randomly and smoothly. The extent of drifting is controlled by the maximal
    drift and the number of drift points.

    Parameters
    ----------
    max_drift : float or tuple, optional
        The maximal amount of drift added to a time series.

        - If float, all series (all channels if `per_channel` is True) are
          drifted with the same maximum.
        - If tuple, the maximal drift added to a time series (a channel if
          `per_channel` is True) is sampled from this interval randomly.

        Default: 0.5.

    n_drift_points : int or list, optional
        The number of time points a new drifting trend is defined in a series.

        - If int, all series (all channels if `per_channel` is True) have the
          same number of drift points.
        - If list, the number of drift points defined in a series (a channel if
          `per_channel` is True) is sampled from this list randomly.

    kind : str, optional
        How the noise is added to the original time series. It must be either
        'additive' or 'multiplicative'. Default: 'additive'.

    per_channel : bool, optional
        Whether to sample independent drifting trends for each channel in a time
        series or to use the same drifting trends for all channels in a time
        series. Default: True.

    normalize : bool, optional
        Whether the drifting trend is added to the normalized time series. If
        True, each channel of a time series is normalized to [0, 1] first.
        Default: True.

    repeats : int, optional
        The number of times a series is augmented. If greater than one, a series
        will be augmented so many times independently. This parameter can also
        be set by operator `*`. Default: 1.

    prob : float, optional
        The probability of a series is augmented. It must be in (0.0, 1.0]. This
        parameter can also be set by operator `@`. Default: 1.0.

    seed : int, optional
        The random seed. Default: None.

    """

    def __init__(
            self,
            max_drift: Union[float, Tuple[float, float]] = 0.5,
            n_drift_points: Union[int, List[int]] = 3,
            kind: str = "additive",
            per_channel: bool = True,
            normalize: bool = True,
            repeats: int = 1,
            prob: float = 1.0,
            seed: Optional[int] = _default_seed,
    ):
        self.max_drift = max_drift
        self.n_drift_points = n_drift_points
        self.kind = kind
        self.per_channel = per_channel
        self.normalize = normalize
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return (
            "max_drift",
            "n_drift_points",
            "kind",
            "per_channel",
            "normalize",
        )

    @property
    def max_drift(self) -> Union[float, Tuple[float, float]]:
        return self._max_drift

    @max_drift.setter
    def max_drift(self, v: Union[float, Tuple[float, float]]) -> None:
        MAX_DRIFT_ERROR_MSG = (
            "Parameter `max_drift` must be a non-negative number "
            "or a 2-tuple of non-negative numbers representing an interval. "
        )
        if not isinstance(v, (float, int)):
            if isinstance(v, tuple):
                if len(v) != 2:
                    raise ValueError(MAX_DRIFT_ERROR_MSG)
                if (not isinstance(v[0], (float, int))) or (
                        not isinstance(v[1], (float, int))
                ):
                    raise TypeError(MAX_DRIFT_ERROR_MSG)
                if v[0] > v[1]:
                    raise ValueError(MAX_DRIFT_ERROR_MSG)
                if (v[0] < 0.0) or (v[1] < 0.0):
                    raise ValueError(MAX_DRIFT_ERROR_MSG)
            else:
                raise TypeError(MAX_DRIFT_ERROR_MSG)
        elif v < 0.0:
            raise ValueError(MAX_DRIFT_ERROR_MSG)
        self._max_drift = v

    @property
    def n_drift_points(self) -> Union[int, List[int]]:
        return self._n_drift_points

    @n_drift_points.setter
    def n_drift_points(self, n: Union[int, List[int]]) -> None:
        N_DRIFT_POINTS_ERROR_MSG = (
            "Parameter `n_drift_points` must be a positive integer "
            "or a list of positive integers."
        )
        if not isinstance(n, int):
            if isinstance(n, list):
                if len(n) == 0:
                    raise ValueError(N_DRIFT_POINTS_ERROR_MSG)
                if not all([isinstance(nn, int) for nn in n]):
                    raise TypeError(N_DRIFT_POINTS_ERROR_MSG)
                if not all([nn > 0 for nn in n]):
                    raise ValueError(N_DRIFT_POINTS_ERROR_MSG)
            else:
                raise TypeError(N_DRIFT_POINTS_ERROR_MSG)
        elif n <= 0:
            raise ValueError(N_DRIFT_POINTS_ERROR_MSG)
        self._n_drift_points = n

    @property
    def per_channel(self) -> bool:
        return self._per_channel

    @per_channel.setter
    def per_channel(self, p: bool) -> None:
        if not isinstance(p, bool):
            raise TypeError("Paremeter `per_channel` must be boolean.")
        self._per_channel = p

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, p: bool) -> None:
        if not isinstance(p, bool):
            raise TypeError("Paremeter `normalize` must be boolean.")
        self._normalize = p

    @property
    def kind(self) -> str:
        return self._kind

    @kind.setter
    def kind(self, k: str) -> None:
        if not isinstance(k, str):
            raise TypeError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        if k not in ("additive", "multiplicative"):
            raise ValueError(
                "Parameter `kind` must be either 'additive' or 'multiplicative'."
            )
        self._kind = k

    def _augment_core(
            self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        N, T, C = X.shape
        rand = np.random.RandomState(self.seed)

        if isinstance(self.n_drift_points, int):
            n_drift_points = set([self.n_drift_points])
        else:
            n_drift_points = set(self.n_drift_points)

        ind = rand.choice(
            len(n_drift_points), N * (C if self.per_channel else 1)
        )  # map series to n_drift_points

        drift = np.zeros((N * (C if self.per_channel else 1), T))
        for i, n in enumerate(n_drift_points):
            if not (ind == i).any():
                continue
            anchors = np.cumsum(
                rand.normal(size=((ind == i).sum(), n + 2)), axis=1
            )  # type: np.ndarray
            interpFuncs = CubicSpline(
                np.linspace(0, T, n + 2), anchors, axis=1
            )  # type: Callable
            drift[ind == i, :] = interpFuncs(np.arange(T))
        drift = drift.reshape((N, -1, T)).swapaxes(1, 2)
        drift = drift - drift[:, 0, :].reshape(N, 1, -1)
        drift = drift / abs(drift).max(axis=1, keepdims=True)
        if isinstance(self.max_drift, (float, int)):
            drift = drift * self.max_drift
        else:
            drift = drift * rand.uniform(
                low=self.max_drift[0],
                high=self.max_drift[1],
                size=(N, 1, C if self.per_channel else 1),
            )

        if self.kind == "additive":
            if self.normalize:
                X_aug = X + drift * (
                        X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)
                )
            else:
                X_aug = X + drift
        else:
            X_aug = X * (1 + drift)

        if Y is not None:
            Y_aug = Y.copy()
        else:
            Y_aug = None

        return X_aug, Y_aug


class Trend(BaseAugmenter):
    "Randomly rotates the sequence along the z-axis"
    def __init__(self, magnitude=0.1, ex=None, **kwargs):
        self.magnitude, self.ex = magnitude, ex
        super().__init__(**kwargs)
    def encodes(self, o: Tensor):
        if not self.magnitude or self.magnitude <= 0: return o
        flat_x = o.reshape(o.shape[0], -1)
        ran = flat_x.max(dim=-1, keepdim=True)[0] - flat_x.min(dim=-1, keepdim=True)[0]
        trend = torch.linspace(0, 1, o.shape[-1], device=o.device) * ran
        t = (1 + self.magnitude * 2 * (np.random.rand() - 0.5) * trend)
        t -= t.mean(-1, keepdim=True)
        if o.ndim == 3: t = t.unsqueeze(1)
        output = o + t
        if self.ex is not None: output[...,self.ex,:] = o[...,self.ex,:]
        return output


