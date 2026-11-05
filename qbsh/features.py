"""Hand-designed features.

Owner: A.
Course methods: standard normalization 9.3, missing-value imputation 9.4,
nonlinear feature engineering 10.5.
Owned experiment: key normalization (experiments/exp_key_normalization.py).

The two histogram features (9.2) live in histograms.py (owner: B), so A and B
never edit the same file. HandDesignedFeatures imports them from there.

Input to everything here is the pitch matrix from data.load_queries():
(n, T) semitones, np.nan where the frame is unvoiced (~27% of frames).

Order inside a pipeline:
    KeyNormalizer -> impute_unvoiced -> HandDesignedFeatures -> StandardScaler
KeyNormalizer comes first so its statistics use only real (voiced) frames.

B, C and D depend on `HandDesignedFeatures` and `StandardScaler`; until they
are ready, use baselines.RawPitch as a stand-in.
"""
from __future__ import annotations

import numpy as np

from .contracts import Transformer
from .histograms import interval_histogram, pitch_histogram  # noqa: F401  (owner: B; used by HandDesignedFeatures)


# ---------------------------------------------------------------- 9.4 missing values

def impute_unvoiced(pitch: np.ndarray, method: str = "linear") -> np.ndarray:
    """9.4 Fill the NaN (unvoiced) frames of each query.

    pitch (n, T) with NaN -> (n, T) with no NaN. Voiced frames are unchanged.
    method:
        "linear" interpolate between the nearest voiced frames on each side
                 (leading/trailing gaps take the nearest voiced value)
        "hold"   repeat the last voiced value
        "mean"   the row's mean voiced pitch
        "zero"   0 (what baselines.RawPitch does; for comparison only)
    A row with no voiced frames at all becomes all zeros.

    TODO(A)
    """
    raise NotImplementedError("A: 9.4 missing-value imputation")


# ---------------------------------------------------------------- owned experiment

class KeyNormalizer(Transformer):
    """Remove the singer's key: subtract one number per query from its voiced frames.

    People hum the same tune in different keys, so the raw contour of one song
    shifts up or down between singers. After normalization, a query and the same
    query transposed by +k semitones must give identical output.

    method:
        "none"   no change (the reference row of the experiment)
        "mean"   subtract the mean voiced pitch
        "median" subtract the median voiced pitch
        "mode"   subtract the most common pitch, rounded to semitones
        "first"  subtract the first voiced frame (first note)
    NaN frames stay NaN. Stateless, so `fit` does nothing.

    TODO(A)
    """

    METHODS = ("none", "mean", "median", "mode", "first")

    def __init__(self, method: str = "median"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method

    def transform(self, pitch):
        raise NotImplementedError("A: key normalization")


# ---------------------------------------------------------------- 10.5 nonlinear features

def contour_statistics(pitch: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """10.5 Hand-designed nonlinear summaries of each contour.

    Ideas: voiced ratio, pitch range, std, mean |frame-to-frame change|,
    number of note changes, overall slope, position of the highest note, ...
    pitch (n, T) with NaN -> (features (n, k), names list of length k).

    TODO(A)
    """
    raise NotImplementedError("A: 10.5 nonlinear feature engineering")


def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """10.5 All monomials of the columns of X up to `degree` (no constant column).

    X (n, d) -> (n, d') where d' = C(d + degree, degree) - 1.

    TODO(A)
    """
    raise NotImplementedError("A: 10.5 polynomial features")


class HandDesignedFeatures(Transformer):
    """The feature vector B, C and D use: concatenates the pieces above
    plus B's pitch_histogram / interval_histogram (histograms.py).

    Until B's histograms exist, A can build and test this with
    use_pitch_hist=False, use_interval_hist=False.

    Input: pitch (n, T) that has already been key-normalized (NaN allowed;
    each piece decides how to treat NaN).
    Output: (n, d) with no NaN, and `feature_names_` (length d) set in `fit`
    so C's feature selection can report which features matter.

    Must work for any T, so B can crop queries in the query-length experiment.

    TODO(A)
    """

    def __init__(self, use_pitch_hist: bool = True, use_interval_hist: bool = True,
                 use_stats: bool = True):
        self.use_pitch_hist = use_pitch_hist
        self.use_interval_hist = use_interval_hist
        self.use_stats = use_stats

    def fit(self, pitch, y=None):
        raise NotImplementedError("A: hand-designed feature vector")

    def transform(self, pitch):
        raise NotImplementedError("A: hand-designed feature vector")


# ---------------------------------------------------------------- 9.3 normalization

class StandardScaler(Transformer):
    """9.3 Standard normalization: (x - mean) / std per column, statistics from TRAIN only.

    Columns with std == 0 must not produce NaN or inf (divide by 1 instead).

    TODO(A)
    """

    def fit(self, X, y=None):
        raise NotImplementedError("A: 9.3 standard normalization")

    def transform(self, X):
        raise NotImplementedError("A: 9.3 standard normalization")
