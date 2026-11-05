"""Histogram features (moved out of features.py so A and B don't share a file).

Owner: B (helping A).
Course method: histogram features 9.2.

A's HandDesignedFeatures imports both functions from here.
Input is the pitch matrix (n, T) in semitones with np.nan for unvoiced frames.
"""
from __future__ import annotations

import numpy as np


def pitch_histogram(pitch: np.ndarray, n_bins: int = 25, value_range=(-12.0, 12.0)) -> np.ndarray:
    """9.2 Histogram of voiced pitch values per query, normalized to sum to 1.

    Meant for key-normalized pitch, so the default range is +-1 octave around 0.
    Values outside value_range go into the edge bins, so every row sums to 1.
    pitch (n, T) with NaN -> (n, n_bins). NaN frames are ignored.

    TODO(B)
    """
    raise NotImplementedError("B: 9.2 pitch histogram")


def interval_histogram(pitch: np.ndarray, max_interval: int = 12) -> np.ndarray:
    """9.2 Histogram of pitch changes between consecutive voiced frames (or notes).

    Key-invariant by construction: adding k semitones to a query doesn't change it.
    Round intervals to whole semitones, clip to [-max_interval, max_interval].
    pitch (n, T) with NaN -> (n, 2 * max_interval + 1), each row sums to 1.

    TODO(B)
    """
    raise NotImplementedError("B: 9.2 interval histogram")
