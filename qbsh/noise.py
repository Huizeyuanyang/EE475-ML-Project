"""Synthetic imperfect humming, for the noise-robustness experiment (exp_noise.py).

Owner: D.

Each function takes a pitch matrix like QueryData.pitch, (n, T) semitones with np.nan for
unvoiced frames, and returns a corrupted copy of the same shape. Unvoiced frames stay NaN.
The inputs are never modified. `seed` makes the corruption reproducible.

The kinds of corruption (jitter, dropped voiced frames, tempo distortion) are the ones in
docs/experiment_plan.md. The exact definitions and strengths are for the team to decide.
"""
from __future__ import annotations

import numpy as np


def add_pitch_jitter(pitch: np.ndarray, std_semitones: float, seed: int = 0) -> np.ndarray:
    """Add independent Gaussian noise (std in semitones) to every voiced frame.

    TODO(D)
    """
    raise NotImplementedError("noise: pitch jitter")


def drop_voiced_frames(pitch: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Turn a random `fraction` (0..1) of each query's voiced frames into NaN.

    TODO(D)
    """
    raise NotImplementedError("noise: dropped voiced frames")


def warp_tempo(pitch: np.ndarray, rate: float) -> np.ndarray:
    """Play each query `rate` times faster (rate > 1) or slower (rate < 1), keeping T frames.

    TODO(D)
    """
    raise NotImplementedError("noise: tempo distortion")
