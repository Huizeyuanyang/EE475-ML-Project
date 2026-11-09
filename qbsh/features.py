"""Hand-designed features.

Owner: A.
Course methods: standard normalization 9.3, missing-value imputation 9.4,
nonlinear feature engineering 10.5.
Owned experiment: key normalization (experiments/exp_key_normalization.py).

The two histogram features (9.2) live in histograms.py (owner: C), so A and C
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

from itertools import combinations_with_replacement

import numpy as np

from .contracts import Transformer
from .histograms import interval_histogram, pitch_histogram  # noqa: F401  (owner: C; used by HandDesignedFeatures)


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
    """
    if method not in ("linear", "hold", "mean", "zero"):
        raise ValueError(f"unknown method {method!r}")
    pitch = np.asarray(pitch, dtype=float)
    out = pitch.copy()                      # never modify the caller's array
    t = np.arange(pitch.shape[1])
    for i, row in enumerate(pitch):
        voiced = ~np.isnan(row)
        if not voiced.any():                # nothing to fill from
            out[i] = 0.0
            continue
        if method == "zero":
            out[i, ~voiced] = 0.0
        elif method == "mean":
            out[i, ~voiced] = row[voiced].mean()
        elif method == "linear":
            # np.interp joins the voiced frames with straight lines and, outside the
            # first/last voiced frame, holds the nearest voiced value.
            out[i] = np.interp(t, t[voiced], row[voiced])
        else:  # "hold"
            # index of the most recent voiced frame at or before each frame;
            # frames before the first voiced one take the first voiced value.
            last = np.where(voiced, t, -1)
            last = np.maximum.accumulate(last)
            last[last < 0] = t[voiced][0]
            out[i] = row[last]
    return out


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
    A row with no voiced frames is left unchanged (all NaN).
    """

    METHODS = ("none", "mean", "median", "mode", "first")

    def __init__(self, method: str = "median"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method

    def reference(self, pitch) -> np.ndarray:
        """The number subtracted from each query, shape (n,). 0 for "none"."""
        pitch = np.asarray(pitch, dtype=float)
        n = pitch.shape[0]
        ref = np.zeros(n)
        if self.method == "none":
            return ref
        for i, row in enumerate(pitch):
            v = row[~np.isnan(row)]
            if v.size == 0:
                continue
            if self.method == "mean":
                ref[i] = v.mean()
            elif self.method == "median":
                ref[i] = np.median(v)
            elif self.method == "first":
                ref[i] = v[0]
            else:  # "mode"
                # floor(x + 0.5) rounds to the nearest semitone and, unlike np.round
                # (round-half-to-even), shifts exactly by k when x shifts by an integer k.
                notes = np.floor(v + 0.5)
                values, counts = np.unique(notes, return_counts=True)
                ref[i] = values[np.argmax(counts)]      # ties -> lowest note
        return ref

    def transform(self, pitch):
        pitch = np.asarray(pitch, dtype=float)
        return pitch - self.reference(pitch)[:, None]   # NaN - x stays NaN


# ---------------------------------------------------------------- 10.5 nonlinear features

N_SEGMENTS = 8          # contour_statistics: number of coarse "melody shape" segments


def _segment_means(v: np.ndarray, n_segments: int) -> np.ndarray:
    """Mean pitch of `n_segments` equal-length pieces of a gap-free contour v."""
    if v.size == 1:
        return np.full(n_segments, v[0])
    # resample to n_segments * 16 points so every segment has samples even for short v
    x = np.interp(np.linspace(0, v.size - 1, n_segments * 16), np.arange(v.size), v)
    return x.reshape(n_segments, 16).mean(axis=1)


def contour_statistics(pitch: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """10.5 Hand-designed nonlinear summaries of each contour.

    Ideas: voiced ratio, pitch range, std, mean |frame-to-frame change|,
    number of note changes, overall slope, position of the highest note, ...
    pitch (n, T) with NaN -> (features (n, k), names list of length k).

    Every feature is computed from the voiced frames only and is scaled by the
    clip length where it depends on time, so the size and meaning of the output
    do not change when queries are cropped (any T works).
    Feed key-normalized pitch: then the pitch-valued features (mean, start, end,
    segment means) are relative to the singer's own key.

    Features (k = 15 + N_SEGMENTS):
        voiced_ratio      fraction of frames that are voiced
        onset_pos         where the first voiced frame is, as a fraction of T
        mean, std         of the voiced pitch
        range, iqr_80     max - min, and 90th - 10th percentile (robust range)
        start, end        first and last voiced pitch
        mean_abs_step     mean |change| between consecutive voiced frames
        change_rate       fraction of those steps of at least half a semitone (note changes)
        up_ratio          among the note changes, the fraction that go up
        max_jump          largest |step|, capped at 12 semitones
        slope             least-squares slope of pitch vs time (time scaled to 0..1)
        argmax_pos        where the highest note is, as a fraction of the voiced span
        argmin_pos        where the lowest note is
        seg_1..seg_8      mean pitch in 8 equal parts of the voiced span: a coarse
                          melody shape that ignores leading silence and small
                          timing differences
    A row with no voiced frames gives all zeros.
    """
    pitch = np.asarray(pitch, dtype=float)
    n, T = pitch.shape
    names = ["voiced_ratio", "onset_pos", "mean", "std", "range", "iqr_80",
             "start", "end", "mean_abs_step", "change_rate", "up_ratio", "max_jump",
             "slope", "argmax_pos", "argmin_pos"] + [f"seg_{k + 1}" for k in range(N_SEGMENTS)]
    feats = np.zeros((n, len(names)))
    t = np.arange(T)
    for i, row in enumerate(pitch):
        voiced = ~np.isnan(row)
        if not voiced.any():
            continue
        idx = t[voiced]
        v = row[voiced]                                   # voiced pitch, in time order
        steps = np.diff(v)
        changes = steps[np.abs(steps) >= 0.5]
        span = row[idx[0]: idx[-1] + 1]                   # first to last voiced frame
        filled = impute_unvoiced(span[None, :], "linear")[0]
        tt = (idx - idx[0]) / max(idx[-1] - idx[0], 1)    # voiced times scaled to 0..1
        slope = np.polyfit(tt, v, 1)[0] if v.size > 1 and np.ptp(tt) > 0 else 0.0
        feats[i] = [
            voiced.mean(),
            idx[0] / T,
            v.mean(),
            v.std(),
            np.ptp(v),
            np.percentile(v, 90) - np.percentile(v, 10),
            v[0],
            v[-1],
            np.abs(steps).mean() if steps.size else 0.0,
            changes.size / steps.size if steps.size else 0.0,
            (changes > 0).mean() if changes.size else 0.5,
            min(np.abs(steps).max(), 12.0) if steps.size else 0.0,
            slope,
            np.argmax(filled) / max(filled.size - 1, 1),
            np.argmin(filled) / max(filled.size - 1, 1),
            *_segment_means(filled, N_SEGMENTS),
        ]
    return feats, names


def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """10.5 All monomials of the columns of X up to `degree` (no constant column).

    X (n, d) -> (n, d') where d' = C(d + degree, degree) - 1.
    Columns are ordered by degree: x1..xd, then x1^2, x1*x2, ..., xd^2, and so on.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    cols = []
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):   # each monomial once
            cols.append(np.prod(X[:, combo], axis=1))
    return np.stack(cols, axis=1) if cols else np.zeros((X.shape[0], 0))


def polynomial_feature_names(names: list[str], degree: int = 2) -> list[str]:
    """Names matching the columns of polynomial_features, e.g. 'std*range', 'std^2'."""
    out = []
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(len(names)), deg):
            parts = [names[j] for j in combo]
            out.append(parts[0] + f"^{deg}" if len(set(combo)) == 1 and deg > 1 else "*".join(parts))
    return out


class HandDesignedFeatures(Transformer):
    """The feature vector B, C and D use: concatenates the pieces above
    plus C's pitch_histogram / interval_histogram (histograms.py).

    Until C's histograms exist, A can build and test this with
    use_pitch_hist=False, use_interval_hist=False.

    Input: pitch (n, T) that has already been key-normalized (NaN allowed;
    each piece decides how to treat NaN).
    Output: (n, d) with no NaN, and `feature_names_` (length d) set in `fit`
    so C's feature selection can report which features matter.

    Must work for any T, so B can crop queries in the query-length experiment.

    Pieces, in this order:
        contour_statistics (if use_stats), expanded with polynomial_features
            up to degree `poly_degree` (1 = the statistics as they are)
        pitch_histogram    (if use_pitch_hist; C)
        interval_histogram (if use_interval_hist; C)
    Follow it with StandardScaler: the pieces have very different scales.
    """

    poly_degree = 2

    def __init__(self, use_pitch_hist: bool = True, use_interval_hist: bool = True,
                 use_stats: bool = True):
        self.use_pitch_hist = use_pitch_hist
        self.use_interval_hist = use_interval_hist
        self.use_stats = use_stats

    def _pieces(self, pitch):
        pitch = np.asarray(pitch, dtype=float)
        blocks, names = [], []
        if self.use_stats:
            stats, stat_names = contour_statistics(pitch)
            blocks.append(polynomial_features(stats, self.poly_degree))
            names += polynomial_feature_names(stat_names, self.poly_degree)
        if self.use_pitch_hist:
            h = pitch_histogram(pitch)
            blocks.append(h)
            names += [f"pitch_hist_{j}" for j in range(h.shape[1])]
        if self.use_interval_hist:
            h = interval_histogram(pitch)
            blocks.append(h)
            names += [f"interval_hist_{j}" for j in range(h.shape[1])]
        if not blocks:
            raise ValueError("HandDesignedFeatures: every piece is switched off")
        return np.hstack(blocks), names

    def fit(self, pitch, y=None):
        _, self.feature_names_ = self._pieces(pitch)
        return self

    def transform(self, pitch):
        X, _ = self._pieces(pitch)
        return np.nan_to_num(X)     # safety net: the output must never contain NaN


# ---------------------------------------------------------------- 9.3 normalization

class StandardScaler(Transformer):
    """9.3 Standard normalization: (x - mean) / std per column, statistics from TRAIN only.

    Columns with std == 0 must not produce NaN or inf (divide by 1 instead).
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)     # constant column -> divide by 1
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_
