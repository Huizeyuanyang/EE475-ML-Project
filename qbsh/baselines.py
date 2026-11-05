"""Reference baselines (shared, implemented). Not course deliverables.

They exist so B, C and D can run their code before A's features are ready,
and so every experiment has a "no processing" reference row.
"""
from __future__ import annotations

import numpy as np

from .contracts import N_SONGS, Classifier, Transformer


class RawPitch(Transformer):
    """No processing: unvoiced (NaN) frames -> 0. Placeholder until A's features exist."""

    def transform(self, X):
        return np.nan_to_num(X, nan=0.0)


class NearestCentroid(Classifier):
    """Score = minus the squared distance to each song's mean training vector."""

    def fit(self, X, y):
        self.centroids_ = np.zeros((N_SONGS, X.shape[1]))
        self.seen_ = np.zeros(N_SONGS, dtype=bool)
        for s in np.unique(y):
            self.centroids_[s] = X[y == s].mean(axis=0)
            self.seen_[s] = True
        return self

    def predict_scores(self, X):
        c = self.centroids_
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ c.T + (c ** 2).sum(1)[None, :]
        return np.where(self.seen_[None, :], -d2, -np.inf)  # songs unseen in training -> -inf
