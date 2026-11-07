"""Reference baselines (shared, implemented). Not course deliverables.

They exist so B, C and D can run their code before A's features are ready,
and so every experiment has a reference row.

    RawPitch + NearestCentroid   no key normalization; the "no processing" reference row.
    NearestQuery                 the similarity baseline: put it after KeyNormalizer and
                                 impute_unvoiced, and it matches a query's contour to the
                                 training queries by Euclidean distance.
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


class NearestQuery(Classifier):
    """Score of a song = minus the squared Euclidean distance to its closest training query.

    No learning: fit just stores the training queries. Meant to follow
    KeyNormalizer + impute_unvoiced, so it compares key-normalized contours.
    Songs unseen in training get -inf.
    """

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.seen_ = np.unique(self.y_)
        return self

    def predict_scores(self, X):
        d2 = ((X ** 2).sum(1)[:, None] - 2 * X @ self.X_.T + (self.X_ ** 2).sum(1)[None, :])
        scores = np.full((len(X), N_SONGS), -np.inf)
        for s in self.seen_:
            scores[:, s] = -d2[:, self.y_ == s].min(axis=1)
        return scores
