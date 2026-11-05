"""Dimensionality reduction and clustering.

Owner: C (together with K-fold CV in evaluation.py).
Course methods: PCA 8.3, K-means 8.5.
Owned experiment: dimensionality reduction (experiments/exp_dimred.py).

All of these are Transformers: fit on TRAIN only, then transform train/val/test.
"""
from __future__ import annotations

import numpy as np

from .contracts import Transformer


class PCA(Transformer):
    """8.3 Principal component analysis.

    fit: center X (store mean_), eigendecompose the covariance (or SVD of centered X).
    Sets components_ (n_components, d), explained_variance_ (n_components,),
    explained_variance_ratio_ (n_components,), sorted largest first.
    transform: (X - mean_) @ components_.T -> (n, n_components).
    n_components=None keeps all d.

    TODO(C)
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X, y=None):
        raise NotImplementedError("C: 8.3 PCA")

    def transform(self, X):
        raise NotImplementedError("C: 8.3 PCA")

    def inverse_transform(self, Z):
        """(n, n_components) -> (n, d) reconstruction."""
        raise NotImplementedError("C: 8.3 PCA")


class KMeans(Transformer):
    """8.5 K-means clustering.

    fit: sets centroids_ (k, d), labels_ (n,), inertia_ (sum of squared distances).
    predict(X) -> (n,) index of the nearest centroid.
    transform(X) -> (n, k) distances to every centroid (usable as features).
    Use several restarts (n_init) and keep the lowest inertia.

    TODO(C)
    """

    def __init__(self, n_clusters: int = 48, max_its: int = 100, n_init: int = 5, seed: int = 0):
        self.n_clusters, self.max_its, self.n_init, self.seed = n_clusters, max_its, n_init, seed

    def fit(self, X, y=None):
        raise NotImplementedError("C: 8.5 K-means")

    def predict(self, X):
        raise NotImplementedError("C: 8.5 K-means")

    def transform(self, X):
        raise NotImplementedError("C: 8.5 K-means")


def cluster_purity(clusters: np.ndarray, y: np.ndarray) -> float:
    """Fraction of points whose cluster's majority song equals their own song (1.0 = pure).

    TODO(C)
    """
    raise NotImplementedError("C: cluster purity")
