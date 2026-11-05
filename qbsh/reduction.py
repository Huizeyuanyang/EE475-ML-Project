"""Dimensionality reduction, clustering and feature selection.

Owner: C (together with K-fold CV in evaluation.py).
Course methods: PCA 8.3, K-means 8.5, PCA-sphering 9.5,
feature selection via boosting 9.6 and via regularization 9.7.
Owned experiment: dimensionality reduction & feature selection analysis
(experiments/exp_dimred_featsel.py).

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


class PCASphering(Transformer):
    """9.5 PCA-sphering: PCA projection divided by sqrt(eigenvalue + epsilon).

    The output has (close to) identity covariance on the training data.
    Compare against features.StandardScaler before a linear classifier.

    TODO(C)
    """

    def __init__(self, n_components: int | None = None, epsilon: float = 1e-8):
        self.n_components, self.epsilon = n_components, epsilon

    def fit(self, X, y=None):
        raise NotImplementedError("C: 9.5 PCA-sphering")

    def transform(self, X):
        raise NotImplementedError("C: 9.5 PCA-sphering")


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


class BoostingSelector(Transformer):
    """9.6 Feature selection via boosting: add one feature at a time.

    Each round, fit a one-feature linear model on the current residual for every
    unused feature and keep the one that lowers the cost most.
    fit sets selected_ (indices, in the order they were picked) and
    cost_history_ (cost after each round). transform keeps only the first n_features.

    For the 48-class problem, use a multi-class cost (e.g. linear.categorical_cross_entropy)
    or run it one-versus-all and report per-song choices.

    TODO(C)
    """

    def __init__(self, n_features: int = 10):
        self.n_features = n_features

    def fit(self, X, y=None):
        raise NotImplementedError("C: 9.6 boosting feature selection")

    def transform(self, X):
        raise NotImplementedError("C: 9.6 boosting feature selection")


class L1Selector(Transformer):
    """9.7 Feature selection via regularization: train a linear model with an L1 penalty
    lam * ||w[1:]||_1 and keep the features whose weights are not (near) zero.

    fit sets coef_ and selected_ (indices with |weight| > tol).
    Sweep lam to get a regularization path (number of features vs accuracy).

    TODO(C)
    """

    def __init__(self, lam: float = 1e-2, tol: float = 1e-4, alpha: float = 0.1, max_its: int = 1000):
        self.lam, self.tol, self.alpha, self.max_its = lam, tol, alpha, max_its

    def fit(self, X, y=None):
        raise NotImplementedError("C: 9.7 L1 feature selection")

    def transform(self, X):
        raise NotImplementedError("C: 9.7 L1 feature selection")
