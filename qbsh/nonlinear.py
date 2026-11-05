"""Nonlinear models and learned features.

Owner: D.
Course methods: kernel methods Ch. 12, fully connected neural networks Ch. 13
(batch normalization 13.6, early stopping 13.7), trees / random forests /
gradient boosting Ch. 14, bagging 11.9.
Owned experiment: hand-designed vs learned features
(experiments/exp_handcrafted_vs_learned.py).

The neural network may be written in NumPy by hand or with PyTorch (installed);
pick one as a team. Every model here follows contracts.Classifier, so it drops
into the same Pipeline and evaluation code as B's linear models.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .contracts import N_SONGS, Classifier


# ---------------------------------------------------------------- Ch. 12 kernels

def polynomial_kernel(A: np.ndarray, B: np.ndarray, degree: int = 2, c: float = 1.0) -> np.ndarray:
    """(A @ B.T + c) ** degree -> (len(A), len(B)).  TODO(D)"""
    raise NotImplementedError("D: Ch. 12 polynomial kernel")


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """exp(-gamma * ||a - b||^2) -> (len(A), len(B)).  TODO(D)"""
    raise NotImplementedError("D: Ch. 12 RBF kernel")


class KernelClassifier(Classifier):
    """Ch. 12 Kernelized multi-class softmax (or one-versus-all kernel SVM).

    Weights live on the training points: scores = K(X, X_train) @ V, V (n_train, N_SONGS).
    lam is the regularizer on V^T K V.
    With ~2700 training queries the kernel matrix is 2700 x 2700, which fits in memory.

    TODO(D)
    """

    def __init__(self, kernel: str = "rbf", gamma: float = 0.1, degree: int = 2,
                 lam: float = 1e-3, alpha: float = 0.1, max_its: int = 500):
        self.kernel, self.gamma, self.degree = kernel, gamma, degree
        self.lam, self.alpha, self.max_its = lam, alpha, max_its

    def fit(self, X, y):
        raise NotImplementedError("D: Ch. 12 kernel classifier")

    def predict_scores(self, X):
        raise NotImplementedError("D: Ch. 12 kernel classifier")


# ---------------------------------------------------------------- Ch. 13 neural network

class MLPClassifier(Classifier):
    """Ch. 13 Fully connected network with a softmax output over the 48 songs.

    hidden:         layer widths, e.g. (128, 64)
    batch_norm:     13.6 normalize each hidden layer's pre-activations
    early_stopping: 13.7 hold out `val_fraction` of the training rows and keep the
                    weights from the epoch with the best validation cost
                    (`patience` epochs without improvement -> stop)
    Trained with mini-batches (7.8) and categorical cross entropy (7.5).

    embed(X) returns the last hidden layer's activations: these are the
    LEARNED FEATURES compared against A's hand-designed ones.

    TODO(D)
    """

    def __init__(self, hidden=(128,), activation: str = "relu", batch_norm: bool = True,
                 early_stopping: bool = True, val_fraction: float = 0.1, patience: int = 10,
                 lr: float = 1e-3, epochs: int = 200, batch_size: int = 64,
                 weight_decay: float = 0.0, seed: int = 0):
        self.hidden, self.activation, self.batch_norm = hidden, activation, batch_norm
        self.early_stopping, self.val_fraction, self.patience = early_stopping, val_fraction, patience
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.weight_decay, self.seed = weight_decay, seed

    def fit(self, X, y):
        raise NotImplementedError("D: Ch. 13 MLP")

    def predict_scores(self, X):
        raise NotImplementedError("D: Ch. 13 MLP")

    def embed(self, X) -> np.ndarray:
        """(n, d) -> (n, hidden[-1]) learned features."""
        raise NotImplementedError("D: Ch. 13 MLP embedding")


# ---------------------------------------------------------------- Ch. 14 trees

class DecisionTree(Classifier):
    """Ch. 14 Classification tree (split on the feature/threshold that most reduces impurity).

    max_features: number of features tried per split (None = all); RandomForest sets it.
    predict_scores: class frequencies in the leaf, as an (n, N_SONGS) array.

    TODO(D)
    """

    def __init__(self, max_depth: int | None = 8, min_samples_leaf: int = 2,
                 max_features: int | None = None, seed: int = 0):
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf
        self.max_features, self.seed = max_features, seed

    def fit(self, X, y):
        raise NotImplementedError("D: Ch. 14 decision tree")

    def predict_scores(self, X):
        raise NotImplementedError("D: Ch. 14 decision tree")


class RandomForest(Classifier):
    """Ch. 14 Random forest: bagged DecisionTrees with random feature subsets; average their scores.

    TODO(D)
    """

    def __init__(self, n_trees: int = 100, max_depth: int | None = None,
                 max_features: str | int = "sqrt", seed: int = 0):
        self.n_trees, self.max_depth, self.max_features, self.seed = n_trees, max_depth, max_features, seed

    def fit(self, X, y):
        raise NotImplementedError("D: Ch. 14 random forest")

    def predict_scores(self, X):
        raise NotImplementedError("D: Ch. 14 random forest")


class GradientBoosting(Classifier):
    """Ch. 14 Gradient boosting with shallow regression trees on the softmax cost
    (one tree per song per round, fit to the negative gradient).

    TODO(D)
    """

    def __init__(self, n_rounds: int = 100, learning_rate: float = 0.1, max_depth: int = 3, seed: int = 0):
        self.n_rounds, self.learning_rate, self.max_depth, self.seed = n_rounds, learning_rate, max_depth, seed

    def fit(self, X, y):
        raise NotImplementedError("D: Ch. 14 gradient boosting")

    def predict_scores(self, X):
        raise NotImplementedError("D: Ch. 14 gradient boosting")


# ---------------------------------------------------------------- 11.9 bagging

class Bagging(Classifier):
    """11.9 Bagging: train `n_models` copies of any Classifier on bootstrap samples
    and average their predict_scores (convert to probabilities first if scales differ).

    make_model: zero-argument factory, e.g. Bagging(lambda: MLPClassifier(hidden=(64,))).

    TODO(D)
    """

    def __init__(self, make_model: Callable[[], Classifier], n_models: int = 10, seed: int = 0):
        self.make_model, self.n_models, self.seed = make_model, n_models, seed

    def fit(self, X, y):
        raise NotImplementedError("D: 11.9 bagging")

    def predict_scores(self, X):
        raise NotImplementedError("D: 11.9 bagging")
