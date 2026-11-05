"""Linear classifiers.

Owner: B.
Course methods: logistic regression 6.2, softmax cost 6.3, perceptron 6.4,
SVM 6.5, one-versus-all 7.2, multi-class perceptron 7.3,
categorical cross entropy 7.5, mini-batch learning 7.8 (via optimizers.py).
Owned experiment: query length (experiments/exp_query_length.py).

Inputs are feature matrices X (n, d) with no NaN, already standard-normalized.
Weights include a bias: use add_bias(X) so w[0] is the intercept.
Every cost function returns (cost, grad) so it plugs straight into optimizers.py.
C's feature selection (9.6, 9.7) may reuse these costs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .contracts import N_SONGS, BinaryClassifier, Classifier


def add_bias(X: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, d + 1) with a leading column of ones."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------- two-class costs, y in {-1, +1}

def softmax_cost(w, Xb, y, lam: float = 0.0):
    """6.2/6.3 Two-class softmax (logistic) cost: mean log(1 + exp(-y * Xb @ w)) + lam * ||w[1:]||^2.

    Xb (n, d+1) with bias column, y (n,) in {-1, +1} -> (cost, grad).
    Use a numerically stable form (np.logaddexp) so large margins don't overflow.

    TODO(B)
    """
    raise NotImplementedError("B: 6.2/6.3 softmax cost")


def perceptron_cost(w, Xb, y):
    """6.4 Perceptron (ReLU) cost: mean max(0, -y * Xb @ w) -> (cost, subgradient).

    TODO(B)
    """
    raise NotImplementedError("B: 6.4 perceptron cost")


def hinge_cost(w, Xb, y, lam: float = 1e-3):
    """6.5 Soft-margin SVM: mean max(0, 1 - y * Xb @ w) + lam * ||w[1:]||^2 -> (cost, subgradient).

    TODO(B)
    """
    raise NotImplementedError("B: 6.5 SVM hinge cost")


# ---------------------------------------------------------------- multi-class costs, y in 0..K-1

def multiclass_perceptron_cost(W, Xb, y, lam: float = 0.0):
    """7.3 Multi-class perceptron: mean[max_c (x W)_c - (x W)_y] + lam * ||W[1:]||^2.

    W (d+1, K), Xb (n, d+1), y (n,) ints -> (cost, grad (d+1, K)).

    TODO(B)
    """
    raise NotImplementedError("B: 7.3 multi-class perceptron cost")


def categorical_cross_entropy(W, Xb, y, lam: float = 0.0):
    """7.5 Categorical cross entropy (multi-class softmax): mean[-log softmax(x W)_y] + lam * ||W[1:]||^2.

    W (d+1, K), Xb (n, d+1), y (n,) ints -> (cost, grad (d+1, K)).
    Subtract the row max before exponentiating.

    TODO(B)
    """
    raise NotImplementedError("B: 7.5 categorical cross entropy")


# ---------------------------------------------------------------- two-class models

class LogisticRegression(BinaryClassifier):
    """6.2 Logistic regression trained with gradient descent ("gd") or Newton's method ("newton").

    TODO(B)
    """

    def __init__(self, optimizer: str = "gd", alpha: float = 0.1, max_its: int = 500, lam: float = 1e-3):
        self.optimizer, self.alpha, self.max_its, self.lam = optimizer, alpha, max_its, lam

    def fit(self, X, y):
        raise NotImplementedError("B: 6.2 logistic regression")

    def decision_function(self, X):
        raise NotImplementedError("B: 6.2 logistic regression")


class Perceptron(BinaryClassifier):
    """6.4 Perceptron trained on perceptron_cost with gradient descent.

    TODO(B)
    """

    def __init__(self, alpha: float = 0.1, max_its: int = 500):
        self.alpha, self.max_its = alpha, max_its

    def fit(self, X, y):
        raise NotImplementedError("B: 6.4 perceptron")

    def decision_function(self, X):
        raise NotImplementedError("B: 6.4 perceptron")


class LinearSVM(BinaryClassifier):
    """6.5 Soft-margin linear SVM trained on hinge_cost.

    TODO(B)
    """

    def __init__(self, lam: float = 1e-3, alpha: float = 0.1, max_its: int = 500):
        self.lam, self.alpha, self.max_its = lam, alpha, max_its

    def fit(self, X, y):
        raise NotImplementedError("B: 6.5 SVM")

    def decision_function(self, X):
        raise NotImplementedError("B: 6.5 SVM")


# ---------------------------------------------------------------- multi-class models

class OneVsAll(Classifier):
    """7.2 One-versus-all: one binary model per song (song s -> +1, rest -> -1).

    make_binary: zero-argument factory returning a fresh BinaryClassifier,
                 e.g. OneVsAll(lambda: LinearSVM(lam=1e-2)).
    predict_scores column s = decision_function of model s.
    Songs absent from the training labels get -inf.
    Optional: normalize each model's weights so scores are comparable across songs.

    TODO(B)
    """

    def __init__(self, make_binary: Callable[[], BinaryClassifier]):
        self.make_binary = make_binary

    def fit(self, X, y):
        raise NotImplementedError("B: 7.2 one-versus-all")

    def predict_scores(self, X):
        raise NotImplementedError("B: 7.2 one-versus-all")


class MulticlassPerceptron(Classifier):
    """7.3 Multi-class perceptron; W has shape (d+1, N_SONGS).

    TODO(B)
    """

    def __init__(self, alpha: float = 0.1, max_its: int = 500, lam: float = 1e-3):
        self.alpha, self.max_its, self.lam = alpha, max_its, lam

    def fit(self, X, y):
        raise NotImplementedError("B: 7.3 multi-class perceptron")

    def predict_scores(self, X):
        raise NotImplementedError("B: 7.3 multi-class perceptron")


class SoftmaxRegression(Classifier):
    """7.5 Multi-class softmax regression with categorical cross entropy.

    batch_size=None -> full-batch gradient descent; an int -> mini-batch (7.8).
    W has shape (d+1, N_SONGS). This is the default linear model the other
    members' experiments use, so finish it first.

    TODO(B)
    """

    def __init__(self, alpha: float = 0.1, max_its: int = 500, lam: float = 1e-3,
                 batch_size: int | None = None, seed: int = 0):
        self.alpha, self.max_its, self.lam = alpha, max_its, lam
        self.batch_size, self.seed = batch_size, seed

    def fit(self, X, y):
        raise NotImplementedError("B: 7.5 softmax regression")

    def predict_scores(self, X):
        raise NotImplementedError("B: 7.5 softmax regression")
