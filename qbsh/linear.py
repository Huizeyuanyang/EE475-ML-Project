"""The linear classifier: multi-class softmax regression.

Owner: B.
Course methods: multi-class softmax / categorical cross entropy 7.5,
L2 regularization, gradient descent 3.5 and mini-batch learning 7.8 (via optimizers.py).
Owned experiment: query length (experiments/exp_query_length.py).

Scope: the team uses one linear model (this one) and one nonlinear model
(nonlinear.MLPClassifier).

Inputs are feature matrices X (n, d) with no NaN, already standard-normalized.
Weights include a bias: use add_bias(X) so W[0] is the intercept row.
The cost returns (cost, grad) so it plugs straight into optimizers.py.
"""
from __future__ import annotations

import numpy as np

from .contracts import N_SONGS, Classifier


def add_bias(X: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, d + 1) with a leading column of ones."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def categorical_cross_entropy(W, Xb, y, lam: float = 0.0):
    """7.5 Categorical cross entropy (multi-class softmax): mean[-log softmax(x W)_y] + lam * ||W[1:]||^2.

    W (d+1, K), Xb (n, d+1), y (n,) ints -> (cost, grad (d+1, K)).
    Subtract the row max before exponentiating.

    TODO(B)
    """
    raise NotImplementedError("B: 7.5 categorical cross entropy")


class SoftmaxRegression(Classifier):
    """7.5 Multi-class softmax regression with categorical cross entropy.

    batch_size=None -> full-batch gradient descent; an int -> mini-batch (7.8).
    W has shape (d+1, N_SONGS). This is the default classifier in A's and C's
    experiments, so finish it first. Songs absent from training get -inf.

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
