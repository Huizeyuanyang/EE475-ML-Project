"""The nonlinear classifier: a fully connected neural network.

Owner: D.
Course methods: fully connected neural networks Ch. 13
(batch normalization 13.6, early stopping 13.7), trained with mini-batches (7.8)
and categorical cross entropy (7.5).
Owned experiment: hand-designed vs learned features
(experiments/exp_handcrafted_vs_learned.py).

Scope: the team uses one linear model (linear.SoftmaxRegression) and one
nonlinear model (this one).

The network may be written in NumPy by hand or with PyTorch; pick one as a team.
It follows contracts.Classifier, so it drops into the same Pipeline and
evaluation code as B's linear model.
"""
from __future__ import annotations

import numpy as np

from .contracts import N_SONGS, Classifier


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
