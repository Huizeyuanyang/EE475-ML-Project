"""Optimizers for the linear model (and reusable by D).

Owner: B.
Course methods: gradient descent 3.5, mini-batch learning 7.8.

Convention: a cost is given as a function `cost_and_grad(w) -> (cost: float, grad: ndarray)`
with grad the same shape as w. Each optimizer returns (w_final, cost_history) so B can
plot cost histories the way the book does.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

CostAndGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]


def gradient_descent(cost_and_grad: CostAndGrad, w0: np.ndarray, alpha: float = 1e-2,
                     max_its: int = 1000, tol: float = 0.0) -> tuple[np.ndarray, list[float]]:
    """3.5 Gradient descent with a fixed step length `alpha`.

    Stop after `max_its` steps, or earlier once the gradient norm falls below `tol`.
    cost_history[0] is the cost at w0.

    TODO(B)
    """
    raise NotImplementedError("B: 3.5 gradient descent")


def minibatch_gradient_descent(batch_cost_and_grad: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]],
                               w0: np.ndarray, n_samples: int, batch_size: int = 64,
                               alpha: float = 1e-2, epochs: int = 50,
                               seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """7.8 Mini-batch gradient descent.

    batch_cost_and_grad(w, idx) -> (cost, grad) on rows `idx` only.
    Each epoch: shuffle 0..n_samples-1, step once per batch.
    cost_history has one entry per epoch (cost over all rows at the end of the epoch).

    TODO(B)
    """
    raise NotImplementedError("B: 7.8 mini-batch gradient descent")
