"""Evaluation: metrics, K-fold cross-validation, saving results.

Owner: C.
  * Metrics, `evaluate` and `save_results` are implemented (shared, needed by everyone
    from day one). Keep their names and outputs.
  * K-fold cross-validation (11.10) is C's TODO. Until it exists, others tune on the
    fixed validation year (config.VAL_YEARS).

Metrics follow the MIREX QBSH task (top-1 / top-10 hit rate, mean reciprocal rank); top-5 is
reported as well.
"""
from __future__ import annotations

import csv
from typing import Callable, Iterator

import numpy as np

from . import config
from .contracts import N_SONGS, Classifier


# ---------------------------------------------------------------- metrics (implemented)

def rank_of_true(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1-based rank of the correct song in each row (ties count against the model)."""
    true = scores[np.arange(len(y)), y]
    return (scores >= true[:, None]).sum(axis=1)


def top_k_accuracy(scores: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Fraction of queries whose correct song is among the k highest scores."""
    return float(np.mean(rank_of_true(scores, y) <= k))


def mean_reciprocal_rank(scores: np.ndarray, y: np.ndarray) -> float:
    """Mean of 1/rank of the correct song (1.0 = always ranked first)."""
    return float(np.mean(1.0 / rank_of_true(scores, y)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_SONGS) -> np.ndarray:
    """(n_classes, n_classes) counts; row = true song, column = predicted song."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def most_confused_pairs(cm: np.ndarray, k: int = 10) -> list[tuple[int, int, int]]:
    """The k largest off-diagonal entries of a confusion matrix.

    -> [(true_song, predicted_song, count), ...], largest count first. Directed: (3, 7) and
    (7, 3) are separate entries. Pairs with a zero count are not returned.
    """
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(-off, axis=None, kind="stable")[:k]
    rows, cols = np.unravel_index(order, off.shape)
    return [(int(r), int(c), int(off[r, c])) for r, c in zip(rows, cols) if off[r, c] > 0]


def score_report(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN; did a feature step leave unvoiced frames unimputed?")
    return {"top1": top_k_accuracy(scores, y, 1),
            "top5": top_k_accuracy(scores, y, 5),
            "top10": top_k_accuracy(scores, y, 10),
            "mrr": mean_reciprocal_rank(scores, y)}


def evaluate(model: Classifier, X_train, y_train, X_test, y_test) -> dict[str, float]:
    """Fit on train, score on test -> {"top1", "top5", "top10", "mrr"}."""
    scores = model.fit(X_train, y_train).predict_scores(X_test)
    if scores.shape != (len(y_test), N_SONGS):
        raise ValueError(f"predict_scores must return (n, {N_SONGS}), got {scores.shape}")
    return score_report(scores, y_test)


def save_results(rows: list[dict], name: str) -> str:
    """Write a list of result dicts to results/<name>.csv and return the path."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{name}.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


# ---------------------------------------------------------------- K-fold CV (C: TODO)

def k_fold_indices(n: int, k: int = 5, groups: np.ndarray | None = None,
                   seed: int = config.SEED) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """11.10 K-fold cross-validation splits -> yields (train_idx, val_idx) k times.

    If `groups` is given (e.g. data.singer_id or data.year), all rows of one group
    must land in the same fold, so no singer is in both train and val.
    Every row is in exactly one val fold.

    TODO(C)
    """
    raise NotImplementedError("C: 11.10 K-fold cross-validation")


def cross_validate(make_model: Callable[[], Classifier], X: np.ndarray, y: np.ndarray,
                   k: int = 5, groups: np.ndarray | None = None) -> dict[str, float]:
    """Average score_report over K folds. `make_model()` must return a fresh, unfitted model.

    Returns {"top1", "top5", "top10", "mrr"} averaged over folds, plus "top1_std".
    Use it to pick hyperparameters (number of PCA components, lambda, tree depth, ...)
    using the training years only.

    TODO(C)
    """
    raise NotImplementedError("C: 11.10 K-fold cross-validation")
