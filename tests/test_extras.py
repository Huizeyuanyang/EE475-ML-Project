"""Checks for top-5, most_confused_pairs and the NearestQuery baseline.

Synthetic data only, so these run without the corpus.
"""
import numpy as np

from qbsh.baselines import NearestQuery
from qbsh.contracts import N_SONGS
from qbsh.evaluation import (confusion_matrix, most_confused_pairs, rank_of_true,
                             score_report, top_k_accuracy)


def test_top5_sits_between_top1_and_top10():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(200, N_SONGS))
    y = rng.integers(0, N_SONGS, size=200)
    r = score_report(scores, y)
    assert r["top1"] <= r["top5"] <= r["top10"]
    assert r["top5"] == top_k_accuracy(scores, y, 5)


def test_top5_counts_rank_five_but_not_six():
    y = np.array([0, 0])
    scores = -np.ones((2, N_SONGS))
    scores[:, 0] = 0.0            # the correct song; every other song is below it ...
    scores[0, 1:5] = 1.0          # ... except four songs above it -> rank 5
    scores[1, 1:6] = 1.0          # ... and five songs above it -> rank 6
    assert list(rank_of_true(scores, y)) == [5, 6]
    assert top_k_accuracy(scores, y, 5) == 0.5


def test_most_confused_pairs():
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([1, 1, 0, 1, 0, 2, 2, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert most_confused_pairs(cm, k=2) == [(0, 1, 2), (1, 0, 1)]
    assert all(t != p and n > 0 for t, p, n in most_confused_pairs(cm, k=100))


def test_nearest_query_finds_the_closest_training_query():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    y = np.array([3, 3, 7, 7])
    m = NearestQuery().fit(X, y)
    s = m.predict_scores(np.array([[0.2, 0.1], [4.9, 5.2]]))
    assert s.shape == (2, N_SONGS)
    assert list(m.predict(np.array([[0.2, 0.1], [4.9, 5.2]]))) == [3, 7]
    assert np.isneginf(s[:, 0]).all()           # a song never seen in training
    assert not np.isnan(s).any()
