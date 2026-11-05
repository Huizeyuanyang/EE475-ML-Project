"""Checks on the real corpus and the shared plumbing (all implemented, so these must pass).

Skipped automatically if the corpus isn't found (set QBSH_DATA).
"""
from functools import lru_cache

import numpy as np
import pytest

from qbsh import config
from qbsh.baselines import NearestCentroid, RawPitch
from qbsh.contracts import N_FRAMES, N_SONGS
from qbsh.data import crop, load_queries, load_song_titles, load_wav, year_split
from qbsh.evaluation import evaluate, mean_reciprocal_rank, top_k_accuracy
from qbsh.pipeline import Pipeline

pytestmark = pytest.mark.skipif(not (config.DATA_ROOT / "waveFile").is_dir(),
                                reason="MIR-QBSH corpus not found; set QBSH_DATA")


@lru_cache(maxsize=1)
def data():
    return load_queries()


def test_corpus_size_and_labels():
    d = data()
    assert len(d) == 4431
    assert d.pitch.shape == (4431, N_FRAMES)
    assert set(d.song_id) == set(range(N_SONGS))


def test_unvoiced_frames_are_nan():
    p = data().pitch
    assert not (p == 0).any()
    assert 0.2 < np.isnan(p).mean() < 0.35
    assert 25 < np.nanmin(p) and np.nanmax(p) < 90


def test_year_split_sizes_and_no_singer_leak():
    train, val, test = year_split(data())
    assert (len(train), len(val), len(test)) == (2749, 454, 1228)
    assert not set(train.singer_id) & set(val.singer_id)
    assert not set(train.singer_id) & set(test.singer_id)
    assert set(train.song_id) == set(range(N_SONGS))


def test_song_titles():
    titles = load_song_titles()
    assert len(titles) == N_SONGS and all(titles)
    assert titles[0] == "I'm the teapot"


def test_wav_matches_pitch_frames():
    x = load_wav(data().wav_path[0])
    assert x.shape == (64000,) and x.min() >= -1 and x.max() < 1
    assert len(x) // 256 == N_FRAMES


def test_crop():
    p = data().pitch[:3]
    assert crop(p, 8).shape == (3, 250)
    assert crop(p, 4).shape == (3, 125)
    assert np.array_equal(crop(p, 4, start_seconds=4), p[:, 125:], equal_nan=True)


def test_baseline_beats_chance():
    train, val, _ = year_split(data())
    m = evaluate(Pipeline([RawPitch()], NearestCentroid()), train.pitch, train.song_id, val.pitch, val.song_id)
    assert m["top1"] > 2 / N_SONGS


def test_metrics():
    y = np.arange(N_SONGS)
    assert top_k_accuracy(np.eye(N_SONGS), y, 1) == 1.0
    assert mean_reciprocal_rank(np.eye(N_SONGS), y) == 1.0
    reversed_scores = -np.eye(N_SONGS)          # correct song ranked last
    assert top_k_accuracy(reversed_scores, y, 10) == 0.0
    assert np.isclose(mean_reciprocal_rank(reversed_scores, y), 1 / N_SONGS)
