"""Team-wide contract. Change this file only after all four members agree.

Every module passes around only the types and shapes defined here, so each
member can rewrite the inside of their own module without breaking anyone.

Array convention
----------------
X has shape (n_samples, n_features): one row per query.
This is the TRANSPOSE of the textbook's convention (Machine Learning Refined
stores points as columns, x of shape (N, P)). Transpose when porting book code.

Labels
------
y holds song ids 0 .. N_SONGS-1 (file 00012.pv -> label 11).
Binary models inside one-versus-all use y in {-1, +1}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Corpus constants (checked against the real files) ----
N_SONGS = 48                            # midiFile/00001.mid ... 00048.mid
SAMPLE_RATE = 8000                      # Hz; wavs are 8-bit unsigned, mono
FRAME_SIZE = 256                        # samples per pitch frame, no overlap
FRAME_RATE = SAMPLE_RATE / FRAME_SIZE   # 31.25 pitch frames per second
CLIP_SECONDS = 8.0                      # every clip is exactly 8 s long
N_FRAMES = 250                          # pitch frames per clip (8 s * 31.25)


@dataclass
class QueryData:
    """A set of hummed queries stored as parallel arrays (row i = query i).

    pitch     (n, T) float  semitones (MIDI note numbers, ~30-86).
                            Unvoiced/silent frames are np.nan (they are 0 in the .pv files).
                            T = N_FRAMES unless the queries were cropped.
    song_id   (n,)   int    label, 0 .. N_SONGS-1
    singer_id (n,)   str    "year2009/person00005"  (person numbers restart every year,
                            so the year is part of the id)
    year      (n,)   str    "year2009"
    wav_path  (n,)   str    absolute path to the matching .wav
    """
    pitch: np.ndarray
    song_id: np.ndarray
    singer_id: np.ndarray
    year: np.ndarray
    wav_path: np.ndarray

    def __len__(self) -> int:
        return len(self.song_id)

    def subset(self, idx) -> "QueryData":
        """Rows `idx` (an index array or boolean mask) as a new QueryData."""
        return QueryData(self.pitch[idx], self.song_id[idx], self.singer_id[idx],
                         self.year[idx], self.wav_path[idx])


class Transformer:
    """A feature step. Learns from TRAINING data only, then maps arrays to arrays.

    fit(X, y=None) -> self
    transform(X: (n, d_in)) -> (n, d_out)

    Stateless steps (e.g. key normalization) can keep the default fit.
    Optional: set `feature_names_` (list[str], length d_out) in fit so that
    feature-selection results can be reported by name.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Classifier:
    """A multi-class model over the 48 songs.

    fit(X: (n, d), y: (n,) ints in 0..N_SONGS-1) -> self
    predict_scores(X: (n, d)) -> (n, N_SONGS) float; larger = more likely that song.
        Always N_SONGS columns, even if a song never appeared in training
        (such songs must never outrank seen ones; -np.inf is the simplest way).
        No NaN. Ranking metrics (top-k, MRR) need the full row.
    predict(X) -> (n,) argmax of the scores. Inherited; don't override.
    """

    def fit(self, X, y):
        raise NotImplementedError

    def predict_scores(self, X):
        raise NotImplementedError

    def predict(self, X):
        return np.argmax(self.predict_scores(X), axis=1)


class BinaryClassifier:
    """A two-class model, used inside one-versus-all (7.2).

    fit(X: (n, d), y: (n,) in {-1, +1}) -> self
    decision_function(X: (n, d)) -> (n,) float; > 0 means class +1.
    """

    def fit(self, X, y):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, -1)
