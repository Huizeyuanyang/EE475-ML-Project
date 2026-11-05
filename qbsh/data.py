"""Loading and splitting the MIR-QBSH corpus.

Owner: A. Loading is already implemented and checked against the real files,
so B, C and D can start right away; A owns any later change here.

The models use the hand-labelled pitch vectors (.pv). Raw audio is available
through `load_wav` for anyone who wants audio features.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from . import config
from .contracts import FRAME_RATE, N_FRAMES, N_SONGS, QueryData


def _root(root) -> Path:
    root = Path(root) if root is not None else config.DATA_ROOT
    if not (root / "waveFile").is_dir():
        raise FileNotFoundError(
            f"MIR-QBSH corpus not found at {root}. "
            "Set the QBSH_DATA environment variable to the folder that contains waveFile/."
        )
    return root


def load_queries(root=None, years=None) -> QueryData:
    """Load every query's pitch vector and label.

    root:  corpus folder (default config.DATA_ROOT)
    years: optional iterable like ("year2003", "year2009") to load only some years

    Returns a QueryData with pitch of shape (n, 250); unvoiced frames are np.nan.
    Takes under a second for all 4431 clips.
    """
    root = _root(root)
    pv_paths = sorted((root / "waveFile").glob("year*/person*/*.pv"))
    if years is not None:
        years = set(years)
        pv_paths = [p for p in pv_paths if p.parent.parent.name in years]

    pitch = np.empty((len(pv_paths), N_FRAMES))
    for i, p in enumerate(pv_paths):
        values = np.array(p.read_text().split(), dtype=float)
        if values.shape != (N_FRAMES,):
            raise ValueError(f"{p}: expected {N_FRAMES} frames, got {values.shape}")
        pitch[i] = values
    pitch[pitch == 0] = np.nan  # 0 in .pv = no pitch (silence/unvoiced)

    song_id = np.array([int(p.stem) - 1 for p in pv_paths])
    year = np.array([p.parent.parent.name for p in pv_paths])
    singer_id = np.array([f"{p.parent.parent.name}/{p.parent.name}" for p in pv_paths])
    wav_path = np.array([str(p.with_suffix(".wav")) for p in pv_paths])
    assert song_id.min() >= 0 and song_id.max() < N_SONGS
    return QueryData(pitch, song_id, singer_id, year, wav_path)


def load_song_titles(root=None) -> list[str]:
    """Title of each song, indexed by song_id (0..47).

    Uses the English title, or the Chinese one when there is none.
    (midiFile/songList.txt is Big5-encoded with Windows line endings.)
    """
    root = _root(root)
    text = (root / "midiFile" / "songList.txt").read_bytes().decode("big5", errors="replace")
    titles = [""] * N_SONGS
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) >= 3 and cols[0].isdigit():
            english, chinese = cols[1].strip(), cols[2].strip()
            titles[int(cols[0]) - 1] = english if english != "-" else chinese
    return titles


def year_split(data: QueryData,
               train_years=config.TRAIN_YEARS,
               val_years=config.VAL_YEARS,
               test_years=config.TEST_YEARS) -> tuple[QueryData, QueryData, QueryData]:
    """Split by recording year -> (train, val, test). See config.py for the agreed years.

    Tune hyperparameters on val (or with K-fold CV inside train); touch test only
    for the final numbers.
    """
    def pick(years):
        return data.subset(np.isin(data.year, list(years)))
    return pick(train_years), pick(val_years), pick(test_years)


def crop(pitch: np.ndarray, seconds: float, start_seconds: float = 0.0) -> np.ndarray:
    """Keep `seconds` of each query, starting at `start_seconds`, for the query-length experiment.

    pitch (n, T) -> (n, round(seconds * 31.25)).
    """
    a = int(round(start_seconds * FRAME_RATE))
    b = a + int(round(seconds * FRAME_RATE))
    if b > pitch.shape[1]:
        raise ValueError(f"crop to {start_seconds}+{seconds}s exceeds the {pitch.shape[1]}-frame queries")
    return pitch[:, a:b]


def load_wav(path) -> np.ndarray:
    """Read one clip: 8-bit unsigned PCM -> float32 in [-1, 1), shape (64000,) at 8 kHz."""
    with wave.open(str(path)) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8)
    return (raw.astype(np.float32) - 128.0) / 128.0


def midi_path(song_id: int, root=None) -> Path:
    """Path of the ground-truth MIDI file for a song (not used by the classification pipeline)."""
    return _root(root) / "midiFile" / f"{song_id + 1:05d}.mid"
