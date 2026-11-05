"""Audio -> pitch contour, in the same format as the corpus .pv files.

Owner: B.
Not a course method; it exists so the video can show a live demo
(demo/hum_demo.py): a new recording has a .wav but no hand-labelled .pv.

Output format must match data.load_queries().pitch row by row:
    semitones (MIDI note numbers), 31.25 frames/s (256-sample hop at 8 kHz),
    np.nan for unvoiced frames, N_FRAMES (250) frames for 8 s.
If it matches, every model and feature step works on it unchanged.

Needs librosa (optional dependency; see requirements.txt).
"""
from __future__ import annotations

import numpy as np

from .contracts import FRAME_SIZE, N_FRAMES, SAMPLE_RATE


def load_audio(path) -> np.ndarray:
    """Any .wav (phone recording, 44.1 kHz, stereo, 16-bit, ...) -> mono float at 8 kHz.

    data.load_wav only reads the corpus's 8-bit 8 kHz files, so use this for your own
    recordings, e.g. librosa.load(path, sr=SAMPLE_RATE, mono=True)[0].

    TODO(B)
    """
    raise NotImplementedError("B: load_audio")


def wav_to_pitch(path, n_frames: int = N_FRAMES) -> np.ndarray:
    """One recording -> pitch contour of shape (n_frames,).

    Suggested: librosa.pyin(y, sr=SAMPLE_RATE, hop_length=FRAME_SIZE, fmin=..., fmax=...),
    then librosa.hz_to_midi(f0); unvoiced frames -> np.nan.
    Crop to the first n_frames, or pad the end with np.nan if the recording is shorter.
    (With librosa's default center=True, an 8 s clip gives 251 frames, so cropping is needed.)
    Tune fmin / fmax / frame_length by comparing against the corpus .pv labels.

    TODO(B)
    """
    raise NotImplementedError("B: pYIN pitch extraction")


def wavs_to_pitch(paths, n_frames: int = N_FRAMES) -> np.ndarray:
    """Several recordings -> (n, n_frames). Implemented on top of wav_to_pitch."""
    return np.stack([wav_to_pitch(p, n_frames) for p in paths])


def pitch_agreement(auto: np.ndarray, labelled: np.ndarray, tol: float = 0.5) -> dict[str, float]:
    """Compare extracted pitch with the corpus labels (optional analysis for B).

    voicing_agreement: fraction of frames where both say voiced, or both say unvoiced.
    pitch_accuracy:    among frames both call voiced, fraction within `tol` semitones
                       (after removing each row's median offset, since labels may be
                       an octave off).

    TODO(B)
    """
    raise NotImplementedError("B: pitch agreement")


__all__ = ["load_audio", "wav_to_pitch", "wavs_to_pitch", "pitch_agreement",
           "SAMPLE_RATE", "FRAME_SIZE"]
