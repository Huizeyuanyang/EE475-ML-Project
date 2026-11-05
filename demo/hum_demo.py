"""Live demo for the video: hum into a .wav file, get the top-10 songs.          Owner: B

    python -m demo.hum_demo my_hum.wav            # uses the MLP
    python -m demo.hum_demo my_hum.wav --linear   # uses softmax regression

Steps
    1. Train the chosen model on the training + validation years (more data than the
       experiments use; the demo isn't a reported number, so that's fine).
    2. audio.wav_to_pitch(my_hum.wav) -> (250,) contour in the .pv format.
    3. Score it and print the 10 most likely song titles.

Hum the beginning of the song for about 8 seconds: every corpus query starts at the
beginning of its song, so the models never saw middle sections.
Note: the models were trained on hand-labelled pitch, and the demo feeds them
automatically extracted pitch, so expect it to be less accurate than the experiments.
"""
import argparse

import numpy as np

from qbsh.audio import wav_to_pitch
from qbsh.data import load_queries, load_song_titles, year_split
from qbsh.features import HandDesignedFeatures, KeyNormalizer, StandardScaler
from qbsh.linear import SoftmaxRegression
from qbsh.nonlinear import MLPClassifier
from qbsh.pipeline import Pipeline


def build_model(linear: bool) -> Pipeline:
    steps = [KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()]
    return Pipeline(steps, SoftmaxRegression() if linear else MLPClassifier())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", help="your recording (any sample rate; mono or stereo)")
    p.add_argument("--linear", action="store_true", help="use softmax regression instead of the MLP")
    args = p.parse_args()

    train, val, _ = year_split(load_queries())
    X = np.vstack([train.pitch, val.pitch])
    y = np.concatenate([train.song_id, val.song_id])
    try:
        model = build_model(args.linear).fit(X, y)
        query = wav_to_pitch(args.wav)[None, :]          # (1, 250)
        scores = model.predict_scores(query)[0]
    except NotImplementedError as e:
        raise SystemExit(f"Demo needs code that is still a stub: {e}")

    titles = load_song_titles()
    print(f"Voiced frames: {np.sum(~np.isnan(query))} / {query.shape[1]}")
    for rank, s in enumerate(np.argsort(-scores)[:10], start=1):
        print(f"{rank:2d}. {titles[s]}")


if __name__ == "__main__":
    main()
