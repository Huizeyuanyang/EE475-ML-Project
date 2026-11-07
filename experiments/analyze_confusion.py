"""Confusion analysis: which songs get mistaken for which.          Owner: C

Question
    Which pairs of songs does each model confuse most often, and why?

Design
    Fit each model on the training years, take the top-1 prediction for every scored
    query, build the 48x48 confusion matrix (evaluation.confusion_matrix) and list its
    largest off-diagonal entries (evaluation.most_confused_pairs) with the song titles.
    Compare the lists across models. Use the fitted models' results from the experiment
    with the best setting; the settings below are the defaults.

Output
    results/confusion_pairs_{val,test}.csv
        (model, true_song, true_title, predicted_song, predicted_title, count)
    The "why" (similar openings, similar melodic contour, too few training clips, ...) is
    written up by hand in the report from these rows and the songs' MIDI files.

Run
    python -m experiments.analyze_confusion [--final]
"""
import numpy as np

from qbsh.baselines import NearestQuery
from qbsh.data import load_song_titles
from qbsh.evaluation import confusion_matrix, most_confused_pairs, save_results
from qbsh.features import HandDesignedFeatures, KeyNormalizer, StandardScaler, impute_unvoiced
from qbsh.linear import SoftmaxRegression
from qbsh.nonlinear import MLPClassifier
from qbsh.pipeline import FunctionStep, Pipeline

from .common import load_split, parse_args, results_name, run_or_skip

TOP_PAIRS = 10


def contour_steps():
    return [KeyNormalizer("median"), FunctionStep(impute_unvoiced)]


def hand_steps():
    return [KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()]


MODELS = {
    "nearest_query": lambda: Pipeline(contour_steps(), NearestQuery()),
    "softmax": lambda: Pipeline(hand_steps(), SoftmaxRegression()),
    "mlp": lambda: Pipeline(hand_steps(), MLPClassifier()),
}


def confused_rows(name, model, train, target, titles) -> list[dict]:
    pred = np.argmax(model.fit(train.pitch, train.song_id).predict_scores(target.pitch), axis=1)
    cm = confusion_matrix(target.song_id, pred)
    return [{"model": name, "true_song": t, "true_title": titles[t],
             "predicted_song": p, "predicted_title": titles[p], "count": n}
            for t, p, n in most_confused_pairs(cm, TOP_PAIRS)]


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    titles = load_song_titles()
    rows = []
    for name, make_model in MODELS.items():
        rows += run_or_skip(name, lambda: confused_rows(name, make_model(), train, target, titles))
    if rows:
        print(save_results(rows, results_name("confusion_pairs", args.final)))


if __name__ == "__main__":
    main()
