"""Experiment A: key normalization.          Owner: A

Question
    People hum the same song in different keys. Does removing the key help
    identify the song, and which way of removing it works best?

Design
    Two representations, each with every KeyNormalizer method ("none" = reference):
        contour: KeyNormalizer(m) -> impute_unvoiced -> StandardScaler      (250 numbers)
        hist:    KeyNormalizer(m) -> HandDesignedFeatures -> StandardScaler
    One fixed classifier for every row, so only the normalization changes.
    Robustness check: transpose the scored queries by +2, +4, +6 semitones.
    A key-invariant pipeline gives the same numbers at every shift.

Output
    results/exp_key_normalization_{val,test}.csv  (representation, method, shift, top1, top10, mrr)
    TODO(A): plot top-1 by method, and top-1 vs transposition shift.

Run
    python -m experiments.exp_key_normalization [--final]
"""
from qbsh.baselines import NearestCentroid
from qbsh.evaluation import save_results, score_report
from qbsh.features import HandDesignedFeatures, KeyNormalizer, StandardScaler, impute_unvoiced
from qbsh.pipeline import FunctionStep, Pipeline

from .common import load_split, parse_args, results_name, run_or_skip

SHIFTS = (0, 2, 4, 6)   # semitones added to the scored queries


def make_classifier():
    # Same classifier for every row. TODO(A): switch to linear.SoftmaxRegression() once B's is ready.
    return NearestCentroid()


def make_pipeline(representation: str, method: str) -> Pipeline:
    key = KeyNormalizer(method)
    if representation == "contour":
        steps = [key, FunctionStep(impute_unvoiced), StandardScaler()]
    elif representation == "hist":
        steps = [key, HandDesignedFeatures(), StandardScaler()]
    else:
        raise ValueError(representation)
    return Pipeline(steps, make_classifier())


def run_setting(train, target, representation: str, method: str) -> list[dict]:
    """Fit once on train, then score the target queries at every transposition shift."""
    model = make_pipeline(representation, method).fit(train.pitch, train.song_id)
    return [{"representation": representation, "method": method, "shift": shift,
             **score_report(model.predict_scores(target.pitch + shift), target.song_id)}
            for shift in SHIFTS]


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    rows = []
    for representation in ("contour", "hist"):
        for method in KeyNormalizer.METHODS:
            rows += run_or_skip(f"{representation}/{method}",
                                lambda: run_setting(train, target, representation, method))
    if rows:
        print(save_results(rows, results_name("exp_key_normalization", args.final)))


if __name__ == "__main__":
    main()
