"""Experiment E: noise robustness.          Owner: D

Question
    How much does accuracy drop when the hum is inaccurate?

Design (proposed)
    Fit every model once on the clean training queries, then score target queries that
    have been corrupted by qbsh.noise, one corruption kind at a time, at several strengths.
    Strength 0 is the clean reference.
        kinds:  jitter (std in semitones), dropped voiced frames (fraction), tempo (rate)
        models: nearest query (similarity baseline), softmax regression, MLP
    Same pipeline for every row, so only the noise changes.

Output
    results/exp_noise_{val,test}.csv  (model, kind, level, top1, top5, top10, mrr)
    TODO: plot top-1 vs noise strength, one line per model, one panel per kind.

Run
    python -m experiments.exp_noise [--final]
"""
from qbsh.baselines import NearestQuery
from qbsh.evaluation import save_results, score_report
from qbsh.features import HandDesignedFeatures, KeyNormalizer, StandardScaler, impute_unvoiced
from qbsh.linear import SoftmaxRegression
from qbsh.noise import add_pitch_jitter, drop_voiced_frames, warp_tempo
from qbsh.nonlinear import MLPClassifier
from qbsh.pipeline import FunctionStep, Pipeline

from .common import load_split, parse_args, results_name, run_or_skip

# TODO(team): the strengths below are placeholders. Decide them in the experiment plan.
LEVELS = {
    "jitter": (0.0, 0.5, 1.0, 2.0),          # std, semitones
    "dropout": (0.0, 0.1, 0.2, 0.4),         # fraction of voiced frames removed
    "tempo": (1.0, 0.9, 0.8, 0.7),           # playback rate
}
CORRUPT = {"jitter": add_pitch_jitter, "dropout": drop_voiced_frames, "tempo": warp_tempo}


NO_NOISE = {"jitter": 0.0, "dropout": 0.0, "tempo": 1.0}   # the level that leaves queries unchanged


def corrupt(kind: str, pitch, level):
    return pitch if level == NO_NOISE[kind] else CORRUPT[kind](pitch, level)


def contour_steps():
    return [KeyNormalizer("median"), FunctionStep(impute_unvoiced)]


def hand_steps():
    return [KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()]


MODELS = {
    "nearest_query": lambda: Pipeline(contour_steps(), NearestQuery()),
    "softmax": lambda: Pipeline(hand_steps(), SoftmaxRegression()),
    "mlp": lambda: Pipeline(hand_steps(), MLPClassifier()),
}


def run_model(train, target, name: str) -> list[dict]:
    model = MODELS[name]().fit(train.pitch, train.song_id)
    return [{"model": name, "kind": kind, "level": level,
             **score_report(model.predict_scores(corrupt(kind, target.pitch, level)), target.song_id)}
            for kind, levels in LEVELS.items() for level in levels]


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    rows = []
    for name in MODELS:
        rows += run_or_skip(name, lambda: run_model(train, target, name))
    if rows:
        print(save_results(rows, results_name("exp_noise", args.final)))


if __name__ == "__main__":
    main()
