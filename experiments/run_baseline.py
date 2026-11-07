"""Baselines that run before any model exists, scored on the year split.

    python -m experiments.run_baseline            # validation year
    python -m experiments.run_baseline --final    # test years

Two rows:
    raw pitch + nearest centroid          no key normalization (reference; runs today)
    key-normalized contour + nearest query  the similarity baseline; needs A's KeyNormalizer
                                            and impute_unvoiced, and is skipped until they exist

Every experiment should beat these rows; chance is 1/48 = 0.021 top-1.
"""
from qbsh.baselines import NearestCentroid, NearestQuery, RawPitch
from qbsh.evaluation import evaluate, save_results
from qbsh.features import KeyNormalizer, impute_unvoiced
from qbsh.pipeline import FunctionStep, Pipeline

from .common import load_split, parse_args, results_name, run_or_skip


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    settings = [
        ("raw pitch + nearest centroid", lambda: Pipeline([RawPitch()], NearestCentroid())),
        ("key-normalized contour + nearest query",
         lambda: Pipeline([KeyNormalizer("median"), FunctionStep(impute_unvoiced)], NearestQuery())),
    ]
    rows = []
    for name, make_model in settings:
        rows += run_or_skip(name, lambda: {
            "model": name,
            **evaluate(make_model(), train.pitch, train.song_id, target.pitch, target.song_id)})
    if rows:
        print(save_results(rows, results_name("baseline", args.final)))


if __name__ == "__main__":
    main()
