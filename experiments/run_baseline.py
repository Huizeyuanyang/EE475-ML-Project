"""Sanity check that runs today: raw pitch -> nearest centroid, year split.

    python -m experiments.run_baseline            # validation year
    python -m experiments.run_baseline --final    # test years

Every experiment should beat this row; chance is 1/48 = 0.021 top-1.
"""
from qbsh.baselines import NearestCentroid, RawPitch
from qbsh.evaluation import evaluate, save_results
from qbsh.pipeline import Pipeline

from .common import load_split, parse_args, results_name


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    model = Pipeline([RawPitch()], NearestCentroid())
    metrics = evaluate(model, train.pitch, train.song_id, target.pitch, target.song_id)
    print("raw pitch + nearest centroid: " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print(save_results([{"model": "raw pitch + nearest centroid", **metrics}], results_name("baseline", args.final)))


if __name__ == "__main__":
    main()
