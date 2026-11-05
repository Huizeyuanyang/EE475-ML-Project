"""Experiment B: query length.          Owner: B

Question
    How much of a song does someone need to hum before it can be identified,
    and does the linear model degrade differently from the nonlinear one as
    queries get shorter?

Design
    Crop every query to its first L seconds, L = 1..8 (data.crop).
    Two protocols:
        matched:    train and score on L-second queries
        full_train: train on full 8 s queries, score on L-second ones
                    (needs length-independent features, i.e. A's HandDesignedFeatures;
                    skipped while the RawPitch placeholder is in use)
    Models: softmax regression (full batch and mini-batch), D's MLP,
    plus the nearest-centroid reference.
    Optional part 2: cost history of full-batch vs mini-batch gradient descent on one
    setting (plot cost vs iteration, as in the book).

Output
    results/exp_query_length_{val,test}.csv  (protocol, seconds, model, top1, top10, mrr)
    TODO(B): plot top-1 vs seconds, one line per model.

Run
    python -m experiments.exp_query_length [--final]
"""
from qbsh.baselines import NearestCentroid, RawPitch
from qbsh.data import crop
from qbsh.evaluation import evaluate, save_results
from qbsh.linear import SoftmaxRegression
from qbsh.nonlinear import MLPClassifier
from qbsh.pipeline import Pipeline

from .common import load_split, parse_args, results_name, run_or_skip

SECONDS = (1, 2, 3, 4, 5, 6, 7, 8)

MODELS = {
    "nearest_centroid (reference)": NearestCentroid,
    "softmax_gd": SoftmaxRegression,
    "softmax_minibatch": lambda: SoftmaxRegression(batch_size=64),
    "mlp": MLPClassifier,
}

# TODO(B): switch to A's features once they exist:
#   from qbsh.features import KeyNormalizer, HandDesignedFeatures, StandardScaler
#   def make_features(): return [KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()]
#   LENGTH_INDEPENDENT = True
def make_features():
    return [RawPitch()]


LENGTH_INDEPENDENT = False   # RawPitch's output size depends on the query length


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    protocols = ["matched"] + (["full_train"] if LENGTH_INDEPENDENT else [])
    rows = []
    for protocol in protocols:
        for seconds in SECONDS:
            X_te = crop(target.pitch, seconds)
            X_tr = crop(train.pitch, seconds) if protocol == "matched" else train.pitch
            for name, make_model in MODELS.items():
                rows += run_or_skip(
                    f"{protocol}/{seconds}s/{name}",
                    lambda: {"protocol": protocol, "seconds": seconds, "model": name,
                             **evaluate(Pipeline(make_features(), make_model()),
                                        X_tr, train.song_id, X_te, target.song_id)})
    if rows:
        print(save_results(rows, results_name("exp_query_length", args.final)))


if __name__ == "__main__":
    main()
