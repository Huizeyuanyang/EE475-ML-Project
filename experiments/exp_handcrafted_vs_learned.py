"""Experiment D: hand-designed vs learned features.          Owner: D

Question
    Do features a neural network learns from the pitch contour beat A's
    hand-designed histogram/statistics features?

Design
    hand-designed: KeyNormalizer -> HandDesignedFeatures -> StandardScaler -> MLP
    learned:       KeyNormalizer -> impute_unvoiced -> StandardScaler (250-d contour) -> MLP
    learned -> linear: MLP.embed() activations fed to B's SoftmaxRegression
                   (are the learned features linearly separable?)
    Ablations on the learned MLP: batch norm on/off (13.6), early stopping on/off (13.7).
    Pick hyperparameters on the validation year or with C's cross_validate.

Output
    results/exp_handcrafted_vs_learned_{val,test}.csv  (features, model, top1, top10, mrr)
    TODO(D): bar chart of top-1 per (features, model); PCA 2-D plot of embed() colored by song
             (use C's reduction.PCA).

Run
    python -m experiments.exp_handcrafted_vs_learned [--final]
"""
from qbsh.evaluation import evaluate, save_results
from qbsh.features import HandDesignedFeatures, KeyNormalizer, StandardScaler, impute_unvoiced
from qbsh.nonlinear import MLPClassifier
from qbsh.pipeline import FunctionStep, Pipeline

from .common import load_split, parse_args, results_name, run_or_skip


def hand_steps():
    return [KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()]


def contour_steps():
    return [KeyNormalizer("median"), FunctionStep(impute_unvoiced), StandardScaler()]


SETTINGS = [
    # (features, model name, steps factory, model factory)
    ("hand", "mlp", hand_steps, lambda: MLPClassifier()),
    ("learned", "mlp", contour_steps, lambda: MLPClassifier(hidden=(256, 128))),
    ("learned", "mlp_no_batchnorm", contour_steps, lambda: MLPClassifier(hidden=(256, 128), batch_norm=False)),
    ("learned", "mlp_no_early_stop", contour_steps, lambda: MLPClassifier(hidden=(256, 128), early_stopping=False)),
]


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    rows = []
    for features, name, make_steps, make_model in SETTINGS:
        rows += run_or_skip(f"{features}/{name}", lambda: {"features": features, "model": name, **evaluate(
            Pipeline(make_steps(), make_model()), train.pitch, train.song_id, target.pitch, target.song_id)})
    # TODO(D): learned -> linear. Fit MLPClassifier on the contour, take embed() of train/target,
    #          then evaluate linear.SoftmaxRegression on those embeddings.
    if rows:
        print(save_results(rows, results_name("exp_handcrafted_vs_learned", args.final)))


if __name__ == "__main__":
    main()
