"""Experiment C: dimensionality reduction & feature selection.          Owner: C

Questions
    1. How many dimensions do the features really need? (PCA, 8.3)
    2. Does PCA-sphering help a linear classifier more than standard normalization? (9.5)
    3. Do queries cluster by song without labels? (K-means, 8.5)
    4. Which hand-designed features matter? (boosting 9.6, L1 regularization 9.7)
    Every hyperparameter (n_components, lam, n_features) is picked with K-fold CV
    (11.10) on the training years, grouped by singer. The val/test year is scored once.

Design
    Inputs: the imputed key-normalized contour (250-d) and A's hand-designed features.
    Classifier: B's SoftmaxRegression (nearest centroid until it is ready).
    part1_pca:        explained-variance curve; top-1 vs n_components
    part2_sphering:   StandardScaler vs PCASphering before the classifier
                      (accuracy and, with B, number of iterations to converge)
    part3_kmeans:     K-means with k = 48: cluster purity vs song labels;
                      K-means distances as features
    part4_selection:  boosting order and L1 path: top-1 vs number of features kept,
                      and the names of the top features (HandDesignedFeatures.feature_names_)

Output
    results/exp_dimred_<part>_{val,test}.csv
    TODO(C): plots of explained variance, top-1 vs n_components, top-1 vs #features.

Run
    python -m experiments.exp_dimred_featsel [--final]
"""
import numpy as np

from qbsh.baselines import NearestCentroid
from qbsh.evaluation import cross_validate, evaluate, save_results
from qbsh.features import HandDesignedFeatures, KeyNormalizer, StandardScaler, impute_unvoiced
from qbsh.pipeline import FunctionStep, Pipeline
from qbsh.reduction import PCA, BoostingSelector, KMeans, L1Selector, PCASphering, cluster_purity

from .common import load_split, parse_args, results_name, run_or_skip

N_COMPONENTS = (2, 5, 10, 20, 50, 100)
N_SELECTED = (1, 2, 5, 10, 20)


def make_classifier():
    # TODO(C): switch to linear.SoftmaxRegression() once B's is ready.
    return NearestCentroid()


def contour_steps():
    return [KeyNormalizer("median"), FunctionStep(impute_unvoiced), StandardScaler()]


def hand_steps():
    return [KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()]


def part1_pca(train, target):
    rows = []
    for n in N_COMPONENTS:
        rows += run_or_skip(f"pca n={n}", lambda: {
            "n_components": n,
            "cv_top1": cross_validate(lambda: Pipeline(contour_steps() + [PCA(n)], make_classifier()),
                                      train.pitch, train.song_id, k=5, groups=train.singer_id)["top1"],
            **evaluate(Pipeline(contour_steps() + [PCA(n)], make_classifier()),
                       train.pitch, train.song_id, target.pitch, target.song_id)})
    # TODO(C): explained_variance_ratio_ of PCA() fit on the training contour, saved for a plot.
    return rows


def part2_sphering(train, target):
    rows = []
    for name, extra in [("standard", []), ("pca_sphering", [PCASphering()])]:
        rows += run_or_skip(name, lambda: {"normalization": name, **evaluate(
            Pipeline(contour_steps() + extra, make_classifier()),
            train.pitch, train.song_id, target.pitch, target.song_id)})
    return rows


def part3_kmeans(train, target):
    def run():
        X_tr = train.pitch
        for step in contour_steps():
            X_tr = step.fit_transform(X_tr)
        km = KMeans(n_clusters=48).fit(X_tr)
        return {"k": 48, "purity": cluster_purity(km.labels_, train.song_id), "inertia": float(km.inertia_)}
    # TODO(C): K-means distances (km.transform) as features -> classifier.
    return run_or_skip("kmeans", run)


def part4_selection(train, target):
    rows = []
    for n in N_SELECTED:
        rows += run_or_skip(f"boosting n={n}", lambda: {"selector": "boosting", "n_features": n, **evaluate(
            Pipeline(hand_steps() + [BoostingSelector(n)], make_classifier()),
            train.pitch, train.song_id, target.pitch, target.song_id)})
    for lam in np.logspace(-4, 0, 5):
        rows += run_or_skip(f"l1 lam={lam:.0e}", lambda: {"selector": "l1", "lam": float(lam), **evaluate(
            Pipeline(hand_steps() + [L1Selector(lam)], make_classifier()),
            train.pitch, train.song_id, target.pitch, target.song_id)})
    # TODO(C): print the names of the selected features (feature_names_ from HandDesignedFeatures).
    return rows


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    for part in (part1_pca, part2_sphering, part3_kmeans, part4_selection):
        print(part.__name__)
        rows = part(train, target)
        if rows:
            print(save_results(rows, results_name(f"exp_dimred_{part.__name__}", args.final)))


if __name__ == "__main__":
    main()
