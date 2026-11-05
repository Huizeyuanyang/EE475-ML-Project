"""Experiment C: dimensionality reduction and clustering.          Owner: C

Questions
    1. How many dimensions do the features really need? (PCA, 8.3)
    2. Do queries cluster by song without labels? (K-means, 8.5)
    Every hyperparameter (n_components) is picked with K-fold CV (11.10) on the
    training years, grouped by singer. The val/test year is scored once.

Design
    Input: the imputed key-normalized contour (250-d).
    Classifier: B's SoftmaxRegression (nearest centroid until it is ready).
    part1_pca:     explained-variance curve; top-1 vs n_components
    part2_kmeans:  K-means with k = 48: cluster purity vs song labels;
                   optional: K-means distances as features

Output
    results/exp_dimred_<part>_{val,test}.csv
    TODO(C): plots of explained variance and top-1 vs n_components.

Run
    python -m experiments.exp_dimred [--final]
"""
from qbsh.baselines import NearestCentroid
from qbsh.evaluation import cross_validate, evaluate, save_results
from qbsh.features import KeyNormalizer, StandardScaler, impute_unvoiced
from qbsh.pipeline import FunctionStep, Pipeline
from qbsh.reduction import PCA, KMeans, cluster_purity

from .common import load_split, parse_args, results_name, run_or_skip

N_COMPONENTS = (2, 5, 10, 20, 50, 100)


def make_classifier():
    # TODO(C): switch to linear.SoftmaxRegression() once B's is ready.
    return NearestCentroid()


def contour_steps():
    return [KeyNormalizer("median"), FunctionStep(impute_unvoiced), StandardScaler()]


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


def part2_kmeans(train, target):
    def run():
        X_tr = train.pitch
        for step in contour_steps():
            X_tr = step.fit_transform(X_tr)
        km = KMeans(n_clusters=48).fit(X_tr)
        return {"k": 48, "purity": cluster_purity(km.labels_, train.song_id), "inertia": float(km.inertia_)}
    # TODO(C, optional): K-means distances (km.transform) as features -> classifier.
    return run_or_skip("kmeans", run)


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    for part in (part1_pca, part2_kmeans):
        print(part.__name__)
        rows = part(train, target)
        if rows:
            print(save_results(rows, results_name(f"exp_dimred_{part.__name__}", args.final)))


if __name__ == "__main__":
    main()
