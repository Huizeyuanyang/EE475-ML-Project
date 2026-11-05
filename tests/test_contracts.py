"""Contract tests for each member's module, on small synthetic data.

While a function is still a stub (raises NotImplementedError) its test is reported
as SKIPPED, not failed. Once you implement it, the test runs and must pass before
you merge. You may tune hyperparameters inside your own tests, but don't weaken the
shape/invariance checks: other members' code relies on them.
"""
import functools

import numpy as np
import pytest

from qbsh.baselines import NearestCentroid
from qbsh.contracts import N_SONGS


def skip_if_stub(test):
    """Report a test as skipped (not failed) while the code it exercises is still a stub."""
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        try:
            return test(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(f"not implemented yet: {e}")
    return wrapper


# ---------------------------------------------------------------- synthetic data

def blobs(n_per_class=40, n_classes=3, d=5, spread=0.5, seed=0):
    """Well-separated Gaussian blobs; labels 0..n_classes-1."""
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(n_classes) / n_classes
    centers = np.zeros((n_classes, d))
    centers[:, 0], centers[:, 1] = 5 * np.cos(angles), 5 * np.sin(angles)
    X = np.vstack([c + spread * rng.normal(size=(n_per_class, d)) for c in centers])
    return X, np.repeat(np.arange(n_classes), n_per_class)


def circles(n=200, seed=0):
    """Inner disk (label 0) vs outer ring (label 1): not linearly separable."""
    rng = np.random.default_rng(seed)
    r = np.r_[rng.uniform(0, 1, n // 2), rng.uniform(2, 3, n // 2)]
    t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t)], np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]


def pitch_batch(n=4, T=250, seed=0):
    """Fake pitch contours shaped like data.load_queries().pitch (integer semitones, NaN gaps)."""
    rng = np.random.default_rng(seed)
    p = 55 + np.cumsum(rng.integers(-1, 2, size=(n, T)), axis=1).astype(float)
    p[rng.random((n, T)) < 0.25] = np.nan
    p[:, :5] = np.nan                       # leading silence
    return p


def accuracy(model, X, y):
    return np.mean(model.predict(X) == y)


def check_scores(scores, n):
    assert scores.shape == (n, N_SONGS)
    assert not np.isnan(scores).any()


# ================================================================ A: features

@skip_if_stub
def test_A_impute_fills_only_missing():
    from qbsh.features import impute_unvoiced
    p = pitch_batch()
    voiced = ~np.isnan(p)
    for method in ("linear", "hold", "mean", "zero"):
        out = impute_unvoiced(p, method)
        assert out.shape == p.shape and not np.isnan(out).any(), method
        assert np.allclose(out[voiced], p[voiced]), method
    assert np.allclose(impute_unvoiced(np.full((1, 10), np.nan)), 0)


@skip_if_stub
def test_A_impute_linear_interpolates():
    from qbsh.features import impute_unvoiced
    p = np.array([[np.nan, 60.0, np.nan, 64.0, np.nan]])
    assert np.allclose(impute_unvoiced(p, "linear"), [[60, 60, 62, 64, 64]])


@skip_if_stub
def test_A_key_normalizer_removes_transposition():
    from qbsh.features import KeyNormalizer
    p = pitch_batch()
    assert np.allclose(KeyNormalizer("none").transform(p), p, equal_nan=True)
    for method in KeyNormalizer.METHODS[1:]:
        kn = KeyNormalizer(method).fit(p)
        a, b = kn.transform(p), kn.transform(p + 5)
        assert np.array_equal(np.isnan(a), np.isnan(p)), method
        assert np.allclose(a, b, equal_nan=True), method


@skip_if_stub
def test_A_histograms():
    from qbsh.features import KeyNormalizer, interval_histogram, pitch_histogram
    p = pitch_batch()
    h = pitch_histogram(KeyNormalizer("median").transform(p), n_bins=25)
    assert h.shape == (4, 25) and np.allclose(h.sum(axis=1), 1)
    ih = interval_histogram(p, max_interval=12)
    assert ih.shape == (4, 25) and np.allclose(ih.sum(axis=1), 1)
    assert np.allclose(interval_histogram(p + 7, max_interval=12), ih)   # key-invariant


@skip_if_stub
def test_A_contour_statistics_and_polynomial_features():
    from qbsh.features import contour_statistics, polynomial_features
    feats, names = contour_statistics(pitch_batch())
    assert feats.shape == (4, len(names)) and not np.isnan(feats).any()
    out = polynomial_features(np.array([[2.0, 3.0]]), degree=2)        # x1, x2, x1^2, x1x2, x2^2
    assert sorted(out[0]) == [2, 3, 4, 6, 9]


@skip_if_stub
def test_A_hand_designed_features():
    from qbsh.features import HandDesignedFeatures, KeyNormalizer
    p = KeyNormalizer("median").transform(pitch_batch(n=6))
    f = HandDesignedFeatures().fit(p)
    X = f.transform(p)
    assert X.ndim == 2 and X.shape[0] == 6 and not np.isnan(X).any()
    assert len(f.feature_names_) == X.shape[1]
    assert f.transform(p[:, :62]).shape == X.shape      # cropped queries -> same feature size


@skip_if_stub
def test_A_standard_scaler_uses_train_statistics():
    from qbsh.features import StandardScaler
    X, _ = blobs()
    X[:, 2] = 3.0                                       # constant column
    s = StandardScaler().fit(X)
    Z = s.transform(X)
    assert np.isfinite(Z).all()
    assert np.allclose(Z.mean(axis=0), 0)
    assert np.allclose(Z[:, [0, 1, 3, 4]].std(axis=0), 1)
    assert np.allclose(s.transform(X + 1)[:, 0], Z[:, 0] + 1 / X[:, 0].std())   # train stats reused


# ================================================================ B: optimizers & linear models

def quadratic():
    A = np.array([[3.0, 0.5], [0.5, 1.0]])
    b = np.array([1.0, -2.0])
    return (lambda w: (0.5 * w @ A @ w - b @ w, A @ w - b)), A, np.linalg.solve(A, b)


def numeric_grad(f, w, h=1e-6):
    g = np.zeros_like(w)
    for i in np.ndindex(w.shape):
        e = np.zeros_like(w)
        e[i] = h
        g[i] = (f(w + e)[0] - f(w - e)[0]) / (2 * h)
    return g


@skip_if_stub
def test_B_gradient_descent_converges():
    from qbsh.optimizers import gradient_descent
    f, _, w_star = quadratic()
    w, hist = gradient_descent(f, np.zeros(2), alpha=0.1, max_its=2000)
    assert np.allclose(w, w_star, atol=1e-4) and hist[-1] < hist[0]


@skip_if_stub
def test_B_newton_solves_quadratic_in_one_step():
    from qbsh.optimizers import newtons_method
    f, A, w_star = quadratic()
    w, _ = newtons_method(f, lambda w: A, np.zeros(2), max_its=1)
    assert np.allclose(w, w_star, atol=1e-5)


@skip_if_stub
def test_B_minibatch_gradient_descent():
    from qbsh.optimizers import minibatch_gradient_descent
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    w_true = np.array([1.0, -2.0, 0.5])
    y = X @ w_true

    def batch(w, idx):
        r = X[idx] @ w - y[idx]
        return float(np.mean(r ** 2)), 2 * X[idx].T @ r / len(idx)

    w, hist = minibatch_gradient_descent(batch, np.zeros(3), n_samples=200, batch_size=32,
                                         alpha=0.05, epochs=50)
    assert np.allclose(w, w_true, atol=1e-3) and len(hist) == 50


@skip_if_stub
def test_B_cost_gradients_match_numeric():
    from qbsh import linear
    rng = np.random.default_rng(0)
    Xb = linear.add_bias(rng.normal(size=(30, 4)))
    y_pm, y_k = rng.choice([-1, 1], size=30), rng.integers(0, 3, size=30)
    w, W = rng.normal(size=5), rng.normal(size=(5, 3))
    cases = [
        ("softmax", lambda v: linear.softmax_cost(v, Xb, y_pm, lam=0.1), w),
        ("perceptron", lambda v: linear.perceptron_cost(v, Xb, y_pm), w),
        ("hinge", lambda v: linear.hinge_cost(v, Xb, y_pm, lam=0.1), w),
        ("multiclass_perceptron", lambda v: linear.multiclass_perceptron_cost(v, Xb, y_k, lam=0.1), W),
        ("cce", lambda v: linear.categorical_cross_entropy(v, Xb, y_k, lam=0.1), W),
    ]
    checked = []
    for name, cost, w0 in cases:
        try:
            grad = cost(w0)[1]
        except NotImplementedError:
            continue                                    # check the costs that exist so far
        assert np.allclose(grad, numeric_grad(cost, w0), atol=1e-5), name
        checked.append(name)
    if not checked:
        raise NotImplementedError("B: no cost functions yet")


@skip_if_stub
def test_B_binary_classifiers_separate_two_blobs():
    from qbsh.linear import LinearSVM, LogisticRegression, Perceptron
    X, y = blobs(n_classes=2, d=2)
    y = np.where(y == 1, 1, -1)
    for model in (LogisticRegression("gd"), LogisticRegression("newton"), Perceptron(), LinearSVM()):
        model.fit(X, y)
        assert model.decision_function(X).shape == (len(y),)
        assert accuracy(model, X, y) >= 0.95, type(model).__name__


@skip_if_stub
def test_B_multiclass_models():
    from qbsh.linear import LinearSVM, LogisticRegression, MulticlassPerceptron, OneVsAll, SoftmaxRegression
    X, y = blobs()
    for model in (OneVsAll(lambda: LogisticRegression()), OneVsAll(lambda: LinearSVM()),
                  MulticlassPerceptron(), SoftmaxRegression(), SoftmaxRegression(batch_size=16)):
        check_scores(model.fit(X, y).predict_scores(X), len(y))
        assert accuracy(model, X, y) >= 0.95, type(model).__name__


# ================================================================ C: reduction & evaluation

def stretched(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(200, 3)) * [5.0, 2.0, 0.1] + 10


@skip_if_stub
def test_C_pca():
    from qbsh.reduction import PCA
    X = stretched()
    pca = PCA(2).fit(X)
    assert pca.components_.shape == (2, 3)
    assert abs(pca.components_[0, 0]) > 0.99                     # first axis = the widest one
    assert np.all(np.diff(pca.explained_variance_ratio_) <= 0) and pca.explained_variance_ratio_.sum() > 0.99
    assert pca.transform(X).shape == (200, 2)
    assert np.abs(pca.inverse_transform(pca.transform(X)) - X).mean() < 0.2
    full = PCA().fit(X)
    assert np.allclose(full.inverse_transform(full.transform(X)), X)


@skip_if_stub
def test_C_pca_sphering_whitens():
    from qbsh.reduction import PCASphering
    Z = PCASphering().fit_transform(stretched())
    assert np.allclose(np.cov(Z.T), np.eye(3), atol=0.02)


@skip_if_stub
def test_C_cluster_purity():
    from qbsh.reduction import cluster_purity
    assert cluster_purity(np.array([0, 0, 1, 1]), np.array([5, 5, 5, 7])) == 0.75


@skip_if_stub
def test_C_kmeans_finds_blobs():
    from qbsh.reduction import KMeans, cluster_purity
    X, y = blobs()
    km = KMeans(n_clusters=3).fit(X)
    assert cluster_purity(km.labels_, y) >= 0.99
    assert np.array_equal(km.predict(X), km.labels_)
    assert km.transform(X).shape == (len(X), 3)


@skip_if_stub
def test_C_k_fold_indices_respect_groups():
    from qbsh.evaluation import k_fold_indices
    groups = np.repeat(np.arange(20), 5)
    seen = []
    folds = list(k_fold_indices(100, k=5, groups=groups))
    assert len(folds) == 5
    for tr, va in folds:
        assert not set(tr) & set(va)
        assert len(tr) + len(va) == 100
        assert not set(groups[tr]) & set(groups[va])           # no group in both
        seen += list(va)
    assert sorted(seen) == list(range(100))                     # each row validated once


@skip_if_stub
def test_C_cross_validate():
    from qbsh.evaluation import cross_validate
    X, y = blobs()
    out = cross_validate(NearestCentroid, X, y, k=5)
    assert {"top1", "top10", "mrr", "top1_std"} <= set(out)
    assert out["top1"] > 0.95


@skip_if_stub
def test_C_feature_selectors_find_the_informative_feature():
    from qbsh.reduction import BoostingSelector, L1Selector
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = (X[:, 2] > 0).astype(int)                                # only feature 2 matters
    boost = BoostingSelector(n_features=1).fit(X, y)
    assert boost.selected_[0] == 2 and boost.transform(X).shape == (300, 1)
    l1 = L1Selector(lam=0.05).fit(X, y)
    assert 2 in l1.selected_ and l1.transform(X).shape == (300, len(l1.selected_))


# ================================================================ D: nonlinear models

@skip_if_stub
def test_D_kernels():
    from qbsh.nonlinear import polynomial_kernel, rbf_kernel
    A, B = np.random.default_rng(0).normal(size=(4, 3)), np.random.default_rng(1).normal(size=(5, 3))
    assert np.allclose(polynomial_kernel(A, B, degree=2, c=1.0), (A @ B.T + 1) ** 2)
    K = rbf_kernel(A, A, gamma=0.5)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T) and (K > 0).all()


@skip_if_stub
def test_D_kernel_classifier_solves_circles():
    from qbsh.nonlinear import KernelClassifier
    X, y = circles()
    model = KernelClassifier("rbf", gamma=1.0).fit(X, y)
    check_scores(model.predict_scores(X), len(y))
    assert accuracy(model, X, y) >= 0.95


@skip_if_stub
def test_D_mlp_and_embedding():
    from qbsh.nonlinear import MLPClassifier
    X, y = blobs()
    model = MLPClassifier(hidden=(32,), lr=1e-2, epochs=200).fit(X, y)   # tune lr/epochs to your optimizer
    check_scores(model.predict_scores(X), len(y))
    assert accuracy(model, X, y) >= 0.9
    assert model.embed(X).shape == (len(X), 32)


@skip_if_stub
def test_D_trees_and_bagging_solve_circles():
    from qbsh.nonlinear import Bagging, DecisionTree, GradientBoosting, RandomForest
    X, y = circles()
    for model, threshold in ((DecisionTree(max_depth=8), 0.95), (RandomForest(n_trees=20), 0.95),
                             (GradientBoosting(n_rounds=30), 0.9),
                             (Bagging(lambda: DecisionTree(max_depth=5), n_models=5), 0.9)):
        check_scores(model.fit(X, y).predict_scores(X), len(y))
        assert accuracy(model, X, y) >= threshold, type(model).__name__


# ================================================================ shared plumbing

def test_pipeline_with_baseline():
    from qbsh.baselines import RawPitch
    from qbsh.pipeline import Pipeline
    X, y = blobs()
    model = Pipeline([RawPitch()], NearestCentroid()).fit(X, y)
    check_scores(model.predict_scores(X), len(y))
    assert accuracy(model, X, y) == 1.0
