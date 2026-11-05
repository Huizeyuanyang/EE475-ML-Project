"""Chain feature steps and a classifier so everything is fit on training data only.

Shared; implemented. Example:

    model = Pipeline(
        steps=[KeyNormalizer("median"), FunctionStep(impute_unvoiced),
               HandDesignedFeatures(), StandardScaler()],
        model=SoftmaxRegression(),
    )
    model.fit(train.pitch, train.song_id)
    scores = model.predict_scores(test.pitch)     # (n_test, 48)

The input X is the raw pitch matrix (n, T) with np.nan for unvoiced frames, so the
steps before the model must remove the NaNs.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .contracts import Classifier, Transformer


class FunctionStep(Transformer):
    """Wrap a stateless function f(X) -> X' as a pipeline step."""

    def __init__(self, fn: Callable[..., np.ndarray], **kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def transform(self, X):
        return self.fn(X, **self.kwargs)

    def __repr__(self):
        return f"FunctionStep({self.fn.__name__})"


class Pipeline(Classifier):
    def __init__(self, steps: Sequence[Transformer], model: Classifier):
        self.steps = list(steps)
        self.model = model

    def fit(self, X, y):
        for step in self.steps:
            X = step.fit(X, y).transform(X)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        """Run only the (already fitted) feature steps."""
        for step in self.steps:
            X = step.transform(X)
        return X

    def predict_scores(self, X):
        return self.model.predict_scores(self.transform(X))
