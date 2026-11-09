"""Experiment A: key normalization.          Owner: A

Question
    People hum the same song in different keys. Does removing the key help
    identify the song, and which way of removing it works best?

Design
    Two representations, each with every KeyNormalizer method ("none" = reference):
        contour: KeyNormalizer(m) -> impute_unvoiced -> StandardScaler      (250 numbers)
        hand:    KeyNormalizer(m) -> HandDesignedFeatures -> StandardScaler
    One fixed classifier for every row, so only the normalization changes.
    Robustness check: transpose the scored queries by +2, +4, +6 semitones.
    A key-invariant pipeline gives the same numbers at every shift.

Output
    results/exp_key_normalization_{val,test}.csv  (representation, method, shift, top1, top5, top10, mrr)
    results/exp_key_normalization_{val,test}.png  left: top-1 by method (no shift);
                                                  right: top-1 vs transposition shift

Run
    python -m experiments.exp_key_normalization [--final]
"""
from qbsh import config
from qbsh.baselines import NearestCentroid
from qbsh.evaluation import save_results, score_report
from qbsh.features import HandDesignedFeatures, KeyNormalizer, StandardScaler, impute_unvoiced
from qbsh.pipeline import FunctionStep, Pipeline

from .common import load_split, parse_args, results_name, run_or_skip

SHIFTS = (0, 2, 4, 6)   # semitones added to the scored queries
REPRESENTATIONS = ("contour", "hand")

# C's pitch/interval histograms are part of HandDesignedFeatures but are still stubs.
# TODO(A): set to True once histograms.py is implemented, then rerun.
USE_HISTOGRAMS = False


def make_classifier():
    # Same classifier for every row. TODO(A): switch to linear.SoftmaxRegression() once B's is ready.
    return NearestCentroid()


def make_pipeline(representation: str, method: str) -> Pipeline:
    key = KeyNormalizer(method)
    if representation == "contour":
        steps = [key, FunctionStep(impute_unvoiced), StandardScaler()]
    elif representation == "hand":
        steps = [key, HandDesignedFeatures(use_pitch_hist=USE_HISTOGRAMS,
                                           use_interval_hist=USE_HISTOGRAMS), StandardScaler()]
    else:
        raise ValueError(representation)
    return Pipeline(steps, make_classifier())


def run_setting(train, target, representation: str, method: str) -> list[dict]:
    """Fit once on train, then score the target queries at every transposition shift."""
    model = make_pipeline(representation, method).fit(train.pitch, train.song_id)
    return [{"representation": representation, "method": method, "shift": shift,
             **score_report(model.predict_scores(target.pitch + shift), target.song_id)}
            for shift in SHIFTS]


def plot_results(rows: list[dict], name: str) -> str:
    """Two panels: top-1 by method at shift 0 (bars), and top-1 vs shift (lines)."""
    import matplotlib
    matplotlib.use("Agg")                       # write a file; no window needed
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [m for m in KeyNormalizer.METHODS if any(r["method"] == m for r in rows)]
    reps = [r for r in REPRESENTATIONS if any(row["representation"] == r for row in rows)]

    def top1(rep, method, shift):
        for r in rows:
            if r["representation"] == rep and r["method"] == method and r["shift"] == shift:
                return r["top1"]
        return np.nan

    fig, axes = plt.subplots(1, 1 + len(reps), figsize=(5 * (1 + len(reps)), 4))
    ax = axes[0]
    width = 0.8 / len(reps)
    x = np.arange(len(methods))
    for j, rep in enumerate(reps):
        vals = [top1(rep, m, 0) for m in methods]
        bars = ax.bar(x + (j - (len(reps) - 1) / 2) * width, vals, width, label=rep)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1)
    ax.set_xlabel("key normalization method")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Top-1 by method (no transposition)")
    ax.legend(title="representation")

    for ax, rep in zip(axes[1:], reps):
        for m in methods:
            ax.plot(SHIFTS, [top1(rep, m, s) for s in SHIFTS], marker="o", label=m)
        ax.set_xticks(SHIFTS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("test queries transposed by (semitones)")
        ax.set_ylabel("top-1 accuracy")
        ax.set_title(f"Robustness to transposition ({rep})")
        ax.legend(title="method", fontsize=8, ncol=3, loc="upper center")

    fig.tight_layout()
    path = config.RESULTS_DIR / f"{name}.png"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main():
    args = parse_args(__doc__)
    train, target = load_split(args.final)
    rows = []
    for representation in REPRESENTATIONS:
        for method in KeyNormalizer.METHODS:
            rows += run_or_skip(f"{representation}/{method}",
                                lambda: run_setting(train, target, representation, method))
    if rows:
        name = results_name("exp_key_normalization", args.final)
        print(save_results(rows, name))
        print(plot_results(rows, name))


if __name__ == "__main__":
    main()
