# Query by Singing/Humming on MIR-QBSH (EE 475 group project)

Identify which of **48 songs** someone is humming from an 8-second clip, using the
hand-labelled pitch contours in the MIR-QBSH corpus, and run four experiments,
one per member.

## Who owns what

| Member | Module file(s) | Course methods | Experiment script |
|---|---|---|---|
| **A** | `qbsh/data.py`, `qbsh/features.py` | histogram features 9.2, standard normalization 9.3, missing-value imputation 9.4, nonlinear feature engineering 10.5 | `experiments/exp_key_normalization.py` |
| **B** | `qbsh/optimizers.py`, `qbsh/linear.py` | gradient descent 3.5, Newton 4.3, logistic/softmax 6.2–6.3, perceptron 6.4, SVM 6.5, one-vs-all 7.2, multi-class perceptron 7.3, cross entropy 7.5, mini-batch 7.8 | `experiments/exp_query_length.py` |
| **C** | `qbsh/evaluation.py`, `qbsh/reduction.py` | PCA 8.3, K-means 8.5, PCA-sphering 9.5, feature selection by boosting 9.6 and regularization 9.7, K-fold CV 11.10 | `experiments/exp_dimred_featsel.py` |
| **D** | `qbsh/nonlinear.py` | kernels Ch. 12, neural nets Ch. 13 (batch norm 13.6, early stopping 13.7), trees / random forests / boosting Ch. 14, bagging 11.9 | `experiments/exp_handcrafted_vs_learned.py` |
| shared | `qbsh/contracts.py`, `qbsh/config.py`, `qbsh/pipeline.py`, `qbsh/baselines.py`, `experiments/common.py`, `experiments/run_baseline.py` | | |

## Layout

```
qbsh-project/
├── qbsh/
│   ├── contracts.py    shared: constants, QueryData, Transformer / Classifier interfaces
│   ├── config.py       shared: data path, train/val/test years, seed
│   ├── data.py         A: load corpus, year split, crop, load_wav            [implemented]
│   ├── features.py     A: imputation, key normalization, histograms, scaler  [stubs]
│   ├── optimizers.py   B: gradient descent, Newton, mini-batch                [stubs]
│   ├── linear.py       B: costs, logistic, perceptron, SVM, OvA, softmax      [stubs]
│   ├── evaluation.py   C: metrics + save_results [implemented], K-fold CV     [stubs]
│   ├── reduction.py    C: PCA, sphering, K-means, boosting / L1 selection     [stubs]
│   ├── nonlinear.py    D: kernels, MLP, trees, forests, boosting, bagging     [stubs]
│   ├── pipeline.py     shared: Pipeline(steps, model), FunctionStep           [implemented]
│   └── baselines.py    shared: RawPitch, NearestCentroid placeholders         [implemented]
├── experiments/
│   ├── run_baseline.py               runs today
│   ├── exp_key_normalization.py      A
│   ├── exp_query_length.py           B
│   ├── exp_dimred_featsel.py         C
│   └── exp_handcrafted_vs_learned.py D
├── tests/
│   ├── test_data.py        real-corpus + plumbing checks (must pass now)
│   └── test_contracts.py   one block per member; skipped until your code exists
└── results/                CSV output (git-ignored)
```

## Setup and running

Put the corpus at `./MIR-QBSH-corpus` (the folder holding `midiFile/` and `waveFile/`).
It is git-ignored, so each teammate downloads it themselves and nothing from it is ever committed.
Anywhere else: `export QBSH_DATA=/path/to/MIR-QBSH-corpus`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # stubs show up as "skipped"
python -m experiments.run_baseline       # raw pitch + nearest centroid
python -m experiments.exp_query_length   # any experiment; unfinished settings print "skip"
```

Run from the project root, using `python -m experiments.<name>`, not `python experiments/<name>.py`.
Experiments score the **validation year** by default. Add `--final` to score the **test years**;
do that once, at the end, for the report. Results go to `results/<name>_val.csv` / `_test.csv`.

Baseline (raw pitch, nearest centroid): **top-1 0.156 / top-10 0.608 / MRR 0.297** on the
validation year (2007), **0.143 / 0.603 / 0.279** on the test years. Chance top-1 is 0.021.

## Rules

1. **`contracts.py` and `config.py` change only when all four members agree.**
2. Keep the **names, arguments and return shapes** of the functions you own; change only their insides.
3. Before merging: `pytest` passes (your block is no longer skipped) and your experiment script runs.
4. Fit every step on the **training years only**. Never `fit` a scaler, PCA or model on val/test data.

## Conventions (details in `qbsh/contracts.py`)

- **X is `(n_samples, n_features)`.** The textbook stores points as columns (`x` of shape `(N, P)`),
  so transpose when porting book code.
- **Labels** are song ids `0..47` (`00012.pv` → 11). Binary models inside one-vs-all use `{-1, +1}`.
- **Unvoiced frames are `np.nan`** in `QueryData.pitch` (0 in the raw `.pv` files, ~27% of frames).
  Some step before the model has to remove them (`impute_unvoiced`, the histograms, or `RawPitch`).
- **Every classifier** has `fit(X, y) -> self` and `predict_scores(X) -> (n, 48)`; `predict` is inherited.
  Metrics are MIREX-style: top-1, top-10 and mean reciprocal rank (`evaluation.py`).
- **Every feature step / reducer** has `fit(X, y=None) -> self` and `transform(X)`.
- Pipelines start from the raw pitch matrix:
  `Pipeline([KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()], SoftmaxRegression())`.

## Build order (what unblocks whom)

The experiment scripts already run end to end with placeholders (`RawPitch`, `NearestCentroid`).
Each place to swap in the real piece is marked with a `TODO` in the script.

1. **A:** `impute_unvoiced`, `KeyNormalizer`, `StandardScaler`, then `HandDesignedFeatures`.
   B, C and D's experiments all use these.
2. **B:** `softmax_cost` / `categorical_cross_entropy`, `gradient_descent`, then `SoftmaxRegression`.
   It is the default classifier in A's and C's experiments, and C's selectors reuse B's costs.
3. **C:** `k_fold_indices` / `cross_validate`. Everyone uses them to tune hyperparameters
   (until then, tune on the validation year).
4. **D:** `MLPClassifier` (with `embed`) first; the trees and kernels can come later.

## The split (`config.py`)

| Split | Years | Clips | Songs |
|---|---|---|---|
| train | 2003, 2004a, 2004b, 2005, 2006a, 2006b | 2749 | all 48 |
| val | 2007 | 454 | all 48 |
| test | 2008, 2009 | 1228 | all 48 |

Splitting by year means no singer is in two splits. Person numbers restart every year,
so `singer_id` includes the year. For K-fold CV inside the training years, pass
`groups=train.singer_id` so one singer's clips stay in one fold.

## Things about the data that matter

- Every clip is exactly 8.0 s: 250 pitch frames at 31.25 frames/s (256-sample frames at 8 kHz).
- The classes are very unbalanced: songs 1–10 (English nursery rhymes, mostly from 2006a) have 11–19 clips each;
  most of the others have 50–170. Look at per-song accuracy (`evaluation.confusion_matrix`), not only the average.
- The pitch labels were made by hand by the students who recorded the clips, so some are wrong.
- `waveFile/year*/personDirInfo.txt` maps people to their real names. Don't load it, commit it or share it.
- The MIDI files aren't used by the classification pipeline (`data.midi_path` points to them if you want them).
