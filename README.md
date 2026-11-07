# Query by Singing/Humming on MIR-QBSH (EE 475 group project)

Identify which of **48 songs** someone is humming from an 8-second clip, using the
hand-labelled pitch contours in the MIR-QBSH corpus, and run four experiments,
one per member.

**Scope:** one linear model (`linear.SoftmaxRegression`) and one nonlinear model
(`nonlinear.MLPClassifier`). Plus a live demo for the video: a new recording is turned
into a pitch contour (`audio.py`) and fed to the trained model (`demo/hum_demo.py`).

## Who owns what

| Member | Module file(s) | Course methods | Experiment script |
|---|---|---|---|
| **A** | `qbsh/data.py`, `qbsh/features.py` | standard normalization 9.3, missing-value imputation 9.4, nonlinear feature engineering 10.5 | `experiments/exp_key_normalization.py` |
| **B** | `qbsh/optimizers.py`, `qbsh/linear.py`, `qbsh/audio.py`, `demo/hum_demo.py` | gradient descent 3.5, multi-class softmax / cross entropy 7.5, mini-batch 7.8; pitch extraction (pYIN) for the demo | `experiments/exp_query_length.py` |
| **C** | `qbsh/histograms.py`, `qbsh/evaluation.py`, `qbsh/reduction.py` | histogram features 9.2, PCA 8.3, K-means 8.5, K-fold CV 11.10 | `experiments/exp_dimred.py`, `experiments/analyze_confusion.py` (confused songs) |
| **D** | `qbsh/nonlinear.py`, `qbsh/noise.py` | neural nets Ch. 13 (batch norm 13.6, early stopping 13.7) | `experiments/exp_handcrafted_vs_learned.py`, `experiments/exp_noise.py` (noise robustness) |
| shared | `qbsh/contracts.py`, `qbsh/config.py`, `qbsh/pipeline.py`, `qbsh/baselines.py`, `experiments/common.py`, `experiments/run_baseline.py` | | |

## Layout

```
qbsh-project/
├── qbsh/
│   ├── contracts.py    shared: constants, QueryData, Transformer / Classifier interfaces
│   ├── config.py       shared: data path, train/val/test years, seed
│   ├── data.py         A: load corpus, year split, crop, load_wav            [implemented]
│   ├── features.py     A: imputation, key normalization, contour stats,
│   │                      polynomial features, HandDesignedFeatures, scaler  [stubs]
│   ├── histograms.py   C: pitch histogram, interval histogram                [stubs]
│   ├── audio.py        B: wav -> pitch (pYIN), same format as .pv            [stubs]
│   ├── optimizers.py   B: gradient descent, mini-batch                       [stubs]
│   ├── linear.py       B: categorical cross entropy, SoftmaxRegression       [stubs]
│   ├── evaluation.py   C: metrics + save_results [implemented], K-fold CV     [stubs]
│   ├── noise.py        D: jitter, dropped frames, tempo distortion           [stubs]
│   ├── reduction.py    C: PCA, K-means                                        [stubs]
│   ├── nonlinear.py    D: MLP (batch norm, early stopping, embed)             [stubs]
│   ├── pipeline.py     shared: Pipeline(steps, model), FunctionStep           [implemented]
│   └── baselines.py    shared: RawPitch, NearestCentroid, NearestQuery        [implemented]
├── experiments/
│   ├── run_baseline.py               raw baseline runs today; similarity baseline needs A's features
│   ├── exp_key_normalization.py      A: normalization methods, softmax
│   ├── exp_query_length.py           B: 1-8 s, softmax vs MLP vs nearest centroid
│   ├── exp_dimred.py                 C: PCA components, K-means purity
│   ├── exp_handcrafted_vs_learned.py D: MLP on hand features vs raw contour, embed -> softmax
│   ├── exp_noise.py                  D: accuracy vs jitter / dropped frames / tempo
│   └── analyze_confusion.py          C: most confused song pairs per model
├── docs/
│   └── experiment_plan.md            algorithms, experiments, metrics, analysis
├── demo/
│   └── hum_demo.py                   B: your .wav -> audio.py -> trained model -> top-10 songs
├── tests/
│   ├── test_data.py        real-corpus + plumbing checks (must pass now)
│   ├── test_contracts.py   one block per member; skipped until your code exists
│   └── test_extras.py      top-5, most_confused_pairs, NearestQuery (no corpus needed)
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
python -m experiments.run_baseline       # raw pitch + nearest centroid; key-normalized nearest query once A's features exist
python -m experiments.exp_query_length   # any experiment; unfinished settings print "skip"
python -m demo.hum_demo my_hum.wav       # needs librosa (B), A's features, and a trained model
```

Run from the project root, using `python -m experiments.<name>`, not `python experiments/<name>.py`.
Experiments score the **validation year** by default. Add `--final` to score the **test years**;
do that once, at the end, for the report. Results go to `results/<name>_val.csv` / `_test.csv`.

Raw reference (raw pitch, nearest centroid, no key normalization): **top-1 0.156 / top-10 0.608 / MRR 0.297** on the
validation year (2007), **0.143 / 0.603 / 0.279** on the test years. Chance top-1 is 0.021.
These numbers were measured before top-5 was added to the metrics.

The similarity baseline is `NearestQuery` after `KeyNormalizer("median")` and `impute_unvoiced`:
it matches a query's key-normalized contour to the training queries by Euclidean distance.
It is the baseline the learned models are compared against. `experiments.run_baseline` prints it
once A's two steps are implemented.

## Rules

1. **`contracts.py` and `config.py` change only when all four members agree.**
2. Keep the **names, arguments and return shapes** of the functions you own; change only their insides.
3. Before merging: `pytest` passes (your block is no longer skipped) and your experiment script runs.
4. Fit every step on the **training years only**. Never `fit` a scaler, PCA or model on val/test data.

## Conventions (details in `qbsh/contracts.py`)

- **X is `(n_samples, n_features)`.** The textbook stores points as columns (`x` of shape `(N, P)`),
  so transpose when porting book code.
- **Labels** are song ids `0..47` (`00012.pv` → 11).
- **Unvoiced frames are `np.nan`** in `QueryData.pitch` (0 in the raw `.pv` files, ~27% of frames).
  Some step before the model has to remove them (`impute_unvoiced`, the histograms, or `RawPitch`).
- **Every classifier** has `fit(X, y) -> self` and `predict_scores(X) -> (n, 48)`; `predict` is inherited.
  Metrics are MIREX-style: top-1, top-10 and mean reciprocal rank, plus top-5 (`evaluation.py`).
- **Every feature step / reducer** has `fit(X, y=None) -> self` and `transform(X)`.
- Pipelines start from the raw pitch matrix:
  `Pipeline([KeyNormalizer("median"), HandDesignedFeatures(), StandardScaler()], SoftmaxRegression())`.
- **`audio.wav_to_pitch` output must look like one row of `QueryData.pitch`**: semitones,
  31.25 frames/s, NaN for unvoiced, 250 frames. Then the models take it unchanged.

## Build order (what unblocks whom)

The experiment scripts already run end to end with placeholders (`RawPitch`, `NearestCentroid`).
Each place to swap in the real piece is marked with a `TODO` in the script.

1. **A:** `impute_unvoiced`, `KeyNormalizer`, `StandardScaler`, then `HandDesignedFeatures`
   (start with the histograms switched off). B, C and D's experiments all use these.
2. **B:** `categorical_cross_entropy`, `gradient_descent`, `SoftmaxRegression`, mini-batch.
   `SoftmaxRegression` is the default classifier in A's and C's experiments.
   `audio.py` and the demo come last.
3. **C:** `pitch_histogram` / `interval_histogram` first (A's `HandDesignedFeatures` uses them),
   then `k_fold_indices` / `cross_validate`. Everyone uses them to tune hyperparameters
   (until then, tune on the validation year). Then `PCA` (D's embedding plot uses it) and `KMeans`.
4. **D:** `MLPClassifier` with `embed`, then `qbsh/noise.py` for the noise experiment.

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
- Every query starts at the beginning of its song, so the demo works best if you hum the opening.
- `waveFile/year*/personDirInfo.txt` maps people to their real names. Don't load it, commit it or share it.
- The MIDI files aren't used by the classification pipeline (`data.midi_path` points to them if you want them).
