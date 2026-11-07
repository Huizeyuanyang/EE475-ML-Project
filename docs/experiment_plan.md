# EE 475 Experiment Plan: Query by Humming on MIR-QBSH

## 1. Problem Statement

Given a short, imperfect hum, the system ranks 48 songs by how likely each one is the song being hummed. Beyond building the system, the goal is to find out experimentally which design choices matter most for retrieval accuracy.

## 2. Data and Evaluation Setup

- **Data:** MIR-QBSH corpus, 4,431 queries of 48 songs. Each query is an 8-second pitch contour (250 frames at 31.25 frames per second).
- **Split:** by recording year, so no singer appears in more than one split.
  - Train: 2003-2006 (2,749 queries)
  - Validation: 2007 (454 queries)
  - Test: 2008-2009 (1,228 queries)
- **Protocol:** all preprocessing steps and models are fit on the training years only. The validation year is used to choose settings. The test years are scored once, at the end.

## 3. System Overview and Demo

The system works in the following stages:

hum input -> pitch extraction -> key normalization -> feature extraction -> classifier -> ranking of 48 songs

- **Training and experiments:** the input is the pitch contour provided by the corpus.
- **Demo:** pitch is extracted directly from a newly recorded hum (.wav) with pYIN. The output is converted to the same format as the training data (semitones, 31.25 frames per second, unvoiced frames as missing values, 250 frames) and passed to the trained model, which shows the top-ranked songs.
- **Known limitation:** the corpus pitch contours were labeled by hand by the students who recorded the clips, while the demo uses automatically extracted pitch. The input distributions differ, so demo accuracy may be lower than the experimental results.

## 4. Algorithms

| Component | Method |
|---|---|
| Linear model | Softmax regression (multinomial logistic regression) with L2 regularization, trained by gradient descent and mini-batch gradient descent |
| Nonlinear model | Multilayer perceptron (MLP) with batch normalization and early stopping |
| Baseline | Euclidean-distance matching between a key-normalized query contour and the training queries (`baselines.NearestQuery`) |
| Dimensionality reduction and clustering | PCA, K-means |
| Model selection | K-fold cross-validation, grouped by singer |
| Pitch extraction (demo) | pYIN |

## 5. Experiments

| Experiment | Question | Variables | Script |
|---|---|---|---|
| A. Key normalization | Does removing the singer's key help? Which method works best? | Normalization method (none, mean, median, etc.); representation (raw contour, histogram features); robustness check by transposing test queries by +2, +4, +6 semitones | `exp_key_normalization.py` |
| B. Query length | How much of a song must be hummed to identify it? Do the linear and nonlinear models degrade differently? | First 1 to 8 seconds of each query; training and testing on the same length vs. training on 8 s and testing on shorter queries | `exp_query_length.py` |
| C. Dimensionality reduction and clustering | How many dimensions do the features really need? Do queries cluster by song without labels? | Number of PCA components (2 to 100); K-means with k = 48, measured by cluster purity | `exp_dimred.py` |
| D. Hand-designed vs. learned features | Do features learned by a neural network beat hand-designed features? | Hand-designed features; raw contour fed directly to the MLP; MLP embedding fed to the linear model; ablation of batch normalization and early stopping | `exp_handcrafted_vs_learned.py` |
| E. Noise robustness | How much does accuracy drop on inaccurate humming? | Synthetic pitch jitter, dropped voiced frames, and tempo distortion added to queries, at several strengths | `exp_noise.py` |

All experiments use the same pipeline (normalization, features, model) and change one factor at a time.

## 6. Evaluation Metrics and Analysis

- **Metrics:** Top-1, Top-5, and Top-10 accuracy, and mean reciprocal rank (MRR). Per-song accuracy is also reported, because the classes are imbalanced.
- **Confusion analysis:** extract the most commonly confused song pairs from the confusion matrix and analyze why they are confused, compared across models and conditions (`analyze_confusion.py`).

## 7. Team Roles

| Member | Code | Experiments and analysis | Other |
|---|---|---|---|
| A | `features.py`: missing-value imputation, key normalization, contour statistics, polynomial and hand-designed features, scaler | Key normalization | Introduction, data, and system sections of the report; integration (merging branches, final test-set run) |
| B | Softmax regression and optimizers; pitch extraction (pYIN) and the demo | Query length | Demo video |
| C | K-fold cross-validation, PCA, K-means; histogram features | Dimensionality reduction and clustering; confusion analysis | Evaluation metrics section of the report |
| D | MLP; noise module | Hand-designed vs. learned features; noise robustness | TA office hours; keeping this plan up to date |

Each member also produces the figures and written results for their own experiments.
