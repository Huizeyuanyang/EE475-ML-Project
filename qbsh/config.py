"""Paths, the agreed train/val/test split, and the random seed.

The split is a team decision; change it only together (the corpus authors ask
that papers state which years were used for training, validation and testing).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Corpus location (the folder containing midiFile/ and waveFile/). It is git-ignored.
# Order: $QBSH_DATA, then ./MIR-QBSH-corpus inside the project, then ../MIR-QBSH-corpus.
def _find_data_root() -> Path:
    if "QBSH_DATA" in os.environ:
        return Path(os.environ["QBSH_DATA"])
    for candidate in (PROJECT_ROOT / "MIR-QBSH-corpus", PROJECT_ROOT.parent / "MIR-QBSH-corpus"):
        if (candidate / "waveFile").is_dir():
            return candidate
    return PROJECT_ROOT / "MIR-QBSH-corpus"   # not found: data.py raises a clear error


DATA_ROOT = _find_data_root()

RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 0

# ---- Year-based split ----
# Splitting by year means no singer appears in two splits.
# Every one of the 48 songs appears in TRAIN (songs 1-10 come mostly from year2006a,
# and songs 3, 6, 9 exist only in 2006a, 2007 and 2008).
TRAIN_YEARS = ("year2003", "year2004a", "year2004b", "year2005", "year2006a", "year2006b")  # 2749 clips
VAL_YEARS = ("year2007",)                                                                   #  454 clips
TEST_YEARS = ("year2008", "year2009")                                                       # 1228 clips
