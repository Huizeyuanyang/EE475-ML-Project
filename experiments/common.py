"""Helpers shared by the experiment scripts (implemented)."""
from __future__ import annotations

import argparse
from typing import Callable

from qbsh.contracts import QueryData
from qbsh.data import load_queries, year_split


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--final", action="store_true",
                   help="score on the TEST years (do this once, at the end). Default: validation year.")
    return p.parse_args()


def load_split(final: bool) -> tuple[QueryData, QueryData]:
    """-> (train, target). target is the test years with --final, else the validation year."""
    train, val, test = year_split(load_queries())
    target = test if final else val
    print(f"train {len(train)} queries | {'TEST' if final else 'val'} {len(target)} queries")
    return train, target


def results_name(name: str, final: bool) -> str:
    """results/<name>_val.csv or results/<name>_test.csv, so test runs never overwrite val runs."""
    return f"{name}_{'test' if final else 'val'}"


def run_or_skip(label: str, fn: Callable[[], dict | list[dict]]) -> list[dict]:
    """Run one setting and return its result rows.

    `fn` returns one row (dict) or several (list of dicts), e.g.
    {"method": "median", "top1": 0.41, ...}. If the setting needs code that is
    still a stub, print a skip line and return [] instead of crashing, so a
    sweep runs with whatever is implemented so far.
    """
    try:
        out = fn()
    except NotImplementedError as e:
        print(f"  skip  {label:40s} (not implemented yet: {e})")
        return []
    rows = [out] if isinstance(out, dict) else list(out)
    for row in rows:
        print("  done  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in row.items()))
    return rows
