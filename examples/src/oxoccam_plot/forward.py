"""Helpers for loading forward-response CSV files."""

from __future__ import annotations

import glob
import os
import re

import pandas as pd


def load_forward_data(
    example_dir: str,
    start_name: str = "forward.csv",
) -> tuple[pd.DataFrame, pd.DataFrame | None, int | None]:
    """Load the starting-model and (optional) final-iteration CSV data.

    Returns ``(df_start, df_final, final_iter)`` where *df_final* and
    *final_iter* are ``None`` when no iteration CSVs are found.
    """
    start_csv = os.path.join(example_dir, start_name)
    if not os.path.exists(start_csv):
        raise FileNotFoundError(
            f"Starting-model CSV not found: {start_csv}\nRun 'occam1d forward' first."
        )
    df_start = pd.read_csv(start_csv)

    iter_csvs = sorted(glob.glob(os.path.join(example_dir, "iter_????.csv")))
    df_final: pd.DataFrame | None = None
    final_iter: int | None = None
    if iter_csvs:
        df_final = pd.read_csv(iter_csvs[-1])
        final_iter = int(re.search(r"iter_(\d+)", iter_csvs[-1]).group(1))

    return df_start, df_final, final_iter
