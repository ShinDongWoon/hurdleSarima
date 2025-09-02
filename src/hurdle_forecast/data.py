from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
import pandas as pd
import numpy as np

@dataclass
class Dataset:
    train: pd.DataFrame
    tests: Dict[str, pd.DataFrame]  # map filename -> df

def _ensure_columns(df: pd.DataFrame, series_cols: Tuple[str, str], date_col: str, target_col: str | None):
    missing = [c for c in ([*series_cols, date_col] + ([target_col] if target_col else [])) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got columns: {df.columns.tolist()}")


def maybe_split_series(
    df: pd.DataFrame,
    series_cols: Tuple[str, str],
    joined_col: str = "영업장명_메뉴명",
) -> None:
    """If `joined_col` exists in ``df``, split it into the two ``series_cols``.

    This lets datasets provide a single combined column (e.g. ``영업장명_메뉴명``)
    instead of two separate columns. The function mutates ``df`` in place and
    creates/overwrites the ``series_cols``.
    """
    if joined_col in df.columns:
        df[list(series_cols)] = df[joined_col].str.split("_", n=1, expand=True)

def load_datasets(train_csv: str, test_dir: str, series_cols: Tuple[str, str], date_col: str, target_col: str) -> Dataset:
    train = pd.read_csv(train_csv)
    maybe_split_series(train, series_cols)
    _ensure_columns(train, series_cols, date_col, target_col)
    # keep original date string
    train[date_col + "_str"] = train[date_col].astype(str)
    train[date_col] = pd.to_datetime(train[date_col])
    train["DOW"] = train[date_col].dt.weekday  # Monday=0
    train["series_id"] = train[series_cols[0]].astype(str) + "_" + train[series_cols[1]].astype(str)
    # sort
    train = train.sort_values([ "series_id", date_col ]).reset_index(drop=True)

    tests = {}
    for p in glob.glob(os.path.join(test_dir, "TEST_*.csv")):
        df = pd.read_csv(p)
        maybe_split_series(df, series_cols)
        _ensure_columns(df, series_cols, date_col, None)
        df[date_col + "_str"] = df[date_col].astype(str)
        df[date_col] = pd.to_datetime(df[date_col])
        df["DOW"] = df[date_col].dt.weekday
        df["series_id"] = df[series_cols[0]].astype(str) + "_" + df[series_cols[1]].astype(str)
        df = df.sort_values(["series_id", date_col]).reset_index(drop=True)
        tests[os.path.basename(p)] = df
    if not tests:
        raise FileNotFoundError(f"No TEST_*.csv found in {test_dir}")
    return Dataset(train=train, tests=tests)

def cutoff_train(train: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    return train.loc[train["영업일자"] < cutoff_date].copy()

def future_dates(df_test: pd.DataFrame, date_col: str) -> List[pd.Timestamp]:
    # expects the test file to contain exactly the H future dates per (series, ...). We use the unique sorted dates.
    return sorted(df_test[date_col].unique())

def basic_calendar(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    dt = pd.to_datetime(df[date_col])
    out["DOW"] = dt.dt.weekday
    # cyclical encodings if needed
    out["woy"] = dt.dt.isocalendar().week.astype(int)
    out["woy_sin"] = np.sin(2 * np.pi * out["woy"] / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * out["woy"] / 52.0)
    out["month"] = dt.dt.month.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out
