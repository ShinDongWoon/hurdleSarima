from __future__ import annotations
import os
import glob
import logging
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np

@dataclass
class Dataset:
    train: pd.DataFrame
    tests: Dict[str, pd.DataFrame]  # map filename -> df
    winsor_limits: pd.DataFrame | None = None

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


def clean_sales(df: pd.DataFrame, target_col: str, quantile: float) -> pd.DataFrame:
    """Replace missing or negative values with 0 and clip extreme positives.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the target column.
    target_col : str
        Name of the sales column.
    quantile : float
        Upper-tail quantile for clipping.
    """
    logger = logging.getLogger(__name__)
    neg_count = df[target_col].lt(0).sum()
    zero_count = df[target_col].eq(0).sum()
    if neg_count:
        logger.warning("%d negative %s values clipped to 0", neg_count, target_col)
    logger.info("%d zero %s values", zero_count, target_col)

    df[target_col] = df[target_col].fillna(0)
    df[target_col] = df[target_col].clip(lower=0)
    upper = df[target_col].quantile(quantile)
    df[target_col] = df[target_col].clip(lower=0, upper=upper)
    return df


def winsorize_by_dow(
    df: pd.DataFrame,
    target_col: str,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize ``target_col`` per (series_id, DOW).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``series_id`` and ``DOW`` columns.
    target_col : str
        Column to clip.
    lower : float, optional
        Lower quantile. Default 0.01.
    upper : float, optional
        Upper quantile. Default 0.99.

    Returns
    -------
    pd.DataFrame
        Mapping of quantile thresholds with columns
        ``['series_id', 'DOW', 'lower', 'upper']``.
    """
    q = (
        df.groupby(["series_id", "DOW"])[target_col]
        .quantile([lower, upper])
        .unstack()
        .rename(columns={lower: "lower", upper: "upper"})
        .reset_index()
    )
    df_limits = df.merge(q, on=["series_id", "DOW"], how="left")
    df[target_col] = df_limits[target_col].clip(df_limits["lower"], df_limits["upper"])
    return q

def load_datasets(
    train_csv: str,
    test_dir: str,
    series_cols: Tuple[str, str],
    date_col: str,
    target_col: str,
    clip_sales_quantile: float,
) -> Dataset:
    train = pd.read_csv(train_csv)
    maybe_split_series(train, series_cols)
    _ensure_columns(train, series_cols, date_col, target_col)
    clean_sales(train, target_col, clip_sales_quantile)
    train[date_col + "_str"] = train[date_col].astype(str)
    train[date_col] = pd.to_datetime(train[date_col])
    train["DOW"] = train[date_col].dt.weekday  # Monday=0

    tmp = train.copy()
    tmp["series_id"] = tmp[series_cols[0]].astype(str) + "_" + tmp[series_cols[1]].astype(str)
    winsor_limits = winsorize_by_dow(tmp, target_col)
    train[target_col] = tmp[target_col]
    train["series_id"] = tmp["series_id"]
    train = train.sort_values(["series_id", date_col]).reset_index(drop=True)

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
    return Dataset(train=train, tests=tests, winsor_limits=winsor_limits)

def cutoff_train(train: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    return train.loc[train["영업일자"] < cutoff_date].copy()


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
