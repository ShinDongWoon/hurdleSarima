from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

def combine_expectation(P: np.ndarray, mu: np.ndarray, cap: Optional[float], train_positive: Optional[np.ndarray] = None) -> np.ndarray:
    yhat = P * mu
    yhat = np.maximum(yhat, 0.0)
    if cap is not None and train_positive is not None and len(train_positive) > 0:
        thresh = np.quantile(train_positive, cap)
        yhat = np.minimum(yhat, thresh)
    return yhat

def fill_submission_skeleton(skel: pd.DataFrame, pred_df: pd.DataFrame, date_col: str, series_cols: tuple, value_col: str) -> pd.DataFrame:
    # Try joins by id if present
    if "id" in skel.columns and "id" in pred_df.columns:
        out = skel.merge(pred_df[["id", value_col]], on="id", how="left")
        return out
    # Otherwise join on keys
    keys = [*series_cols, date_col]
    out = skel.merge(pred_df[keys + [value_col]], on=keys, how="left")
    # preserve original order of skeleton
    out = out.loc[skel.index]
    return out
