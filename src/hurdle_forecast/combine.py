from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional cupy
    import cupy as cp
except ImportError:  # pragma: no cover - optional cupy
    cp = None


def _as_same_backend(x, reference):
    """Cast ``x`` to the same backend/type as ``reference`` if possible."""
    # Torch tensor
    try:  # pragma: no cover - optional torch
        import torch
        if isinstance(reference, torch.Tensor) and not isinstance(x, torch.Tensor):
            return torch.as_tensor(x, device=reference.device, dtype=reference.dtype)
    except Exception:
        pass
    # CuPy array
    try:  # pragma: no cover - optional cupy
        import cupy as cp
        if isinstance(reference, cp.ndarray) and not isinstance(x, cp.ndarray):
            return cp.asarray(x)
    except Exception:
        pass
    return x


def combine_expectation(
    P,
    mu,
    cap_values: Optional[np.ndarray] = None,
):
    """Combine nonzero probability ``P`` and intensity ``mu``.

    Parameters
    ----------
    P : array-like
        Probability of a non-zero sale.
    mu : array-like
        Forecasted intensity of sales when non-zero.
    cap_values : array-like, optional
        Element-wise cap applied to the expectation ``P * mu``. The array is
        broadcast against the resulting expectation. ``None`` disables
        capping.

    Supports :class:`torch.Tensor` and CuPy arrays; operations are carried out
    on the device of the inputs. Other array types are not supported.
    """

    # Torch branch ---------------------------------------------------------
    try:  # pragma: no cover - torch optional
        import torch
        if isinstance(P, torch.Tensor) or isinstance(mu, torch.Tensor):
            P = torch.as_tensor(P, dtype=torch.float32, device=getattr(P, "device", None))
            mu = _as_same_backend(mu, P)
            yhat = P * mu
            zero = torch.tensor(0.0, device=yhat.device, dtype=yhat.dtype)
            yhat = torch.maximum(yhat, zero)
            if cap_values is not None:
                caps = _as_same_backend(cap_values, yhat)
                yhat = torch.minimum(yhat, caps)
            return yhat
    except Exception:
        pass

    # CuPy branch ----------------------------------------------------------
    if cp is not None and (
        isinstance(P, cp.ndarray) or isinstance(mu, cp.ndarray)
    ):
        P = cp.asarray(P)
        mu = _as_same_backend(mu, P)
        if not isinstance(P, cp.ndarray) or not isinstance(mu, cp.ndarray):
            raise TypeError("P and mu must be CuPy arrays")
        try:
            np.broadcast_shapes(P.shape, mu.shape)
        except ValueError as e:
            raise ValueError("P and mu must have broadcastable shapes") from e
        yhat = P * mu
        yhat = cp.maximum(yhat, 0.0)
        if cap_values is not None:
            caps = _as_same_backend(cap_values, yhat)
            yhat = cp.minimum(yhat, caps)
        return yhat

    raise TypeError("Unsupported array types for P and mu")

def fill_submission_skeleton(
    skel: pd.DataFrame,
    pred_df: pd.DataFrame,
    date_col: str,
    series_cols: tuple,
    value_col: str,
) -> pd.DataFrame:
    """Fill a sample submission skeleton with prediction values.

    The helper supports two skeleton layouts:

    1. **Long format** with explicit ``series_cols`` and a value column.
       Predictions are merged on ``date_col`` and ``series_cols``.
    2. **Wide format** with one column per series and one row per date.
       In this case ``series_cols`` are absent from ``skel`` and the
       predictions are pivoted so they align with the existing series
       columns without adding a new ``value_col`` field.
    """

    # Try joins by id if present -------------------------------------------
    if "id" in skel.columns and "id" in pred_df.columns:
        out = skel.merge(pred_df[["id", value_col]], on="id", how="left")
        return out

    # Detect wide-format skeleton (no series_cols present) -----------------
    if set(series_cols).isdisjoint(skel.columns):
        pred_df = pred_df.copy()
        if len(series_cols) > 1:
            pred_df["series_id"] = (
                pred_df[list(series_cols)].astype(str).agg("_".join, axis=1)
            )
        else:
            pred_df["series_id"] = pred_df[series_cols[0]].astype(str)
        wide = (
            pred_df.pivot(index=date_col, columns="series_id", values=value_col)
            .sort_index()
        )
        out = skel.set_index(date_col)
        out.loc[wide.index, wide.columns] = wide
        out = out.reset_index()
        return out

    # Otherwise join on keys (long-format skeleton) ------------------------
    keys = [*series_cols, date_col]
    out = skel.merge(pred_df[keys + [value_col]], on=keys, how="left")
    # preserve original order of skeleton
    out = out.loc[skel.index]
    return out
