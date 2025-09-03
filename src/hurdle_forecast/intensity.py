from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Sequence, Union
import warnings
import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _make_exog_dow(idx: pd.DatetimeIndex) -> pd.DataFrame:
    dow = idx.weekday
    exog = pd.get_dummies(dow, prefix="dow", drop_first=False)
    exog.index = idx
    return exog


def _aicc(aic: float, n: int, k: int) -> float:
    # n = observations used, k = number of parameters
    if n - k - 1 <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def _candidate_orders(grid: str = "full"):
    Ps = [0, 1]
    Ds = [0, 1]
    Qs = [0, 1]
    ps = [0, 1]
    ds = [0, 1]
    qs = [0, 1]
    if grid == "small":
        # small but diverse subset
        seasonal = [(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0)]
        nonseasonal = [(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
        return [
            (p, d, q, P, D, Q) for (p, d, q) in nonseasonal for (P, D, Q) in seasonal
        ]
    # limit the search to lower-order combinations to avoid unstable models
    return [
        (p, d, q, P, D, Q)
        for p in ps
        for d in ds
        for q in qs
        for P in Ps
        for D in Ds
        for Q in Qs
        if (p + q) <= 1 and (P + Q) <= 1
    ]


def _fit_sarimax(
    y_log: pd.Series, exog: Optional[pd.DataFrame], order, seasonal_order, m: int
):
    # suppress convergence warnings for quick grid search
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_log,
            order=order,
            seasonal_order=(*seasonal_order, m),
            exog=exog,
            enforce_stationarity=True,
            enforce_invertibility=True,
            initialization="approximate_diffuse",
            # missing='drop' is implicit for NaNs
        )
        res = model.fit(disp=False, maxiter=200)
    return res


def _seasonal_naive(y: pd.Series, horizon: int, m: int = 7) -> np.ndarray:
    if len(y) >= m:
        last_week = y.iloc[-m:]
        reps = int(np.ceil(horizon / m))
        pred = np.tile(last_week.values, reps)[:horizon]
        return pred
    # if shorter than m, repeat last value
    return np.repeat(y.iloc[-1], horizon) if len(y) else np.zeros(horizon)


def _forecast_intensity_single(
    train_cut: pd.DataFrame,
    series_id: str,
    future_dates: List[pd.Timestamp],
    m: int = 7,
    grid: str = "full",
    val_weeks: int = 4,
    fallback: str = "ets",
    target_col: str = "매출수량",
) -> np.ndarray:
    """Original single-series intensity forecast implementation."""
    sdf = train_cut.loc[train_cut["series_id"] == series_id].copy()
    if sdf.empty:
        return np.zeros(len(future_dates))

    # Build time series with date index
    y = sdf.set_index("영업일자")[target_col].astype(float).sort_index()
    # Mask zeros as NaN to model only positive magnitudes while keeping regular spacing
    y_log = np.log1p(y.where(y > 0, np.nan))

    # If too short, fallback immediately
    if y_log.notna().sum() < max(10, 2 * m):
        if fallback == "ets":
            try:
                hw = ExponentialSmoothing(
                    y.where(y > 0, np.nan).dropna(),
                    trend=None,
                    seasonal="add",
                    seasonal_periods=m,
                )
                hw_fit = hw.fit(optimized=True, use_brute=False)
                pred = hw_fit.forecast(len(future_dates)).values
                return np.maximum(pred, 0.0)
            except Exception:
                pass
        # seasonal naive fallback
        return _seasonal_naive(y, horizon=len(future_dates), m=m)

    # Candidate orders
    cands = _candidate_orders(grid=grid)

    exog = _make_exog_dow(y.index)
    exog_future = _make_exog_dow(pd.DatetimeIndex(future_dates))

    best = {"aicc": np.inf, "res": None, "order": None, "sorder": None}
    nobs = int(y_log.notna().sum())

    for p, d, q, P, D, Q in cands:
        try:
            res = _fit_sarimax(y_log, exog.loc[y_log.index], (p, d, q), (P, D, Q), m)
            k = res.params.size
            aicc = _aicc(res.aic, nobs, k)
            if aicc < best["aicc"]:
                best = {
                    "aicc": aicc,
                    "res": res,
                    "order": (p, d, q),
                    "sorder": (P, D, Q),
                }
        except Exception:
            continue

    if best["res"] is None:
        # fallbacks
        if fallback == "ets":
            try:
                hw = ExponentialSmoothing(
                    y.where(y > 0, np.nan).dropna(),
                    trend=None,
                    seasonal="add",
                    seasonal_periods=m,
                )
                hw_fit = hw.fit(optimized=True, use_brute=False)
                pred = hw_fit.forecast(len(future_dates)).values
                return np.maximum(pred, 0.0)
            except Exception:
                pass
        return _seasonal_naive(y, horizon=len(future_dates), m=m)

    # Forecast on future horizon
    res = best["res"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = res.get_forecast(steps=len(future_dates), exog=exog_future)
        mu_log = fc.predicted_mean
    mu_log_max = mu_log.max()
    if mu_log_max > 100:
        logging.getLogger(__name__).warning(
            "mu_log max %s exceeds threshold for series_id %s",
            mu_log_max,
            series_id,
        )
    mu_log = np.clip(mu_log.values, -700, 700)
    mu = np.expm1(mu_log)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0)
    mu = np.maximum(mu, 0.0)  # clip negatives and handle infs
    return mu


def forecast_intensity(
    train_cut: pd.DataFrame,
    series_id: Union[str, Sequence[str]],
    future_dates: Union[List[pd.Timestamp], Sequence[List[pd.Timestamp]]],
    m: int = 7,
    grid: str = "full",
    val_weeks: int = 4,
    fallback: str = "ets",
    target_col: str = "매출수량",
    batch_size: int = 128,
) -> np.ndarray:
    """Return mu_t = E[y | y>0] for each future date.

    Supports single-series input for backward compatibility or lists of
    ``series_id``/``future_dates`` for simple batch processing.  When a batch is
    provided the output has shape ``(n_series, horizon)``.
    """
    if isinstance(series_id, (list, tuple)):
        mus = []
        for sid, fdates in zip(series_id, future_dates):
            mus.append(
                _forecast_intensity_single(
                    train_cut,
                    sid,
                    list(fdates),
                    m=m,
                    grid=grid,
                    val_weeks=val_weeks,
                    fallback=fallback,
                    target_col=target_col,
                )
            )
        return np.stack(mus, axis=0)

    return _forecast_intensity_single(
        train_cut,
        series_id,
        list(future_dates),
        m=m,
        grid=grid,
        val_weeks=val_weeks,
        fallback=fallback,
        target_col=target_col,
    )


def _forecast_intensity_gpu_single(
    train_cut: pd.DataFrame,
    series_id: str,
    future_dates: List[pd.Timestamp],
    m: int = 7,
    target_col: str = "매출수량",
) -> "cp.ndarray":
    try:
        from cuml.tsa.arima import ARIMA as cuARIMA  # type: ignore
        import cupy as cp
    except Exception as e:  # pragma: no cover - depends on optional GPU libs
        raise RuntimeError("cuml ARIMA not available") from e

    sdf = train_cut.loc[train_cut["series_id"] == series_id].copy()
    if sdf.empty:
        return cp.zeros(len(future_dates))
    y = (
        sdf.set_index("영업일자")[target_col]
        .astype(float)
        .sort_index()
        .fillna(0.0)
    )

    if (y > 0).sum() < max(10, 2 * m):
        if (y > 0).sum() > 0:
            try:
                hw = ExponentialSmoothing(
                    y.where(y > 0, np.nan).dropna(),
                    trend=None,
                    seasonal="add",
                    seasonal_periods=m,
                )
                hw_fit = hw.fit(optimized=True, use_brute=False)
                pred = hw_fit.forecast(len(future_dates)).values
                return cp.asarray(np.maximum(pred, 0.0))
            except Exception:
                pass
        return cp.asarray(_seasonal_naive(y, horizon=len(future_dates), m=m))

    arr = cp.asarray(y.values)
    arr = cp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = cp.where(arr > 0, arr, 0)
    y_log = cp.log1p(arr)

    model = cuARIMA(y_log, order=(1, 0, 0), seasonal_order=(0, 0, 0, m))
    model.fit()
    # cuml's ARIMA forecasts may return a 2D array with shape (1, steps).
    # Flatten to 1D so downstream stacking yields (n_series, horizon).
    fc = model.forecast(len(future_dates)).reshape(-1)
    mu = cp.expm1(fc)
    mu = cp.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    mu = cp.maximum(mu, 0.0)
    return mu


def forecast_intensity_gpu(
    train_cut: pd.DataFrame,
    series_id: Union[str, Sequence[str]],
    future_dates: Union[List[pd.Timestamp], Sequence[List[pd.Timestamp]]],
    m: int = 7,
    grid: str = "full",
    val_weeks: int = 4,
    fallback: str = "ets",
    target_col: str = "매출수량",
    batch_size: int = 128,
) -> "cp.ndarray":
    """GPU-accelerated alternative to :func:`forecast_intensity`.

    Always returns a 2D array with shape ``(n_series, horizon)``.  Single-series
    inputs are supported and will return an array where ``n_series`` is 1.
    """
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover - depends on optional GPU libs
        raise RuntimeError("cuml ARIMA not available") from e

    if isinstance(series_id, (list, tuple)):
        mus = []
        for sid, fdates in zip(series_id, future_dates):
            mus.append(
                _forecast_intensity_gpu_single(
                    train_cut,
                    sid,
                    list(fdates),
                    m=m,
                    target_col=target_col,
                )
            )
    else:
        mus = [
            _forecast_intensity_gpu_single(
                train_cut,
                series_id,
                list(future_dates),
                m=m,
                target_col=target_col,
            )
        ]
    # Stack along axis 0 so output is (n_series, horizon)
    return cp.stack(mus, axis=0)
