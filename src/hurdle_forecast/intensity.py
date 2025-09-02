from __future__ import annotations
from typing import Tuple, Optional, List, Dict
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


def forecast_intensity(
    train_cut: pd.DataFrame,
    series_id: str,
    future_dates: List[pd.Timestamp],
    m: int = 7,
    grid: str = "full",
    val_weeks: int = 4,
    fallback: str = "ets",
    target_col: str = "매출수량",
) -> np.ndarray:
    """Return mu_t = E[y | y>0] for each future date.
    Fit SARIMAX on log1p(y) with zeros set to NaN (no leakage), seasonal period m.
    Fallback to ETS or SeasonalNaive when data is too short or SARIMA unstable.
    """
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
