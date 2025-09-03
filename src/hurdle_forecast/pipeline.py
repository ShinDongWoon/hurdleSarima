from __future__ import annotations
from typing import Dict, Optional
import os
import numpy as np
import pandas as pd

from .config import Config
from .data import load_datasets, cutoff_train
from .classifier import (
    beta_smooth_probs,
    logistic_global_calendar,
    _series_dow_lookup,
)
from .intensity import forecast_intensity
from .combine import combine_expectation, fill_submission_skeleton
from .mps_utils import to_numpy


def smooth_probs(P: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    """Blend predicted probabilities ``P`` with ``prior`` using weight ``lam``.

    Parameters
    ----------
    P : np.ndarray
        Array of predicted probabilities.
    prior : np.ndarray
        Array of prior probabilities to blend with ``P``. Must have the same
        shape as ``P``.
    lam : float
        Blending weight. ``lam=1`` keeps ``P`` unchanged, ``lam=0`` returns the
        prior.

    Returns
    -------
    np.ndarray
        Blended probabilities clipped to ``[0, 1]``.
    """

    blended = lam * P + (1.0 - lam) * prior
    return np.clip(blended, 0.0, 1.0)


def train_models(cfg: Config) -> Dict[str, Dict]:
    """Train classifier/intensity models for each test file.

    Returns a nested dictionary with per-file and per-series outputs
    required for prediction.
    """
    ds = load_datasets(
        cfg.train_csv,
        cfg.test_dir,
        cfg.series_cols,
        cfg.date_col,
        cfg.target_col,
        cfg.clip_sales_quantile,
    )

    models: Dict[str, Dict] = {"files": {}}

    for fname, df_test in ds.tests.items():
        cutoff_date = df_test[cfg.date_col].min()
        train_cut = cutoff_train(ds.train, cutoff_date)
        train_full = pd.concat([train_cut, df_test], ignore_index=True)

        file_models: Dict[str, Dict] = {"series": {}}

        # build future calendar for next 7 days per series
        fut_parts = []
        for sid in df_test["series_id"].unique():
            last_date = train_full.loc[train_full["series_id"] == sid, cfg.date_col].max()
            fdates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
            fut_parts.append(
                pd.DataFrame(
                    {
                        cfg.date_col: fdates,
                        "DOW": [d.weekday() for d in fdates],
                        "series_id": sid,
                    }
                )
            )
        fut_cal = pd.concat(fut_parts, ignore_index=True)

        # if logistic classifier is selected, fit once globally per test file
        lookup_prior = None
        if cfg.classifier_kind == "logit":
            P_all = logistic_global_calendar(
                train_cut=train_full,
                future_calendar=fut_cal,
                lr=cfg.logit_lr,
                epochs=cfg.logit_epochs,
                l2=cfg.logit_l2,
                batch_size=cfg.logit_batch_size,
                window_weeks=cfg.dow_window_weeks,
                alpha=cfg.beta_alpha,
                beta=cfg.beta_beta,
                calib_lambda=cfg.calib_lambda,
                class_weight=cfg.class_weight,
            )
            fut_cal = fut_cal.copy()
            fut_cal["P_nonzero"] = to_numpy(P_all)
        else:
            lookup_prior = _series_dow_lookup(
                train_full,
                window_weeks=cfg.dow_window_weeks,
                alpha=cfg.beta_alpha,
                beta=cfg.beta_beta,
                date_col=cfg.date_col,
                target_col=cfg.target_col,
            )

        for sid, tdf in df_test.groupby("series_id"):
            sc = fut_cal.loc[fut_cal["series_id"] == sid]
            fut_dates = sc[cfg.date_col].tolist()
            fut_dows = sc["DOW"].tolist()

            if cfg.classifier_kind == "beta":
                P_raw = beta_smooth_probs(
                    train_cut=train_full,
                    series_id=sid,
                    future_dows=fut_dows,
                    window_weeks=cfg.dow_window_weeks,
                    alpha=cfg.beta_alpha,
                    beta=cfg.beta_beta,
                    date_col=cfg.date_col,
                    target_col=cfg.target_col,
                )
                prior = np.array([lookup_prior(sid, d) for d in fut_dows])
                P = smooth_probs(P_raw, prior, cfg.calib_lambda)
            else:
                P = sc["P_nonzero"].values

            mu = forecast_intensity(
                train_cut=train_full,
                series_id=sid,
                future_dates=fut_dates,
                grid=cfg.sarima_grid,
                val_weeks=cfg.val_weeks,
                fallback=cfg.fallback,
                target_col=cfg.target_col,
                batch_size=cfg.intensity_batch_size,
            )

            # p99 cap of positive sales per (series_id, DOW)
            if cfg.cap_quantile is not None:
                train_pos = train_cut[
                    (train_cut["series_id"] == sid)
                    & (train_cut[cfg.target_col] > 0)
                ]
                if len(train_pos) > 0:
                    q = (
                        train_pos.assign(DOW=train_pos[cfg.date_col].dt.weekday)
                        .groupby("DOW")[cfg.target_col]
                        .quantile(cfg.cap_quantile)
                    )
                    caps = np.array([q.get(d, 0.0) for d in fut_dows])
                else:
                    caps = np.zeros(len(fut_dows))
            else:
                caps = None

            fut_out = pd.DataFrame({cfg.date_col: [d.strftime("%Y-%m-%d") for d in fut_dates]})
            for col in cfg.series_cols:
                fut_out[col] = tdf.iloc[0][col]
            fut_out = fut_out[[*cfg.series_cols, cfg.date_col]]

            file_models["series"][sid] = {"P": P, "mu": mu, "cap": caps, "out": fut_out}

        models["files"][fname] = file_models

    return models


def predict_with_models(cfg: Config, models: Dict[str, Dict]) -> Optional[pd.DataFrame]:
    """Generate predictions using trained models and write outputs.

    Returns a filled wide-format submission DataFrame when a sample
    submission is supplied via ``cfg.sample_submission``. Otherwise ``None``
    is returned."""
    os.makedirs(cfg.out_dir, exist_ok=True)

    for fname, fmods in models["files"].items():
        preds = []
        for sid, smod in fmods["series"].items():
            P = smod["P"]
            mu = smod["mu"]
            out = smod["out"].copy()
            yhat = combine_expectation(P, mu, smod.get("cap"))
            out["예측값"] = yhat
            # enforce column ordering for downstream compatibility
            out = out[[*cfg.series_cols, cfg.date_col, "예측값"]]
            preds.append(out)

        pred_df = pd.concat(preds, axis=0, ignore_index=True)
        pred_df = pred_df[[*cfg.series_cols, cfg.date_col, "예측값"]]
        out_path = os.path.join(cfg.out_dir, f"pred_{fname}")
        pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    if cfg.sample_submission is not None:
        skel = pd.read_csv(cfg.sample_submission)
        all_preds = []
        for p in sorted(os.listdir(cfg.out_dir)):
            if p.startswith("pred_TEST_") and p.endswith(".csv"):
                all_preds.append(pd.read_csv(os.path.join(cfg.out_dir, p)))
        if all_preds:
            pred_all = pd.concat(all_preds, ignore_index=True)
            pred_all = pred_all[[*cfg.series_cols, cfg.date_col, "예측값"]]
            filled = fill_submission_skeleton(
                skel,
                pred_all,
                date_col=cfg.date_col,
                series_cols=cfg.series_cols,
                value_col="예측값",
            )
            filled_path = os.path.join(cfg.out_dir, "submission_filled.csv")
            filled.to_csv(filled_path, index=False, encoding="utf-8-sig")
            return filled

    return None


def run_forecast(cfg: Config):
    """Backward compatible wrapper for CLI entry point."""
    models = train_models(cfg)
    predict_with_models(cfg, models)
