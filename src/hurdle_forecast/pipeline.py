from __future__ import annotations
from typing import Dict
import os
import numpy as np
import pandas as pd

from .config import Config
from .data import load_datasets, cutoff_train, future_dates
from .classifier import beta_smooth_probs, logistic_global_calendar
from .intensity import forecast_intensity
from .combine import combine_expectation, fill_submission_skeleton
from .mps_utils import to_numpy


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

    # Pre-compute global positives (for p99 cap) from full train
    train_pos = ds.train.loc[ds.train[cfg.target_col] > 0, cfg.target_col].values

    models: Dict[str, Dict] = {"train_pos": train_pos, "files": {}}

    for fname, df_test in ds.tests.items():
        test_dates = future_dates(df_test, cfg.date_col)
        cutoff_date = min(test_dates)
        train_cut = cutoff_train(ds.train, cutoff_date)

        file_models: Dict[str, Dict] = {"series": {}}

        # if logistic classifier is selected, fit once globally per test file
        if cfg.classifier_kind == "logit":
            fut_cal = df_test[[cfg.date_col, "DOW", "series_id"]].copy()
            P_all = logistic_global_calendar(
                train_cut=train_cut,
                future_calendar=fut_cal,
                lr=cfg.logit_lr,
                epochs=cfg.logit_epochs,
                l2=cfg.logit_l2,
                batch_size=cfg.logit_batch_size,
            )
            df_test = df_test.copy()
            df_test["P_nonzero"] = to_numpy(P_all)

        for sid, tdf in df_test.groupby("series_id"):
            fut_dates = tdf[cfg.date_col].tolist()
            fut_dows = tdf["DOW"].tolist()

            if cfg.classifier_kind == "beta":
                P = beta_smooth_probs(
                    train_cut=train_cut,
                    series_id=sid,
                    future_dows=fut_dows,
                    window_weeks=cfg.dow_window_weeks,
                    alpha=cfg.beta_alpha,
                    beta=cfg.beta_beta,
                    date_col=cfg.date_col,
                    target_col=cfg.target_col,
                )
            else:
                P = tdf["P_nonzero"].values

            mu = forecast_intensity(
                train_cut=train_cut,
                series_id=sid,
                future_dates=fut_dates,
                m=cfg.seasonal_m,
                grid=cfg.sarima_grid,
                val_weeks=cfg.val_weeks,
                fallback=cfg.fallback,
                target_col=cfg.target_col,
            )

            out = tdf[[*cfg.series_cols, cfg.date_col + "_str"]].copy()
            out.rename(columns={cfg.date_col + "_str": cfg.date_col}, inplace=True)

            file_models["series"][sid] = {"P": P, "mu": mu, "out": out}

        models["files"][fname] = file_models

    return models


def predict_with_models(cfg: Config, models: Dict[str, Dict]):
    """Generate predictions using trained models and write outputs."""
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_pos = models["train_pos"]

    for fname, fmods in models["files"].items():
        preds = []
        for sid, smod in fmods["series"].items():
            P = smod["P"]
            mu = smod["mu"]
            out = smod["out"].copy()
            yhat = combine_expectation(P, mu, cfg.cap_quantile, train_positive=train_pos)
            out["예측값"] = yhat
            preds.append(out)

        pred_df = pd.concat(preds, axis=0, ignore_index=True)
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
            filled = fill_submission_skeleton(
                skel,
                pred_all,
                date_col=cfg.date_col,
                series_cols=cfg.series_cols,
                value_col="예측값",
            )
            filled_path = os.path.join(cfg.out_dir, "submission_filled.csv")
            filled.to_csv(filled_path, index=False, encoding="utf-8-sig")


def run_forecast(cfg: Config):
    """Backward compatible wrapper for CLI entry point."""
    models = train_models(cfg)
    predict_with_models(cfg, models)
