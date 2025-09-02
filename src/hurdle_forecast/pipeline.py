from __future__ import annotations
from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd

from .config import Config
from .data import load_datasets, cutoff_train, future_dates
from .classifier import beta_smooth_probs, logistic_global_calendar
from .intensity import forecast_intensity
from .combine import combine_expectation, fill_submission_skeleton

def run_forecast(cfg: Config):
    ds = load_datasets(cfg.train_csv, cfg.test_dir, cfg.series_cols, cfg.date_col, cfg.target_col)

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Pre-compute global positives (for p99 cap) from full train
    train_pos = ds.train.loc[ds.train[cfg.target_col] > 0, cfg.target_col].values

    for fname, df_test in ds.tests.items():
        # Cutoff is strictly before first test date in this file
        test_dates = future_dates(df_test, cfg.date_col)
        cutoff_date = min(test_dates)
        train_cut = cutoff_train(ds.train, cutoff_date)

        # Prepare outputs
        preds = []

        # if logistic classifier is selected, fit once globally per test file
        if cfg.classifier_kind == "logit":
            # Build future calendar rows aligned with df_test rows
            fut_cal = df_test[[cfg.date_col, "DOW", "series_id"]].copy()
            P_all = logistic_global_calendar(
                train_cut=train_cut,
                future_calendar=fut_cal,
                lr=cfg.logit_lr,
                epochs=cfg.logit_epochs,
                l2=cfg.logit_l2,
                batch_size=cfg.logit_batch_size,
            )
            # attach back to df_test by row
            df_test = df_test.copy()
            df_test["P_nonzero"] = P_all

        # Iterate per series present in this test file
        for sid, tdf in df_test.groupby("series_id"):
            fut_dates = tdf[cfg.date_col].tolist()
            fut_dows = tdf["DOW"].tolist()

            # 1) P_t
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
                # already computed globally; slice by index
                P = tdf["P_nonzero"].values

            # 2) mu_t
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

            # 3) combine
            yhat = combine_expectation(P, mu, cfg.cap_quantile, train_positive=train_pos)

            out = tdf[[*cfg.series_cols, cfg.date_col + "_str"]].copy()
            out.rename(columns={cfg.date_col + "_str": cfg.date_col}, inplace=True)  # keep original string
            out["예측값"] = yhat
            preds.append(out)

        pred_df = pd.concat(preds, axis=0, ignore_index=True)

        # write outputs
        out_path = os.path.join(cfg.out_dir, f"pred_{fname}")
        pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # If a sample submission is provided, also produce a filled skeleton using the concatenated predictions
    if cfg.sample_submission is not None:
        skel = pd.read_csv(cfg.sample_submission)
        # read back all per-file outputs and concat
        all_preds = []
        for p in sorted(os.listdir(cfg.out_dir)):
            if p.startswith("pred_TEST_") and p.endswith(".csv"):
                all_preds.append(pd.read_csv(os.path.join(cfg.out_dir, p)))
        if all_preds:
            pred_all = pd.concat(all_preds, ignore_index=True)
            filled = fill_submission_skeleton(skel, pred_all, date_col=cfg.date_col, series_cols=cfg.series_cols, value_col="예측값")
            filled_path = os.path.join(cfg.out_dir, "submission_filled.csv")
            filled.to_csv(filled_path, index=False, encoding="utf-8-sig")
