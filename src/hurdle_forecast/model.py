from __future__ import annotations

"""GPU-only hurdle forecasting utilities.

This module requires a CUDA-capable GPU with CuPy and cuML installed.
CPU execution paths have been removed.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import os
import pickle
import logging
import pandas as pd
import numpy as np

from .config import Config
from .data import cutoff_train, maybe_split_series, clean_sales
from .classifier import beta_smooth_probs, logistic_global_calendar
from .intensity import forecast_intensity_gpu
from .combine import combine_expectation, fill_submission_skeleton
from .mps_utils import gpu_available, to_numpy


def _prepare_train(cfg: Config) -> pd.DataFrame:
    """Load and preprocess the training CSV similar to `load_datasets`."""
    use_cudf = False
    try:  # pragma: no cover - optional GPU libs
        import cudf  # type: ignore
        gdf = cudf.read_csv(cfg.train_csv)
        series_cols = cfg.series_cols
        date_col = cfg.date_col
        target_col = cfg.target_col
        maybe_split_series(gdf, series_cols)
        missing = [c for c in [*series_cols, date_col, target_col] if c not in gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns in train data: {missing}")
        gdf[date_col + "_str"] = gdf[date_col].astype(str)
        gdf[date_col] = cudf.to_datetime(gdf[date_col])
        gdf["DOW"] = gdf[date_col].dt.weekday
        gdf["series_id"] = gdf[series_cols[0]].astype(str) + "_" + gdf[series_cols[1]].astype(str)
        gdf = gdf.sort_values(["series_id", date_col]).reset_index(drop=True)
        train = gdf.to_pandas()
        use_cudf = True
    except Exception:
        train = pd.read_csv(cfg.train_csv)
        series_cols = cfg.series_cols
        date_col = cfg.date_col
        target_col = cfg.target_col
        maybe_split_series(train, series_cols)
        missing = [c for c in [*series_cols, date_col, target_col] if c not in train.columns]
        if missing:
            raise ValueError(f"Missing required columns in train data: {missing}")
        train[date_col + "_str"] = train[date_col].astype(str)
        train[date_col] = pd.to_datetime(train[date_col])
        train["DOW"] = train[date_col].dt.weekday
        train["series_id"] = train[series_cols[0]].astype(str) + "_" + train[series_cols[1]].astype(str)
        train = train.sort_values(["series_id", date_col]).reset_index(drop=True)
    return train


def _to_placeholder_dates(df: pd.DataFrame, date_col: str, test_id: str) -> pd.DataFrame:
    """Map actual dates in ``date_col`` to placeholder strings for submissions.

    The sample submission provided by the competition uses placeholder date
    labels such as ``TEST_00+1일`` instead of real calendar dates.  This helper
    replaces the dates in ``df`` with those placeholders so that predictions
    can be merged back into the skeleton without key errors.
    """

    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    unique_dates = sorted(pd.unique(dates))
    mapping = {d: f"{test_id}+{i + 1}일" for i, d in enumerate(unique_dates)}
    out[date_col] = dates.map(mapping)
    return out


@dataclass
class HurdleForecastModel:
    cfg: Config
    train: pd.DataFrame

    @classmethod
    def from_config(cls, cfg: Config) -> "HurdleForecastModel":
        train = _prepare_train(cfg)
        return cls(cfg=cfg, train=train)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HurdleForecastModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_dir(
        self, test_dir: str, out_dir: str, sample_submission: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Run forecasts for all TEST_*.csv in ``test_dir`` and write outputs.

        If ``sample_submission`` is provided, a filled wide-format submission
        DataFrame is returned in addition to being written to ``out_dir``.
        """
        os.makedirs(out_dir, exist_ok=True)
        series_cols = self.cfg.series_cols
        date_col = self.cfg.date_col
        target_col = self.cfg.target_col

        if not gpu_available():
            raise RuntimeError(
                "GPU not detected. Install CuPy/cuML and ensure a CUDA-capable device is available."
            )

        files = sorted(os.listdir(test_dir))
        total = len(files)
        for idx, fname in enumerate(files, 1):
            logging.info("(%d/%d) Processing %s", idx, total, fname)
            if not (fname.startswith("TEST_") and fname.endswith(".csv")):
                continue
            path = os.path.join(test_dir, fname)
            df_test = pd.read_csv(path)
            maybe_split_series(df_test, series_cols)
            missing = [c for c in [*series_cols, date_col, target_col] if c not in df_test.columns]
            if missing:
                raise ValueError(f"Missing required columns in test data {fname}: {missing}")
            clean_sales(df_test, target_col, self.cfg.clip_sales_quantile)
            df_test[date_col + "_str"] = df_test[date_col].astype(str)
            df_test[date_col] = pd.to_datetime(df_test[date_col])
            df_test["DOW"] = df_test[date_col].dt.weekday
            df_test["series_id"] = (
                df_test[series_cols[0]].astype(str) + "_" + df_test[series_cols[1]].astype(str)
            )
            df_test = df_test.sort_values(["series_id", date_col]).reset_index(drop=True)

            train_cut = cutoff_train(self.train, df_test[date_col].min())
            train_full = pd.concat([train_cut, df_test])

            preds: List[pd.DataFrame] = []

            groups = list(df_test.groupby("series_id"))
            series_ids = [sid for sid, _ in groups]
            fut_dates_list: List[List[pd.Timestamp]] = []
            fut_dows_list: List[List[int]] = []
            future_rows: List[Dict[str, object]] = []
            for j, (sid, tdf) in enumerate(groups, 1):
                logging.debug("  Series %s (%d/%d)", sid, j, len(groups))
                last_date = tdf[date_col].max()
                fut_dates = [
                    last_date + pd.Timedelta(days=i)
                    for i in range(1, self.cfg.horizon + 1)
                ]
                fut_dates_list.append(fut_dates)
                dows = [d.weekday() for d in fut_dates]
                fut_dows_list.append(dows)
                for d, dow in zip(fut_dates, dows):
                    future_rows.append({"series_id": sid, date_col: d, "DOW": dow})

            if self.cfg.classifier_kind == "logit":
                fut_cal = pd.DataFrame(future_rows)
                P_all = logistic_global_calendar(
                    train_cut=train_full,
                    future_calendar=fut_cal,
                    lr=self.cfg.logit_lr,
                    epochs=self.cfg.logit_epochs,
                    l2=self.cfg.logit_l2,
                    batch_size=self.cfg.logit_batch_size,
                    window_weeks=self.cfg.dow_window_weeks,
                    alpha=self.cfg.beta_alpha,
                    beta=self.cfg.beta_beta,
                    calib_lambda=self.cfg.calib_lambda,
                    class_weight=self.cfg.class_weight,
                )
                P_batch = []
                for i in range(len(series_ids)):
                    start = i * self.cfg.horizon
                    P_batch.append(to_numpy(P_all[start : start + self.cfg.horizon]))
                P_batch = np.stack(P_batch, axis=0)
            else:
                P_list = [
                    beta_smooth_probs(
                        train_cut=train_cut,
                        series_id=sid,
                        future_dows=dows,
                        window_weeks=self.cfg.dow_window_weeks,
                        alpha=self.cfg.beta_alpha,
                        beta=self.cfg.beta_beta,
                        date_col=date_col,
                        target_col=target_col,
                    )
                    for sid, dows in zip(series_ids, fut_dows_list)
                ]
                P_batch = np.stack(P_list, axis=0)

            try:
                mu_batch = forecast_intensity_gpu(
                    train_cut=train_full,
                    series_id=series_ids,
                    future_dates=fut_dates_list,
                    grid=self.cfg.sarima_grid,
                    val_weeks=self.cfg.val_weeks,
                    fallback=self.cfg.fallback,
                    target_col=target_col,
                    batch_size=self.cfg.intensity_batch_size,
                )
            except Exception as exc:  # pragma: no cover - propagate GPU errors
                raise RuntimeError(f"GPU intensity failed: {exc}") from exc

            cap_batch = None
            if self.cfg.cap_quantile is not None:
                cap_list = []
                for sid, dows in zip(series_ids, fut_dows_list):
                    train_pos = train_cut[
                        (train_cut["series_id"] == sid)
                        & (train_cut[target_col] > 0)
                    ]
                    if len(train_pos) > 0:
                        q = (
                            train_pos.groupby("DOW")[target_col]
                            .quantile(self.cfg.cap_quantile)
                        )
                        cap_vals = np.array([q.get(d, 0.0) for d in dows])
                    else:
                        cap_vals = np.zeros(len(dows))
                    cap_list.append(cap_vals)
                cap_batch = np.stack(cap_list, axis=0)

            yhat_gpu = combine_expectation(P_batch, mu_batch, cap_batch)
            yhat_batch = to_numpy(yhat_gpu)

            for (sid, tdf), yhat, fut_dates in zip(groups, yhat_batch, fut_dates_list):
                series_vals = tdf.iloc[0][list(series_cols)]
                out = pd.DataFrame({
                    series_cols[0]: series_vals[series_cols[0]],
                    series_cols[1]: series_vals[series_cols[1]],
                    date_col: [d.strftime("%Y-%m-%d") for d in fut_dates],
                    "예측값": yhat,
                })
                # ensure consistent column order for downstream merging
                out = out[[*series_cols, date_col, "예측값"]]
                preds.append(out)

            pred_df = pd.concat(preds, axis=0, ignore_index=True)
            pred_df = pred_df[[*series_cols, date_col, "예측값"]]
            out_path = os.path.join(out_dir, f"pred_{fname}")
            pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Saved predictions to %s", out_path)

        if sample_submission is not None:
            skel = pd.read_csv(sample_submission)
            all_preds = []
            for p in sorted(os.listdir(out_dir)):
                if p.startswith("pred_TEST_") and p.endswith(".csv"):
                    test_id = p[len("pred_") : -4]
                    df_pred = pd.read_csv(os.path.join(out_dir, p))
                    df_pred = _to_placeholder_dates(df_pred, date_col, test_id)
                    all_preds.append(df_pred)
            if all_preds:
                pred_all = pd.concat(all_preds, ignore_index=True)
                pred_all = pred_all[[*series_cols, date_col, "예측값"]]
                filled = fill_submission_skeleton(
                    skel,
                    pred_all,
                    date_col=date_col,
                    series_cols=series_cols,
                    value_col="예측값",
                )
                filled_path = os.path.join(out_dir, "submission_filled.csv")
                filled.to_csv(filled_path, index=False, encoding="utf-8-sig")
                logging.info("Saved filled submission to %s", filled_path)
                logging.info("All predictions saved to %s", out_dir)
                return filled

        logging.info("All predictions saved to %s", out_dir)
        return None
