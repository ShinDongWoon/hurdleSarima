from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import pickle
import pandas as pd
import numpy as np
import logging

from .config import Config
from .data import cutoff_train, future_dates, maybe_split_series
from .classifier import beta_smooth_probs, logistic_global_calendar
from .intensity import forecast_intensity, forecast_intensity_gpu
from .combine import combine_expectation, fill_submission_skeleton
from .mps_utils import gpu_available, torch_device, to_numpy


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


@dataclass
class HurdleForecastModel:
    cfg: Config
    train: pd.DataFrame
    train_pos: np.ndarray

    @classmethod
    def from_config(cls, cfg: Config) -> "HurdleForecastModel":
        train = _prepare_train(cfg)
        train_pos = train.loc[train[cfg.target_col] > 0, cfg.target_col].values
        return cls(cfg=cfg, train=train, train_pos=train_pos)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HurdleForecastModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_dir(self, test_dir: str, out_dir: str, sample_submission: Optional[str] = None) -> None:
        """Run forecasts for all TEST_*.csv in `test_dir` and write outputs."""
        os.makedirs(out_dir, exist_ok=True)
        series_cols = self.cfg.series_cols
        date_col = self.cfg.date_col
        target_col = self.cfg.target_col
        use_gpu = gpu_available()
        logger = logging.getLogger(__name__)

        for fname in sorted(os.listdir(test_dir)):
            if not (fname.startswith("TEST_") and fname.endswith(".csv")):
                continue
            path = os.path.join(test_dir, fname)
            df_test = pd.read_csv(path)
            maybe_split_series(df_test, series_cols)
            missing = [c for c in [*series_cols, date_col] if c not in df_test.columns]
            if missing:
                raise ValueError(f"Missing required columns in test data {fname}: {missing}")
            df_test[date_col + "_str"] = df_test[date_col].astype(str)
            df_test[date_col] = pd.to_datetime(df_test[date_col])
            df_test["DOW"] = df_test[date_col].dt.weekday
            df_test["series_id"] = df_test[series_cols[0]].astype(str) + "_" + df_test[series_cols[1]].astype(str)
            df_test = df_test.sort_values(["series_id", date_col]).reset_index(drop=True)

            test_dates = future_dates(df_test, date_col)
            cutoff_date = min(test_dates)
            train_cut = cutoff_train(self.train, cutoff_date)

            preds = []

            if self.cfg.classifier_kind == "logit":
                fut_cal = df_test[[date_col, "DOW", "series_id"]].copy()
                P_all = logistic_global_calendar(
                    train_cut=train_cut,
                    future_calendar=fut_cal,
                    lr=self.cfg.logit_lr,
                    epochs=self.cfg.logit_epochs,
                    l2=self.cfg.logit_l2,
                    batch_size=self.cfg.logit_batch_size,
                )

            for sid, tdf in df_test.groupby("series_id"):
                fut_dates = tdf[date_col].tolist()
                fut_dows = tdf["DOW"].tolist()

                if self.cfg.classifier_kind == "beta":
                    P = beta_smooth_probs(
                        train_cut=train_cut,
                        series_id=sid,
                        future_dows=fut_dows,
                        window_weeks=self.cfg.dow_window_weeks,
                        alpha=self.cfg.beta_alpha,
                        beta=self.cfg.beta_beta,
                        date_col=date_col,
                        target_col=target_col,
                    )
                else:
                    start = tdf.index[0]
                    P = P_all[start : start + len(tdf)]
                if use_gpu:
                    try:
                        mu = forecast_intensity_gpu(
                            train_cut=train_cut,
                            series_id=sid,
                            future_dates=fut_dates,
                            m=self.cfg.seasonal_m,
                            grid=self.cfg.sarima_grid,
                            val_weeks=self.cfg.val_weeks,
                            fallback=self.cfg.fallback,
                            target_col=target_col,
                        )
                    except Exception as exc:  # pragma: no cover - GPU optional
                        logger.warning(
                            "GPU intensity failed for %s; falling back to CPU: %s",
                            sid,
                            exc,
                        )
                        mu = forecast_intensity(
                            train_cut=train_cut,
                            series_id=sid,
                            future_dates=fut_dates,
                            m=self.cfg.seasonal_m,
                            grid=self.cfg.sarima_grid,
                            val_weeks=self.cfg.val_weeks,
                            fallback=self.cfg.fallback,
                            target_col=target_col,
                        )
                else:
                    mu = forecast_intensity(
                        train_cut=train_cut,
                        series_id=sid,
                        future_dates=fut_dates,
                        m=self.cfg.seasonal_m,
                        grid=self.cfg.sarima_grid,
                        val_weeks=self.cfg.val_weeks,
                        fallback=self.cfg.fallback,
                        target_col=target_col,
                    )

                if use_gpu:
                    try:
                        yhat_gpu = combine_expectation(
                            P,
                            mu,
                            self.cfg.cap_quantile,
                            train_positive=self.train_pos,
                        )
                        yhat = to_numpy(yhat_gpu)
                    except Exception:
                        yhat = combine_expectation(
                            to_numpy(P),
                            to_numpy(mu),
                            self.cfg.cap_quantile,
                            train_positive=self.train_pos,
                        )
                else:
                    yhat = combine_expectation(
                        P, mu, self.cfg.cap_quantile, train_positive=self.train_pos
                    )

                out = tdf[[*series_cols, date_col + "_str"]].copy()
                out.rename(columns={date_col + "_str": date_col}, inplace=True)
                out["예측값"] = yhat
                preds.append(out)

            pred_df = pd.concat(preds, axis=0, ignore_index=True)
            out_path = os.path.join(out_dir, f"pred_{fname}")
            pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        if sample_submission is not None:
            skel = pd.read_csv(sample_submission)
            all_preds = []
            for p in sorted(os.listdir(out_dir)):
                if p.startswith("pred_TEST_") and p.endswith(".csv"):
                    all_preds.append(pd.read_csv(os.path.join(out_dir, p)))
            if all_preds:
                pred_all = pd.concat(all_preds, ignore_index=True)
                filled = fill_submission_skeleton(
                    skel,
                    pred_all,
                    date_col=date_col,
                    series_cols=series_cols,
                    value_col="예측값",
                )
                filled_path = os.path.join(out_dir, "submission_filled.csv")
                filled.to_csv(filled_path, index=False, encoding="utf-8-sig")
