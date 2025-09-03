import sys
import types
import numpy as np
import pandas as pd
from hurdle_forecast import intensity


def _patch_gpu(monkeypatch):
    dummy_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        array=np.array,
        asarray=np.asarray,
        zeros=np.zeros,
        nan_to_num=np.nan_to_num,
        where=np.where,
        log1p=np.log1p,
        expm1=np.expm1,
        maximum=np.maximum,
        stack=np.stack,
    )
    monkeypatch.setitem(sys.modules, "cupy", dummy_cp)

    dummy_arima_mod = types.ModuleType("cuml.tsa.arima")

    class DummyARIMA:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self):
            pass

        def forecast(self, steps):
            # Simulate CuML ARIMA returning 2D array (1, steps)
            return np.zeros((1, steps))

    dummy_arima_mod.ARIMA = DummyARIMA
    dummy_tsa_mod = types.ModuleType("cuml.tsa")
    dummy_tsa_mod.arima = dummy_arima_mod
    dummy_cuml_mod = types.ModuleType("cuml")
    dummy_cuml_mod.tsa = dummy_tsa_mod

    monkeypatch.setitem(sys.modules, "cuml", dummy_cuml_mod)
    monkeypatch.setitem(sys.modules, "cuml.tsa", dummy_tsa_mod)
    monkeypatch.setitem(sys.modules, "cuml.tsa.arima", dummy_arima_mod)


def test_forecast_intensity_gpu_shape_single(monkeypatch):
    _patch_gpu(monkeypatch)
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    train = pd.DataFrame({
        "series_id": "A",
        "\uc601\uc5c5\uc77c\uc790": dates,
        "\ub9e4\ucd9c\uc218\ub7c9": 1.0,
    })
    horizon = 3
    future_dates = [dates[-1] + pd.Timedelta(days=i + 1) for i in range(horizon)]
    mu = intensity.forecast_intensity_gpu(train, "A", future_dates)
    mu_np = np.asarray(mu)
    assert mu_np.shape == (1, horizon)


def test_forecast_intensity_gpu_shape_multi(monkeypatch):
    _patch_gpu(monkeypatch)
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    train = pd.DataFrame({
        "series_id": ["A"] * 15 + ["B"] * 15,
        "\uc601\uc5c5\uc77c\uc790": list(dates) + list(dates),
        "\ub9e4\ucd9c\uc218\ub7c9": 1.0,
    })
    horizon = 4
    future_dates = [
        [dates[-1] + pd.Timedelta(days=i + 1) for i in range(horizon)],
        [dates[-1] + pd.Timedelta(days=i + 1) for i in range(horizon)]
    ]
    mu = intensity.forecast_intensity_gpu(train, ["A", "B"], future_dates)
    mu_np = np.asarray(mu)
    assert mu_np.shape == (2, horizon)

