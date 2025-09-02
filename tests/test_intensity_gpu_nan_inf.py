import sys
import types
import numpy as np
import pandas as pd
from hurdle_forecast import intensity


def test_forecast_intensity_gpu_handles_nan_zero_series(monkeypatch):
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
            return np.zeros(steps)
    dummy_arima_mod.ARIMA = DummyARIMA
    dummy_tsa_mod = types.ModuleType("cuml.tsa")
    dummy_tsa_mod.arima = dummy_arima_mod
    dummy_cuml_mod = types.ModuleType("cuml")
    dummy_cuml_mod.tsa = dummy_tsa_mod
    monkeypatch.setitem(sys.modules, "cuml", dummy_cuml_mod)
    monkeypatch.setitem(sys.modules, "cuml.tsa", dummy_tsa_mod)
    monkeypatch.setitem(sys.modules, "cuml.tsa.arima", dummy_arima_mod)

    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    train = pd.DataFrame({
        "series_id": "A",
        "영업일자": dates,
        "매출수량": [np.nan, 0.0, np.nan, 0.0, np.nan],
    })
    future_dates = [pd.Timestamp("2023-01-06") + pd.Timedelta(days=i) for i in range(2)]

    mu = intensity.forecast_intensity_gpu(train, "A", future_dates)
    mu_np = np.asarray(mu)
    assert np.all(np.isfinite(mu_np))
