import sys
import types
import pandas as pd
import pytest

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - optional GPU libs
    cp = None
    GPU_AVAILABLE = False

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required")


@requires_gpu
def test_forecast_intensity_gpu_single_returns_cupy(monkeypatch):
    from hurdle_forecast.intensity import _forecast_intensity_gpu_single

    dummy_arima_mod = types.ModuleType("cuml.tsa.arima")

    class DummyARIMA:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self):
            pass

        def forecast(self, steps):
            return cp.zeros((1, steps))

    dummy_arima_mod.ARIMA = DummyARIMA
    dummy_tsa_mod = types.ModuleType("cuml.tsa")
    dummy_tsa_mod.arima = dummy_arima_mod
    dummy_cuml_mod = types.ModuleType("cuml")
    dummy_cuml_mod.tsa = dummy_tsa_mod

    monkeypatch.setitem(sys.modules, "cuml", dummy_cuml_mod)
    monkeypatch.setitem(sys.modules, "cuml.tsa", dummy_tsa_mod)
    monkeypatch.setitem(sys.modules, "cuml.tsa.arima", dummy_arima_mod)

    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    train = pd.DataFrame(
        {
            "series_id": "A",
            "영업일자": dates,
            "매출수량": 1.0,
        }
    )
    horizon = 3
    future_dates = [dates[-1] + pd.Timedelta(days=i + 1) for i in range(horizon)]
    mu = _forecast_intensity_gpu_single(train, "A", future_dates)
    assert isinstance(mu, cp.ndarray)
    assert mu.shape == (horizon,)


@requires_gpu
def test_combine_expectation_cupy_broadcast():
    from hurdle_forecast.combine import combine_expectation

    P = cp.asarray([[0.5], [0.8]])
    mu = cp.asarray([1.0, 2.0, 3.0])
    train_pos = cp.asarray([1.0, 2.0, 3.0, 4.0])
    yhat = combine_expectation(P, mu, cap=0.5, train_positive=train_pos)
    assert isinstance(yhat, cp.ndarray)
    assert yhat.shape == (2, 3)
    expected = cp.minimum(P * mu, cp.quantile(train_pos, 0.5))
    cp.testing.assert_allclose(yhat, expected)
