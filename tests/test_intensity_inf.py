import numpy as np
import pandas as pd

from hurdle_forecast import intensity


def test_forecast_intensity_handles_inf_from_expm1(monkeypatch):
    """np.expm1 may overflow; ensure resulting infinities are set to zero."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    train = pd.DataFrame(
        {
            "series_id": "A",
            "영업일자": dates,
            "매출수량": 1.0,
        }
    )
    future_dates = [pd.Timestamp("2023-01-21") + pd.Timedelta(days=i) for i in range(2)]

    # Avoid early fallback
    original_notna = pd.Series.notna

    def fake_notna(self):
        if self.name == "매출수량":
            return pd.Series([True] * len(self), index=self.index)
        return original_notna(self)

    monkeypatch.setattr(pd.Series, "notna", fake_notna, raising=False)

    class DummyRes:
        params = np.array([0.0])
        aic = 0.0
        mle_retvals = {"converged": True}

        def get_forecast(self, steps, exog=None):
            class DummyForecast:
                def __init__(self, steps):
                    self.predicted_mean = pd.Series([0.0] * steps)

            return DummyForecast(steps)

    monkeypatch.setattr(intensity, "_fit_sarimax", lambda *args, **kwargs: DummyRes())
    monkeypatch.setattr(intensity, "_candidate_orders", lambda: [(0, 0, 0, 0, 0, 0)])

    # Force np.expm1 to overflow
    monkeypatch.setattr(intensity.np, "expm1", lambda x: np.full_like(x, np.inf))

    mu = intensity.forecast_intensity(train, "A", future_dates)
    assert np.array_equal(mu, np.zeros_like(mu))
