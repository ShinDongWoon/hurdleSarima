import numpy as np
import pandas as pd
import pytest

from hurdle_forecast import intensity


def test_forecast_intensity_handles_nan_predictions(monkeypatch):
    """Even if the underlying model returns NaNs, the forecast should be finite.
    This scenario can occur when the time series contains only zeros."""
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    train = pd.DataFrame({
        'series_id': 'A',
        '영업일자': dates,
        '매출수량': 0,
    })
    future_dates = [pd.Timestamp('2023-01-21') + pd.Timedelta(days=i) for i in range(3)]

    # Pretend we have enough non-zero observations to avoid early fallback
    original_notna = pd.Series.notna

    def fake_notna(self):
        if self.name == '매출수량':
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
                    self.predicted_mean = pd.Series([np.nan] * steps)

            return DummyForecast(steps)

    monkeypatch.setattr(intensity, "_fit_sarimax", lambda *args, **kwargs: DummyRes())
    monkeypatch.setattr(intensity, "_candidate_orders", lambda: [(0, 0, 0, 0, 0, 0)])

    mu = intensity.forecast_intensity(train, 'A', future_dates)
    assert np.all(np.isfinite(mu))
