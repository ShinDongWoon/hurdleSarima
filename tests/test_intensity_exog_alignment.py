import numpy as np
import pandas as pd
import logging

from hurdle_forecast import intensity


def test_forecast_intensity_aligns_exog(monkeypatch, caplog):
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    train = pd.DataFrame({
        'series_id': 'A',
        '영업일자': dates,
        '매출수량': np.arange(1, 21),
    })
    future_dates = [pd.Timestamp('2023-01-21'), pd.Timestamp('2023-01-22')]

    def fake_make_exog_dow(idx):
        if len(idx) > 2:
            return pd.DataFrame({'a': np.ones(len(idx)), 'b': np.ones(len(idx))}, index=idx)
        return pd.DataFrame({'b': np.ones(len(idx)), 'c': np.ones(len(idx))}, index=idx)

    monkeypatch.setattr(intensity, '_make_exog_dow', fake_make_exog_dow)
    monkeypatch.setattr(intensity, '_candidate_orders', lambda: [(0, 0, 0, 0, 0, 0)])

    class DummyRes:
        params = np.array([0.0])
        aic = 0.0
        mle_retvals = {'converged': True}

        def get_forecast(self, steps, exog=None):
            self.exog = exog
            class DummyForecast:
                def __init__(self, steps):
                    self.predicted_mean = pd.Series([0.0] * steps)
            return DummyForecast(steps)

    dummy_res = DummyRes()
    monkeypatch.setattr(intensity, '_fit_sarimax', lambda *args, **kwargs: dummy_res)

    with caplog.at_level(logging.WARNING):
        intensity._forecast_intensity_single(train, 'A', future_dates)

    assert list(dummy_res.exog.columns) == ['a', 'b']
    assert (dummy_res.exog['a'] == 0).all()
    assert 'Exogenous feature columns misaligned' in caplog.text
