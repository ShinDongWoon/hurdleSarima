import numpy as np
import pandas as pd

from hurdle_forecast import intensity


def test_weekend_dummy_coefficients_positive():
    idx = pd.date_range("2023-01-01", periods=70, freq="D")
    y_log = pd.Series(np.where(idx.weekday >= 5, 1.0, 0.0), index=idx)
    exog = intensity._make_exog_dow(idx)
    res = intensity._fit_sarimax(y_log, exog, (0, 0, 0), (0, 0, 0), 7)
    params = res.params
    assert params["dow_5"] > 0
    assert params["dow_6"] > 0

