import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from hurdle_forecast.intensity import _make_exog_dow


def test_make_exog_dow_has_all_columns_and_order():
    # Training dates without Sundays
    train_idx = pd.date_range('2024-01-01', '2024-01-27', freq='D')
    train_idx = train_idx[train_idx.weekday != 6]

    # Future dates that include a Sunday
    future_idx = pd.date_range('2024-01-28', '2024-02-03', freq='D')

    exog_train = _make_exog_dow(train_idx)
    exog_future = _make_exog_dow(future_idx)

    expected_cols = [f"dow_{i}" for i in range(7)]
    assert list(exog_train.columns) == expected_cols
    assert list(exog_future.columns) == expected_cols
    assert exog_train.shape[1] == 7
    assert exog_future.shape[1] == 7

    # Optional: ensure forecasting works with these matrices
    y = pd.Series(np.arange(len(train_idx)), index=train_idx)
    model = SARIMAX(
        y,
        order=(0, 0, 0),
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=len(future_idx), exog=exog_future)
    mu = fc.predicted_mean
    assert len(mu) == len(future_idx)
