import pandas as pd
import pytest

from hurdle_forecast.data import clean_sales


def test_clean_sales_replaces_negatives_and_clips_upper_quantile():
    df = pd.DataFrame({"y": [-5, 0, 5, 10, 100]})
    q = 0.8
    expected_upper = pd.Series([0, 0, 5, 10, 100]).quantile(q)
    clean_sales(df, "y", q)
    assert (df["y"] >= 0).all()
    assert df.loc[0, "y"] == 0
    assert df.loc[4, "y"] == pytest.approx(expected_upper)
    assert df["y"].max() == pytest.approx(expected_upper)
