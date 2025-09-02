import numpy as np
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


def test_clean_sales_with_missing_and_extreme_values():
    df = pd.DataFrame({"y": [-10, -1, np.nan, 0, 50, 1000]})
    q = 0.9
    expected_upper = pd.Series([0, 0, 0, 0, 50, 1000]).quantile(q)
    clean_sales(df, "y", q)
    assert not df["y"].isna().any()
    assert (df.loc[:1, "y"] == 0).all()  # negatives to zero
    assert df.loc[5, "y"] == pytest.approx(expected_upper)  # extreme clipped
    assert df["y"].max() == pytest.approx(expected_upper)


def test_clean_sales_handles_nans_negatives_and_clipping():
    df = pd.DataFrame({"y": [-5, np.nan, 5, 10, 100]})
    q = 0.8
    expected_upper = pd.Series([0, 0, 5, 10, 100]).quantile(q)
    clean_sales(df, "y", q)
    assert not df["y"].isna().any()
    assert df.loc[0, "y"] == 0
    assert df.loc[1, "y"] == 0
    assert df.loc[4, "y"] == pytest.approx(expected_upper)
    assert df["y"].max() == pytest.approx(expected_upper)
