import pandas as pd
from hurdle_forecast.combine import fill_submission_skeleton

def test_fill_submission_skeleton_wide_two_keys():
    skel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2),
            "0_0": [None, None],
            "0_1": [None, None],
        }
    )
    pred_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2).repeat(2),
            "store": [0, 0, 0, 0],
            "item": [0, 1, 0, 1],
            "pred": [1, 2, 3, 4],
        }
    )
    out = fill_submission_skeleton(
        skel,
        pred_df,
        date_col="date",
        series_cols=("store", "item"),
        value_col="pred",
    )
    assert out.loc[0, "0_0"] == 1
    assert out.loc[0, "0_1"] == 2
    assert out.loc[1, "0_0"] == 3
    assert out.loc[1, "0_1"] == 4
