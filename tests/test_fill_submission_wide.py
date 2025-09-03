import pandas as pd
from hurdle_forecast.combine import fill_submission_skeleton


def test_fill_submission_skeleton_wide_korean_columns():
    # Wide sample submission with Korean column names combining store and menu
    sample_submission = pd.DataFrame(
        {
            "\uC601\uC5C5\uC77C\uC790": pd.date_range("2024-01-01", periods=2),
            "\uAC00\uAC8C1_\uAE40\uCE58": [None, None],
            "\uAC00\uAC8C1_\uBD88\uACE0\uAE30": [None, None],
        }
    )

    # Long prediction dataframe with separate store, menu and date columns
    pred_df = pd.DataFrame(
        {
            "\uC601\uC5C5\uC77C\uC790": pd.date_range("2024-01-01", periods=2).repeat(2),
            "\uC601\uC5C5\uC7A5\uBA85": ["\uAC00\uAC8C1", "\uAC00\uAC8C1", "\uAC00\uAC8C1", "\uAC00\uAC8C1"],
            "\uBA54\uB274\uBA85": ["\uAE40\uCE58", "\uBD88\uACE0\uAE30", "\uAE40\uCE58", "\uBD88\uACE0\uAE30"],
            "\uC608\uCE21": [1, 2, 3, 4],
        }
    )

    out = fill_submission_skeleton(
        sample_submission,
        pred_df,
        date_col="\uC601\uC5C5\uC77C\uC790",
        series_cols=("\uC601\uC5C5\uC7A5\uBA85", "\uBA54\uB274\uBA85"),
        value_col="\uC608\uCE21",
    )

    assert out.loc[0, "\uAC00\uAC8C1_\uAE40\uCE58"] == 1
    assert out.loc[0, "\uAC00\uAC8C1_\uBD88\uACE0\uAE30"] == 2
    assert out.loc[1, "\uAC00\uAC8C1_\uAE40\uCE58"] == 3
    assert out.loc[1, "\uAC00\uAC8C1_\uBD88\uACE0\uAE30"] == 4
