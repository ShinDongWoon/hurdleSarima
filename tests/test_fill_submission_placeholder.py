import pandas as pd
from hurdle_forecast.model import _to_placeholder_dates
from hurdle_forecast.combine import fill_submission_skeleton


def test_fill_submission_placeholder_dates():
    # Skeleton with placeholder date labels
    skel = pd.DataFrame({
        "영업일자": ["TEST_00+1일", "TEST_00+2일", "TEST_01+1일"],
        "A_BBQ": [None, None, None],
    })

    # Predictions for TEST_00 with actual calendar dates
    pred0 = pd.DataFrame({
        "영업일자": ["2024-01-01", "2024-01-02"],
        "영업장명": ["A", "A"],
        "메뉴명": ["BBQ", "BBQ"],
        "예측값": [1, 2],
    })
    pred0 = _to_placeholder_dates(pred0, "영업일자", "TEST_00")

    # Predictions for TEST_01
    pred1 = pd.DataFrame({
        "영업일자": ["2024-01-03"],
        "영업장명": ["A"],
        "메뉴명": ["BBQ"],
        "예측값": [3],
    })
    pred1 = _to_placeholder_dates(pred1, "영업일자", "TEST_01")

    pred_all = pd.concat([pred0, pred1], ignore_index=True)

    out = fill_submission_skeleton(
        skel,
        pred_all,
        date_col="영업일자",
        series_cols=("영업장명", "메뉴명"),
        value_col="예측값",
    )

    assert pred0["영업일자"].tolist() == ["TEST_00+1일", "TEST_00+2일"]
    assert pred1["영업일자"].tolist() == ["TEST_01+1일"]
    assert out.loc[out["영업일자"] == "TEST_00+1일", "A_BBQ"].iloc[0] == 1
    assert out.loc[out["영업일자"] == "TEST_00+2일", "A_BBQ"].iloc[0] == 2
    assert out.loc[out["영업일자"] == "TEST_01+1일", "A_BBQ"].iloc[0] == 3
