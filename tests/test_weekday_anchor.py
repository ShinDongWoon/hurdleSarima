import pandas as pd
import numpy as np

from hurdle_forecast import intensity, classifier


def test_weekday_anchor_future_dows():
    # Create a simple dataset with two series
    dates_a = pd.date_range("2024-01-01", periods=10, freq="D")
    dates_b = pd.date_range("2024-02-01", periods=10, freq="D")
    train = pd.DataFrame(
        {
            "series_id": ["A"] * len(dates_a) + ["B"] * len(dates_b),
            "영업일자": list(dates_a) + list(dates_b),
            "매출수량": [1] * len(dates_a) + [2] * len(dates_b),
        }
    )
    train["DOW"] = train["영업일자"].dt.weekday

    # Record last date per series
    last_dates = train.groupby("series_id")["영업일자"].max().to_dict()

    for sid, last_date in last_dates.items():
        # Compute future dates and DOWs as in the pipeline
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        fut_dows = [d.weekday() for d in future_dates]

        d0 = last_date.weekday()
        expected = [(d0 + i) % 7 for i in range(1, 8)]
        assert fut_dows == expected

        # Verify one-hot exogenous matrix
        exog = intensity._make_exog_dow(pd.DatetimeIndex(future_dates))
        assert list(exog.to_numpy().argmax(axis=1)) == fut_dows

        # Verify classifier output length and future calendar DOWs
        probs = classifier.beta_smooth_probs(
            train_cut=train,
            series_id=sid,
            future_dows=fut_dows,
            date_col="영업일자",
            target_col="매출수량",
        )
        assert len(probs) == len(fut_dows)

        future_calendar = pd.DataFrame(
            {"영업일자": future_dates, "DOW": fut_dows, "series_id": sid}
        )
        assert future_calendar["DOW"].tolist() == fut_dows

