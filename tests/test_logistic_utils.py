import numpy as np
import pandas as pd
import pytest

from hurdle_forecast.classifier import logistic_global_calendar
from hurdle_forecast.mps_utils import has_torch, to_numpy

@pytest.mark.skipif(not has_torch(), reason="PyTorch not installed")
def test_logistic_global_calendar_cpu_returns_numpy():
    train = pd.DataFrame({
        "영업일자": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]),
        "DOW": [0, 1, 0, 1],
        "series_id": ["A", "A", "B", "B"],
        "매출수량": [1, 0, 0, 1],
    })
    future = pd.DataFrame({
        "영업일자": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "DOW": [2, 2],
        "series_id": ["A", "B"],
    })
    probs = logistic_global_calendar(train, future, epochs=1, batch_size=2)
    assert isinstance(probs, np.ndarray)
    assert probs.shape[0] == len(future)


def test_to_numpy_handles_torch():
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0, 2.0])
    arr = to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, [1.0, 2.0])
