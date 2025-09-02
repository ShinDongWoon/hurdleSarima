from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class Config:
    train_csv: str
    test_dir: str
    out_dir: str
    sample_submission: Optional[str] = None

    horizon: int = 7
    # expects separate columns, but a joined `영업장명_메뉴명` will be split automatically
    series_cols: tuple = ("영업장명", "메뉴명")
    date_col: str = "영업일자"
    target_col: str = "매출수량"

    # Classifier settings
    classifier_kind: Literal["beta", "logit"] = "beta"
    dow_window_weeks: int = 8
    beta_alpha: float = 0.5
    beta_beta: float = 0.5

    # Logistic (PyTorch) hyperparams (only used if classifier_kind == "logit")
    logit_lr: float = 0.05
    logit_epochs: int = 200
    logit_l2: float = 1e-4
    logit_batch_size: int = 4096

    # Intensity settings
    seasonal_m: int = 7
    sarima_grid: Literal["full", "small"] = "full"
    val_weeks: int = 4
    fallback: Literal["ets", "snaive"] = "ets"

    # Post-processing
    cap_quantile: Optional[float] = 0.99  # set None to disable

    # Misc
    n_jobs: int = 1  # reserved; current implementation is single-process for reproducibility
