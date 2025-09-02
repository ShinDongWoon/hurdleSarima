from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .mps_utils import has_torch, torch_device

# ---------- Beta-smoothed frequency classifier ----------

def beta_smooth_probs(
    train_cut: pd.DataFrame,
    series_id: str,
    future_dows: List[int],
    window_weeks: int = 8,
    alpha: float = 0.5,
    beta: float = 0.5,
    date_col: str = "영업일자",
    target_col: str = "매출수량",
) -> np.ndarray:
    """Compute P(y>0 | DOW) for a given series_id using last `window_weeks` window.
    Fallbacks: series all-time, then global all-time.
    Returns probs for each future DOW in order.
    """
    # series subset up to cutoff
    sdf = train_cut.loc[train_cut["series_id"] == series_id]
    if sdf.empty:
        # no history; use global fallback
        g = _beta_counts(train_cut, window=None, date_col=date_col, target_col=target_col)
        return _probs_from_counts(g, future_dows, alpha, beta)

    cutoff_date = train_cut[date_col].max()
    win_start = cutoff_date - pd.Timedelta(days=7*window_weeks) if window_weeks is not None else None

    # last window per series
    sc = _beta_counts(sdf, window=(win_start, cutoff_date), date_col=date_col, target_col=target_col)
    if sc["total"].sum() == 0:
        # series all-time
        sc = _beta_counts(sdf, window=None, date_col=date_col, target_col=target_col)

    if sc["total"].sum() == 0:
        # global fallback
        sc = _beta_counts(train_cut, window=None, date_col=date_col, target_col=target_col)

    return _probs_from_counts(sc, future_dows, alpha, beta)

def _beta_counts(df: pd.DataFrame, window: Optional[Tuple[pd.Timestamp, pd.Timestamp]], date_col: str, target_col: str):
    if window is not None:
        start, end = window
        m = (df[date_col] >= start) & (df[date_col] <= end)
        df = df.loc[m]
    grp = df.groupby("DOW")[target_col]
    total = grp.size().reindex(range(7), fill_value=0).astype(float)
    nonzero = grp.apply(lambda s: (s > 0).sum()).reindex(range(7), fill_value=0).astype(float)
    return {"total": total, "nonzero": nonzero}

def _probs_from_counts(counts, future_dows: List[int], alpha: float, beta: float) -> np.ndarray:
    total = counts["total"]
    nonzero = counts["nonzero"]
    p = (nonzero + alpha) / (total + alpha + beta)
    p = p.clip(0.0, 1.0)
    return p.reindex(range(7)).values[future_dows]

# ---------- Optional global logistic regression (PyTorch) ----------

def logistic_global_calendar(
    train_cut: pd.DataFrame,
    future_calendar: pd.DataFrame,
    prior_weight: float = 1.0,
    lr: float = 0.05,
    epochs: int = 200,
    l2: float = 1e-4,
    batch_size: int = 4096,
    seed: int = 42,
) -> np.ndarray:
    """Calendar-only global logistic regression on (y>0).
    Features: DOW one-hot + (optional) series prior (logit(p_nonzero_series)).
    Runs on MPS if available.
    """
    if not has_torch():
        raise RuntimeError("PyTorch not installed. Install torch>=2.1 to use logistic classifier.")

    import torch
    g = train_cut.groupby("series_id")["매출수량"]
    p_series = (g.apply(lambda s: (s > 0).mean())).to_dict()
    def prior_logit(series_ids: pd.Series):
        p = series_ids.map(p_series).fillna(g.apply(lambda s: (s > 0).mean()).mean())
        eps = 1e-4
        return np.log((p + eps) / (1 - (p + eps)))

    # Build training features
    X_dow = pd.get_dummies(train_cut["DOW"], prefix="dow", drop_first=False)
    X = X_dow.astype(float)
    X["prior"] = prior_weight * prior_logit(train_cut["series_id"])
    y = (train_cut["매출수량"] > 0).astype(float).values

    # Future features (same columns)
    Xf_dow = pd.get_dummies(future_calendar["DOW"], prefix="dow", drop_first=False)
    Xf = Xf_dow.astype(float).reindex(columns=X_dow.columns, fill_value=0.0)
    # For prior in future, we need series_id column in future_calendar
    if "series_id" in future_calendar.columns:
        Xf["prior"] = prior_weight * prior_logit(future_calendar["series_id"])
    else:
        Xf["prior"] = prior_weight * np.mean(list(p_series.values()))

    # Torch tensors
    device = torch_device(prefer_mps=True)
    torch.manual_seed(seed)
    Xt = torch.tensor(X.values, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

    Xft = torch.tensor(Xf.values, dtype=torch.float32, device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(Xt.shape[1], 1),
        torch.nn.Sigmoid(),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    bce = torch.nn.BCELoss()

    # mini-batch training
    n = Xt.shape[0]
    idx = torch.randperm(n, device=device)
    for epoch in range(epochs):
        # shuffle each epoch
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            sel = idx[start:start+batch_size]
            xb, yb = Xt[sel], yt[sel]
            opt.zero_grad()
            pred = model(xb)
            loss = bce(pred, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pf = model(Xft).squeeze(1).clamp(0, 1).detach().cpu().numpy()

    return pf
