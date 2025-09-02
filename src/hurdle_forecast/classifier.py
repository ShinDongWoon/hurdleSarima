from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union
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
) -> Union[np.ndarray, "torch.Tensor"]:
    """Calendar-only global logistic regression on (y>0).
    Features: DOW one-hot + (optional) series prior (logit(p_nonzero_series)).
    Runs on MPS if available.
    """
    if not has_torch():
        raise RuntimeError("PyTorch not installed. Install torch>=2.1 to use logistic classifier.")

    import torch
    use_cudf = False
    try:  # pragma: no cover - optional GPU libs
        import cudf  # type: ignore
        import cupy as cp  # type: ignore
        train_c = cudf.from_pandas(train_cut) if not isinstance(train_cut, cudf.DataFrame) else train_cut
        future_c = cudf.from_pandas(future_calendar) if not isinstance(future_calendar, cudf.DataFrame) else future_calendar
        g = train_c.groupby("series_id")["매출수량"]
        p_series_c = g.gt(0).mean()
        global_mean = float(p_series_c.mean())
        p_series = p_series_c.to_pandas().to_dict()

        def prior_logit_gpu(series_ids):
            p = series_ids.map(p_series).fillna(global_mean)
            eps = 1e-4
            return cp.log((p + eps) / (1 - (p + eps)))

        X_dow = cudf.get_dummies(train_c["DOW"], prefix="dow", dtype="float32")
        X = X_dow
        X["prior"] = prior_weight * prior_logit_gpu(train_c["series_id"])
        y = train_c["매출수량"].gt(0).astype("float32")

        Xf_dow = cudf.get_dummies(future_c["DOW"], prefix="dow", dtype="float32")
        for c in X_dow.columns:
            if c not in Xf_dow.columns:
                Xf_dow[c] = 0.0
        Xf = Xf_dow[X_dow.columns]
        if "series_id" in future_c.columns:
            Xf["prior"] = prior_weight * prior_logit_gpu(future_c["series_id"])
        else:
            eps = 1e-4
            prior_val = cp.log((global_mean + eps) / (1 - (global_mean + eps)))
            Xf["prior"] = prior_weight * prior_val

        device = torch_device(prefer_mps=True)
        torch.manual_seed(seed)
        Xt = torch.as_tensor(X.to_cupy(), dtype=torch.float32, device=device)
        yt = torch.as_tensor(y.to_cupy(), dtype=torch.float32, device=device).view(-1, 1)
        Xft = torch.as_tensor(Xf.to_cupy(), dtype=torch.float32, device=device)
        use_cudf = True
    except Exception:
        g = train_cut.groupby("series_id")["매출수량"]
        p_series = (g.apply(lambda s: (s > 0).mean())).to_dict()

        def prior_logit(series_ids: pd.Series):
            p = series_ids.map(p_series).fillna(g.apply(lambda s: (s > 0).mean()).mean())
            eps = 1e-4
            return np.log((p + eps) / (1 - (p + eps)))

        X_dow = pd.get_dummies(train_cut["DOW"], prefix="dow", drop_first=False)
        X = X_dow.astype(float)
        X["prior"] = prior_weight * prior_logit(train_cut["series_id"])
        y = (train_cut["매출수량"] > 0).astype(float).values

        Xf_dow = pd.get_dummies(future_calendar["DOW"], prefix="dow", drop_first=False)
        Xf = Xf_dow.astype(float).reindex(columns=X_dow.columns, fill_value=0.0)
        if "series_id" in future_calendar.columns:
            Xf["prior"] = prior_weight * prior_logit(future_calendar["series_id"])
        else:
            Xf["prior"] = prior_weight * np.mean(list(p_series.values()))

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
        preds = []
        for start in range(0, Xft.shape[0], batch_size):
            preds.append(model(Xft[start:start + batch_size]).squeeze(1))
        pf = torch.cat(preds, dim=0).clamp(0, 1).detach()

    if device == "cpu":
        return pf.cpu().numpy()
    return pf
