from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

from .mps_utils import has_torch, torch_device


def _series_dow_lookup(
    train_cut: pd.DataFrame,
    window_weeks: int = 8,
    alpha: float = 0.5,
    beta: float = 0.5,
    date_col: str = "영업일자",
    target_col: str = "매출수량",
):
    """Return a lookup function for P(y>0 | series_id, DOW) with Beta smoothing."""
    cutoff_date = train_cut[date_col].max()
    win_start = (
        cutoff_date - pd.Timedelta(days=7 * window_weeks)
        if window_weeks is not None
        else None
    )
    if win_start is not None:
        df_win = train_cut.loc[train_cut[date_col] >= win_start]
    else:
        df_win = train_cut

    grp = df_win.groupby(["series_id", "DOW"])[target_col]
    total = grp.size().astype(float)
    nonzero = grp.apply(lambda s: (s > 0).sum()).astype(float)
    p_win = ((nonzero + alpha) / (total + alpha + beta)).to_dict()

    grp_all = train_cut.groupby(["series_id", "DOW"])[target_col]
    total_all = grp_all.size().astype(float)
    nonzero_all = grp_all.apply(lambda s: (s > 0).sum()).astype(float)
    p_all = ((nonzero_all + alpha) / (total_all + alpha + beta)).to_dict()

    grp_g = train_cut.groupby("DOW")[target_col]
    total_g = grp_g.size().reindex(range(7), fill_value=0).astype(float)
    nonzero_g = (
        grp_g.apply(lambda s: (s > 0).sum())
        .reindex(range(7), fill_value=0)
        .astype(float)
    )
    p_g = ((nonzero_g + alpha) / (total_g + alpha + beta)).to_dict()

    def lookup(series_id, dow):
        key = (series_id, dow)
        if key in p_win:
            return p_win[key]
        if key in p_all:
            return p_all[key]
        return p_g.get(dow, alpha / (alpha + beta))

    return lookup

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
    window_weeks: int = 8,
    alpha: float = 0.5,
    beta: float = 0.5,
    calib_lambda: float = 0.8,
    class_weight: bool = True,
) -> Union[np.ndarray, "torch.Tensor"]:
    """Calendar-only global logistic regression on (y>0).

    Features: DOW one-hot + Beta-smoothed per-series/DOW prior logits.
    """
    if not has_torch():
        raise RuntimeError(
            "PyTorch not installed. Install torch>=2.1 to use logistic classifier."
        )

    import torch
    import torch.nn.functional as F

    lookup = _series_dow_lookup(
        train_cut,
        window_weeks=window_weeks,
        alpha=alpha,
        beta=beta,
    )

    eps = 1e-4
    train_prior = np.array(
        [lookup(s, d) for s, d in zip(train_cut["series_id"], train_cut["DOW"])]
    )
    train_prior_logit = np.log((train_prior + eps) / (1 - (train_prior + eps)))

    X_dow = pd.get_dummies(train_cut["DOW"], prefix="dow", drop_first=False)
    X = X_dow.astype(float)
    X["prior"] = prior_weight * train_prior_logit
    y = (train_cut["매출수량"] > 0).astype(float).values

    Xf_dow = pd.get_dummies(future_calendar["DOW"], prefix="dow", drop_first=False)
    Xf = Xf_dow.astype(float).reindex(columns=X_dow.columns, fill_value=0.0)
    future_prior = np.array(
        [lookup(s, d) for s, d in zip(future_calendar["series_id"], future_calendar["DOW"])]
    )
    Xf["prior"] = prior_weight * np.log((future_prior + eps) / (1 - (future_prior + eps)))

    device = torch_device(prefer_mps=True)
    torch.manual_seed(seed)
    Xt = torch.tensor(X.values, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    Xft = torch.tensor(Xf.values, dtype=torch.float32, device=device)

    if class_weight:
        n_total = yt.shape[0]
        n_pos = (yt > 0).sum().item()
        n_neg = n_total - n_pos
        w_pos = n_neg / n_total
        w_neg = n_pos / n_total
    else:
        w_pos = w_neg = 1.0
    w_pos_t = torch.tensor(w_pos, device=device)
    w_neg_t = torch.tensor(w_neg, device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(Xt.shape[1], 1),
        torch.nn.Sigmoid(),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)

    n = Xt.shape[0]
    for epoch in range(epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            sel = idx[start : start + batch_size]
            xb, yb = Xt[sel], yt[sel]
            wb = torch.where(yb > 0, w_pos_t, w_neg_t)
            opt.zero_grad()
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb, weight=wb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        preds = []
        for start in range(0, Xft.shape[0], batch_size):
            preds.append(model(Xft[start : start + batch_size]).squeeze(1))
        pf = torch.cat(preds, dim=0).clamp(0, 1)

    pf_np = pf.detach().cpu().numpy()
    pf_np = calib_lambda * pf_np + (1 - calib_lambda) * future_prior
    pf_np = np.clip(pf_np, 0.0, 1.0)

    if device == "cpu":
        return pf_np
    return torch.tensor(pf_np, dtype=torch.float32, device=device)
