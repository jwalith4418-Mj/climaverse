"""Honest gridded-forecast metrics (NaN-aware)."""
from __future__ import annotations
import numpy as np


def _flat(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def rmse(pred, obs) -> float:
    p, o = _flat(np.asarray(pred), np.asarray(obs))
    return float(np.sqrt(np.mean((p - o) ** 2))) if p.size else float("nan")


def mae(pred, obs) -> float:
    p, o = _flat(np.asarray(pred), np.asarray(obs))
    return float(np.mean(np.abs(p - o))) if p.size else float("nan")


def bias(pred, obs) -> float:
    p, o = _flat(np.asarray(pred), np.asarray(obs))
    return float(np.mean(p - o)) if p.size else float("nan")


def skill_score(model_rmse: float, ref_rmse: float) -> float:
    """Murphy skill score vs a reference (climatology). 1=perfect, 0=no better, <0=worse."""
    if ref_rmse <= 0 or np.isnan(ref_rmse):
        return float("nan")
    return float(1.0 - model_rmse / ref_rmse)
