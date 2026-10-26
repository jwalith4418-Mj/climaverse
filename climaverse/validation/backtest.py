"""Rolling-origin (walk-forward) validation = the twin's COMPARE step, made real.

We never test on the past we trained on. We train on [0:split], then for each day in
the holdout we predict next-day from the true recent window, score it against the
withheld observation, and accumulate. The reference is day-of-year climatology fit on
the TRAIN period only. Reported skill is model-vs-climatology on unseen days.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from ..model import Predictor, baselines
from .metrics import rmse, mae, bias, skill_score


@dataclass
class BacktestResult:
    n_days: int
    model_rmse: float
    clim_rmse: float
    persist_rmse: float
    model_mae: float
    model_bias: float
    skill_vs_clim: float
    skill_vs_persist: float
    backend: str
    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def walk_forward(field: np.ndarray, split: float = 0.7, seq_len: int = 7,
                 epochs: int = 8, start_doy: int = 0, log=lambda *a: None) -> BacktestResult:
    n = field.shape[0]
    cut = int(n * split)
    train = field[:cut]
    pred = Predictor(seq_len=seq_len, epochs=epochs).fit(train, start_doy=start_doy, log=log)
    clim = baselines.day_of_year_climatology(train, start_doy)

    m_pred, c_pred, p_pred, obs = [], [], [], []
    for i in range(max(cut, seq_len), n):
        target_doy = (i + start_doy) % 365
        recent = field[i - seq_len:i]
        m_pred.append(pred.predict_next(recent, target_doy))
        c_pred.append(baselines.climatology_forecast(clim, target_doy))
        p_pred.append(baselines.persistence_forecast(field[i - 1]))
        obs.append(field[i])
    m_pred, c_pred, p_pred, obs = map(np.asarray, (m_pred, c_pred, p_pred, obs))

    mr, cr, pr = rmse(m_pred, obs), rmse(c_pred, obs), rmse(p_pred, obs)
    return BacktestResult(
        n_days=len(obs), model_rmse=mr, clim_rmse=cr, persist_rmse=pr,
        model_mae=mae(m_pred, obs), model_bias=bias(m_pred, obs),
        skill_vs_clim=skill_score(mr, cr), skill_vs_persist=skill_score(mr, pr),
        backend=pred.backend,
    )
