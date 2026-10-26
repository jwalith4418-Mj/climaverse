"""Scientific baselines. A climate PoC that does not beat these has shown nothing.

  climatology : day-of-year mean (the bar every weather/climate model is judged on)
  persistence : tomorrow == today (the bar for short-horizon prediction)
Skill is reported RELATIVE to climatology, so 'good RMSE' can never hide a model that
merely memorised the seasonal cycle.
"""
from __future__ import annotations
import numpy as np


def day_of_year_climatology(train: np.ndarray, start_doy: int = 0) -> np.ndarray:
    """Mean field for each day-of-year (mod 365) from the training period."""
    n = train.shape[0]
    doy = (np.arange(n) + start_doy) % 365
    clim = np.full((365, *train.shape[1:]), np.nan, "float32")
    for d in range(365):
        sel = train[doy == d]
        if sel.size:
            clim[d] = np.nanmean(sel, axis=0)
    # fill any empty doy with overall mean
    overall = np.nanmean(train, axis=0)
    empty = np.all(np.isnan(clim.reshape(365, -1)), axis=1)
    clim[empty] = overall
    return clim


def climatology_forecast(clim: np.ndarray, target_doy: int) -> np.ndarray:
    return clim[target_doy % 365]


def persistence_forecast(last_obs: np.ndarray) -> np.ndarray:
    return last_obs.copy()
