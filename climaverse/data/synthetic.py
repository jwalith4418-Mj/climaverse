"""Format-identical SYNTHETIC climate fields.

>>> THIS IS NOT REAL DATA. <<<
It exists so the entire twin pipeline runs offline and the validation numbers are
meaningful (a model that beats climatology on this field is doing real work). It is
physically *plausible*, not physically *correct*: seasonal cycle + latitude gradient
+ a monsoon (JJAS) rainfall regime + spatial smoothness + day-to-day persistence.
Swap in real IMD .grd via data/loader.py and every downstream number becomes real.
"""
from __future__ import annotations
import numpy as np
from ..config import Grid


def _smooth(field: np.ndarray, passes: int = 2) -> np.ndarray:
    f = field.copy()
    for _ in range(passes):
        f = (f
             + np.roll(f, 1, 0) + np.roll(f, -1, 0)
             + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5.0
    return f


def synth_temperature(grid: Grid, n_days: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nlat, nlon = grid.shape
    lat = grid.lats[:, None]                        # south is hotter (lower lat)
    doy = np.arange(n_days)
    season = 7.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)   # peak ~ late Apr
    base = 32.0 - 0.45 * (lat - lat.mean())                   # ~26-36 C span
    field = np.empty((n_days, nlat, nlon), "float32")
    state = np.broadcast_to(base, (nlat, nlon)).astype("float32").copy()
    for t in range(n_days):
        target = base + season[t]
        # AR(1) persistence around the seasonal target + smooth spatial noise
        noise = _smooth(rng.normal(0, 1.4, (nlat, nlon)).astype("float32"))
        state = 0.75 * state + 0.25 * (target + noise)
        field[t] = state
    return field


def synth_rainfall(grid: Grid, n_days: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nlat, nlon = grid.shape
    doy = np.arange(n_days) % 365
    # Monsoon envelope: SW monsoon Jun-Sep (doy ~152-273), plus weak NE monsoon.
    monsoon = np.exp(-((doy - 210) ** 2) / (2 * 38.0 ** 2))
    ne = 0.25 * np.exp(-((doy - 305) ** 2) / (2 * 22.0 ** 2))
    intensity = 18.0 * monsoon + 6.0 * ne + 0.4
    field = np.zeros((n_days, nlat, nlon), "float32")
    for t in range(n_days):
        wet = rng.random((nlat, nlon)) < (0.12 + 0.55 * monsoon[t] + 0.2 * ne[t])
        amount = rng.gamma(shape=1.3, scale=intensity[t], size=(nlat, nlon))
        field[t] = _smooth((wet * amount).astype("float32"))
    return np.clip(field, 0, None)
