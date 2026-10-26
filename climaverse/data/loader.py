"""Unified observation loader: REAL IMD .grd if present, else synthetic fallback.

Returns an Observation: the full-grid daily field + grid metadata + a `real` flag so
every downstream report can state honestly whether it is running on observations or
on the offline stand-in.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..config import Grid, RAIN_GRID, TEMP_GRID
from . import synthetic, imd_grid

RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

VARIABLES = {
    "rain":  ("Rainfall (mm/day)",        RAIN_GRID, synthetic.synth_rainfall),
    "tmax":  ("Max temperature (deg C)",  TEMP_GRID, synthetic.synth_temperature),
}


@dataclass
class Observation:
    var: str
    label: str
    grid: Grid
    data: np.ndarray          # [day, lat, lon], NaN = missing/ocean
    real: bool


def load_variable(var: str, n_days: int = 730, grd_path: str | None = None) -> Observation:
    if var not in VARIABLES:
        raise KeyError(f"unknown variable {var!r}; choose from {list(VARIABLES)}")
    label, grid, synth_fn = VARIABLES[var]
    path = grd_path
    if path is None:
        cand = list(RAW.glob(f"{var}*.grd"))
        path = str(cand[0]) if cand else None
    if path and Path(path).exists():
        data = imd_grid.read_grd(path, grid, n_days=n_days)
        return Observation(var, label, grid, data, real=True)
    data = synth_fn(grid, n_days)
    return Observation(var, label, grid, data, real=False)
