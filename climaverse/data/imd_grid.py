"""Readers for the OFFICIAL IMD gridded binary (.grd) products.

These honour the byte layout documented by IMD Pune. Per day the stream is written
Fortran-style with longitude as the fastest index:
    do j = 1, nlat:  do i = 1, nlon:  write value(i, j)
so a year file is float32 of length n_days * nlat * nlon, and
    np.fromfile(...).reshape(n_days, nlat, nlon)
yields [day, lat(south->north), lon(west->east)].

No real file is bundled (the IMD portal requires manual download). Drop a .grd into
data/raw/ and point load_variable() at it to use real observations; the parser is
exercised by a round-trip test in tests/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..config import Grid


def read_grd(path: str | Path, grid: Grid, n_days: int | None = None) -> np.ndarray:
    """Read an IMD .grd file into [day, lat, lon] with missing -> np.nan."""
    raw = np.fromfile(str(path), dtype="<f4")
    nlat, nlon = grid.shape
    per_day = nlat * nlon
    if raw.size % per_day != 0:
        raise ValueError(
            f"{path}: {raw.size} floats not divisible by {per_day} (grid {grid.shape}). "
            "Wrong grid for this product?"
        )
    days = raw.size // per_day
    arr = raw.reshape(days, nlat, nlon).astype("float32")
    if n_days is not None:
        arr = arr[:n_days]
    arr[np.isclose(arr, grid.missing)] = np.nan
    return arr


def write_grd(path: str | Path, arr: np.ndarray, grid: Grid) -> None:
    """Inverse of read_grd — used to manufacture format-identical test fixtures."""
    out = np.where(np.isnan(arr), grid.missing, arr).astype("<f4")
    out.tofile(str(path))
