"""Grid + pilot-region configuration.

Grids match the OFFICIAL IMD gridded products named in the problem statement:
  * Rainfall 0.25 deg  (Pai et al. 2014) : lat 6.5..38.5 (129), lon 66.5..100.0 (135)
  * Max/Min Temp 1.0 deg                  : lat 7.5..37.5 (31),  lon 67.5..97.5  (31)

We never invent grids. If you drop the real .grd files into data/raw/ the readers
in data/imd_grid.py parse them with these exact conventions; otherwise the loader
falls back to a clearly-labelled synthetic field of identical shape so the whole
pipeline runs offline.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    name: str
    lat0: float
    lat1: float
    lon0: float
    lon1: float
    step: float
    missing: float

    @property
    def lats(self) -> np.ndarray:
        return np.round(np.arange(self.lat0, self.lat1 + 1e-6, self.step), 4)

    @property
    def lons(self) -> np.ndarray:
        return np.round(np.arange(self.lon0, self.lon1 + 1e-6, self.step), 4)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.lats), len(self.lons))


# Official IMD product grids.
RAIN_GRID = Grid("IMD-0.25-rain", 6.5, 38.5, 66.5, 100.0, 0.25, -999.0)
TEMP_GRID = Grid("IMD-1.0-temp", 7.5, 37.5, 67.5, 97.5, 1.0, 99.9)


@dataclass(frozen=True)
class PilotRegion:
    """The PoC pilot region. The twin scales to all-India by widening this box;
    nothing in the pipeline is hard-coded to the pilot extent."""
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# Peninsular SE India: Telangana + coastal Andhra. Strong monsoon + heat-stress
# signal -> a pilot where rainfall AND temperature prediction both matter, and
# where 'what-if' (monsoon deficit, +2C) maps to real agricultural decisions.
PILOT = PilotRegion("Telangana-CoastalAP", 13.0, 20.0, 76.0, 85.0)


def subset_indices(grid: Grid, region: PilotRegion) -> tuple[slice, slice]:
    lats, lons = grid.lats, grid.lons
    la = np.where((lats >= region.lat_min) & (lats <= region.lat_max))[0]
    lo = np.where((lons >= region.lon_min) & (lons <= region.lon_max))[0]
    return slice(la[0], la[-1] + 1), slice(lo[0], lo[-1] + 1)
