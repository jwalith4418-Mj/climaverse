"""The living twin state.

The twin is NOT the forecast. It is a continuously-updated estimate of the *current*
climate state over the pilot grid, with a freshness clock and a per-cell confidence
that decays as observations age and rises when fresh data confirms the estimate.
This is the object the dashboard renders and the simulator perturbs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
from ..config import Grid, PilotRegion, subset_indices


@dataclass
class TwinState:
    grid: Grid
    region: PilotRegion
    var: str
    value: np.ndarray            # current best estimate [lat, lon]
    confidence: np.ndarray       # 0..1 per cell
    last_obs_day: int            # index of the most recent assimilated day
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def empty(cls, grid: Grid, region: PilotRegion, var: str) -> "TwinState":
        la, lo = subset_indices(grid, region)
        shape = (la.stop - la.start, lo.stop - lo.start)
        return cls(grid, region, var,
                   value=np.full(shape, np.nan, "float32"),
                   confidence=np.zeros(shape, "float32"),
                   last_obs_day=-1)

    def fingerprint(self) -> str:
        """Stable content hash of the state for the audit ledger."""
        import hashlib
        v = np.nan_to_num(self.value, nan=-9999.0).tobytes()
        c = np.round(self.confidence, 4).tobytes()
        return hashlib.sha256(v + c + str(self.last_obs_day).encode()).hexdigest()[:16]
