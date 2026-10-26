"""Synchronisation: the heartbeat of the twin.

assimilate one new observation day -> update each cell's estimate and confidence ->
append an audit record. Confidence is a simple, honest model: fresh observation sets
near-1, and unobserved cells decay geometrically. (A production twin swaps this for a
Kalman/EnKF update; the interface is identical.)
"""
from __future__ import annotations
import numpy as np
from ..config import subset_indices
from .state import TwinState
from .ledger import SyncLedger

DECAY = 0.85            # per-day confidence decay for cells that get no new obs
OBS_CONF = 0.98         # confidence assigned to a freshly observed cell


def synchronise(state: TwinState, obs_field_full: np.ndarray, obs_day: int,
                ledger: SyncLedger | None = None) -> TwinState:
    la, lo = subset_indices(state.grid, state.region)
    obs = obs_field_full[la, lo].astype("float32")
    valid = ~np.isnan(obs)

    # decay everywhere, then snap observed cells to the observation
    state.confidence *= DECAY
    new_val = state.value.copy()
    new_val[valid] = obs[valid]
    # initialise never-seen cells so the field is renderable from day 0
    cold = np.isnan(new_val) & valid
    new_val[cold] = obs[cold]
    state.confidence[valid] = OBS_CONF

    state.value = new_val
    state.last_obs_day = obs_day
    import time as _t
    state.updated_at = _t.time()

    if ledger is not None:
        ledger.append(obs_day, state.fingerprint(),
                      int(valid.sum()), float(np.nanmean(new_val)))
    return state
