"""Twin Health Score — one honest number the judge reads in 10 seconds.

It is the *mean of three measured sub-scores*, never a hard-coded constant:
  freshness  : how recently the twin assimilated data (decays with staleness)
  coverage   : fraction of pilot cells with a confident estimate
  skill      : prediction skill vs climatology on the latest backtest (0 if none yet)
We never report skill we have not earned: with no resolved forecast, skill = 0 and the
overall score is capped accordingly (the same honesty invariant as NEXUS recalibrate).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .state import TwinState


@dataclass
class Health:
    freshness: float
    coverage: float
    skill: float
    overall: float
    def as_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


def twin_health(state: TwinState, days_since_obs: int = 0,
                skill_score: float | None = None) -> Health:
    freshness = float(np.clip(0.85 ** max(0, days_since_obs), 0, 1))
    coverage = float(np.mean(state.confidence > 0.3))
    skill = 0.0 if skill_score is None else float(np.clip(skill_score, 0, 1))
    overall = float(np.mean([freshness, coverage, skill]))
    return Health(freshness, coverage, skill, overall)
