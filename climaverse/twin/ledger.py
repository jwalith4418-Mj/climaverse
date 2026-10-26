"""Append-only, hash-chained sync ledger.

Every synchronisation appends a record whose hash includes the previous record's
hash, so the twin's update history is tamper-evident and replayable. (Carried over
from the NEXUS Forward Ledger: an auditable state history is what lets us later
prove 'the twin said X at time T, reality returned Y' — the basis of validation.)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib, json, time
from typing import Any


@dataclass
class SyncRecord:
    seq: int
    obs_day: int
    state_hash: str
    n_cells_updated: int
    mean_value: float
    prev_hash: str
    ts: float = field(default_factory=time.time)
    hash: str = ""

    def compute_hash(self) -> str:
        body = json.dumps({k: v for k, v in asdict(self).items() if k != "hash"},
                          sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


class SyncLedger:
    GENESIS = "0" * 16

    def __init__(self) -> None:
        self._rows: list[SyncRecord] = []

    def append(self, obs_day: int, state_hash: str, n_updated: int, mean_value: float) -> SyncRecord:
        prev = self._rows[-1].hash if self._rows else self.GENESIS
        rec = SyncRecord(len(self._rows), obs_day, state_hash, n_updated,
                         round(float(mean_value), 4), prev)
        rec.hash = rec.compute_hash()
        self._rows.append(rec)
        return rec

    def verify(self) -> bool:
        prev = self.GENESIS
        for r in self._rows:
            if r.prev_hash != prev or r.hash != r.compute_hash():
                return False
            prev = r.hash
        return True

    def __len__(self) -> int:
        return len(self._rows)

    def tail(self, n: int = 5) -> list[dict[str, Any]]:
        return [asdict(r) for r in self._rows[-n:]]
