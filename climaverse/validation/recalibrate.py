"""The LEARN step: an online bias-corrector the twin fits on its own resolved errors.

After each holdout day resolves, we know (prediction, reality). We fit a per-magnitude
bias correction (isotonic, monotone) so systematic over/under-prediction is removed as
evidence accumulates. Honesty invariant inherited from NEXUS: below MIN_N resolved
pairs the correction is the IDENTITY — we do not 'learn' a curve we have not earned.
"""
from __future__ import annotations
import numpy as np

MIN_N = 60


class BiasCorrector:
    def __init__(self, n_bins: int = 8):
        self.n_bins = n_bins
        self.edges: np.ndarray | None = None
        self.delta: np.ndarray | None = None
        self._n = 0

    def fit(self, pred: np.ndarray, obs: np.ndarray) -> "BiasCorrector":
        p, o = pred.ravel(), obs.ravel()
        m = ~(np.isnan(p) | np.isnan(o)); p, o = p[m], o[m]
        self._n = p.size
        if p.size < MIN_N:
            self.edges = self.delta = None
            return self
        self.edges = np.quantile(p, np.linspace(0, 1, self.n_bins + 1))
        self.edges[0] -= 1e-6; self.edges[-1] += 1e-6
        idx = np.clip(np.digitize(p, self.edges) - 1, 0, self.n_bins - 1)
        delta = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            sel = idx == b
            if sel.any():
                delta[b] = float(np.mean(o[sel] - p[sel]))
        # enforce monotone (PAVA-lite) so the correction can't invert ordering
        for b in range(1, self.n_bins):
            delta[b] = max(delta[b], delta[b - 1] - abs(delta[b - 1]) * 0)
        self.delta = delta
        return self

    def apply(self, pred: np.ndarray) -> np.ndarray:
        if self.delta is None:
            return pred                      # identity until evidence earned
        idx = np.clip(np.digitize(pred, self.edges) - 1, 0, self.n_bins - 1)
        return pred + self.delta[idx]

    @property
    def active(self) -> bool:
        return self.delta is not None
