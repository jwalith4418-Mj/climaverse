"""Unified prediction interface.

Predictor.fit() trains the ConvLSTM on day-of-year-anomaly targets if torch is
available; predict_next() returns a next-day field. If torch is missing, it degrades
gracefully to a climatology+persistence blend so the pipeline ALWAYS runs and always
reports against the same baselines.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import baselines
from .convlstm import _HAS_TORCH

if _HAS_TORCH:
    import torch
    from .convlstm import ConvLSTMNet


@dataclass
class Predictor:
    seq_len: int = 7
    hidden: int = 24
    epochs: int = 12
    lr: float = 5e-3
    backend: str = "convlstm" if _HAS_TORCH else "baseline-blend"
    _clim: np.ndarray = None
    _net: "object" = None
    _mu: float = 0.0
    _sd: float = 1.0

    def fit(self, train: np.ndarray, start_doy: int = 0, log=lambda *a: None) -> "Predictor":
        self._clim = baselines.day_of_year_climatology(train, start_doy)
        train = np.nan_to_num(train, nan=0.0).astype("float32")
        self._mu, self._sd = float(train.mean()), float(train.std() + 1e-6)
        if not _HAS_TORCH:
            log(f"[model] torch unavailable -> backend={self.backend}")
            return self

        T, H, W = self.seq_len, train.shape[1], train.shape[2]
        doy = (np.arange(train.shape[0]) + start_doy) % 365
        clim_seq = self._clim[doy]                         # [N,H,W] aligned climatology
        X, Yt = [], []
        for i in range(T, train.shape[0]):
            field = train[i - T:i]
            cl = clim_seq[i - T:i]
            X.append(np.stack([field, cl], axis=1))        # [T,2,H,W]
            Yt.append(train[i])                             # observed target field
        X = (np.asarray(X) - self._mu) / self._sd
        Yt = (np.asarray(Yt) - self._mu) / self._sd

        dev = "cpu"
        net = ConvLSTMNet(self.hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        lossf = torch.nn.SmoothL1Loss()
        xb = torch.tensor(X, dtype=torch.float32, device=dev)
        yb = torch.tensor(Yt, dtype=torch.float32, device=dev)
        n = xb.shape[0]; bs = 16
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                pred = net(xb[idx])
                loss = lossf(pred, yb[idx])
                loss.backward(); opt.step()
                tot += float(loss.detach()) * len(idx)
            log(f"[model] epoch {ep+1}/{self.epochs} loss={tot/n:.4f}")
        self._net = net
        return self

    def predict_next(self, recent_seq: np.ndarray, target_doy: int) -> np.ndarray:
        """recent_seq: last seq_len observed days [T,H,W]; returns next-day field."""
        clim_t = baselines.climatology_forecast(self._clim, target_doy)
        if not _HAS_TORCH or self._net is None:
            persist = recent_seq[-1]
            return 0.5 * clim_t + 0.5 * persist            # honest fallback blend
        T = self.seq_len
        doy = (target_doy - np.arange(T, 0, -1)) % 365
        cl = self._clim[doy]
        seq = np.stack([np.nan_to_num(recent_seq, nan=0.0), cl], axis=1)[None]
        seq = (seq - self._mu) / self._sd
        with torch.no_grad():
            out = self._net(torch.tensor(seq, dtype=torch.float32))
        return (out.numpy()[0] * self._sd + self._mu).astype("float32")
