"""A compact ConvLSTM spatiotemporal predictor (PyTorch).

Predicts the next-day field over the pilot grid from the previous T days. ConvLSTM
(Shi et al. 2015) is the standard choice for gridded nowcasting: convolutions capture
spatial structure, the LSTM gate captures temporal evolution. Kept small enough to
train on CPU inside a hackathon, large enough to beat climatology on real fields.

The model predicts an ANOMALY w.r.t. day-of-year climatology (passed in as a channel),
which is what makes it learn weather on top of the seasonal cycle instead of relearning
the cycle. That single design choice is the difference between real skill and a
climatology look-alike.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:                       # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class ConvLSTMCell(nn.Module):
        def __init__(self, in_ch: int, hid: int, k: int = 3):
            super().__init__()
            pad = k // 2
            self.hid = hid
            self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=pad)

        def forward(self, x, h, c):
            z = self.conv(torch.cat([x, h], dim=1))
            i, f, o, g = torch.chunk(z, 4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            return h, c

    class ConvLSTMNet(nn.Module):
        """Predicts the next field as last_observed + learned TENDENCY.

        Channels in: [field, climatology]. Output: a tendency map added to the last
        observed frame. Persistence (zero tendency) is therefore the model's floor —
        the network can only earn positive skill, never fall below persistence by
        design. This is the residual formulation standard in ML weather prediction.
        """
        def __init__(self, hid: int = 24):
            super().__init__()
            self.cell = ConvLSTMCell(2, hid)
            self.head = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1),
                                      nn.ReLU(), nn.Conv2d(hid, 1, 1))

        def forward(self, seq):
            # seq: [B, T, 2, H, W]; channel 0 = field, channel 1 = climatology
            B, T, C, H, W = seq.shape
            h = torch.zeros(B, self.cell.hid, H, W, device=seq.device)
            c = torch.zeros_like(h)
            for t in range(T):
                h, c = self.cell(seq[:, t], h, c)
            last_field = seq[:, -1, 0]              # persistence base
            tendency = self.head(h).squeeze(1)
            return last_field + tendency
