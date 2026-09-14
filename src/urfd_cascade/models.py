"""Temporal classifiers evaluated on DDL feature windows: a two-parameter
physics threshold, a small LSTM, and the proposed Liquid Time-Constant
(LTC) network (Hasani et al., 2021), used as a compact edge-deployable
alternative to the LSTM.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Small single-layer LSTM classifier over a feature-window."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)


class LTCCell(nn.Module):
    """Simplified closed-form Liquid Time-Constant cell (Hasani et al.,
    2021, AAAI), discretised with a fixed unit time step:

        h <- (h + dt * f * A) / (1 + dt * (1/tau + gate))

    This is a compact approximation used for the edge-deployable filter;
    it is not a full ODE solver, trading some fidelity for parameter count
    and inference speed, which is the point of using LTC here.
    """

    def __init__(self, in_dim: int, hidden: int = 24):
        super().__init__()
        self.hidden = hidden
        self.w_f = nn.Linear(in_dim + hidden, hidden)
        self.w_tau = nn.Linear(in_dim + hidden, hidden)
        self.a = nn.Parameter(torch.randn(hidden) * 0.1)
        self.tau0 = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        h = torch.zeros(batch, self.hidden, device=x.device)
        for t in range(steps):
            inp = torch.cat([x[:, t], h], dim=-1)
            f = torch.tanh(self.w_f(inp))
            gate = torch.sigmoid(self.w_tau(inp))
            tau = torch.nn.functional.softplus(self.tau0) + 1e-3
            h = (h + f * self.a) / (1.0 + (1.0 / tau + gate))
        return h


class LTCClassifier(nn.Module):
    """LTC-based fall classifier: the edge-deployable filter proposed in
    the paper's first cascade."""

    def __init__(self, in_dim: int, hidden: int = 24):
        super().__init__()
        self.cell = LTCCell(in_dim, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cell(x)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
