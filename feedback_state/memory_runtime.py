"""Online memory over one event stream: projected addresses + Kalman state + read/write helpers.

Shared by the record and every analysis of it, so all run exactly the same
decide-then-update protocol: ``read(t)`` returns the memory's log-odds and
evidence counts for the real candidates of event ``t`` before any label of
that event is seen; ``write(t, labels)`` applies the feedback afterwards.
"""
from __future__ import annotations

import torch

from feedback_state.addresses import Projection, design_dim, design_rows, memory_evidence
from feedback_state.feature_streams import FeatureStream
from feedback_state.kalman_memory import KalmanMemory


class MemoryRuntime:
    def __init__(self, *, design: str, proj_q: Projection, proj_c: Projection, num_peers: int, lam: float, rho: float = 1.0, device=None) -> None:
        self.design, self.proj_q, self.proj_c = str(design), proj_q, proj_c
        self.P, self.lam, self.rho = int(num_peers), float(lam), float(rho)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.per_peer = self.design == "q"
        self.D = design_dim(self.design, self.P, proj_q.dim, proj_c.dim)
        self.mem = KalmanMemory(self.D, self.P if self.per_peer else 1, lam=self.lam, rho=self.rho, device=self.device)
        self.psi_q = self.psi_c = self.z = self.real = None

    def attach(self, fs: FeatureStream, *, q_source: str = "sem") -> None:
        src = fs.sem if q_source == "sem" else fs.q_mean
        N, P = len(fs), fs.num_peers
        self.psi_q = self.proj_q(src).to(torch.float64)
        self.psi_c = self.proj_c(fs.peer_hidden.reshape(N * P, -1)).reshape(N, P, -1).to(torch.float64)
        self.z = fs.margins.to(self.device, torch.float64)
        self.real = fs.real.tolist()

    @classmethod
    def from_addresses(cls, saved: dict, device=None) -> "MemoryRuntime":
        """A cold runtime over the addresses pipeline.record --save-addresses wrote (no features, no PCA fit)."""
        self = cls.__new__(cls)
        self.design, self.P, self.lam, self.rho = str(saved["design"]), int(saved["num_peers"]), float(saved["lam"]), float(saved.get("rho", 1.0))
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.per_peer = self.design == "q"
        self.proj_q = self.proj_c = None
        self.psi_q, self.psi_c = saved["psi_q"].to(torch.float64), saved["psi_c"].to(torch.float64)
        self.z, self.real = saved["z"].to(self.device, torch.float64), [int(r) for r in saved["real"]]
        self.D = design_dim(self.design, self.P, self.psi_q.shape[-1], self.psi_c.shape[-1])
        self.mem = KalmanMemory(self.D, self.P if self.per_peer else 1, lam=self.lam, rho=self.rho, device=self.device)
        return self

    def addresses(self) -> dict:
        """What from_addresses needs: the projected addresses of every event."""
        return {"design": self.design, "num_peers": self.P, "lam": self.lam, "rho": self.rho, "psi_q": self.psi_q.cpu(),
                "psi_c": self.psi_c.cpu(), "z": self.z.cpu(), "real": list(self.real)}

    def reset(self) -> None:
        self.mem.reset()

    def rows(self, t: int) -> torch.Tensor:
        r = self.real[t]
        return design_rows(self.design, self.psi_q[t], self.psi_c[t, :r], self.z[t, :r], list(range(r)), self.P)

    @torch.no_grad()
    def read(self, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (ell [r] log-odds, n_eff [r], evidence [r, 4], rows X) for the real candidates of event t."""
        r = self.real[t]
        X = self.rows(t)
        mu, var = self.mem.read(X)
        if self.per_peer:
            mu, var = mu[0, :r], var.expand(r)
            Xe = X.expand(r, -1)
        else:
            mu, Xe = mu[:, 0], X
        p = self.mem.prob(mu, var)
        ell = self.mem.logit(p)
        n_eff = self.mem.evidence(Xe, var)
        return ell, n_eff, memory_evidence(ell, n_eff), X

    @torch.no_grad()
    def write(self, t: int, X: torch.Tensor, labels) -> None:
        r = self.real[t]
        s = torch.tensor([1.0 if int(labels[c]) else -1.0 for c in range(r)], device=self.device, dtype=torch.float64)
        if self.per_peer:
            sv = torch.zeros(self.P, device=self.device, dtype=torch.float64)
            sv[:r] = s
            self.mem.write(X[0], sv)
        else:
            for c in range(r):
                self.mem.write(X[c], s[c : c + 1])

    @torch.no_grad()
    def write_targets(self, X: torch.Tensor, targets: dict[int, float]) -> None:
        """Write only the observed candidates, each with a target in [-1, 1] (2 * accuracy - 1 for an average of samples)."""
        if self.per_peer:
            raise ValueError("write_targets needs a shared-state design (one row per candidate)")
        for c, s in sorted(targets.items()):
            self.mem.write(X[c], torch.tensor([float(s)], device=self.device, dtype=torch.float64))

    def config(self) -> dict:
        return {"design": self.design, "num_peers": self.P, "lam": self.lam, "rho": self.rho,
                "dim_q": self.psi_q.shape[-1] if self.psi_q is not None else None, "dim_c": self.psi_c.shape[-1] if self.psi_c is not None else None}
