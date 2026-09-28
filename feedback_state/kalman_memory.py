"""Kalman (recursive least squares) competence memory.

The memory is the exact Bayesian posterior of a linear-Gaussian model of peer
correctness.  For an address ``psi`` (a vector built from the frozen center
model's hidden states) and a signed correctness ``s in {-1, +1}``:

    s = w^T psi + eps,      w ~ N(0, I / lam),      eps ~ N(0, 1).

Sufficient statistics (additive, hence order-invariant for rho = 1):

    Lambda = lam I + sum_tau rho^{t - tau} psi_tau psi_tau^T        (shared precision)
    b_h    = sum_tau rho^{t - tau} s_{h, tau} psi_tau               (one per head h)

Read-out for a query address psi:

    mu_h  = psi^T Lambda^{-1} b_h          posterior mean of w_h^T psi
    v     = psi^T Lambda^{-1} psi          posterior variance of that mean
    P(s_h > 0 | history) = Phi( mu_h / sqrt(1 + v) )       (probit predictive)

Write (rho = 1): Sherman-Morrison on P = Lambda^{-1}; this is the delta rule
``S <- S + k (s - S psi)`` with the Kalman gain ``k = P psi / (1 + psi^T P psi)``
instead of a fixed step, i.e. the closed-form optimum of the online least
squares objective that DeltaNet / Gated DeltaNet / KDA optimise by one gradient
step per token.  With rho < 1 the precision decays (Gated DeltaNet's alpha, or
RLS with a forgetting factor) and the state is kept as Lambda with a Cholesky
solve at read time so the prior ``lam I`` is preserved.

Heads: ``heads = P`` gives one competence function per peer written once per
event (the same address, one label per peer); ``heads = 1`` gives a single
function written once per candidate (candidate-conditioned addresses), which is
how the memory learns to verify answers instead of only rating peers.
"""
from __future__ import annotations

import math

import torch


class KalmanMemory:
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        lam: float = 100.0,
        rho: float = 1.0,
        device=None,
        dtype=torch.float64,
    ) -> None:
        if dim < 1 or heads < 1 or lam <= 0.0 or not 0.0 < rho <= 1.0:
            raise ValueError("invalid memory configuration")
        self.dim, self.heads, self.lam, self.rho = int(dim), int(heads), float(lam), float(rho)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.reset()

    # ---- state ----
    def reset(self) -> None:
        eye = torch.eye(self.dim, device=self.device, dtype=self.dtype)
        self.b = torch.zeros(self.heads, self.dim, device=self.device, dtype=self.dtype)
        self.writes = 0
        if self.rho >= 1.0:
            self.P = eye / self.lam          # Lambda^{-1}, updated by Sherman-Morrison
            self.Lam = None
        else:
            self.Lam = eye * self.lam        # Lambda itself, decayed towards lam I
            self.P = None
            self._chol = None

    def snapshot(self) -> dict:
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in
                (("P", self.P), ("Lam", self.Lam), ("b", self.b), ("writes", self.writes))}

    def restore(self, snap: dict) -> None:
        self.P = snap["P"].clone() if snap["P"] is not None else None
        self.Lam = snap["Lam"].clone() if snap["Lam"] is not None else None
        self._chol = None
        self.b = snap["b"].clone()
        self.writes = snap["writes"]

    # ---- read ----
    def _solve(self, x: torch.Tensor) -> torch.Tensor:
        """Lambda^{-1} x for x of shape [..., dim]."""
        if self.P is not None:
            return x @ self.P
        if self._chol is None:
            self._chol = torch.linalg.cholesky(self.Lam)
        return torch.cholesky_solve(x.reshape(-1, self.dim).T, self._chol).T.reshape(x.shape)

    @torch.no_grad()
    def read(self, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """psi [..., dim] -> (mu [..., heads], var [...])."""
        psi = psi.to(self.device, self.dtype)
        ppsi = self._solve(psi)
        mu = ppsi @ self.b.T
        var = (ppsi * psi).sum(-1)
        return mu, var

    @staticmethod
    def prob(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Predictive P(s > 0) = Phi(mu / sqrt(1 + var)); uncertain reads shrink to 1/2."""
        v = var.unsqueeze(-1) if mu.dim() > var.dim() else var
        z = mu / torch.sqrt(1.0 + v)
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    @staticmethod
    def logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def evidence(self, psi: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Effective number of past observations along the query direction:
        n(x) = psi^T (lam I)^{-1} psi / (psi^T Lambda^{-1} psi) - 1  (0 at cold start, grows with coverage)."""
        prior_var = (psi.to(self.device, self.dtype) ** 2).sum(-1) / self.lam
        return (prior_var / var.clamp_min(1e-12) - 1.0).clamp_min(0.0)

    # ---- write ----
    @torch.no_grad()
    def write(self, psi: torch.Tensor, s: torch.Tensor) -> None:
        """One observation: psi [dim], s [heads] with entries in {-1, 0, +1}; 0 = head not observed
        (its cross-moment is untouched; the shared precision still absorbs the address)."""
        psi = psi.to(self.device, self.dtype).reshape(self.dim)
        s = s.to(self.device, self.dtype).reshape(self.heads)
        if self.P is not None:
            ppsi = self.P @ psi
            k = ppsi / (1.0 + psi @ ppsi)
            self.P -= torch.outer(k, ppsi)
        else:
            self.Lam.mul_(self.rho).add_(torch.eye(self.dim, device=self.device, dtype=self.dtype), alpha=(1.0 - self.rho) * self.lam)
            self.Lam.addr_(psi, psi)
            self.b.mul_(self.rho)
            self._chol = None
        self.b.addr_(s, psi)
        self.writes += 1

    @torch.no_grad()
    def decay(self) -> None:
        """Advance the clock without an observation (only matters for rho < 1)."""
        if self.Lam is not None:
            self.Lam.mul_(self.rho).add_(torch.eye(self.dim, device=self.device, dtype=self.dtype), alpha=(1.0 - self.rho) * self.lam)
            self.b.mul_(self.rho)
            self._chol = None


class DeltaMemory:
    """First-order counterpart: the (gated) delta rule with a fixed step.

    S <- alpha * S + beta * (s - S psi) psi^T  (S: [heads, dim]).  This is DeltaNet's
    update with scalar gates; it is one gradient step of the least-squares objective
    that KalmanMemory solves exactly.  Kept for the ablation "exact posterior vs. one
    gradient step".
    """

    def __init__(self, dim: int, heads: int, *, beta: float = 0.05, alpha: float = 1.0, device=None, dtype=torch.float64) -> None:
        self.dim, self.heads, self.beta, self.alpha = int(dim), int(heads), float(beta), float(alpha)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.S = torch.zeros(self.heads, self.dim, device=self.device, dtype=self.dtype)
        self.writes = 0

    @torch.no_grad()
    def read(self, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        psi = psi.to(self.device, self.dtype)
        mu = psi @ self.S.T
        return mu, torch.zeros(psi.shape[:-1], device=self.device, dtype=self.dtype)

    prob = staticmethod(KalmanMemory.prob)
    logit = staticmethod(KalmanMemory.logit)

    @torch.no_grad()
    def write(self, psi: torch.Tensor, s: torch.Tensor) -> None:
        psi = psi.to(self.device, self.dtype).reshape(self.dim)
        s = s.to(self.device, self.dtype).reshape(self.heads)
        err = s - self.S @ psi
        err = torch.where(s == 0, torch.zeros_like(err), err)
        self.S.mul_(self.alpha).addr_(err, psi, alpha=self.beta / float(psi @ psi + 1e-12))
        self.writes += 1

    @torch.no_grad()
    def decay(self) -> None:
        self.S.mul_(self.alpha)
