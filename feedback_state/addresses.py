"""Memory addresses built from the frozen center model's hidden states.

``Projection`` is an unsupervised standardise + PCA map fitted once on the unlabeled
features (no labels, no dataset names).  ``design_rows`` turns one event's
projected question address ``psi_q`` [d], candidate addresses ``psi_c`` [r, d]
and judge log-odds ``z`` [r] into the feature rows the memory reads and writes:

  q     [psi_q ; 1]                                   one row, per-peer heads
  c     [psi_c(p) ; 1]                                one row per candidate, shared head
  cm    [psi_c(p) ; z_p ; 1]
  qc    [e_p (x) psi_q ; psi_c(p) ; 1]                peer-blocked question part + shared candidate part
  qcm   [e_p (x) psi_q ; psi_c(p) ; z_p ; e_p z_p ; 1]

The peer block index is the peer's identity (canonical id), never its slot in
the prompt, so the memory stays identity-aware while the judge sees anonymous,
permutable slots.
"""
from __future__ import annotations

import math

import torch


class Projection:
    """Standardise + PCA (fitted on unlabeled features) + global scale."""

    def __init__(self, X: torch.Tensor | None = None, dim: int = 256, device=None, *, state: dict | None = None, seed: int = 0) -> None:
        if state is not None:
            for k in ("mean", "std", "center", "basis"):
                setattr(self, k, state[k].to(device))
            self.scale = float(state["scale"])
            self.dim = int(self.basis.shape[1])
            return
        X = X.to(device, torch.float32)
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-3
        Z = (X - self.mean) / self.std
        self.center = Z.mean(0)
        Zc = Z - self.center
        q = min(int(dim) + 16, Zc.shape[0], Zc.shape[1])
        # pca_lowrank draws a random sketch: seed it, or every run gets a slightly different basis (and record). Records
        # built before 2026-09-13 were unseeded, so they cannot be regenerated bit for bit; their quality can (AUC to 1e-4).
        devices = [Zc.device] if Zc.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            _, _, V = torch.pca_lowrank(Zc, q=q, center=False, niter=6)
        self.basis = V[:, : int(dim)].contiguous()
        self.scale = float((Zc @ self.basis).std()) + 1e-6
        self.dim = int(self.basis.shape[1])     # fewer than asked when the fitting set is smaller than dim (a smoke run)

    def state(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "center": self.center.cpu(), "basis": self.basis.cpu(), "scale": self.scale}

    def __call__(self, X: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        out = []
        for i in range(0, X.shape[0], chunk):
            x = X[i : i + chunk].to(self.basis.device, torch.float32)
            out.append((((x - self.mean) / self.std - self.center) @ self.basis) / self.scale)
        return torch.cat(out)


DESIGNS = ("q", "c", "cm", "qc", "qcm")


def design_dim(design: str, num_peers: int, dq: int, dc: int) -> int:
    P = int(num_peers)
    return {"q": dq + 1, "c": dc + 1, "cm": dc + 2, "qc": P * dq + dc + 1, "qcm": P * dq + dc + 2 + P}[design]


def design_rows(design: str, psi_q: torch.Tensor, psi_c: torch.Tensor, z: torch.Tensor, peer_ids, num_peers: int) -> torch.Tensor:
    """Feature rows for the candidates of one event (design 'q': a single shared row [1, dq+1]).

    psi_q [dq]; psi_c [r, dc] (rows in candidate order); z [r]; peer_ids: canonical peer id per candidate.
    """
    dev, dt = psi_q.device, torch.float64
    psi_q = psi_q.to(dt); psi_c = psi_c.to(dev, dt); z = z.to(dev, dt).reshape(-1, 1)
    r = psi_c.shape[0]
    one = torch.ones(r, 1, device=dev, dtype=dt)
    if design == "q":
        return torch.cat([psi_q, one[:1, 0]]).unsqueeze(0)
    if design == "c":
        return torch.cat([psi_c, one], 1)
    if design == "cm":
        return torch.cat([psi_c, z, one], 1)
    dq = psi_q.shape[0]
    blocks = torch.zeros(r, int(num_peers) * dq, device=dev, dtype=dt)
    for row, p in enumerate(peer_ids[:r]):
        blocks[row, int(p) * dq : (int(p) + 1) * dq] = psi_q
    if design == "qc":
        return torch.cat([blocks, psi_c, one], 1)
    if design == "qcm":
        zb = torch.zeros(r, int(num_peers), device=dev, dtype=dt)
        for row, p in enumerate(peer_ids[:r]):
            zb[row, int(p)] = z[row, 0]
        return torch.cat([blocks, psi_c, z, zb, one], 1)
    raise ValueError(design)


def memory_evidence(ell: torch.Tensor, n_eff: torch.Tensor, *, n_scale: float = 100.0) -> torch.Tensor:
    """Per-candidate evidence vector [r, 4] handed to the judge:
    [own log-odds, saturating evidence count, best other candidate's log-odds, mean of the others']."""
    r = ell.shape[0]
    nu = torch.log1p(n_eff.clamp_min(0.0)) / torch.log1p(torch.tensor(n_scale, dtype=ell.dtype, device=ell.device))
    rows = []
    for p in range(r):
        others = torch.cat([ell[:p], ell[p + 1 :]]) if r > 1 else torch.zeros(1, dtype=ell.dtype, device=ell.device)
        rows.append(torch.stack([ell[p], nu[p].clamp(max=1.0), others.max(), others.mean()]))
    return torch.stack(rows)


EVIDENCE_DIM = 4


class AddressMap(torch.nn.Module):
    """Learned address maps on top of the fixed standardisation, all with an explicit inverse on their range.

    arch = "linear"      z = W^T x / s                              (PCA-initialised)
    arch = "orthogonal"  z = Q^T x / s with Q = qr(W) orthonormal   norm-preserving on the retained subspace, so the
                         address can be decoded back into the hidden-state space exactly: x_hat = Q z s
    arch = "flow"        z = f(W^T x / s) with f two affine-coupling layers (RealNVP): a bijection of the PCA
                         coordinates, so nonlinearly encoded structure can become linear for the memory while
                         nothing is lost (f^{-1} is closed form)
    """

    def __init__(self, pca: "Projection", arch: str = "linear", hidden: int = 128) -> None:
        super().__init__()
        self.arch = str(arch)
        self.register_buffer("mean", pca.mean.clone())
        self.register_buffer("std", pca.std.clone())
        self.register_buffer("center", pca.center.clone())
        self.W = torch.nn.Parameter(pca.basis.clone())
        self.scale = float(pca.scale)
        self.dim = int(pca.basis.shape[1])
        if self.arch == "flow":
            d = self.dim; h = int(hidden); half = d // 2
            self.nets = torch.nn.ModuleList()
            for _ in range(2):
                self.nets.append(torch.nn.Sequential(torch.nn.Linear(half, h), torch.nn.Tanh(), torch.nn.Linear(h, 2 * (d - half))))
                torch.nn.init.zeros_(self.nets[-1][-1].weight); torch.nn.init.zeros_(self.nets[-1][-1].bias)  # start at identity

    def basis(self) -> torch.Tensor:
        if self.arch == "orthogonal":
            q, r = torch.linalg.qr(self.W)
            return q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        return self.W

    def _couple(self, z: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        d = self.dim; half = d // 2
        nets = list(self.nets)[::-1] if inverse else list(self.nets)
        for i, net in enumerate(nets):
            flip = ((len(self.nets) - 1 - i) if inverse else i) % 2 == 1
            a, b = (z[:, half:], z[:, :half]) if flip else (z[:, :half], z[:, half:])
            st = net(a); s, tt = st.chunk(2, dim=-1); s = torch.tanh(s)
            b = (b - tt) * torch.exp(-s) if inverse else b * torch.exp(s) + tt
            z = torch.cat([b, a], -1) if flip else torch.cat([a, b], -1)
        return z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = X.to(self.W.device, torch.float32)
        z = (((x - self.mean) / self.std - self.center) @ self.basis()) / self.scale
        return self._couple(z) if self.arch == "flow" else z

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """Decode an address back into standardised hidden-state space (exact on the map's range)."""
        if self.arch == "flow":
            z = self._couple(z, inverse=True)
        return (z * self.scale) @ self.basis().T + self.center

    def to_projection(self, device=None) -> "Projection":
        return Projection(state={"mean": self.mean.cpu(), "std": self.std.cpu(), "center": self.center.cpu(), "basis": self.basis().detach().cpu(), "scale": self.scale}, device=device)

    def state(self) -> dict:
        return {"arch": self.arch, "dim": self.dim, "scale": self.scale, "module": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    @classmethod
    def from_state(cls, st: dict, device=None) -> "AddressMap":
        mod = st["module"]
        pca = Projection(state={"mean": mod["mean"], "std": mod["std"], "center": mod["center"], "basis": mod["W"], "scale": st["scale"]}, device="cpu")
        m = cls(pca, arch=st["arch"])
        m.load_state_dict(mod)
        return m.to(device) if device is not None else m

    def __call__(self, X: torch.Tensor, chunk: int = 4096) -> torch.Tensor:  # Projection-compatible chunked interface
        with torch.no_grad():
            return torch.cat([self.forward(X[i : i + chunk]) for i in range(0, X.shape[0], chunk)])

