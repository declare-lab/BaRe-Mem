"""Does the record's covariance stay symmetric and positive definite over a whole stream?

The deployed record (rho = 1) keeps P = Lambda^{-1} explicitly and updates it by Sherman-Morrison,
P <- P - k x^T P, once per peer answer: about 106,000 rank-one updates per pass over a six-peer
stream of 17,709 events.  The short form is only symmetric at the exact gain, so in floating point the
asymmetry can accumulate.  This measures it, against two references:

  exact      Lambda = lam I + sum x x^T accumulated separately, inverted once at the end
  joseph     the Joseph form P <- (I - k x^T) P (I - x k^T) + sigma^2 k k^T, symmetric by construction

Reports, every few thousand writes: relative asymmetry ||P - P^T||_F / ||P||_F, the smallest
eigenvalue (must stay > 0), and the relative error of P and of the readout mu against `exact`.
Addresses imitate the deployed ones: a peer-blocked question part plus a shared answer part plus a
constant, drawn from a low-rank-plus-noise covariance so that, as in the real stream, a few
directions carry most of the variance.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def make_addresses(n: int, dq: int, dc: int, peers: int, rng) -> np.ndarray:
    """[n*peers, D] addresses with the deployed block layout and anisotropic statistics."""
    D = peers * dq + dc + 1
    # anisotropic factors: 8 strong shared directions + isotropic remainder, as measured on the judge features
    Uq = rng.normal(size=(dq, 8)) / np.sqrt(dq)
    Uc = rng.normal(size=(dc, 8)) / np.sqrt(dc)
    rows = np.zeros((n * peers, D))
    for t in range(n):
        zq = rng.normal(size=8) * 3.0
        psi_q = Uq @ zq + rng.normal(size=dq) * 0.3
        for i in range(peers):
            zc = rng.normal(size=8) * 3.0
            psi_c = Uc @ zc + rng.normal(size=dc) * 0.3
            r = rows[t * peers + i]
            r[i * dq:(i + 1) * dq] = psi_q
            r[peers * dq:peers * dq + dc] = psi_c
            r[-1] = 1.0
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=17709)
    ap.add_argument("--peers", type=int, default=6)
    ap.add_argument("--dq", type=int, default=256)
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/analysis/kalman_numerics.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    X = make_addresses(args.events, args.dq, args.dc, args.peers, rng)
    N, D = X.shape
    s = rng.choice([-1.0, 1.0], size=N)
    print(f"{N} writes of dimension {D} ({time.time() - t0:.0f}s)", flush=True)

    P_sm = np.eye(D) / args.lam          # Sherman-Morrison, the deployed path
    P_jo = np.eye(D) / args.lam          # Joseph form, symmetric by construction
    m_sm = np.zeros(D)
    m_jo = np.zeros(D)
    Lam = args.lam * np.eye(D)           # exact reference, inverted only at the report points
    b = np.zeros(D)
    log = []

    for t in range(N):
        x, st = X[t], s[t]
        # --- deployed: Sherman-Morrison short form
        Px = P_sm @ x
        k = Px / (args.sigma2 + x @ Px)
        m_sm = m_sm + k * (st - m_sm @ x)
        P_sm = P_sm - np.outer(k, x @ P_sm)
        # --- Joseph form, expanded to rank-one pieces so it stays O(D^2):
        #     (I - k x^T) P (I - x k^T) + s2 k k^T = P - k (P^T x)^T - (P x) k^T + (v + s2) k k^T
        u = P_jo @ x
        ut = x @ P_jo                     # = (P_jo^T x)^T, differs from u only by the accumulated asymmetry
        vt = x @ u
        kj = u / (args.sigma2 + vt)
        m_jo = m_jo + kj * (st - m_jo @ x)
        P_jo = P_jo - np.outer(kj, ut) - np.outer(u, kj) + (vt + args.sigma2) * np.outer(kj, kj)
        # --- exact sufficient statistics
        Lam += np.outer(x, x) / args.sigma2
        b += st * x / args.sigma2

        if (t + 1) % args.every == 0 or t == N - 1:
            P_ex = np.linalg.inv(Lam)
            m_ex = P_ex @ b
            def rep(P, m, name):
                asym = np.linalg.norm(P - P.T) / np.linalg.norm(P)
                ev = np.linalg.eigvalsh((P + P.T) / 2)
                dP = np.linalg.norm(P - P_ex) / np.linalg.norm(P_ex)
                dm = np.linalg.norm(m - m_ex) / max(np.linalg.norm(m_ex), 1e-12)
                return {"path": name, "asymmetry": float(asym), "min_eig": float(ev[0]),
                        "rel_err_P": float(dP), "rel_err_m": float(dm)}
            row = {"writes": t + 1, "paths": [rep(P_sm, m_sm, "sherman_morrison"), rep(P_jo, m_jo, "joseph")]}
            log.append(row)
            for r in row["paths"]:
                print(f"  writes={t+1:>7} {r['path']:<17} asym={r['asymmetry']:.2e} "
                      f"min_eig={r['min_eig']:.3e} relP={r['rel_err_P']:.2e} relm={r['rel_err_m']:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    json.dump({"dim": D, "writes": N, "lam": args.lam, "sigma2": args.sigma2, "log": log},
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
