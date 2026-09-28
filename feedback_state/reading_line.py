"""The central model's reading line: on an event, is Advisors + memory worth more than its no-consultation answer?

For the central model c, T is the record's trust in the evidence it reads (the top estimate among the peers) and kappa the
record's estimate of c's own no-consultation answer, addressed like every other answer. Reading succeeds with

    A(T) = T rho + (1 - T)(kappa - delta)

rho: c's reading accuracy when the evidence is trustworthy; delta: how far untrustworthy evidence pulls it below its own
answer. With u = [T, T - 1] and z = y - (1 - T) kappa (y: the verified reading outcome), z = u^T (rho, delta), so the two
parameters are a Bayesian linear regression, a 2x2 state of its own (not the record's Lambda):

    P = lam I + sum u u^T      q = lam theta0 + sum u z      (rho, delta) = P^-1 q

read before write, one state per task type. c reads when A(T) >= kappa. The two lines cross at T* = delta / (rho + delta - kappa):
with delta > 0 and rho > kappa, c reads when T >= T*; reads_when() names every case.
"""
from __future__ import annotations

import numpy as np


class ReadingLine:
    def __init__(self, prior: tuple[float, float] = (0.5, 0.0), lam: float = 1.0):
        self.P = float(lam) * np.eye(2)
        self.q = float(lam) * np.asarray(prior, dtype=float)
        self.events = 0

    def estimate(self) -> tuple[float, float]:
        """(rho, delta) = P^-1 q from the events written so far."""
        rho, delta = np.linalg.solve(self.P, self.q)
        return float(rho), float(delta)

    def value(self, trust: float, own: float) -> float:
        """A(T): the expected success of reading at this trust, for an own answer estimated at `own`."""
        rho, delta = self.estimate()
        return trust * rho + (1.0 - trust) * (own - delta)

    def reads(self, trust: float, own: float) -> bool:
        return self.value(trust, own) >= own

    def switch(self, own: float) -> float | None:
        """T*, the trust where reading and the own answer are worth the same; None when the two lines are parallel."""
        rho, delta = self.estimate()
        slope = rho + delta - own
        return delta / slope if slope != 0 else None

    def reads_when(self, own: float) -> str:
        """Where on T in [0, 1] reading is worth at least the own answer: always, never, T >= T* or T <= T*."""
        rho, delta = self.estimate()
        at0, at1 = -delta, rho - own            # A(T) - own at T = 0 and at T = 1; a straight line in between
        if at0 >= 0 and at1 >= 0:
            return "always"
        if at0 < 0 and at1 < 0:
            return "never"
        return f"T >= {delta / (rho + delta - own):.2f}" if at1 >= 0 else f"T <= {delta / (rho + delta - own):.2f}"

    def write(self, trust: float, own: float, correct: int | bool) -> None:
        """Add one verified reading outcome: P += u u^T, q += u z."""
        u = np.array([trust, trust - 1.0])
        self.P += np.outer(u, u)
        self.q += u * (float(correct) - (1.0 - trust) * own)
        self.events += 1
