"""The record inside a lead-worker team: whom the lead gives a sub-task to, and whom it asks next.

A lead agent splits a task into sub-tasks and gives each to ONE source of its pool (a worker, or itself: the autonomous option).
The source reports, and the report is verified before the commit; a report that fails sends the sub-task to the next source
(consultation, or the lead itself), a few sources at most; the last report is committed. Two verifications, two illustrations:
  the lead's check      the lead holds the report against what it expected of it (pipeline.review: a plain, specific answer of the
                        right kind; a model's check, it sees no evidence and not the record). What a deployed team can do.
  the dataset's check   the benchmark's evaluator (the sub-task's gold answer) verifies the report. Exact, so what is left to learn
                        is whom to ask, and every failure is known at once: the setting that shows how fast the record adapts.
After the commit the benchmark's gold answer labels EVERY report that was produced, the failed ones as much as the committed one (a
failed report is what says the source it came from was unreliable here), and those labels are written into the record; a source
that was not called produced nothing, so nothing is written for it (unlike the QA streams, where every peer answers and is verified).

The record's one job is the order in which sources are asked. It is read once per sub-task, before anyone is called, from the
sub-task alone: source s's row is the sub-task's address in s's block, [e_s (x) [psi_q ; 1]] (psi_q: the lead model's
question-only features, PCA-projected; the constant is the source's overall track record). No report enters an address, and the
record takes no part in the check.

`replay` runs one policy along a stream whose every source answered every sub-task offline and had its report checked. That full
table is the evaluation device, not the protocol: a policy only ever sees, and the record is only ever written with, the reports of
the sources it called, so every policy is replayed exactly on the same events.

Policies (the order in which sources are asked):
  memory          by the record's P(right), read from the sub-task
  success_counts  by Thompson sampling on Beta success counts (a track record that ignores the sub-task)
  random          at random
"""
from __future__ import annotations

import numpy as np
import torch

from feedback_state.kalman_memory import KalmanMemory

POLICIES = ("memory", "success_counts", "random")


def rows_of(psi_q_t: torch.Tensor, sources: int) -> torch.Tensor:
    """The record's rows of one sub-task, one per source: the sub-task's address and a constant, in that source's block."""
    q1 = torch.cat([psi_q_t, torch.ones(1, dtype=psi_q_t.dtype, device=psi_q_t.device)])
    return torch.kron(torch.eye(sources, dtype=psi_q_t.dtype, device=psi_q_t.device), q1.unsqueeze(0))


@torch.no_grad()
def replay(policy: str, psi_q: torch.Tensor, labels: np.ndarray, order: np.ndarray, *, rejected: np.ndarray | None = None, calls: int = 2, write: str = "all",
           lam: float = 100.0, seed: int = 0, tasks: list[np.ndarray] | None = None, own: int | None = None, windows: int = 8, device=None) -> dict:
    """One policy along one stream order, from a cold record. psi_q [N, d]; labels [N, S] in {0, 1}: the gold verdict on source s's
    report for sub-task t; rejected [N, S] bool: the verification of that report before the commit (the lead's check, or the dataset's:
    labels == 0; None: no verification, the first report is committed); calls: the sources a sub-task may go to at most; write: "all"
    (every report that was produced is written) or "final" (only the committed one: the ablation that throws the failed reports away)."""
    if policy not in POLICIES:
        raise ValueError(f"policy {policy!r} is not one of {POLICIES}")
    rng = np.random.default_rng(seed)
    N, S = labels.shape
    mem = KalmanMemory(S * (psi_q.shape[1] + 1), 1, lam=lam, device=device) if policy == "memory" else None
    beta = np.ones((S, 2))
    hit, used, first_hit = np.zeros(N, dtype=int), np.ones(N, dtype=int), np.zeros(N, dtype=int)
    chosen, committed = np.zeros(S), np.zeros(S)
    for t in order.tolist():
        if mem is not None:                          # the record's P(right) of every source, from the sub-task alone
            X = rows_of(psi_q[t], S)
            mu, var = mem.read(X)
            rank = np.argsort(-(mem.prob(mu[:, 0], var).cpu().numpy() + 1e-9 * rng.random(S))).tolist()   # ties (a cold record) at random
        elif policy == "success_counts":
            rank = np.argsort(-rng.beta(beta[:, 0], beta[:, 1])).tolist()
        else:
            rank = rng.permutation(S).tolist()
        called = [rank[0]]
        if rejected is not None:                     # a report the lead rejects sends the sub-task to the next source in the order
            for nxt in rank[1: int(calls)]:
                if not rejected[t, called[-1]]:
                    break
                called.append(nxt)
        s = called[-1]
        hit[t], used[t], first_hit[t] = int(labels[t, s]), len(called), int(labels[t, rank[0]])
        chosen[rank[0]] += 1
        committed[s] += 1
        for k in (called if write == "all" else [s]):   # feedback: the gold verdict on every report that was produced, and on nothing else
            if mem is not None:
                mem.write(X[k], torch.tensor([1.0 if labels[t, k] else -1.0], dtype=torch.float64))
            beta[k, 0 if labels[t, k] else 1] += 1
    out = {"accuracy": 100 * float(hit.mean()), "worker_calls": float(used.mean()), "chosen": (chosen / N).round(4).tolist(), "committed": (committed / N).round(4).tolist()}
    if tasks is not None:                            # a task is solved when every one of its sub-tasks' committed reports is right
        out["task_accuracy"] = 100 * float(np.mean([hit[idx].all() for idx in tasks]))
    if own is not None:
        out["autonomous"] = float(committed[own] / N)
    edges = [round(i * N / windows) for i in range(windows + 1)]
    curve = lambda v: [round(100 * float(v[order][a:b].mean()), 1) for a, b in zip(edges[:-1], edges[1:])]
    out.update(first_accuracy=100 * float(first_hit.mean()), along=curve(hit), along_first=curve(first_hit))   # along the stream: how fast the choice improves
    return out


def check_quality(labels: np.ndarray, rejected: np.ndarray) -> dict:
    """How good the lead's check is, over every source's report (per source and overall): how often it rejects, how many of the
    wrong reports it catches, how many right ones it rejects."""
    rej, right = rejected.astype(bool), labels.astype(bool)

    def stats(mask):
        r, y = rej[mask], right[mask]
        pct = lambda a, b: round(100 * float(a.sum()) / max(1, int(b.sum())), 2)
        return {"right": round(100 * float(y.mean()), 2), "rejected": round(100 * float(r.mean()), 2), "wrong_caught": pct(r & ~y, ~y), "right_rejected": pct(r & y, y)}

    S = labels.shape[1]
    every = np.ones_like(rej, dtype=bool)
    return {"overall": stats(every), "by_source": [stats(every & (np.arange(S)[None, :] == s)) for s in range(S)]}


def task_index(ids: list[str]) -> list[np.ndarray]:
    """The sub-task indices of every task, for sub-tasks named <task>/h<k> (in the stream's order)."""
    index: dict[str, list[int]] = {}
    for t, name in enumerate(ids):
        index.setdefault(str(name).rsplit("/h", 1)[0], []).append(t)
    return [np.array(v) for v in index.values()]


def stream_order(n: int, seed: int, tasks: list[np.ndarray] | None) -> np.ndarray:
    """A stream order: a permutation of the sub-tasks, or of the tasks with each task's sub-tasks kept together and in order."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n) if tasks is None else np.concatenate([tasks[i] for i in rng.permutation(len(tasks))])
