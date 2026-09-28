"""Record: the Bayesian reliability record along a stream, read before write, and its quality against the labels.

The record is run exactly as at test time: cold start; for every event, in the stream order, the estimate p_i of every
peer is read from the state built by the earlier events, the event's row is written, and only then are its verified
peer labels written into the state. --feedback-ratio r keeps the reads and writes only a share r of the events (sparse
feedback; the events are nested in r and fixed by --feedback-seed, so every question is still answered and scored). The PCA addresses are fit on --fit-features (label-free; the stream's own features
when omitted). Peer slots are permuted per event (the record is identity-indexed, the prompt is not).

With --own-slot K the answer in slot K is the central model's own no-consultation answer (pipeline.streams add --eval): the
record estimates and writes it like every answer, but it stays out of the prompt and the tilt, and its estimate goes to
own_prob (pipeline.combination reads it). --save-addresses writes every event's projected address, so a later process
(a later analysis) can run the same record without the features or the PCA fit.

    PYTHONPATH=. python -m pipeline.record --stream data/capability_supported/test.jsonl --features outputs/features/q3_4b/capability_supported \
        --fit-stream data/address_fit/stream.jsonl --fit-features outputs/features/q3_4b/address_fit \
        --order shuffled0 --out outputs/record/q3_4b/capability_supported/shuffled0.jsonl

Writes <out> (one row per event: pos, id, task_type, source, peer_order, peer_correct, memory_prob, memory_evidence,
messages_peers, messages_solo; with --own-slot also own_prob, own_correct, own_evidence) and <out without .jsonl>.quality.json.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
from pathlib import Path

import numpy as np


def order_of(n: int, order: str) -> np.ndarray:
    """The stream order: 'fixed' = file order, 'shuffledK' = numpy.random.default_rng(K).permutation(n)."""
    return np.arange(n) if order == "fixed" else np.random.default_rng(int(order.replace("shuffled", ""))).permutation(n)


def build(args) -> list[dict]:
    import torch

    from feedback_state.addresses import Projection
    from feedback_state.feature_streams import load_stream_from
    from feedback_state.memory_generator import build_messages, prob_from_logit
    from feedback_state.memory_runtime import MemoryRuntime

    device = torch.device(args.device)
    fs = load_stream_from(args.stream, args.features, num_peers=args.peers)
    fit = fs if args.fit_features is None else load_stream_from(args.fit_stream or args.stream, args.fit_features, num_peers=args.peers)
    proj_q = Projection(fit.sem, args.dim, device)
    proj_c = Projection(fit.peer_hidden.reshape(-1, fit.peer_hidden.shape[-1]), args.dim, device)
    runtime = MemoryRuntime(design=args.design, proj_q=proj_q, proj_c=proj_c, num_peers=fs.num_peers, lam=args.lam, device=device)
    runtime.attach(fs)
    runtime.reset()
    if getattr(args, "save_addresses", None):
        args.save_addresses.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**runtime.addresses(), "ids": list(fs.ids), "labels": fs.labels.clone(), "task": list(fs.task)}, args.save_addresses)
    order = order_of(len(fs), args.order)
    if args.limit:
        order = order[: args.limit]
    rng = np.random.default_rng(args.seed)
    labels = fs.labels.numpy()
    own = getattr(args, "own_slot", None)
    ratio = float(getattr(args, "feedback_ratio", 1.0) or 0.0)
    fb_seed = int(getattr(args, "feedback_seed", 0) or 0)
    rows = []
    for pos, t in enumerate(order.tolist()):
        r = int(fs.real[t])
        if r < 1:
            continue
        if own is not None and r <= own:
            raise ValueError(f"event {fs.ids[t]} has {r} answers, so no own answer in slot {own}")
        y = labels[t, :r]
        ell, n_eff, _, X = runtime.read(t)
        perm = prompt_slots(r, own, int(rng.integers(1 << 30)))
        texts = [fs.texts[t][p] for p in perm]
        rec = fs.records[t]
        row = {
            "pos": pos, "id": fs.ids[t], "task_type": fs.task[t], "source": fs.source[t],
            "peer_order": [int(p) for p in perm], "peer_correct": [int(y[p]) for p in perm],
            "memory_prob": [round(prob_from_logit(float(ell[p])), 4) for p in perm],
            "memory_evidence": [round(float(n_eff[p]), 1) for p in perm],
            "messages_peers": build_messages(rec, texts, mode="peers"),
            "messages_solo": build_messages(rec, texts, mode="solo"),
        }
        if own is not None:
            row.update(own_prob=round(prob_from_logit(float(ell[own])), 4), own_correct=int(y[own]), own_evidence=round(float(n_eff[own]), 1))
        fed = feedback(fs.ids[t], ratio, fb_seed)
        if ratio < 1.0:
            row["feedback"] = int(fed)
        rows.append(row)
        if fed:
            runtime.write(t, X, y)   # every answer's label, the own answer's included
    return rows


def feedback(event_id, ratio: float, seed: int) -> bool:
    """Does this event's outcome reach the record? A fixed share of the stream, the same events for every run of the same
    seed and nested in the ratio: an event fed at ratio r is fed at every larger ratio (u < r implies u < r')."""
    if ratio >= 1.0:
        return True
    if ratio <= 0.0:
        return False
    import hashlib

    h = hashlib.sha256(f"{seed}:{event_id}".encode()).digest()[:8]
    return int.from_bytes(h, "big") / float(1 << 64) < ratio


def prompt_slots(real: int, own: int | None, seed: int) -> list[int]:
    """The answers shown in the prompt, in slot order: a permutation of the peers, without the own answer."""
    from feedback_state.permutations import random_order

    peers = [i for i in range(real) if i != own]
    return [peers[j] for j in random_order(len(peers), seed)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stream", type=Path, required=True, help="the stream JSONL")
    ap.add_argument("--features", type=Path, required=True, help="its feature directory (shard*.pt, pipeline.features)")
    ap.add_argument("--fit-stream", type=Path, default=None, help="the stream the addresses are fit on (default: --stream)")
    ap.add_argument("--fit-features", type=Path, default=None, help="its features (default: fit on --features)")
    ap.add_argument("--peers", type=int, default=6)
    ap.add_argument("--order", default="shuffled0", help="fixed | shuffledK")
    ap.add_argument("--design", default="qc")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0, help="the per-event peer-slot permutation")
    ap.add_argument("--own-slot", type=int, default=None, help="the slot of the central model's own answer: recorded, not shown")
    ap.add_argument("--save-addresses", type=Path, default=None, help="also write every event's projected address (torch file)")
    ap.add_argument("--feedback-ratio", type=float, default=1.0, help="the share of events whose verified labels reach the record "
                    "(read happens for every event; nested in the ratio, fixed by --feedback-seed)")
    ap.add_argument("--feedback-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    rows = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    res = quality(rows, args.windows, name=str(args.out))
    if args.own_slot is not None:
        own = [(r["own_prob"], r["own_correct"]) for r in rows]
        res["own_answer"] = {"auc": auc(own), "mean_prob": sum(p for p, _ in own) / max(1, len(own)), "accuracy": sum(y for _, y in own) / max(1, len(own))}
    res["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("device",)}
    quality_path(args.out).write_text(json.dumps(res, indent=1))
    tmp.replace(args.out)   # the record file appears last: its presence means the stage finished
    print(f"[record] wrote {len(rows)} rows to {args.out}", flush=True)


def quality_path(out: Path) -> Path:
    return out.with_name(out.name[: -len(".jsonl")] + ".quality.json" if out.name.endswith(".jsonl") else out.name + ".quality.json")


def auc(pairs):
    pos = sorted(p for p, y in pairs if y == 1)
    neg = sorted(p for p, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos) * len(neg))


def quality(rows: list[dict], windows: int = 8, name: str = "record") -> dict:
    """AUC / favourite-right of the record against the labels, next to trivial histories, overall and along the stream."""
    rows = sorted(rows, key=lambda r: r["pos"])
    tot = collections.defaultdict(lambda: [0, 0]); tot_t = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for slot, pid in enumerate(r["peer_order"]):
            y = r["peer_correct"][slot]; tot[pid][0] += 1; tot[pid][1] += y; tot_t[(pid, r["task_type"])][0] += 1; tot_t[(pid, r["task_type"])][1] += y
    run = collections.defaultdict(lambda: [0, 0]); run_t = collections.defaultdict(lambda: [0, 0])
    keys = ("memory", "running_peer", "running_peer_task", "hindsight_peer_task")
    preds = {k: [] for k in keys}; mixed = {k: [0, 0] for k in keys}; per_task = collections.defaultdict(lambda: {k: [] for k in keys})
    n_mixed = 0; chance = 0.0
    part = collections.Counter(); minority_fav_right = [0, 0]
    win = []
    N = len(rows); edges = [round(i * N / windows) for i in range(windows + 1)]
    for i, r in enumerate(rows):
        ys = r["peer_correct"]; task = r["task_type"]; probs = list(r["memory_prob"])
        cand = {"memory": probs, "running_peer": [(run[p][1] + 1) / (run[p][0] + 2) for p in r["peer_order"]],
                "running_peer_task": [(run_t[(p, task)][1] + 1) / (run_t[(p, task)][0] + 2) for p in r["peer_order"]],
                "hindsight_peer_task": [tot_t[(p, task)][1] / max(1, tot_t[(p, task)][0]) for p in r["peer_order"]]}
        for k, ps in cand.items():
            for p, y in zip(ps, ys):
                preds[k].append((p, y)); per_task[task][k].append((p, y))
        if 0 < sum(ys) < len(ys):
            n_mixed += 1; chance += sum(ys) / len(ys)
            for k, ps in cand.items():
                top = max(range(len(ps)), key=lambda s: ps[s]); mixed[k][0] += 1; mixed[k][1] += ys[top]
        # partition by peer agreement on correctness (labels only; answer groups need the texts)
        flat = max(probs) - min(probs) <= 0.1
        if len(set(ys)) == 1:
            part["unanimous (all right or all wrong)"] += 1
        elif flat:
            part["split, flat record"] += 1
        else:
            top = max(range(len(probs)), key=lambda s: probs[s])
            maj = 1 if sum(ys) * 2 > len(ys) else 0
            if ys[top] == maj:
                part["split, favourite agrees with the majority label"] += 1
            else:
                part["split, favourite against the majority"] += 1
                minority_fav_right[0] += 1; minority_fav_right[1] += ys[top]
        for slot, pid in enumerate(r["peer_order"]):
            run[pid][0] += 1; run[pid][1] += ys[slot]; run_t[(pid, task)][0] += 1; run_t[(pid, task)][1] += ys[slot]
    res = {"n_events": N, "num_peers": len(rows[0]["peer_order"]), "n_mixed": n_mixed, "chance_on_mixed": 100 * chance / max(1, n_mixed),
           "peer_accuracy": {str(p): 100 * v[1] / v[0] for p, v in sorted(tot.items())},
           "peer_accuracy_by_task": {f"{p}/{t}": 100 * v[1] / v[0] for (p, t), v in sorted(tot_t.items())},
           "partition": dict(part), "minority_favourite_right": 100 * minority_fav_right[1] / max(1, minority_fav_right[0])}
    for k in keys:
        res[k] = {"auc": auc(preds[k]), "favourite_acc_mixed": 100 * mixed[k][1] / max(1, mixed[k][0]), "auc_by_task": {t: auc(per_task[t][k]) for t in sorted(per_task)}}
    # along the stream
    for a, b in zip(edges[:-1], edges[1:]):
        seg = rows[a:b]
        mp = [(p, y) for r in seg for p, y in zip(r["memory_prob"], r["peer_correct"])]
        mx = [r for r in seg if 0 < sum(r["peer_correct"]) < len(r["peer_correct"])]
        fav = sum(r["peer_correct"][max(range(len(r["memory_prob"])), key=lambda s: r["memory_prob"][s])] for r in mx) / max(1, len(mx))
        win.append({"events": f"{a}-{b - 1}", "memory_auc": auc(mp), "memory_favourite_right": 100 * fav, "n_mixed": len(mx), "mean_evidence": sum(e for r in seg for e in r["memory_evidence"]) / max(1, sum(len(r["memory_evidence"]) for r in seg))})
    res["windows"] = win
    print(f"== {name}: {N} events, {res['num_peers']} peers, mixed {n_mixed} ({100 * n_mixed / N:.1f}%), chance pick {res['chance_on_mixed']:.1f}%")
    print("   peer accuracy:", {k: round(v, 1) for k, v in res["peer_accuracy"].items()})
    for k in keys:
        print(f"   {k:20s} AUC {res[k]['auc']:.3f}  favourite right on mixed {res[k]['favourite_acc_mixed']:.1f}%  by task {{{', '.join(f'{t}: {v:.2f}' for t, v in res[k]['auc_by_task'].items())}}}")
    print("   partition:", res["partition"], f"| favourite against the majority is right {res['minority_favourite_right']:.1f}%")
    print("   along the stream: " + " | ".join(f"{w['events']}: AUC {w['memory_auc']:.3f}, fav {w['memory_favourite_right']:.1f}%, evidence {w['mean_evidence']:.0f}" for w in win))
    return res


if __name__ == "__main__":
    main()
