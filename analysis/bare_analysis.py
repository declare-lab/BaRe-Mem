"""Tally the three BaRe-Mem analysis experiments into one JSON (prints JSON to stdout).

    PYTHONPATH=. python analysis/bare_analysis.py [--models q3_4b:Qwen3-4B qwen3_8b:Qwen3-8B ...] > bare_analysis.json

Capability-challenging stream (capability_challenging) only; every model needs the combination experiment and analysis/sparse_feedback.sh.
  1  autonomous-ability estimation: kappa_hat = own_prob against question_alone_correct, 10 equal-count bins
  2  when to consult: predicted gain A(T) - kappa against the observed gain, pooled over the misleading ratios
  3  sparse feedback: accuracy gain over 0% feedback, from outputs/sparse
"""
import argparse
import glob
import json
import os

_ap = argparse.ArgumentParser(description="Tally the BaRe-Mem analysis experiments into one JSON.")
_ap.add_argument("--models", nargs="+", default=["q3_4b:Qwen3-4B", "qwen3_8b:Qwen3-8B", "qwen3_14b:Qwen3-14B", "phi4:Phi-4"],
                 help="registered model key:display name")
MODELS = [tuple(m.split(":", 1)) for m in _ap.parse_args().models]
RATIOS = ["p000", "p025", "p050", "p075", "p100"]
NBINS = 24            # the pooled curves
PANEL_BINS = 16       # one misleading information ratio on its own
STREAM_BINS = 24      # kappa along the stream


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def comb(tag, ratio):
    return rows(f"outputs/eval/{tag}/capability_challenging_misleading_{ratio}+own/combination/generations.jsonl")


def bins(points, nbins=NBINS):
    """Equal-count bins over the first coordinate; returns (x mean, y mean, n) per bin."""
    points = sorted(points)
    n = len(points)
    out = []
    for i in range(nbins):
        lo, hi = n * i // nbins, n * (i + 1) // nbins
        chunk = points[lo:hi]
        if not chunk:
            continue
        out.append({"x": sum(a for a, _ in chunk) / len(chunk), "y": sum(b for _, b in chunk) / len(chunk), "n": len(chunk)})
    return out


def consulted(r):
    return "peers" in str(r.get("choice", "")).lower() or "memory" in str(r.get("choice", "")).lower()


# ---------------------------------------------------------------- 1
exp1 = {}
for tag, name in MODELS:
    per_ratio, per_ratio_panel = {}, {}
    for ratio in RATIOS:
        g = comb(tag, ratio)
        if not g:
            continue
        per_ratio[ratio] = bins([(r["own_prob"], r["question_alone_correct"]) for r in g])
        per_ratio_panel[ratio] = bins([(r["own_prob"], r["question_alone_correct"]) for r in g], PANEL_BINS)
    g0 = comb(tag, "p000")
    if not g0:
        continue
    # kappa along the stream, in 20 position bins, against the autonomous accuracy there
    g0 = sorted(g0, key=lambda r: r["pos"])
    n, nb = len(g0), STREAM_BINS
    along = []
    for i in range(nb):
        chunk = g0[n * i // nb:n * (i + 1) // nb]
        along.append({"pos": sum(r["pos"] for r in chunk) / len(chunk),
                      "kappa": sum(r["own_prob"] for r in chunk) / len(chunk),
                      "acc": sum(r["question_alone_correct"] for r in chunk) / len(chunk), "n": len(chunk)})
    exp1[tag] = {"model": name, "n": n, "bins": per_ratio["p000"], "by_ratio": per_ratio,
                 "by_ratio_panel": per_ratio_panel, "along": along,
                 "overall_acc": sum(r["question_alone_correct"] for r in g0) / n,
                 "overall_kappa": sum(r["own_prob"] for r in g0) / n}

# ---------------------------------------------------------------- 2
exp2 = {}
for tag, name in MODELS:
    pooled, per_ratio, sel = [], {}, {}
    for ratio in RATIOS:
        g = comb(tag, ratio)
        if not g:
            continue
        pts = [(r["peers_memory_value"] - r["own_prob"], r["peers_memory_correct"] - r["question_alone_correct"]) for r in g]
        pooled += pts
        per_ratio[ratio] = {"bins": bins(pts, PANEL_BINS), "n": len(pts)}
        help_c = [r for r in g if r["peers_memory_correct"] and not r["question_alone_correct"]]
        help_a = [r for r in g if r["question_alone_correct"] and not r["peers_memory_correct"]]
        sel[ratio] = {"consult_cases": len(help_c), "consult_picked": sum(1 for r in help_c if consulted(r)),
                      "alone_cases": len(help_a), "alone_picked": sum(1 for r in help_a if not consulted(r)),
                      "consult_share": sum(1 for r in g if consulted(r)) / len(g), "n": len(g),
                      "acc_comb": sum(r["correct"] for r in g) / len(g),
                      "acc_tilt": sum(r["peers_memory_correct"] for r in g) / len(g),
                      "acc_solo": sum(r["question_alone_correct"] for r in g) / len(g),
                      "oracle": sum(max(r["peers_memory_correct"], r["question_alone_correct"]) for r in g) / len(g)}
    if not pooled:
        continue
    pos = [(a, b) for a, b in pooled if a >= 0]
    neg = [(a, b) for a, b in pooled if a < 0]
    exp2[tag] = {"model": name, "n": len(pooled), "bins": bins(pooled), "by_ratio": per_ratio, "selection": sel,
                 "sign": {"pos_n": len(pos), "pos_obs": sum(b for _, b in pos) / len(pos) if pos else None,
                          "neg_n": len(neg), "neg_obs": sum(b for _, b in neg) / len(neg) if neg else None}}

# ---------------------------------------------------------------- 3
exp3 = {}
for tag, name in MODELS:
    cells = {}
    for d in sorted(glob.glob(f"outputs/sparse/{tag}/capability_challenging_misleading_p000/r*")):
        r = os.path.basename(d)[1:]
        m = os.path.join(d, "combination", "eval_metrics.json")
        t = os.path.join(d, "tilt", "eval_metrics.json")
        if not os.path.exists(m):
            continue
        M = json.load(open(m))
        T = json.load(open(t)) if os.path.exists(t) else {}
        g = rows(os.path.join(d, "combination", "generations.jsonl"))
        cells[r] = {"accuracy": M.get("accuracy"), "n": M.get("num_samples") or M.get("n"),
                    "accuracy_tilt": T.get("accuracy"),
                    "consult_share": (sum(1 for x in g if consulted(x)) / len(g)) if g else M.get("share_peers_memory"),
                    "accuracy_solo": (sum(x["question_alone_correct"] for x in g) / len(g)) if g else None}
    solo = json.load(open(f"outputs/sparse/{tag}/capability_challenging_misleading_p000/solo/eval_metrics.json")) if os.path.exists(
        f"outputs/sparse/{tag}/capability_challenging_misleading_p000/solo/eval_metrics.json") else {}
    exp3[tag] = {"model": name, "cells": cells, "solo": solo.get("accuracy")}

print(json.dumps({"exp1": exp1, "exp2": exp2, "exp3": exp3}, ensure_ascii=False))
