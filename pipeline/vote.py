"""Vote: majority vote over the peers' answers, with or without an answer of the central model (a multi-agent baseline).

    python -m pipeline.vote --record outputs/record/q3_4b/capability_challenging_misleading_p050+own/shuffled0.fit-self.jsonl \
        --stream data/capability_challenging_misleading_p050/test.jsonl --own outputs/eval/q3_4b/capability_challenging/solo \
        --condition vote_all --output outputs/eval/q3_4b/capability_challenging_misleading_p050+own/vote_all

The candidates of an event are the peers' answers in the stream and, with --own, the central model's answer in that
evaluation (its no-consultation answer, or its last debate answer). The largest group of equal answers wins; a tie is
broken uniformly at random and scored in expectation (feedback_state.baselines.plurality). Correctness is the stored
label of each answer, graded by the same rules as every other condition; no model runs. The record fixes which events
are voted on. Writes generations.jsonl (per event: expected correctness, the winning group's size, the tied groups) and
eval_metrics.json (accuracy, per task, how often the vote was tied, and the settings).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.config import shown


def peer_labels(record: dict) -> list[float]:
    labels = record.get("correctness_by_peer") or record.get("peer_correct") or {}
    return [float(labels[k]) for k in sorted(labels)]


def vote_rows(rows: list[dict], events: dict[str, dict], own: dict[str, dict] | None) -> list[dict]:
    from feedback_state.baselines import plurality

    out = []
    for r in sorted(rows, key=lambda r: int(r["pos"])):
        rec = events[str(r["id"])]
        keys = sorted(rec["peer_responses"])
        texts = [str(rec["peer_responses"][k]) for k in keys]
        labels = peer_labels(rec)
        if len(labels) != len(texts):
            raise SystemExit(f"event {r['id']}: {len(texts)} answers but {len(labels)} labels")
        if own is not None:
            mine = own.get(str(r["id"]))
            if mine is None:
                raise SystemExit(f"event {r['id']} has no answer in the --own evaluation")
            texts.append(str(mine["generation"]))
            labels.append(float(mine["correct"]))
        v = plurality(rec, texts, labels)
        out.append({"pos": r["pos"], "id": str(r["id"]), "task_type": str(rec.get("task_type") or r.get("task_type") or ""),
                    "source": str(rec.get("source") or ""), "candidates": len(texts), **v})
    return out


def summary(rows: list[dict]) -> dict:
    hits = np.array([float(r["correct"]) for r in rows])
    return {"accuracy": float(hits.mean()), "num_samples": int(len(rows)),
            "by_task": {t: float(np.mean([float(r["correct"]) for r in rows if r["task_type"] == t])) for t in sorted({r["task_type"] for r in rows})},
            "tied": float(np.mean([r["tied"] > 1 for r in rows]))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=Path, required=True, help="the record of the stream: which events, in which order")
    ap.add_argument("--stream", type=Path, required=True, help="the stream whose peers' answers and labels are voted on")
    ap.add_argument("--own", type=Path, default=None, help="an evaluation of the central model whose answer joins the vote")
    ap.add_argument("--condition", default="vote")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    from feedback_state.data import JsonlDataset

    events = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.stream).records}
    rows = [json.loads(line) for line in args.record.open()]
    own = {str(g["id"]): g for g in map(json.loads, (args.own / "generations.jsonl").open())} if args.own else None
    out = vote_rows(rows, events, own)
    metrics = {"condition": args.condition, "mode": "vote", "record": shown(args.record), "stream": shown(args.stream),
               "own": shown(args.own) if args.own else None, **summary(out)}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "generations.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    (args.output / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[vote] {args.condition}: accuracy {100 * metrics['accuracy']:.2f} over {len(out)} events "
          f"({out[0]['candidates'] if out else 0} candidates, tied {100 * metrics['tied']:.1f}%), "
          f"by task { {k: round(100 * v, 1) for k, v in metrics['by_task'].items()} }", flush=True)


if __name__ == "__main__":
    main()
