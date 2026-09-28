"""BaRe-Mem: per event, the central model's final answer is Advisors + memory or No consultation, chosen by its reading line.

    python -m pipeline.combination --record outputs/record/q3_4b/capability_supported_misleading_p050+own/shuffled0.fit-self.jsonl \
        --peers-memory outputs/eval/q3_4b/capability_supported_misleading_p050+own/tilt --question-alone outputs/eval/q3_4b/capability_supported/solo \
        --output outputs/eval/q3_4b/capability_supported_misleading_p050+own/combination

The record estimates the peers and the central model's own answer (its no-consultation answer) alike (pipeline.record
--own-slot): per event T is the top estimate among the peers and kappa the estimate of the own answer. Walking the stream
in the record's order, read before write: with the reading line of the event's task type (feedback_state.reading_line),
take Advisors + memory when T rho + (1 - T)(kappa - delta) >= kappa, otherwise No consultation; then write whether
Advisors + memory was right into the reading line. That outcome feeds nothing else, neither the record nor the tilt, so this pass
gives exactly what an online loop would. Writes generations.jsonl (the final answer and the choice per event) and
eval_metrics.json (accuracy, the share that took Advisors + memory, both options' accuracy, the reading line per task type,
and the same by trust band).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from pipeline.config import shown
from pipeline.evaluate import summarise

BANDS = 5
PEERS_MEMORY, QUESTION_ALONE = "Advisors + memory", "No consultation"


def load_generations(directory: Path) -> dict[str, dict]:
    return {str(g["id"]): g for g in map(json.loads, (directory / "generations.jsonl").open())}


def combine(rows: list[dict], peers_memory: dict[str, dict], question_alone: dict[str, dict], prior: tuple[float, float],
            lam: float, skip_missing: bool = False) -> tuple[list[dict], dict]:
    from feedback_state.reading_line import ReadingLine

    lines: dict[str, ReadingLine] = {}
    out, mismatched, skipped = [], 0, 0
    for r in sorted(rows, key=lambda r: int(r["pos"])):
        rid = str(r["id"])
        if rid not in peers_memory or rid not in question_alone:
            if skip_missing:      # the record walks the whole stream; only a subsample was generated
                skipped += 1
                continue
            raise SystemExit(f"event {rid} has no {'Advisors + memory' if rid not in peers_memory else 'no-consultation'} answer")
        trust, kappa = max(float(p) for p in r["memory_prob"]), float(r["own_prob"])
        line = lines.setdefault(r["task_type"], ReadingLine(prior, lam))
        rho, delta = line.estimate()
        value = line.value(trust, kappa)
        take_peers = value >= kappa
        pm_ok, qa_ok = int(peers_memory[rid]["correct"]), int(question_alone[rid]["correct"])
        mismatched += qa_ok != int(r["own_correct"])
        chosen = peers_memory[rid] if take_peers else question_alone[rid]
        out.append({"pos": r["pos"], "id": rid, "task_type": r["task_type"], "source": r.get("source"), "correct": int(chosen["correct"]),
                    "choice": PEERS_MEMORY if take_peers else QUESTION_ALONE, "trust": round(trust, 4), "own_prob": round(kappa, 4),
                    "peers_memory_value": round(value, 4), "rho": round(rho, 4), "delta": round(delta, 4),
                    "peers_memory_correct": pm_ok, "question_alone_correct": qa_ok, "peer_correct": r["peer_correct"],
                    "generation": chosen["generation"]})
        line.write(trust, kappa, pm_ok)
    by_task = collections.defaultdict(list)
    for o in out:
        by_task[o["task_type"]].append(o)
    summary = {"own_labels_mismatched": mismatched, "skipped_without_answer": skipped, "reading_line": {}}
    for task, line in sorted(lines.items()):
        rho, delta = line.estimate()
        mean_own = float(np.mean([o["own_prob"] for o in by_task[task]]))
        summary["reading_line"][task] = {"events": line.events, "rho": rho, "delta": delta, "mean_own_prob": mean_own,
                                         "switch_at_mean_own_prob": line.switch(mean_own), "peers_memory_wins_at_mean_own_prob": line.reads_when(mean_own),
                                         "share_peers_memory": float(np.mean([o["choice"] == PEERS_MEMORY for o in by_task[task]]))}
    return out, summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=Path, required=True, help="the record with the own answer (pipeline.record --own-slot)")
    ap.add_argument("--peers-memory", type=Path, required=True, help="the central model's Advisors + memory evaluation")
    ap.add_argument("--question-alone", type=Path, required=True, help="its no-consultation evaluation")
    ap.add_argument("--prior", default="0.5,0.0", help="rho, delta before any event")
    ap.add_argument("--lam", type=float, default=1.0, help="the prior's precision")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--skip-missing", action="store_true",
                    help="ignore record events with no generation (the record walked the whole stream, a subsample was answered)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    prior = tuple(float(x) for x in args.prior.split(","))
    rows = [json.loads(line) for line in args.record.open()]
    out, summary = combine(rows, load_generations(args.peers_memory), load_generations(args.question_alone), prior, args.lam,
                           skip_missing=args.skip_missing)
    mean = lambda key, sel=out: float(np.mean([o[key] for o in sel])) if sel else None
    share = lambda sel: float(np.mean([o["choice"] == PEERS_MEMORY for o in sel]))
    bands = collections.defaultdict(list)
    for o in out:
        bands[min(BANDS - 1, int(o["trust"] * BANDS))].append(o)
    metrics = {"condition": "combination", "record": shown(args.record), "peers_memory_eval": shown(args.peers_memory),
               "question_alone_eval": shown(args.question_alone), "prior": list(prior), "lam": args.lam}
    metrics.update(summarise(out, args.windows))
    metrics.update({"share_peers_memory": share(out), "peers_memory_accuracy": mean("peers_memory_correct"),
                    "question_alone_accuracy": mean("question_alone_correct"),
                    "either_right": float(np.mean([max(o["peers_memory_correct"], o["question_alone_correct"]) for o in out])),
                    **summary,
                    "by_trust": {f"{b / BANDS:.1f}-{(b + 1) / BANDS:.1f}": {"events": len(sel), "share_peers_memory": share(sel),
                                                                          "peers_memory": mean("peers_memory_correct", sel),
                                                                          "question_alone": mean("question_alone_correct", sel),
                                                                          "combination": mean("correct", sel)}
                                 for b, sel in sorted(bands.items())}})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "generations.jsonl").open("w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    (args.output / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    lines = "; ".join(f"{t} rho {v['rho']:.2f} delta {v['delta']:.2f}" for t, v in metrics["reading_line"].items())
    print(f"[combination] {args.output}: accuracy {100 * metrics['accuracy']:.2f} (Advisors + memory {100 * metrics['peers_memory_accuracy']:.2f}, "
          f"No consultation {100 * metrics['question_alone_accuracy']:.2f}), Advisors + memory taken on {100 * metrics['share_peers_memory']:.1f}% "
          f"of events; {lines}", flush=True)


if __name__ == "__main__":
    main()
