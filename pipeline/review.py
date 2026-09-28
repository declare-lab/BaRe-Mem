"""Review: the lead checks every report against what it expected of it.

The lead (`team.checker`, a registered model; default: the central model) reads a sub-task and one source's report and replies with
a one-sentence critique and ACCEPT or REJECT, following the task's `lead_check` instruction (configs/tasks/<task>.yaml): it rejects
a report that gives no answer, says the information is missing, answers a different question or gives the wrong kind of thing. It
sees no evidence, no gold answer and not the record, and only the sub-task, not the whole task: a small model otherwise holds a
right sub-answer against the task's question (2026-09-19: it rejected 27% of the right reports that way).

Every source's report is checked offline so that any choice of source can be replayed on the same events (pipeline.team replay); in
the protocol only the reports of the sources that were called are ever checked.

    PYTHONPATH=. python -m pipeline.review --stream data/musique_team+q3_4b/test.jsonl --sources 7 --model /models/Qwen3-4B \
        --output outputs/review/q3_4b/musique_team+own/lead_check/verdicts --shards 7 --shard 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

VERDICT = re.compile(r"verdict\s*:\s*\**\s*(accept|reject)", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[str, str]:
    """(ACCEPT | REJECT, critique) from the checker's reply; a reply without a verdict accepts."""
    found = VERDICT.findall(text or "")
    verdict = found[-1].upper() if found else "ACCEPT"
    m = re.search(r"critique\s*:\s*(.+)", text or "", re.IGNORECASE)
    critique = (m.group(1) if m else (text or "")).split("\n")[0].strip()
    return verdict, critique[:500]


def check_prompt(record: dict, report: str, instruction: str = "lead_check") -> str:
    from feedback_state.tasks import task_config, task_type_of

    return f"{task_config(task_type_of(record))[instruction]}\n\nSub-question: {record.get('problem', '')}\n\nWorker's answer:\n{str(report).strip()[:1500]}"


def load_rows(directory: Path) -> dict[str, dict]:
    rows = {}
    for f in sorted(glob.glob(str(directory / "shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stream", type=Path, required=True, help="the team's stream, every source's report in it")
    ap.add_argument("--sources", type=int, required=True, help="reports per event")
    ap.add_argument("--model", required=True, help="the checker: the lead's model")
    ap.add_argument("--instruction", default="lead_check", help="the task's instruction the checker follows (configs/tasks/<task>.yaml)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=None, help="per shard")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from feedback_state.data import JsonlDataset
    from feedback_state.peer_generation import prompt_ids
    from feedback_state.permutations import canonical_peer_view

    t0 = time.time()
    records = [r for i, r in enumerate(JsonlDataset(args.stream).records) if i % args.shards == args.shard]
    records = records[: args.max_examples] if args.max_examples else records
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, seed=0,
              disable_sliding_window=True)     # as pipeline.evaluate: one attention pattern for every family of lead
    items = [(i, k, text) for i, r in enumerate(records) for k, text in enumerate(canonical_peer_view(r, args.sources)["texts"][: args.sources])]
    outs = llm.generate([prompt_ids(tok, check_prompt(records[i], text, args.instruction)) for i, _, text in items],
                        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0), use_tqdm=True)
    rows = [{"id": r.get("id"), "verdicts": [None] * args.sources, "critiques": [None] * args.sources} for r in records]
    for (i, k, _), o in zip(items, outs):
        rows[i]["verdicts"][k], rows[i]["critiques"][k] = parse_verdict(o.outputs[0].text)
    rejected = 100 * sum(v == "REJECT" for r in rows for v in r["verdicts"]) / max(1, len(items))
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"shard{args.shard}of{args.shards}"
    tmp = args.output / f"{name}.jsonl.tmp"
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / f"summary.{name}.json").write_text(json.dumps({"checker": args.model, "instruction": args.instruction, "events": len(rows), "reports": len(items),
                                                                  "rejected_pct": rejected, "seconds": time.time() - t0}, indent=1))
    tmp.replace(args.output / f"{name}.jsonl")
    print(f"[review] {Path(args.model).name} checked {len(items)} reports, rejected {rejected:.1f}% ({time.time() - t0:.0f}s) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
