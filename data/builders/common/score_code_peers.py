"""Offline per-peer code pass@1 scorer.

Reads a Setting-A JSONL with code-task records (task_type="code") that already
contain ``peer_responses``, executes each peer's extracted code against the
record's unit tests, and writes a ``peer_correct`` map ({peer_key: 1.0/0.0}) back
into each record. The pipeline then reads that label and never
executes code itself.

Records whose task_type != "code" are passed through unchanged, so you can run
this on a mixed Math+Code JSONL safely.

SECURITY: this executes model-generated code. Run on a disposable/containerised
host and set FEEDBACK_CODE_EXEC_ALLOW=1 to acknowledge.

    FEEDBACK_CODE_EXEC_ALLOW=1 python -m data.builders.common.score_code_peers \
      --input data/code_setting_a.jsonl --output data/code_setting_a_scored.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from data.builders.common.code_grading import score_code_record
from feedback_state.tasks import code_extract_answer
from feedback_state.utils import append_jsonl, completed_ids, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score per-peer code pass@1 offline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    if args.max_samples is not None:
        records = records[: int(args.max_samples)]
    done = completed_ids(args.output)

    pending = [r for r in records if str(r.get("id", "")) not in done]
    buffer: list[dict] = []
    for record in tqdm(pending, desc="score_code_peers"):
        if str(record.get("task_type", "math")) != "code":
            buffer.append(record)
            continue
        peer_correct: dict[str, float] = {}
        for key, text in dict(record.get("peer_responses", {})).items():
            code = code_extract_answer(str(text))
            result = score_code_record(record, code, timeout=float(args.timeout))
            peer_correct[key] = 1.0 if result.passed else 0.0
        record["peer_correct"] = peer_correct
        buffer.append(record)
        if len(buffer) >= 64:
            append_jsonl(args.output, buffer)
            buffer = []
    if buffer:
        append_jsonl(args.output, buffer)


if __name__ == "__main__":
    main()
