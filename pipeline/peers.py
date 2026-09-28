"""Peers: a peer model answers every event of a stream, and each answer is graded.

    --mode honest       the task's peer prompt, sampled as the released peers were (temperature 0.2, top-p 0.95)
                        --turns N (N > 1): the peer is an agent -- after each answer it observes the task's visible check
                        (its feedback_fn, e.g. the docstring's examples run on its code) and revises, up to N answers; the
                        last one is graded (feedback_state.peer_generation.agentic_answers)
    --mode misleading   a confident, relevant, verified-wrong answer: generated, graded and re-generated until usable
                        (feedback_state.adversarial: acceptance rules, escalating retries, a target value for math,
                        a rewritten conclusion only as the last resort, flagged ``forced``)

    PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python -m pipeline.peers --mode misleading \
        --model /models/Meta-Llama-3.1-8B-Instruct --stream data/capability_supported/test.jsonl \
        --output outputs/peers/capability_supported/misleading/Meta-Llama-3.1-8B-Instruct --shards 8 --shard 0

Writes <output>/shard<k>of<N>.jsonl, one row per event ({id, source, task_type, response, target, correct}; misleading
rows add accepted, forced, attempts, reasons, soft) and <output>/summary.shard<k>of<N>.json, which records the
generation settings (pipeline.streams copies them into the stream) and the acceptance statistics.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

from pipeline.config import shown

FORCEABLE = {"math", "mcqa", "boolqa"}   # tasks whose conclusion can be rewritten without rewriting the argument


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["honest", "misleading"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--stream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=None, help="per shard")
    p.add_argument("--sources", default=None, help="comma-separated source datasets to keep")
    p.add_argument("--max-tokens", default=None, help="per task, e.g. math=512,rag=256,code=768 (default: the released peers' budgets; "
                   "misleading answers get 256 on the short-answer tasks)")
    p.add_argument("--reasoning", action="store_true", help="a thinking peer: 4096-token budget, graded after the last </think>")
    p.add_argument("--no-context", action="store_true", help="reading tasks without the passage")
    p.add_argument("--temperature", type=float, default=0.2, help="honest mode")
    p.add_argument("--turns", type=int, default=1, help="honest mode: answers per event, revising on the task's visible check")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-attempts", type=int, default=3, help="misleading mode: generations per event before the forcing pass")
    p.add_argument("--temperatures", default="0.2,0.7,1.0", help="misleading mode: one per attempt")
    p.add_argument("--no-force", action="store_true", help="misleading mode: never rewrite a conclusion")
    p.add_argument("--grade-workers", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args(argv)


def budgets(spec: str | None, mode: str = "honest") -> dict:
    from feedback_state.peer_generation import DEFAULT_MAX_TOKENS, MISLEADING_MAX_TOKENS

    out = dict(MISLEADING_MAX_TOKENS if mode == "misleading" else DEFAULT_MAX_TOKENS)
    if not spec:
        return out
    for part in spec.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM

    from feedback_state.data import JsonlDataset

    args.output.mkdir(parents=True, exist_ok=True)
    records = [r for i, r in enumerate(JsonlDataset(args.stream).records) if i % args.shards == args.shard]
    if args.sources:
        keep = set(args.sources.split(","))
        records = [r for r in records if r.get("source") in keep]
    if args.max_examples:
        records = records[: args.max_examples]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.trust_remote_code)
    t0 = time.time()
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enable_prefix_caching=not args.no_prefix_caching,
              trust_remote_code=args.trust_remote_code, enforce_eager=args.enforce_eager, seed=0)
    run = misleading if args.mode == "misleading" else agentic if args.turns > 1 else honest
    rows, extra = run(args, records, tok, llm)
    del llm
    name = f"shard{args.shard}of{args.shards}"
    with (args.output / f"{name}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = max(1, len(rows))
    by_source = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_source[r["source"]][0] += 1; by_source[r["source"]][1] += r["correct"]
    summary = {"mode": args.mode, "model": args.model, "stream": shown(args.stream), "n": len(rows),
               "generation_params": {"temperature": args.temperature if args.mode == "honest" else args.temperatures,
                                     "top_p": args.top_p, "max_new_tokens": "reasoning 4096" if args.reasoning else budgets(args.max_tokens, args.mode),
                                     "context": not args.no_context, "reasoning": args.reasoning, "backend": "vllm",
                                     "prompt": "chat template as token ids, no special tokens added", "turns": args.turns},
               "accuracy_pct": 100 * sum(r["correct"] for r in rows) / n,
               "by_source": {s: {"n": c, "accuracy_pct": 100 * k / c} for s, (c, k) in by_source.items()},
               "seconds": time.time() - t0, **extra}
    (args.output / f"summary.{name}.json").write_text(json.dumps(summary, indent=1))
    print(f"[peers] {args.mode} {Path(args.model).name}: {len(rows)} answers, graded correct {summary['accuracy_pct']:.1f}%"
          + (f", usable {extra['accepted_pct']:.1f}% ({extra['forced']} forced)" if args.mode == "misleading" else "")
          + f" ({summary['seconds']:.0f}s) -> {args.output / (name + '.jsonl')}", flush=True)


def honest(args, records, tok, llm):
    from vllm import SamplingParams

    from feedback_state.memory_generator import strip_thinking
    from feedback_state.peer_generation import grade_all, max_tokens, prompt_ids
    from feedback_state.tasks import build_peer_prompt, task_type_of

    b = budgets(args.max_tokens, "honest")
    prompts = [prompt_ids(tok, build_peer_prompt(r, with_context=not args.no_context)) for r in records]
    params = [SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=max_tokens(r, b, args.reasoning), seed=0) for r in records]
    texts = [o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=True)]
    values = grade_all([(r, strip_thinking(t) if args.reasoning else t) for r, t in zip(records, texts)], args.grade_workers)
    rows = [{"id": r.get("id"), "source": r.get("source"), "task_type": task_type_of(r), "response": t, "target": v,
             "correct": int(round(v))} for r, t, v in zip(records, texts, values)]
    return rows, {}


def agentic(args, records, tok, llm):
    """Honest answers from a peer that revises on the task's visible check (--turns N): each turn batches the events still
    revising; the last answer is graded by the hidden rule, the visible checks are kept with it."""
    from vllm import SamplingParams

    from feedback_state.memory_generator import strip_thinking
    from feedback_state.peer_generation import agentic_answers, grade_all, max_tokens, messages_ids
    from feedback_state.tasks import get_task, task_type_of

    b = budgets(args.max_tokens, "honest")
    missing = sorted({task_type_of(r) for r in records if get_task(task_type_of(r)).feedback_fn is None})
    if missing:
        raise SystemExit(f"--turns {args.turns}: task(s) {missing} have no visible check (TaskSpec.feedback_fn) to revise on")
    answer_of = (lambda t: strip_thinking(t)) if args.reasoning else (lambda t: t)
    turn = [0]

    def generate(items):
        prompts = [messages_ids(tok, conv) for _, conv in items]
        params = [SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=max_tokens(records[i], b, args.reasoning),
                                 seed=turn[0]) for i, _ in items]
        turn[0] += 1
        print(f"[peers] turn {turn[0]}: {len(items)} events", flush=True)
        return [o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=True)]

    feedback = lambda r, text: get_task(task_type_of(r)).feedback_fn(r, text)
    answers = agentic_answers(records, generate, feedback, args.turns, answer_of, workers=args.grade_workers)
    values = grade_all([(r, answer_of(a["response"])) for r, a in zip(records, answers)], args.grade_workers)
    rows = [{"id": r.get("id"), "source": r.get("source"), "task_type": task_type_of(r), "response": a["response"], "target": v,
             "correct": int(round(v)), "turns": a["turns"], "visible": a["visible"]} for r, a, v in zip(records, answers, values)]
    n = max(1, len(rows))
    first = sum(a["visible"][0]["passed"] for a in answers if a["visible"])
    last = sum(a["visible"][-1]["passed"] for a in answers if a["visible"])
    extra = {"agentic": {"turns": args.turns, "mean_turns": sum(a["turns"] for a in answers) / n,
                         "visible_passed_first_pct": 100 * first / n, "visible_passed_last_pct": 100 * last / n,
                         "hidden_passed_by_turns": {str(k): 100 * sum(r["correct"] for r in rows if r["turns"] == k) / max(1, sum(r["turns"] == k for r in rows))
                                                    for k in sorted({r["turns"] for r in rows})}}}
    return rows, extra


def misleading(args, records, tok, llm):
    from vllm import SamplingParams

    from feedback_state.adversarial import accept, force_wrong, misleading_prompt, plausible_wrong_number
    from feedback_state.memory_generator import strip_thinking
    from feedback_state.peer_generation import grade_all, max_tokens, prompt_ids
    from feedback_state.tasks import task_type_of

    b = budgets(args.max_tokens, "misleading")
    temps = [float(x) for x in args.temperatures.split(",")] or [0.2]
    answer_of = (lambda t: strip_thinking(t)) if args.reasoning else (lambda t: t)
    done: dict[int, dict] = {}
    # "reasons": faults of the latest attempt (what the next prompt complains about);
    # "best" / "best_reasons": the attempt kept for the forcing pass, and its own faults
    state = {i: {"reasons": [], "attempts": 0, "best": None, "best_value": None, "best_reasons": []} for i in range(len(records))}
    pending = list(range(len(records)))
    attempt_stats = []
    total = max(1, args.max_attempts)
    only_correct = lambda rs: set(rs) <= {"still_correct"}   # fails only by being right: can still be turned around
    for attempt in range(total):
        if not pending:
            break
        last = attempt == total - 1   # on the last attempt a soft fault no longer costs the answer
        temp = temps[min(attempt, len(temps) - 1)]
        # a math peer that keeps solving the problem is given the wrong value to arrive at: a derivation that reaches it
        # misleads, where a conclusion rewritten afterwards contradicts the lines above it
        targets = {i: plausible_wrong_number(records[i], answer_of(state[i]["best"])) for i in pending
                   if last and task_type_of(records[i]) == "math" and "still_correct" in state[i]["best_reasons"] and state[i]["best"]}
        prompts = [prompt_ids(tok, misleading_prompt(records[i], attempt=attempt, complaints=state[i]["reasons"], target_answer=targets.get(i)))
                   for i in pending]
        params = [SamplingParams(temperature=temp, top_p=args.top_p, max_tokens=max_tokens(records[i], b, args.reasoning), seed=attempt)
                  for i in pending]
        texts = [o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=True)]
        values = grade_all([(records[i], answer_of(t)) for i, t in zip(pending, texts)], args.grade_workers)
        nxt, accepted_now, reasons_now = [], 0, collections.Counter()
        for i, text, value in zip(pending, texts, values):
            st = state[i]
            st["attempts"] = attempt + 1
            verdict = accept(records[i], answer_of(text), value=value, strict=not last)
            if verdict.ok:
                done[i] = {"response": text, "target": value, "attempts": attempt + 1, "reasons": [], "soft": verdict.soft, "forced": False}
                accepted_now += 1
                continue
            reasons_now.update(verdict.reasons)
            if st["best"] is None or (only_correct(verdict.reasons) and not only_correct(st["best_reasons"])):
                st["best"], st["best_value"], st["best_reasons"] = text, value, list(verdict.reasons)
            st["reasons"] = verdict.reasons
            nxt.append(i)
        attempt_stats.append({"attempt": attempt, "temperature": temp, "generated": len(pending), "accepted": accepted_now,
                              "given_a_target": len(targets), "reasons": dict(reasons_now)})
        print(f"[peers] attempt {attempt} (T={temp}): {accepted_now}/{len(pending)} accepted, remaining {len(nxt)}, "
              f"reasons {dict(reasons_now)}", flush=True)
        pending = nxt
    if pending and not args.no_force:   # last resort, counted apart: rewrite the conclusion, re-grade, re-check
        rewritten = {}
        for i in pending:
            if task_type_of(records[i]) in FORCEABLE and state[i]["best"] is not None:
                text, what = force_wrong(records[i], state[i]["best"])
                if what is not None:
                    rewritten[i] = text
        values = grade_all([(records[i], answer_of(t)) for i, t in rewritten.items()], args.grade_workers)
        for (i, text), value in zip(rewritten.items(), values):
            verdict = accept(records[i], answer_of(text), value=value, strict=False)
            if verdict.ok:
                done[i] = {"response": text, "target": value, "attempts": state[i]["attempts"], "reasons": [], "soft": verdict.soft, "forced": True}
    rows, unusable, by_task = [], collections.Counter(), collections.defaultdict(lambda: [0, 0, 0])
    for i, r in enumerate(records):
        got = done.get(i)
        if got is None:
            st = state[i]
            got = {"response": st["best"] or "", "target": st["best_value"] or 0.0, "attempts": st["attempts"],
                   "reasons": st["best_reasons"], "soft": [], "forced": False}
            unusable.update(st["best_reasons"] or ["empty"])
        v = float(got["target"])
        task = task_type_of(r)
        rows.append({"id": r.get("id"), "source": r.get("source"), "task_type": task, "response": got["response"], "target": v,
                     "correct": int(round(v)), "accepted": i in done, "forced": got["forced"], "attempts": got["attempts"],
                     "reasons": got["reasons"], "soft": got["soft"]})
        by_task[task][0] += 1; by_task[task][1] += int(i in done); by_task[task][2] += int(got["forced"])
    n = max(1, len(rows))
    extra = {"max_attempts": args.max_attempts, "forced_allowed": not args.no_force,
             "accepted": sum(r["accepted"] for r in rows), "accepted_pct": 100 * sum(r["accepted"] for r in rows) / n,
             "forced": sum(r["forced"] for r in rows), "mean_attempts": sum(r["attempts"] for r in rows) / n,
             "accepted_with_soft_fault": sum(bool(r["soft"]) for r in rows if r["accepted"]),
             "by_task": {t: {"n": c, "accepted_pct": 100 * a / max(1, c), "forced": f} for t, (c, a, f) in sorted(by_task.items())},
             "attempts": attempt_stats, "unusable_reasons": dict(unusable)}
    return rows, extra


if __name__ == "__main__":
    main()
