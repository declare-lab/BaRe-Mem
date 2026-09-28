#!/usr/bin/env python3
"""Pre-grade per-peer correctness into record['peer_correct'] so eval never runs the
(slow, sometimes-hanging) sympy grader at runtime. Each sample is graded in a
subprocess with a hard wall-clock timeout (sympy can ignore SIGALRM); on timeout the
peer is scored 0. Code datasets already have peer_correct (skip). RAG uses fast F1.

Usage: python -m data.builders.common.precompute_peer_correct <in.jsonl> <out.jsonl> [timeout_s]
"""
import json, sys, multiprocessing as mp
from feedback_state.tasks import task_type_of, get_task

def _grade_one(args):
    record, key, text = args
    task = get_task(task_type_of(record))
    return float(task.target_fn(text, record))


def _worker(args, q):
    try:
        q.put(_grade_one(args))
    except Exception:
        q.put(0.0)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    n = 0
    timeouts = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if "peer_correct" in r and isinstance(r["peer_correct"], dict):
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            pc = {}
            peer_keys = sorted(dict(r.get("peer_responses", {})).keys())
            task_type = task_type_of(r)
            for k in peer_keys:
                text = str(r["peer_responses"][k])
                if task_type != "math":
                    pc[k] = _grade_one((r, k, text))
                    continue
                q = mp.Queue()
                p = mp.Process(target=_worker, args=((r, k, text), q))
                p.start()
                p.join(timeout)
                if p.is_alive():
                    p.terminate()
                    p.join()
                    pc[k] = 0.0
                    timeouts += 1
                else:
                    try:
                        pc[k] = q.get_nowait()
                    except Exception:
                        pc[k] = 0.0
            r["peer_correct"] = pc
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[precompute] {src} -> {dst}: {n} records, {timeouts} grade-timeouts")


if __name__ == "__main__":
    main()
