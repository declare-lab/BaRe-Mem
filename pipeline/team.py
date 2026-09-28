"""Team: a lead-worker team's sub-task stream, and the lead's choice of source replayed along it.

    build    a benchmark's sub-tasks as a stream with one view per tool (`built: musique`): MuSiQue
             gives every multi-hop question its decomposition, the sub-questions with their gold answers, later ones referring
             to earlier answers (#1, #2). One `subqa` event per hop, teacher-forced (every reference filled with the gold
             answer, so a report is verified against the hop's gold answer whatever was committed before), a question's hops
             together and in order. The stream itself is the lead's view; each view is the same events with the passages that
             tool retrieves, and a worker (a registered peer with `tool:`) answers from its view:
               test.jsonl            BM25 top-3 over the question's own 20 paragraphs (the lead's own tool; view local3)
               agents/local1.jsonl   BM25 top-1 over them
               agents/global3.jsonl  BM25 top-3 over every paragraph of the split (an open index: more misses, more distractors)
             A closed-book worker answers test.jsonl without its passages.
             tasks.jsonl is the baseline's stream: one event per WHOLE question (no decomposition), with the question's own 20
             paragraphs ranked by BM25 for the question (the prompt keeps the first 8,000 characters), graded on the final answer.
    replay   feedback_state.agent_team along the stream (its sources: the workers, and the lead's own answer in --own-slot): who
             gets each sub-task (by the record, by success counts, at random), with the first report committed as it is, and with
             the lead's check (pipeline.review) or the dataset's check (--dataset-check: the gold answer) before the commit: a report
             that fails sends the sub-task to the next source in the same order; every report that was produced is written

    PYTHONPATH=. python -m pipeline.team build --source data/musique_src/musique_ans_v1.0_dev.jsonl --out data/musique_hops_q
    PYTHONPATH=. python -m pipeline.team replay --stream data/musique_team+q3_4b/test.jsonl --features outputs/features/q3_4b/musique_team+own \
        --sources 7 --own-slot 6 --by-task --out outputs/eval/q3_4b/musique_team+own/team
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

SOURCE_REPO, SOURCE_FILE = "dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl"


def sub_question(text: str, answers: list[str]) -> str:
    """A hop's question with its references filled and the relation form ('X >> relation') spelled out."""
    text = re.sub(r"#(\d+)", lambda m: answers[int(m.group(1)) - 1] if int(m.group(1)) <= len(answers) else m.group(0), str(text)).strip()
    if ">>" in text:
        subject, relation = (part.strip() for part in text.split(">>", 1))
        return f"What is the '{relation}' of {subject}?"
    return text


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import CountVectorizer

        self.cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w+\b")
        tf = self.cv.fit_transform(docs).astype(np.float32)
        dl, df = np.asarray(tf.sum(1)).ravel(), np.asarray((tf > 0).sum(0)).ravel()
        idf = np.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        w = tf.data * (k1 + 1.0) / (tf.data + k1 * (1.0 - b + b * dl[tf.row] / dl.mean())) * idf[tf.col]
        self.M = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray((self.M @ self.cv.transform([query]).T).todense()).ravel()


def build(args) -> None:
    if not args.source.exists():     # the benchmark's own file, from Hugging Face
        from huggingface_hub import hf_hub_download

        args.source.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(SOURCE_REPO, args.source.name, repo_type="dataset", local_dir=str(args.source.parent))
    rows = [json.loads(l) for l in args.source.open()][: args.limit]
    docs, doc_id = [], {}
    for r in rows:
        for p in r["paragraphs"]:
            key = (p["title"], p["paragraph_text"])
            if key not in doc_id:
                doc_id[key] = len(docs)
                docs.append(f"{p['title']}: {p['paragraph_text']}")
    index = BM25(docs)
    views = {"local3": [], "local1": [], "global3": []}
    found = {k: 0 for k in views}
    whole = []
    for r in rows:
        pool = np.array([doc_id[(p["title"], p["paragraph_text"])] for p in r["paragraphs"]])
        ranked = pool[np.argsort(-index.scores(r["question"])[pool], kind="stable")]
        whole.append({"id": r["id"], "task_type": "subqa", "source": "musique_dev", "problem": r["question"], "answer": r["answer"], "answer_aliases": list(r.get("answer_aliases") or []),
                      "context": [docs[i] for i in ranked], "team": {"sub_tasks": len(r["question_decomposition"]), "kind": r["id"].split("__")[0]},
                      "peer_responses": {}, "correctness_by_peer": {}})
        golds: list[str] = []
        steps = r["question_decomposition"]
        for h, step in enumerate(steps):
            q = sub_question(step["question"], golds)
            s = index.scores(q)
            sup = r["paragraphs"][step["paragraph_support_idx"]] if step.get("paragraph_support_idx") is not None else None
            support = doc_id[(sup["title"], sup["paragraph_text"])] if sup else -1
            local = pool[np.argsort(-s[pool], kind="stable")]
            got = {"local3": local[:3].tolist(), "local1": local[:1].tolist(), "global3": np.argsort(-s, kind="stable")[:3].tolist()}
            last = h == len(steps) - 1
            for name, ids in got.items():
                found[name] += int(support in ids)
                views[name].append({"id": f"{r['id']}/h{h + 1}", "task_type": "subqa", "source": "musique_dev", "problem": q, "answer": step["answer"],
                                    "answer_aliases": list(r.get("answer_aliases") or []) if last else [], "context": [docs[i] for i in ids],
                                    "team": {"task": r["question"], "sub_task": h + 1, "sub_tasks": len(steps), "kind": r["id"].split("__")[0],
                                             "last": last, "support_retrieved": bool(support in ids)},
                                    "peer_responses": {}, "correctness_by_peer": {}})
            golds.append(step["answer"])
    (args.out / "agents").mkdir(parents=True, exist_ok=True)
    for name, events in list(views.items()) + [("tasks", whole)]:
        path = args.out / "test.jsonl" if name == "local3" else args.out / "tasks.jsonl" if name == "tasks" else args.out / "agents" / f"{name}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        tmp.replace(path)
    n = len(views["local3"])
    manifest = {"source": str(args.source), "tasks": len(rows), "events": n, "paragraphs": len(docs),
                "support_retrieved": {k: round(v / max(1, n), 4) for k, v in found.items()}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[team] {json.dumps(manifest)}", flush=True)


def replay_all(args) -> None:
    import torch

    from feedback_state import agent_team as at
    from feedback_state.addresses import Projection
    from feedback_state.feature_streams import load_stream_from
    from pipeline.review import load_rows

    device, t0 = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"), time.time()
    fs = load_stream_from(args.stream, args.features, num_peers=args.sources, limit=args.limit)
    psi_q = Projection(fs.q_mean, args.dim, device)(fs.q_mean).to(torch.float64)      # the sub-task's address: question-only features, label-free PCA
    labels = fs.labels.numpy().astype(int)
    N, S = labels.shape
    tasks = at.task_index(fs.ids) if args.by_task else None
    rejected = None
    if args.verdicts:                                # the lead's check of every report (pipeline.review)
        v = load_rows(args.verdicts)
        rejected = np.array([[v[str(i)]["verdicts"][s] == "REJECT" for s in range(S)] for i in fs.ids])
    names = [str(dict((fs.records[0].get("peer_metadata") or {}).get(f"peer_{s}", {})).get("model") or f"peer_{s}").rsplit("/", 1)[-1] for s in range(S)]
    if args.own_slot is not None:
        names[args.own_slot] = "lead (own answer)"
    ref = {"sources": names, "accuracy": (100 * labels.mean(0)).round(2).tolist(), "any_right": round(100 * float(labels.max(1).mean()), 2)}
    if tasks is not None:
        ref.update(tasks=len(tasks), task_accuracy=[round(100 * float(np.mean([labels[idx, s].all() for idx in tasks])), 2) for s in range(S)])
    if rejected is not None:
        ref["check"] = dict(at.check_quality(labels, rejected), calls=args.calls)
    if args.direct and (args.direct / "eval_metrics.json").exists():     # the baseline without a team: the lead answers the whole task directly
        ref["direct_task_accuracy"] = round(100 * float(json.loads((args.direct / "eval_metrics.json").read_text())["accuracy"]), 2)
    print(f"[team] {N} sub-tasks, {S} sources, dim {args.dim}, lam {args.lam:g} ({time.time() - t0:.0f}s)\n       {json.dumps(ref)}", flush=True)
    res = {"stream": str(args.stream), "sub_tasks": N, "reference": ref, "dim": args.dim, "lam": args.lam, "orders": args.orders, "calls": args.calls, "policies": {}}
    for pol in args.policies or list(at.POLICIES):
        modes = [("first_report", None, "all")] + ([("lead_check", rejected, "all")] if rejected is not None else [])
        if args.dataset_check:       # the benchmark's evaluator verifies: exact; and the ablation that writes the committed report only
            modes += [("dataset_check", labels == 0, "all"), ("dataset_check_final_only", labels == 0, "final")]
        for mode, rj, write in modes:
            runs = [at.replay(pol, psi_q, labels, at.stream_order(N, o, tasks), rejected=rj, calls=args.calls, write=write, lam=args.lam, seed=o, tasks=tasks,
                              own=args.own_slot, device=device) for o in args.orders]
            mean = {m: (np.mean([r[m] for r in runs], 0).round(4).tolist() if isinstance(runs[0][m], list) else round(float(np.mean([r[m] for r in runs])), 4))
                    for m in runs[0]}
            res["policies"].setdefault(pol, {})[mode] = mean
            print(f"       {pol:14s} {mode:12s} worker calls {mean['worker_calls']:.2f}  sub-task {mean['accuracy']:.2f}"
                  + (f"  task {mean['task_accuracy']:.2f}" if "task_accuracy" in mean else "") + f"  ({time.time() - t0:.0f}s)", flush=True)
    if args.dataset_check:           # the dataset's check against the sources a sub-task may go to, 1 ... every source: every order ends at "any source right"
        res["sweep"] = {}
        for pol in args.policies or list(at.POLICIES):
            for k in range(1, S + 1):
                runs = [at.replay(pol, psi_q, labels, at.stream_order(N, o, tasks), rejected=labels == 0, calls=k, lam=args.lam, seed=o, tasks=tasks, own=args.own_slot,
                                  device=device) for o in args.orders]
                row = {"sources_allowed": k, **{m: round(float(np.mean([r[m] for r in runs])), 4) for m in ("accuracy", "worker_calls") + (("task_accuracy",) if tasks is not None else ())}}
                res["sweep"].setdefault(pol, []).append(row)
            print(f"       {pol:14s} dataset's check, 1..{S} sources allowed: sub-task " + " ".join(f"{r['accuracy']:.1f}" for r in res["sweep"][pol]) + f"  ({time.time() - t0:.0f}s)", flush=True)
        if "memory" in res["sweep"]:     # the ablation: the record written with the committed report only (the failed reports thrown away)
            for k in range(1, S + 1):
                runs = [at.replay("memory", psi_q, labels, at.stream_order(N, o, tasks), rejected=labels == 0, calls=k, write="final", lam=args.lam, seed=o, tasks=tasks,
                                  own=args.own_slot, device=device) for o in args.orders]
                res["sweep"].setdefault("memory_final_only", []).append(
                    {"sources_allowed": k, **{m: round(float(np.mean([r[m] for r in runs])), 4) for m in ("accuracy", "worker_calls") + (("task_accuracy",) if tasks is not None else ())}})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "team.json").write_text(json.dumps(res, indent=1))
    (args.out / "team.md").write_text(table(res, args.title or str(args.stream)))
    print(f"[team] wrote {args.out}/team.json and team.md", flush=True)


def table(res: dict, title: str) -> str:
    ref, by_task, checked = res["reference"], "task_accuracy" in res["reference"], "check" in res["reference"]
    out = [f"# {title}", "", f"{res['sub_tasks']} sub-tasks" + (f", {ref['tasks']} tasks" if by_task else "") + f"; record dim {res['dim']}, lam {res['lam']:g}; "
           f"mean of stream orders {res['orders']}; every policy from a cold start; only the reports that were produced are labelled and written.", "",
           "| source | sub-task accuracy |" + (" task accuracy alone |" if by_task else "") + (" rejected by the lead's check | wrong reports caught | right reports rejected |" if checked else ""),
           "|---|---:|" + ("---:|" if by_task else "") + ("---:|---:|---:|" if checked else "")]
    for s, name in enumerate(ref["sources"]):
        q = ref["check"]["by_source"][s] if checked else None
        out.append(f"| {name} | {ref['accuracy'][s]:.1f} |" + (f" {ref['task_accuracy'][s]:.1f} |" if by_task else "")
                   + (f" {q['rejected']:.1f} | {q['wrong_caught']:.1f} | {q['right_rejected']:.1f} |" if checked else ""))
    if checked:
        q = ref["check"]["overall"]
        out.append(f"| every report | {q['right']:.1f} |" + (" |" if by_task else "") + f" {q['rejected']:.1f} | {q['wrong_caught']:.1f} | {q['right_rejected']:.1f} |")
    out.append(f"| any source right | {ref['any_right']:.1f} |" + (" |" if by_task else "") + (" | | |" if checked else ""))
    if "direct_task_accuracy" in ref:
        out += ["", f"Baselines without a team: the lead answers the whole task directly {ref['direct_task_accuracy']:.1f}% of tasks; the lead does every sub-task itself "
                f"{ref['task_accuracy'][-1]:.1f}% (the last source above)."]
    out += ["", "| who gets the sub-task | before the commit | worker calls per sub-task | sub-task accuracy |" + (" task accuracy |" if by_task else "") + " first choice right | first choice right, along the stream (eighths) | committed from the lead itself |",
            "|---|---|---:|---:|" + ("---:|" if by_task else "") + "---:|---|---:|"]
    for pol, modes in res["policies"].items():
        for mode, r in modes.items():
            label = {"first_report": "nothing: the first report is committed", "lead_check": "the lead's check; rejected -> the next source",
                     "dataset_check": "the dataset's check; failed -> the next source", "dataset_check_final_only": "the dataset's check; only the committed report is written"}[mode]
            out.append(f"| {pol} | {label} | {r['worker_calls']:.2f} | {r['accuracy']:.1f} |" + (f" {r['task_accuracy']:.1f} |" if by_task else "")
                       + f" {r['first_accuracy']:.1f} | {' '.join(f'{v:.0f}' for v in r['along_first'])} |" + (f" {100 * r['autonomous']:.0f}% |" if "autonomous" in r else " |"))
    if res.get("sweep"):
        pols = list(res["sweep"])
        out += ["", "The dataset's check against the sources a sub-task may go to: sub-task accuracy" + (" / task accuracy" if by_task else "") + " (worker calls used). Every order ends at "
                "\"any source right\"; the question is how fast.", "", "| sources allowed | " + " | ".join(pols) + " |", "|---:|" + "---:|" * len(pols)]
        for i in range(len(res["sweep"][pols[0]])):
            cells = [res["sweep"][p][i] for p in pols]
            out.append(f"| {cells[0]['sources_allowed']} | " + " | ".join(f"{c['accuracy']:.1f}" + (f" / {c['task_accuracy']:.1f}" if by_task else "") + f" ({c['worker_calls']:.2f})" for c in cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", type=Path, required=True, help=f"{SOURCE_FILE} (fetched from {SOURCE_REPO} when missing)")
    b.add_argument("--out", type=Path, required=True, help="the stream's directory")
    b.add_argument("--limit", type=int, default=None, help="tasks")
    r = sub.add_parser("replay")
    r.add_argument("--stream", type=Path, required=True)
    r.add_argument("--features", type=Path, required=True, help="the stream's features (pipeline.features): the question-only part addresses the record")
    r.add_argument("--sources", type=int, required=True, help="answers per event: the workers, plus the lead's own answer")
    r.add_argument("--own-slot", type=int, default=None, help="the slot of the lead's own answer (the autonomous option)")
    r.add_argument("--verdicts", type=Path, default=None, help="the lead's check of every report (pipeline.review); without it only the first report is committed")
    r.add_argument("--dataset-check", action="store_true", help="also replay with the benchmark's evaluator verifying every report before the commit "
                   "(and its ablation that writes the committed report only)")
    r.add_argument("--direct", type=Path, default=None, help="the lead's no-consultation evaluation of the whole tasks (the baseline without a team)")
    r.add_argument("--calls", type=int, default=2, help="the sources a sub-task may go to at most, when the lead rejects a report")
    r.add_argument("--by-task", action="store_true", help="sub-tasks are named <task>/h<k>: orders shuffle the tasks, task accuracy is reported")
    r.add_argument("--dim", type=int, default=64)
    r.add_argument("--lam", type=float, default=100.0)
    r.add_argument("--orders", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--policies", nargs="+", default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--title", default=None)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    build(args) if args.cmd == "build" else replay_all(args)


if __name__ == "__main__":
    main()
