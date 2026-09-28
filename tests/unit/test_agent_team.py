"""The lead-worker team (feedback_state.agent_team, pipeline.team) and the registries it rests on: tasks registered by file,
workers with tools, the experiment's jobs, the sub-task stream, the replayed loop."""
import json
from pathlib import Path

import numpy as np
import torch

from feedback_state import agent_team as at
from feedback_state.tasks import GRADERS, TASK_CONFIGS, get_task, peer_is_correct
from pipeline.config import load
from pipeline.registry import load_registry
from pipeline.run import Plan

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
OVER = lambda tmp: [f"paths.outputs={tmp}/out", f"paths.data={tmp}/data", "paths.models_root=/models"]


def test_every_task_type_is_registered_by_a_file_and_keeps_the_prompts_of_the_stored_results():
    from feedback_state.memory_generator import INSTRUCTIONS
    from feedback_state.peer_generation import DEFAULT_MAX_TOKENS, MISLEADING_MAX_TOKENS

    assert set(TASK_CONFIGS) >= {"math", "rag", "code", "boolqa", "mcqa", "shortqa", "subqa"}
    assert all(cfg["grader"] in GRADERS and get_task(name).name == name for name, cfg in TASK_CONFIGS.items())
    # the values every stored result was produced with (hard-coded until 2026-09-18)
    assert INSTRUCTIONS["math"] == "Solve the problem. Reason briefly, then end with a line of the form 'Final answer: <number>'."
    assert INSTRUCTIONS["rag"] == "Answer the question using the evidence. End with a line of the form 'Final answer: <short answer>'."
    assert INSTRUCTIONS["mcqa"] == "Choose the correct option. End with a line of the form 'Final answer: (<letter>)'."
    assert INSTRUCTIONS["boolqa"] == "Decide. End with a line of the form 'Final answer: <yes or no>'."
    assert INSTRUCTIONS["shortqa"] == "Answer briefly. End with a line of the form 'Final answer: <answer>'."
    assert INSTRUCTIONS["code"] == ("Write a complete Python program that reads from standard input and writes the answer to standard "
                                    "output (use input()/sys.stdin and print()). Return the program inside a single ```python code block.")
    assert {k: DEFAULT_MAX_TOKENS[k] for k in ("math", "rag", "code", "boolqa", "mcqa", "shortqa")} == \
        {"math": 512, "rag": 256, "code": 768, "boolqa": 96, "mcqa": 96, "shortqa": 96}
    assert {k: MISLEADING_MAX_TOKENS[k] for k in ("math", "rag", "code", "boolqa", "mcqa", "shortqa")} == \
        {"math": 512, "rag": 256, "code": 768, "boolqa": 256, "mcqa": 256, "shortqa": 256}
    assert get_task("code").precomputed and not get_task("subqa").precomputed
    reg = load_registry()
    assert reg.problems() == [] and reg.task("subqa")["grader"] == "subqa" and reg.dataset("musique_team")["task"] == "subqa"


def test_a_workers_report_is_right_when_it_carries_the_entity():
    ev = {"task_type": "subqa", "problem": "Who is the spouse of Steve Hillage?", "answer": "Miquette Giraudy", "answer_aliases": []}
    assert peer_is_correct(ev, None, "Miquette Giraudy")
    assert peer_is_correct(ev, None, "The spouse of Steve Hillage is Miquette Giraudy.")          # a sentence: F1 0.44, but it holds the entity
    assert not peer_is_correct(ev, None, "Tracey Hillage")
    long = "Steve Hillage has worked with many people over the years, among them " + "several musicians and " * 4 + "Miquette Giraudy and others."
    assert not peer_is_correct(ev, None, long)                                                       # a long list that happens to contain it
    assert not peer_is_correct(dict(ev, task_type="rag"), None, "The spouse of Steve Hillage is Miquette Giraudy.")   # the streams' rule is untouched


def test_the_experiment_builds_the_sub_tasks_has_each_worker_use_its_tool_and_replays_the_loop(tmp_path):
    plan = Plan(load(EXPERIMENTS / "agent_team.yaml", OVER(tmp_path)), "agent_team.yaml", smoke=False, gpus=list(range(7)))
    [q] = plan.questions()
    assert "pipeline.team build" in q.cmd and "musique_src/musique_ans_v1.0_dev.jsonl" in q.cmd and q.gpus == 0 and q.cmd.endswith("musique_hops_q")
    peers = {j.name: j.cmd for j in plan.peers()}
    assert len(peers) == 6 * 7                                                                       # six workers, seven shards each
    cmd = lambda model: next(c for n, c in peers.items() if model in n and n.endswith("_0"))
    assert "--no-context" in cmd("gemma-3-4b-it") and "musique_hops_q/test.jsonl" in cmd("gemma-3-4b-it")
    assert "--no-context" not in cmd("Phi-4-mini") and "musique_hops_q/test.jsonl" in cmd("Phi-4-mini")
    assert "agents/local1.jsonl" in cmd("Qwen2.5-Coder") and "agents/global3.jsonl" in cmd("Llama-3.1")
    assert "agents/global3.jsonl" in cmd("R1-Distill") and "--reasoning" in cmd("R1-Distill")
    [s] = plan.streams()
    assert "pipeline.streams add" in s.cmd and "musique_hops_answers" in s.cmd and s.cmd.count("--peer ") == 6
    feats = plan.features()
    assert feats and all("--peers 7" in f.cmd and "musique_team+own/" in f.cmd for f in feats)
    [direct] = plan.direct()
    assert "pipeline.evaluate" in direct.cmd and "--mode solo" in direct.cmd and "musique_hops_q/tasks.jsonl" in direct.cmd and "eval/q3_4b/musique_tasks/solo" in str(direct.done)
    assert str(q.done).endswith("musique_hops_q/tasks.jsonl")                                      # one build writes the sub-tasks, their views and the whole tasks
    verify = plan.verify()
    assert len(verify) == 7 and all("pipeline.review --stream" in j.cmd and "--sources 7 --model /models/Qwen3-4B" in j.cmd and "musique_team+q3_4b/test.jsonl" in j.cmd
                                    and "musique_team+own/lead_check/verdicts" in j.cmd for j in verify)
    assert not hasattr(plan, "revise")
    [t] = plan.team()
    assert "--verdicts" in t.cmd and "musique_team+own/lead_check/verdicts --dataset-check --calls 2" in t.cmd and "--direct" in t.cmd and "musique_tasks/solo" in t.cmd
    assert "pipeline.team replay" in t.cmd and "--sources 7 --own-slot 6 --by-task" in t.cmd and "features/q3_4b/musique_team+own " in t.cmd
    assert "musique_team+q3_4b/test.jsonl" in t.cmd and t.cmd.rstrip().endswith("musique_team+own/team")


def test_every_registered_lead_has_its_own_experiment(tmp_path):
    for key, model in (("llama31", "Meta-Llama-3.1-8B-Instruct"), ("phi4", "phi-4"), ("ministral", "Ministral-8B-Instruct-2410"), ("qwen25", "Qwen2.5-7B-Instruct"), ("qwen3_14b", "Qwen3-14B")):
        plan = Plan(load(EXPERIMENTS / f"agent_team_{key}.yaml", OVER(tmp_path)), f"agent_team_{key}.yaml", smoke=False, gpus=list(range(7)))
        assert all(f"--model /models/{model} " in j.cmd and f"review/{key}/musique_team+own/lead_check/verdicts" in j.cmd for j in plan.verify())
        [t] = plan.team()
        assert f"musique_team+{key}/test.jsonl" in t.cmd and f"features/{key}/musique_team+own " in t.cmd


def test_another_lead_reuses_the_workers_reports_and_checks_them_itself(tmp_path):
    plan = Plan(load(EXPERIMENTS / "agent_team_qwen3_8b.yaml", OVER(tmp_path)), "agent_team_qwen3_8b.yaml", smoke=False, gpus=list(range(7)))
    assert all("musique_hops_answers" in j.cmd and "Qwen3-8B" not in j.cmd for j in plan.peers())        # the workers' reports are the same files
    assert all("--model /models/Qwen3-8B" in j.cmd and "review/qwen3_8b/musique_team+own/lead_check/verdicts" in j.cmd for j in plan.verify())
    [t] = plan.team()
    assert "musique_team+qwen3_8b/test.jsonl" in t.cmd and "features/qwen3_8b/musique_team+own " in t.cmd and "--dataset-check --calls 2" in t.cmd


def test_the_sub_task_stream_fills_references_and_gives_every_tool_its_view(tmp_path):
    from pipeline.team import main, sub_question

    assert sub_question("Green >> performer", []) == "What is the 'performer' of Green?"
    assert sub_question("Who is the spouse of #1?", ["Steve Hillage"]) == "Who is the spouse of Steve Hillage?"
    para = lambda i, title, text, sup=False: {"idx": i, "title": title, "paragraph_text": text, "is_supporting": sup}
    row = {"id": "2hop__1_2", "question": "Who is the spouse of the Green performer?", "answer": "Miquette Giraudy", "answer_aliases": ["Giraudy"],
           "paragraphs": [para(0, "Green (album)", "Green is an album whose performer is Steve Hillage.", True),
                          para(1, "Steve Hillage", "Steve Hillage is married; his spouse is Miquette Giraudy.", True),
                          para(2, "Blue", "Blue is a colour of the sky and the sea.")],
           "question_decomposition": [{"id": 1, "question": "Green >> performer", "answer": "Steve Hillage", "paragraph_support_idx": 0},
                                      {"id": 2, "question": "#1 >> spouse", "answer": "Miquette Giraudy", "paragraph_support_idx": 1}]}
    src = tmp_path / "musique.jsonl"
    src.write_text(json.dumps(row) + "\n")
    main(["build", "--source", str(src), "--out", str(tmp_path / "q")])
    lead = [json.loads(l) for l in (tmp_path / "q" / "test.jsonl").open()]
    top1 = [json.loads(l) for l in (tmp_path / "q" / "agents" / "local1.jsonl").open()]
    [task] = [json.loads(l) for l in (tmp_path / "q" / "tasks.jsonl").open()]
    assert task["id"] == "2hop__1_2" and task["problem"] == row["question"] and task["answer"] == "Miquette Giraudy" and len(task["context"]) == 3 and task["answer_aliases"] == ["Giraudy"]
    assert [e["id"] for e in lead] == ["2hop__1_2/h1", "2hop__1_2/h2"] == [e["id"] for e in top1]
    assert lead[1]["problem"] == "What is the 'spouse' of Steve Hillage?" and lead[1]["answer_aliases"] == ["Giraudy"] and lead[0]["answer_aliases"] == []
    assert all(e["task_type"] == "subqa" and e["peer_responses"] == {} for e in lead) and len(top1[1]["context"]) == 1
    assert top1[1]["context"][0].startswith("Steve Hillage:") and top1[1]["team"]["support_retrieved"]
    assert [list(map(int, idx)) for idx in at.task_index([e["id"] for e in lead] + ["3hop__x/h1"])] == [[0, 1], [2]]


def test_the_leads_check_is_parsed_and_sees_the_sub_task_and_the_report_only():
    from pipeline.review import check_prompt, parse_verdict

    assert parse_verdict("Critique: It gives no answer.\nVerdict: REJECT") == ("REJECT", "It gives no answer.")
    assert parse_verdict("Critique: a person's name, as asked.\nVerdict: **ACCEPT**")[0] == "ACCEPT" and parse_verdict("no verdict here")[0] == "ACCEPT"
    ev = {"task_type": "subqa", "problem": "Who is the spouse of Steve Hillage?", "context": ["Steve Hillage: his spouse is Miquette Giraudy."], "answer": "Miquette Giraudy",
          "team": {"task": "Who is the spouse of the Green performer?"}}
    prompt = check_prompt(ev, "No information in the context.")
    assert "meets that expectation" in prompt and "Worker's answer:\nNo information in the context." in prompt
    assert "Green performer" not in prompt and "Miquette" not in prompt                                # the sub-task only: no whole task, no evidence, no gold


def test_the_record_learns_whom_to_ask_from_what_the_chosen_source_produced():
    rng = np.random.default_rng(0)
    N, S = 600, 4
    kind = rng.integers(0, 2, N)                                   # two kinds of sub-task, visible in the sub-task's address
    psi_q = torch.tensor(np.stack([np.where(kind == 0, 1.0, -1.0), rng.normal(size=N) * 0.1, rng.normal(size=N) * 0.1, rng.normal(size=N) * 0.1], 1))
    labels = np.zeros((N, S), dtype=int)
    labels[:, 0] = (kind == 0) & (rng.random(N) < 0.95)            # source 0 is right on kind 0, source 1 on kind 1, the others rarely
    labels[:, 1] = (kind == 1) & (rng.random(N) < 0.95)
    labels[:, 2:] = rng.random((N, 2)) < 0.15
    run = lambda pol, **kw: at.replay(pol, psi_q, labels, at.stream_order(N, 0, None), lam=1.0, seed=0, **kw)
    one = {pol: run(pol) for pol in at.POLICIES}
    assert all(r["worker_calls"] == 1.0 for r in one.values())
    assert one["memory"]["accuracy"] > 80 > one["success_counts"]["accuracy"] > one["random"]["accuracy"]   # success counts cannot see the kind
    assert one["memory"]["along"][-1] >= one["memory"]["along"][0]
    # the lead's check rejects most wrong reports and a few right ones; a rejected sub-task goes to the next source in the order
    rejected = np.where(labels == 1, rng.random((N, S)) < 0.1, rng.random((N, S)) < 0.8)
    ra = {pol: run(pol, rejected=rejected, calls=2) for pol in at.POLICIES}
    assert all(1.0 < ra[p]["worker_calls"] < 2.0 for p in at.POLICIES) and ra["random"]["accuracy"] > one["random"]["accuracy"]
    assert ra["memory"]["accuracy"] > ra["success_counts"]["accuracy"] > ra["random"]["accuracy"]
    assert ra["memory"]["worker_calls"] < ra["random"]["worker_calls"]                                  # a better first choice is rejected less often
    assert run("random", rejected=rejected, calls=S)["accuracy"] > ra["random"]["accuracy"]
    # the dataset's check is exact; every failed report is written too, and throwing them away (write="final") slows the record down
    exact = {w: run("memory", rejected=labels == 0, calls=2, write=w) for w in ("all", "final")}
    assert exact["all"]["accuracy"] >= ra["memory"]["accuracy"] and len(exact["all"]["along_first"]) == 8
    assert exact["all"]["first_accuracy"] >= exact["final"]["first_accuracy"] - 1
    q = at.check_quality(labels, rejected)["overall"]
    assert 70 < q["wrong_caught"] < 90 and 5 < q["right_rejected"] < 15
    tasks = at.task_index([f"t{i // 2}/h{i % 2 + 1}" for i in range(N)])
    by_task = at.replay("memory", psi_q, labels, at.stream_order(N, 1, tasks), rejected=rejected, lam=1.0, seed=1, tasks=tasks, own=3)
    assert 0 <= by_task["task_accuracy"] <= by_task["accuracy"] and 0 <= by_task["autonomous"] <= 1
