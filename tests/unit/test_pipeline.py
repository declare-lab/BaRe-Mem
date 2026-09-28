"""The pipeline's config, layout and job expansion: what `bash run.sh <experiment> --dry-run` would run.

CPU only, no models and no data: the expansion must be right before anything runs.
"""
import json
from pathlib import Path

import pytest
import yaml

from pipeline.config import REPO, deep_merge, load
from pipeline.layout import Layout
from pipeline.registry import load_registry
from pipeline.run import Plan, stale, streams_command

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"


def test_a_config_inherits_its_base_and_overrides_win():
    cfg = load(EXPERIMENTS / "main.yaml", ["evaluation.max_new_tokens=1024", "eval_conditions=[tilt]"])

    assert cfg["name"] == "main"
    assert cfg["datasets"] == ["capability_supported", "capability_challenging"]                         # the experiment's own value
    assert cfg["record"]["dim"] == 256                                   # inherited from base.yaml
    assert cfg["evaluation"]["max_new_tokens"] == 1024                   # the override
    assert cfg["evaluation"]["engine"] == "vllm"                         # merged, not replaced
    assert cfg["eval_conditions"] == ["tilt"]


def test_deep_merge_merges_mappings_and_replaces_lists():
    assert deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]}, {"a": {"y": 3}, "l": [9]}) == {"a": {"x": 1, "y": 3}, "l": [9]}


def test_every_experiment_config_loads_and_expands(tmp_path):
    for f in sorted(EXPERIMENTS.glob("*.yaml")):
        cfg = load(f, [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", f"paths.logs={tmp_path}/logs"])
        plan = Plan(cfg, str(f), smoke=False, gpus=[0, 1])
        for step in cfg["steps"]:
            jobs = getattr(plan, step)()
            assert jobs, f"{f.name}: step {step} expands to no jobs"


def test_the_registry_is_consistent_and_groups_expand():
    reg = load_registry()

    assert reg.problems() == []
    assert "_misleading" not in reg.datasets                              # a template is not a dataset
    rates = reg.expand(["misleading_rates"])
    assert len(rates) == 10 and rates[0] == "capability_supported_misleading_p000" and rates[-1] == "capability_challenging_misleading_p100"
    assert reg.expand(["capability_supported", "misleading_rates", "capability_supported"]) == ["capability_supported"] + rates   # in order, each once
    assert reg.dataset("capability_challenging_misleading_p025")["answers"] == "capability_challenging_misleading"            # {base} filled from the template
    assert reg.dataset("capability_supported")["peers"] == ["gemma3_4b", "phi4_mini", "qwen25_coder_7b", "llama31", "deepseek_coder_v2_lite", "r1_distill_qwen_7b"]
    assert reg.peer("r1_distill_qwen_7b")["reasoning"] and "reasoning" not in reg.model("r1_distill_qwen_7b")      # a peer-only setting
    with pytest.raises(KeyError, match="did you mean"):
        reg.dataset("capability_supported_misleading_p05")


def test_the_registry_reports_broken_references(tmp_path):
    for sub in ("datasets", "models", "peers"):
        (tmp_path / sub).mkdir()
    (tmp_path / "models/m.yaml").write_text("path: M\n")
    (tmp_path / "peers/p.yaml").write_text("model: m\n")
    (tmp_path / "peers/q.yaml").write_text("model: ghost\n")
    (tmp_path / "datasets/s.yaml").write_text("path: s/test.jsonl\npeers: [p, q, nobody]\n")
    (tmp_path / "datasets/s_bad.yaml").write_text("kind: misleading\nbase: s\nanswers: nowhere\nregime: p150\n")
    problems = "\n".join(load_registry(tmp_path).problems())

    assert "ghost" in problems and "nobody" in problems and "nowhere" in problems and "regime must be pNNN" in problems


def test_registered_datasets_resolve_to_data_and_smoke_builds_are_isolated(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.data={tmp_path}/data", f"paths.outputs={tmp_path}/out"])
    real, smoke = Layout(cfg), Layout(cfg, smoke=True)

    assert real.stream("capability_supported")["path"] == tmp_path / "data/capability_supported/test.jsonl"
    p050 = real.stream("capability_supported_misleading_p050")
    assert p050["path"] == tmp_path / "data/capability_supported_misleading_p050/test.jsonl"
    assert (p050["kind"], p050["base"], p050["answers"], p050["regime"], p050["peers"]) == ("misleading", "capability_supported", "capability_supported_misleading", "p050", 6)
    assert real.stream("capability_supported_misleading")["path"] == tmp_path / "data/capability_supported_misleading"
    assert smoke.stream("capability_supported_misleading_p050")["path"] == tmp_path / "out/smoke/data/capability_supported_misleading_p050/test.jsonl"
    assert smoke.stream("capability_supported")["path"] == real.stream("capability_supported")["path"]          # the released stream is read, never written
    assert str(smoke.eval_dir("q3_4b", "capability_supported", "tilt")).startswith(str(tmp_path / "out/smoke"))
    with pytest.raises(KeyError):
        real.stream("nope")
    with pytest.raises(KeyError, match="group"):
        real.stream("misleading_rates")


def test_the_record_file_names_its_fit_and_any_non_default_setting(tmp_path):
    base = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}"])
    assert Layout(base).record_file("q3_4b", "capability_supported").name == "shuffled0.fit-self.jsonl"
    other = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}", "record.dim=64"])
    assert Layout(other).record_file("q3_4b", "capability_supported").name == "shuffled0.fit-self.qc-d64-lam100.jsonl"


def test_main_expands_to_the_reference_pipeline(tmp_path):
    cfg = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}/out", "paths.models_root=/models"])
    plan = Plan(cfg, "main.yaml", smoke=False, gpus=[0, 1, 2, 3])

    feats = plan.features()
    assert len(feats) == 2 * 4                                            # capability_supported + capability_challenging, four shards each
    assert "--model /models/Qwen3-4B" in feats[0].cmd and "--shards 4 --shard 0" in feats[0].cmd
    rec = [j for j in plan.record() if j.name == "record_q3_4b_capability_supported"][0]
    assert "--fit-features" not in rec.cmd and "--dim 256 --lam 100.0" in rec.cmd          # each stream fits its own addresses
    ev = {j.name: j for j in plan.evaluate()}
    assert len(ev) == 2 * 4
    assert "--mode peers --gamma 3.0 --bias-form logratio" in ev["eval_q3_4b_capability_supported_tilt"].cmd
    assert "--swap" in ev["eval_q3_4b_capability_challenging_swap"].cmd and "--swap" not in ev["eval_q3_4b_capability_challenging_tilt"].cmd
    assert "--mode solo --gamma 0.0" in ev["eval_q3_4b_capability_supported_solo"].cmd


def test_misleading_expands_datasets_into_answers_streams_and_evaluations(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                 "datasets=[capability_supported_misleading_p050, capability_challenging_misleading_p100]"])
    plan = Plan(cfg, "misleading.yaml", smoke=False, gpus=[0])

    assert plan.eval_datasets() == ["capability_supported_misleading_p050", "capability_challenging_misleading_p100"]
    assert plan.answer_datasets() == ["capability_supported_misleading", "capability_challenging_misleading"]
    streams = {j.name: j for j in plan.streams()}
    cmd = streams["stream_capability_supported_misleading_p050"].cmd
    assert '"kind": "fraction"' in cmd and "--regime p050 " in cmd and f"--answers {tmp_path}/data/capability_supported_misleading " in cmd
    assert ("--peer-dirs gemma-3-4b-it,Phi-4-mini-instruct,Qwen2.5-Coder-7B-Instruct,Meta-Llama-3.1-8B-Instruct,"
            "DeepSeek-Coder-V2-Lite-Instruct,DeepSeek-R1-Distill-Qwen-7B ") in cmd
    assert all(j.gpus == 0 for j in streams.values())
    peers = plan.peers()
    assert len(peers) == 2 * 6                                            # two answers datasets, six peers, one shard each
    r1 = [j for j in peers if "DeepSeek-R1" in j.name][0]
    assert "--reasoning" in r1.cmd and "--model /models/DeepSeek-R1-Distill-Qwen-7B " in r1.cmd
    assert r1.done == tmp_path / "data/capability_supported_misleading/DeepSeek-R1-Distill-Qwen-7B/shard0of1.jsonl"
    coder = [j for j in peers if "DeepSeek-Coder" in j.name][0]
    assert coder.env == {"VLLM_USE_V1": "0"} and "--no-prefix-caching" in coder.cmd
    assert all("--no-prefix-caching" not in j.cmd for j in peers if "DeepSeek-Coder" not in j.name)
    assert all("--fit-features" not in j.cmd for j in plan.record())      # fit: self


def test_a_new_regime_is_a_new_dataset_file(tmp_path):
    root = Path(__file__).resolve().parents[2] / "configs"
    for sub in ("models", "peers"):
        (tmp_path / sub).symlink_to(root / sub)
    (tmp_path / "datasets").mkdir()
    for f in ("_misleading.yaml", "capability_supported.yaml", "capability_supported_misleading.yaml"):
        (tmp_path / "datasets" / f).write_text((root / "datasets" / f).read_text())
    (tmp_path / "datasets/capability_supported_saboteurs.yaml").write_text(
        "include: _misleading.yaml\nbase: capability_supported\nregime: {kind: fraction, rate: 1.0, peers: [1, 4]}\ndrop_forced: true\n")
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.registry={tmp_path}", f"paths.data={tmp_path}/data", "datasets=[capability_supported_saboteurs]"])
    L = Layout(cfg)

    assert L.registry.problems() == []
    cmd = streams_command(L, "capability_supported_saboteurs")
    assert '"peers": [1, 4]' in cmd and "--regime capability_supported_saboteurs " in cmd and "--drop-forced" in cmd
    assert Plan(cfg, "misleading.yaml", smoke=False, gpus=[0]).answer_datasets() == ["capability_supported_misleading"]


def test_smoke_uses_one_dataset_one_shard_and_a_small_record(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out"])
    plan = Plan(cfg, "misleading.yaml", smoke=True, gpus=[0, 1, 2, 3])

    assert plan.eval_datasets() == ["capability_supported_misleading_p050"]
    assert all(str(j.done).startswith(str(tmp_path / "out/smoke/data/capability_supported_misleading/")) for j in plan.peers())
    assert all("--shards 1" in j.cmd and "--max-examples 48" in j.cmd for j in plan.features())
    assert all("--dim 32" in j.cmd for j in plan.record())
    assert all("--limit 48" in j.cmd for j in plan.streams())


def test_a_stored_evaluation_with_other_settings_is_reported_not_reused(tmp_path):
    m = tmp_path / "eval_metrics.json"
    m.write_text(json.dumps({"mode": "peers", "gamma": 3.0, "swap_record": False, "max_new_tokens": 768,
                             "record": "r.jsonl", "accuracy": 0.5}))
    same = {"mode": "peers", "gamma": 3.0, "swap_record": False, "max_new_tokens": 768, "record": "r.jsonl"}

    assert stale(m, same) is None
    assert "gamma" in stale(m, dict(same, gamma=5.0))
    assert stale(tmp_path / "missing.json", same) is None
    moved = dict(same, record=str(REPO / "r.jsonl"))
    m.write_text(json.dumps(dict(moved, accuracy=0.5)))
    assert stale(m, same) is None   # an absolute path inside the repository names the same record


def test_the_central_prompts_are_the_ones_every_stored_result_used():
    from feedback_state.memory_generator import build_messages

    rec = {"task_type": "math", "problem": "2 + 2?", "answer": "4"}
    peers = build_messages(rec, ["4", "5"], mode="peers")
    solo = build_messages(rec, ["4", "5"], mode="solo")

    assert peers[0]["content"] == ("You are the central model of a multi-agent system. Several peer models answered the same question. "
                                   "Treat their answers as evidence, verify them yourself, and produce your own final answer.")
    assert solo[0]["content"] == "Answer the question."
    assert peers[1]["content"].startswith("Question:\n2 + 2?\n\nPeer answers:\n\nPeer 1:\n4\n\nPeer 2:\n5\n\nInstruction: ")


def test_paths_written_into_outputs_are_relative_to_the_repository():
    from pipeline.config import REPO, shown

    assert shown(REPO / "data" / "capability_challenging_misleading_p050" / "test.jsonl") == "data/capability_challenging_misleading_p050/test.jsonl"
    assert shown("/models/Qwen3-4B") == "/models/Qwen3-4B"
    assert shown(None) is None
    assert shown("outputs/record/q3_4b/capability_challenging/shuffled0.jsonl") == "outputs/record/q3_4b/capability_challenging/shuffled0.jsonl"


def test_a_failed_job_is_retried_once_before_it_counts_as_failed(tmp_path):
    from pipeline.run import Job, Scheduler

    cfg = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}/out", f"paths.logs={tmp_path}/logs"])
    L = Layout(cfg)
    L.run_dir("t").mkdir(parents=True)
    flag, out = tmp_path / "tried", tmp_path / "result"
    flaky = Job("x", "flaky", f"if [ -e {flag} ]; then touch {out}; else touch {flag}; kill -9 $$; fi", gpus=0, done=out)
    sched = Scheduler([0], "t", L, dry=False)
    sched.run(flaky)
    assert out.exists() and sched.failed == []

    broken = Job("x", "broken", "exit 3", gpus=0, done=tmp_path / "never")
    sched.run(broken)
    assert sched.failed == ["broken"] and "attempt 2" in L.log("t", "broken").read_text()


def test_question_only_is_shared_by_a_stream_and_its_misleading_variants(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data",
                                                 "datasets=[capability_supported_misleading_p025, capability_supported_misleading_p050]"])
    plan = Plan(cfg, "misleading.yaml", smoke=False, gpus=[0])
    ev = {j.name: j for j in plan.evaluate()}

    assert sorted(n for n in ev if n.endswith("_solo")) == ["eval_q3_4b_capability_supported_solo"]          # one job, on the base stream
    assert f"--stream {tmp_path}/data/capability_supported/test.jsonl " in ev["eval_q3_4b_capability_supported_solo"].cmd
    assert "eval_q3_4b_capability_supported_misleading_p050_tilt" in ev and len(ev) == 2 * 2 + 1
    stored = tmp_path / "out/eval/q3_4b/capability_supported/solo"
    stored.mkdir(parents=True)
    (stored / "eval_metrics.json").write_text(json.dumps({"mode": "solo", "gamma": 0.0, "swap_record": False, "max_new_tokens": 768,
                                                          "record": "outputs/record/q3_4b/capability_supported/shuffled0.fit-self.jsonl"}))
    assert ev["eval_q3_4b_capability_supported_solo"].check() is None                                     # the main experiment's result is reused


def test_the_families_experiment_evaluates_every_model_with_its_own_record(tmp_path):
    cfg = load(EXPERIMENTS / "misleading_families.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    plan = Plan(cfg, "misleading_families.yaml", smoke=False, gpus=[0, 1])

    assert cfg["central"] == ["llama31", "ministral", "qwen25", "phi4", "qwen3_14b"]
    assert "peers" not in cfg["steps"]                                                     # nothing is generated
    ev = {j.name: j for j in plan.evaluate()}
    assert len(ev) == 5 * (10 * 2 + 2)                                                     # tilt + peers per dataset, solo per base stream
    job = ev["eval_qwen3_14b_capability_challenging_misleading_p050_tilt"]
    assert "--model /models/Qwen3-14B " in job.cmd and "/record/qwen3_14b/capability_challenging_misleading_p050/shuffled0.fit-self.jsonl" in job.cmd
    assert all("--fit-features" not in j.cmd for j in plan.record())


def test_a_regime_table_has_one_block_per_central_model(tmp_path):
    from pipeline.table import build

    cfg = load(EXPERIMENTS / "misleading_families.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data",
                                                          "central=[llama31, qwen25]", "datasets=[capability_supported_misleading_p050]"])
    L = Layout(cfg)
    for m, acc in (("llama31", 0.61), ("qwen25", 0.58)):
        for c in ("tilt", "peers"):
            d = L.eval_dir(m, "capability_supported_misleading_p050", c)
            d.mkdir(parents=True)
            (d / "eval_metrics.json").write_text(json.dumps({"accuracy": acc}))
    text = build(cfg, smoke=False)

    assert "| llama31 · p050 |  61.0 |  61.0 |" in text and "| qwen25 · p050 |  58.0 |  58.0 |" in text


def test_a_families_smoke_run_reads_the_released_datasets_and_writes_to_smoke(tmp_path):
    cfg = load(EXPERIMENTS / "misleading_families.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "central=[qwen25]"])
    plan = Plan(cfg, "misleading_families.yaml", smoke=True, gpus=[0, 1])

    assert plan.eval_datasets() == ["capability_supported_misleading_p050"]
    feats = plan.features()
    assert all(f"--stream {tmp_path}/data/capability_supported_misleading_p050/test.jsonl " in j.cmd and "--max-examples 48" in j.cmd for j in feats)
    assert all(str(j.done).startswith(str(tmp_path / "out/smoke/")) for j in feats + plan.record() + plan.evaluate())


def test_a_new_peer_is_a_new_file_and_a_stream_lists_its_peers(tmp_path):
    root = Path(__file__).resolve().parents[2] / "configs"
    (tmp_path / "models").symlink_to(root / "models")
    (tmp_path / "peers").mkdir()
    for f in (root / "peers").glob("*.yaml"):
        (tmp_path / "peers" / f.name).write_text(f.read_text())
    (tmp_path / "peers/ministral.yaml").write_text("model: ministral\n")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/new_stream.yaml").write_text(
        "path: new_stream/test.jsonl\npeers: [gemma3_4b, phi4_mini, qwen25_coder_7b, llama31, deepseek_coder_v2_lite, r1_distill_qwen_7b, ministral]\n")
    (tmp_path / "datasets/new_stream_misleading.yaml").write_text("kind: answers\nbase: new_stream\nmode: misleading\n")
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.registry={tmp_path}", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                 "datasets=[new_stream_misleading]"])
    plan = Plan(cfg, "misleading.yaml", smoke=False, gpus=[0])

    assert Layout(cfg).registry.problems() == []
    jobs = plan.peers()
    assert len(jobs) == 7 and "--model /models/Ministral-8B-Instruct-2410 " in jobs[-1].cmd
    assert [j for j in jobs if "DeepSeek-R1" in j.name][0].cmd.count("--reasoning") == 1


def test_combination_adds_the_own_answer_before_the_record_and_chooses_after_peers_and_memory(tmp_path):
    cfg = load(EXPERIMENTS / "combination.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                   "datasets=[capability_supported_misleading_p025, capability_supported_misleading_p050]", "central=[q3_4b]"])
    plan = Plan(cfg, "combination.yaml", smoke=False, gpus=[0, 1])

    own = plan.own()
    solo = [j for j in own if j.wave == 0]
    assert [j.name for j in solo] == ["eval_q3_4b_capability_supported_solo"]                   # once per base stream, needing no record
    assert "--record" not in solo[0].cmd and "--order shuffled0 " in solo[0].cmd
    assert solo[0].done == tmp_path / "out/eval/q3_4b/capability_supported/solo/eval_metrics.json"   # shared with the main experiment
    add = {j.name: j for j in own if j.wave == 2}
    assert add["own_q3_4b_capability_supported_misleading_p050"].cmd == (
        f"python -m pipeline.streams add --base {tmp_path}/data/capability_supported_misleading_p050/test.jsonl "
        f"--eval {tmp_path}/out/eval/q3_4b/capability_supported/solo --out {tmp_path}/data/capability_supported_misleading_p050+q3_4b/test.jsonl")
    feats = plan.features()
    assert len(feats) == 2 * 2 and all("+q3_4b/test.jsonl --model /models/Qwen3-4B " in j.cmd and "--peers 7 " in j.cmd for j in feats)
    rec = {j.name: j for j in plan.record()}["record_q3_4b_capability_supported_misleading_p050"]
    assert "--peers 7 " in rec.cmd and rec.cmd.endswith("--own-slot 6")
    assert rec.done == tmp_path / "out/record/q3_4b/capability_supported_misleading_p050+own/shuffled0.fit-self.jsonl"
    ev = {j.name: j for j in plan.evaluate()}
    assert sorted(ev) == ["eval_q3_4b_capability_supported_misleading_p025_peers", "eval_q3_4b_capability_supported_misleading_p025_tilt",
                          "eval_q3_4b_capability_supported_misleading_p050_peers", "eval_q3_4b_capability_supported_misleading_p050_tilt"]   # No consultation ran in own
    assert ev["eval_q3_4b_capability_supported_misleading_p050_peers"].done == tmp_path / "out/eval/q3_4b/capability_supported_misleading_p050+own/peers/eval_metrics.json"
    tilt = ev["eval_q3_4b_capability_supported_misleading_p050_tilt"]
    assert "/record/q3_4b/capability_supported_misleading_p050+own/" in tilt.cmd
    assert tilt.done == tmp_path / "out/eval/q3_4b/capability_supported_misleading_p050+own/tilt/eval_metrics.json"
    comb = {j.name: j for j in plan.combination()}["combination_q3_4b_capability_supported_misleading_p050"]
    assert (f"--peers-memory {tmp_path}/out/eval/q3_4b/capability_supported_misleading_p050+own/tilt "
            f"--question-alone {tmp_path}/out/eval/q3_4b/capability_supported/solo ") in comb.cmd
    assert "--prior 0.5,0.0 --lam 1.0" in comb.cmd and comb.gpus == 0
    with pytest.raises(SystemExit, match="fit: self"):
        Plan(load(EXPERIMENTS / "combination.yaml", ["record.fit=capability_supported"]), "combination.yaml", smoke=False, gpus=[0])


def test_a_combination_smoke_run_reads_the_released_stream_and_writes_to_smoke(tmp_path):
    cfg = load(EXPERIMENTS / "combination.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data"])
    plan = Plan(cfg, "combination.yaml", smoke=True, gpus=[0])
    own = plan.own()

    assert "--limit 48" in own[0].cmd and f"--base {tmp_path}/data/capability_supported_misleading_p050/test.jsonl " in own[-1].cmd
    jobs = own + plan.features() + plan.record() + plan.evaluate() + plan.combination()
    assert all(str(j.done).startswith(str(tmp_path / "out/smoke/")) for j in jobs)


def test_the_combination_table_names_its_columns_as_the_reports_do(tmp_path):
    from pipeline.table import build

    cfg = load(EXPERIMENTS / "combination.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "datasets=[capability_supported_misleading_p050]",
                                                  "central=[q3_4b]"])
    L = Layout(cfg)
    for c, acc in (("tilt", 0.70), ("combination", 0.74)):
        d = L.eval_dir("q3_4b", "capability_supported_misleading_p050", c)
        d.mkdir(parents=True)
        (d / "eval_metrics.json").write_text(json.dumps({"accuracy": acc, "share_peers_memory": 0.8,
                                                         "reading_line": {"math": {"rho": 0.9, "delta": 0.3, "peers_memory_wins_at_mean_own_prob": "T >= 0.50"}}}))
    d = L.eval_dir("q3_4b", "capability_supported", "solo")
    d.mkdir(parents=True)
    (d / "eval_metrics.json").write_text(json.dumps({"accuracy": 0.69}))
    text = build(cfg, smoke=False)

    assert "| row | Capability-supported: Advisors + memory | Capability-supported: Question + Peers | Capability-supported: No consultation | Capability-supported: BaRe-Mem |" in text
    assert "| p050 |  70.0 |   -   |  69.0 |  74.0 |" in text and "| p050 · Capability-supported | 80% | math 0.90 / 0.30 / T >= 0.50 |" in text


def test_the_baselines_debate_in_rounds_and_vote_on_the_combination_results(tmp_path):
    cfg = load(EXPERIMENTS / "baselines.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                "datasets=[capability_supported_misleading_p050]"])
    plan = Plan(cfg, "baselines.yaml", smoke=False, gpus=[0, 1])

    ev = {j.name: j for j in plan.evaluate()}
    assert sorted(ev) == ["eval_q3_4b_capability_supported_misleading_p050_debate1_0", "eval_q3_4b_capability_supported_misleading_p050_debate1_1",
                          "eval_q3_4b_capability_supported_misleading_p050_debate2_0", "eval_q3_4b_capability_supported_misleading_p050_debate2_1",
                          "merge_q3_4b_capability_supported_misleading_p050_debate1", "merge_q3_4b_capability_supported_misleading_p050_debate2"]
    own = f"{tmp_path}/out/eval/q3_4b/capability_supported_misleading_p050+own"
    first, second = ev["eval_q3_4b_capability_supported_misleading_p050_debate1_0"], ev["eval_q3_4b_capability_supported_misleading_p050_debate2_1"]
    assert "--mode debate --gamma 0.0 " in first.cmd and f"--record {tmp_path}/out/record/q3_4b/capability_supported_misleading_p050+own/" in first.cmd
    assert f"--round 1 --previous {tmp_path}/out/eval/q3_4b/capability_supported/solo --shard 0/2 " in first.cmd       # round 0: the no-consultation answers
    assert f"--round 2 --previous {own}/debate1 --shard 1/2 " in second.cmd
    assert [ev[n].wave for n in ("eval_q3_4b_capability_supported_misleading_p050_debate1_0", "merge_q3_4b_capability_supported_misleading_p050_debate1",
                                 "eval_q3_4b_capability_supported_misleading_p050_debate2_1", "merge_q3_4b_capability_supported_misleading_p050_debate2")] == [0, 1, 2, 3]
    assert ev["merge_q3_4b_capability_supported_misleading_p050_debate2"].done == Path(own) / "debate2" / "eval_metrics.json"

    votes = {j.name: j for j in plan.vote()}
    assert sorted(votes) == ["vote_q3_4b_capability_supported_misleading_p050_debate_vote", "vote_q3_4b_capability_supported_misleading_p050_vote_all",
                             "vote_q3_4b_capability_supported_misleading_p050_vote_peers"]
    assert "--own" not in votes["vote_q3_4b_capability_supported_misleading_p050_vote_peers"].cmd
    assert f"--own {tmp_path}/out/eval/q3_4b/capability_supported/solo --output {own}/vote_all" in votes["vote_q3_4b_capability_supported_misleading_p050_vote_all"].cmd
    assert f"--own {own}/debate2 --output {own}/debate_vote" in votes["vote_q3_4b_capability_supported_misleading_p050_debate_vote"].cmd
    assert all(j.gpus == 0 and f"--stream {tmp_path}/data/capability_supported_misleading_p050/test.jsonl " in j.cmd for j in votes.values())


def test_the_baselines_table_lists_every_method_in_the_configured_order(tmp_path):
    from pipeline.table import build

    cfg = load(EXPERIMENTS / "baselines.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "datasets=[capability_challenging_misleading_p100]"])
    L = Layout(cfg)
    for c, acc in (("vote_peers", 0.31), ("debate2", 0.52), ("combination", 0.69)):
        d = L.eval_dir("q3_4b", "capability_challenging_misleading_p100", c)
        d.mkdir(parents=True)
        (d / "eval_metrics.json").write_text(json.dumps({"accuracy": acc, "share_peers_memory": 0.15, "reading_line": {}}))
    text = build(cfg, smoke=False)

    assert ("| row | Capability-challenging: No consultation | Capability-challenging: Question + Peers | Capability-challenging: Majority vote (advisors) | Capability-challenging: Majority vote (advisors + own) | "
            "Capability-challenging: Debate (1 round) | Capability-challenging: Debate (2 rounds) | Capability-challenging: Debate + vote | Capability-challenging: Advisors + memory | Capability-challenging: BaRe-Mem |") in text
    assert "| p100 |   -   |   -   |  31.0 |   -   |   -   |  52.0 |   -   |   -   |  69.0 |" in text
    assert "Majority vote (advisors + own) (`vote_all`): majority vote over the peers' answers and the central model's answer in `solo`" in text


def test_the_baselines_families_run_the_missing_combination_steps_then_the_baselines_and_record_seven_methods(tmp_path):
    from pipeline.table import build

    cfg = load(EXPERIMENTS / "baselines_families.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                         "datasets=[capability_challenging_misleading_p050]", "central=[qwen3_14b]"])
    plan = Plan(cfg, "baselines_families.yaml", smoke=False, gpus=[0, 1])

    assert cfg["central"] == ["qwen3_14b"] and [j.name for j in plan.own() if j.wave == 0] == ["eval_qwen3_14b_capability_challenging_solo_0", "eval_qwen3_14b_capability_challenging_solo_1"]
    assert [j.name for j in plan.record()] == ["record_qwen3_14b_capability_challenging_misleading_p050"]
    ev = {j.name for j in plan.evaluate() if not j.name.startswith("merge_")}
    assert ev == {f"eval_qwen3_14b_capability_challenging_misleading_p050_{c}_{k}" for c in ("tilt", "peers", "debate1", "debate2") for k in (0, 1)}
    assert sorted(j.name for j in plan.vote()) == ["vote_qwen3_14b_capability_challenging_misleading_p050_vote_all", "vote_qwen3_14b_capability_challenging_misleading_p050_vote_peers"]
    assert [j.name for j in plan.combination()] == ["combination_qwen3_14b_capability_challenging_misleading_p050"]
    assert "--model /models/Qwen3-14B " in next(j for j in plan.evaluate() if j.name.endswith("debate2_0")).cmd

    d = Layout(cfg).eval_dir("qwen3_14b", "capability_challenging_misleading_p050", "combination")
    d.mkdir(parents=True)
    (d / "eval_metrics.json").write_text(json.dumps({"accuracy": 0.7, "share_peers_memory": 0.7, "reading_line": {}}))
    assert ("| row | Capability-challenging: BaRe-Mem | Capability-challenging: Advisors + memory | Capability-challenging: Question + Peers | Capability-challenging: Debate (2 rounds) | "
            "Capability-challenging: Majority vote (advisors + own) | Capability-challenging: Majority vote (advisors) | Capability-challenging: No consultation |") in build(cfg, smoke=False)
