"""Run an experiment: expand its YAML into jobs, schedule them on the GPUs, skip what is already done.

    bash run.sh configs/experiments/main.yaml                     every step of the experiment
    bash run.sh configs/experiments/main.yaml --steps evaluate    one step
    bash run.sh configs/experiments/misleading.yaml --smoke       48 events of one dataset, every step, into outputs/smoke/
    bash run.sh configs/experiments/main.yaml --dry-run           print the jobs and whether each is done
    bash run.sh configs/experiments/main.yaml --set evaluation.max_new_tokens=1024

Steps run in the order questions -> peers -> streams -> own -> direct -> verify -> features -> record -> evaluate -> vote -> combination
-> team -> table; the jobs of a step run in
parallel on the configured GPUs. A job is skipped when its output exists; a sharded
output counts only once all its shards finished (a complete.json is written then); an evaluation is skipped only if the
stored settings match the requested ones, and a mismatch stops the job instead of silently reusing the old result.
The resolved config and every command go to outputs/runs/<experiment>/, logs to logs/<experiment>/.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pipeline.config import load, shown
from pipeline.layout import Layout

ORDER = ["questions", "peers", "streams", "own", "direct", "verify", "features", "record", "evaluate", "vote", "combination", "team", "table"]


def stamp() -> str:
    return time.strftime("%H:%M:%S")


@dataclass
class Job:
    step: str
    name: str
    cmd: str
    gpus: int = 1
    done: Path | None = None              # the output whose existence means the job is finished
    group: Path | None = None             # a sharded output directory: complete.json there means every shard is done
    group_size: int = 0
    check: object = None                  # callable() -> str | None: a reason the stored result cannot be reused
    env: dict = field(default_factory=dict)
    wave: int = 0                         # within a step, waves run in order; the jobs of a wave run in parallel


# --- expansion ------------------------------------------------------------------------------------------------------------
class Plan:
    def __init__(self, cfg: dict, cfg_path: str, smoke: bool, gpus: list[int]):
        self.cfg, self.cfg_path, self.smoke, self.gpus = cfg, cfg_path, smoke, gpus
        self.L = Layout(cfg, smoke)
        self.exp = cfg["name"]
        sm = cfg.get("smoke", {})
        self.events = int(sm.get("events", 48)) if smoke else None
        asked = self.L.registry.expand(cfg.get("datasets", []))
        self.datasets = self.L.registry.expand(sm["datasets"]) if smoke and "datasets" in sm else asked[:1] if smoke else asked
        self.shards = 1 if smoke else int(cfg.get("generation", {}).get("shards", len(gpus)))   # peer answers: shards per peer (default: one per GPU)
        if self.L.own and self.fit() != "self":
            raise SystemExit("own_answer needs record.fit: self (the fit stream has no own answers)")

    def eval_datasets(self) -> list[str]:
        """The datasets the central model is run on (answers datasets only feed the streams)."""
        return [d for d in self.datasets if self.L.stream(d)["kind"] != "answers"]

    def answer_datasets(self) -> list[str]:
        """The answers datasets to generate: those named, and those the named misleading streams are built from."""
        out = []
        for d in self.datasets:
            spec = self.L.stream(d)
            a = d if spec["kind"] == "answers" else spec.get("answers") if spec["kind"] == "misleading" or spec.get("built") == "answers" else None
            if a and a not in out:
                out.append(a)
        return out

    def fit(self) -> str:
        return str(self.cfg.get("record", {}).get("fit", "self"))

    def central(self, m: str) -> dict:
        """A central model's registered spec: the model answers, and its own features address its record."""
        return self.L.model(m)

    def model_env(self, spec: dict) -> str:
        return f'eval "$(conda shell.bash hook)" && conda activate {spec["conda_env"]} && ' if spec.get("conda_env") else ""

    # the steps
    def questions(self) -> list[Job]:
        """Question streams built from a benchmark's own data (CPU), by the stream's `built:` (musique: a team's sub-tasks, one
        view per tool, pipeline.team build)."""
        built: dict[Path, list[dict]] = {}
        for d in self.datasets + ([self.cfg["team"]["direct"]] if self.cfg.get("team", {}).get("direct") else []):
            for name in (d, self.L.stream(d).get("base")):
                spec = self.L.stream(name) if name else None
                if spec is not None and spec.get("built") == "musique" and spec not in built.setdefault(spec["path"].parent, []):
                    built[spec["path"].parent].append(spec)
        jobs = []
        for folder, specs in built.items():      # one build writes every stream of the folder (the sub-tasks, their views, the whole tasks)
            src = self.L._abs(Path(self.cfg.get("paths", {}).get("data", "data")) / specs[0]["source"])
            jobs.append(Job("questions", f"questions_{specs[0]['name']}", f"python -m pipeline.team build --source {src} --out {folder}"
                            + (f" --limit {max(2, self.events // 3)}" if self.events else ""), gpus=0, done=sorted(sp["path"] for sp in specs)[0]))
        return jobs

    @staticmethod
    def view(stream: dict, peer: dict) -> tuple[Path, bool]:
        """What a peer answers from: the stream, or the view its `tool` names (the stream's `views`: a file beside it, or
        {path, context: false} for a worker that gets no passages). -> (path, passages shown)"""
        tool = peer.get("tool")
        if not tool:
            return stream["path"], True
        v = (stream.get("views") or {})[tool]
        v = {"path": v} if isinstance(v, str) else dict(v)
        return stream["path"].parent / v.get("path", stream["path"].name), bool(v.get("context", True))

    def peers(self) -> list[Job]:
        gen = self.cfg.get("generation", {})
        mis = gen.get("misleading", {})
        jobs = []
        for a in self.answer_datasets():
            answers = self.L.stream(a)
            stream, mode = self.L.stream(answers["base"]), answers["mode"]
            for p in self.L.peer_models(stream["peer_names"]):
                out = answers["path"] / p["name"]
                path, context = self.view(stream, p)
                for k in range(self.shards):
                    cmd = (f"python -m pipeline.peers --mode {mode} --model {p['path']} --stream {path} --output {out} "
                           f"--shards {self.shards} --shard {k} --max-model-len {gen.get('max_model_len', 8192)} "
                           f"--gpu-memory-utilization {gen.get('gpu_memory_utilization', 0.85)} --grade-workers {gen.get('grade_workers', 8)}")
                    if mode == "misleading":
                        cmd += f" --max-attempts {mis.get('max_attempts', 3)} --temperatures {mis.get('temperatures', '0.2,0.7,1.0')}"
                        cmd += "" if mis.get("force", True) else " --no-force"
                    if int(answers.get("turns", 1)) > 1:   # agentic peers: they revise on the task's visible check
                        cmd += f" --turns {int(answers['turns'])}"
                    cmd += " --reasoning" if p.get("reasoning") else ""
                    cmd += "" if context else " --no-context"
                    cmd += " --no-prefix-caching" if p.get("prefix_caching") is False else ""
                    cmd += " --trust-remote-code" if p.get("trust_remote_code") else ""
                    cmd += f" --max-examples {self.events}" if self.events else ""
                    jobs.append(Job("peers", f"peers_{a}_{p['name']}_{k}", cmd, done=out / f"shard{k}of{self.shards}.jsonl",
                                    group=out, group_size=self.shards, env={str(x): str(y) for x, y in (p.get("env_vars") or {}).items()}))
        return jobs

    def streams(self) -> list[Job]:
        jobs = []
        for d in self.eval_datasets():
            spec = self.L.stream(d)
            if spec.get("built") == "answers":     # the base's questions with the peers' generated answers, in peer order
                names = " ".join(f"--peer {pm['name']}" for pm in self.L.peer_models(spec["peer_names"]))
                jobs.append(Job("streams", f"stream_{d}", f"python -m pipeline.streams add --base {self.L.stream(spec['base'])['path']} "
                                f"--answers {self.L.stream(spec['answers'])['path']} {names} --out {spec['path']}", gpus=0, done=spec["path"]))
                continue
            if spec["kind"] != "misleading":
                continue
            jobs.append(Job("streams", f"stream_{d}", streams_command(self.L, d, self.events), gpus=0, done=spec["path"]))
        return jobs

    def direct(self) -> list[Job]:
        """The baseline without a team: the lead answers every WHOLE task directly, alone (`team.direct`: the registered stream of
        whole tasks), graded on the final answer. pipeline.team replay reads its accuracy next to the team's."""
        d = self.cfg.get("team", {}).get("direct")
        if not d:
            raise SystemExit("step direct needs team.direct: the registered stream of whole tasks")
        jobs = []
        for m in self.cfg.get("central", []):
            jobs += self.eval_jobs(m, self.central(m), m, [d], conditions=["solo"], step="direct")
        return jobs

    def verify(self) -> list[Job]:
        """The lead's check of every source's report on every sub-task (pipeline.review): does the report meet what the lead expected
        of it? Run by `team.checker` (a registered model; default: the central model), sharded like the peers' answers."""
        tc = self.cfg.get("team", {})
        ev, jobs = self.L.model(tc.get("checker") or self.cfg["central"][0]), []
        for m in self.cfg.get("central", []):
            for d in self.eval_datasets():
                (path, sources), out = self.source(m, d), self.L.review_dir(m, d) / "lead_check" / "verdicts"
                for k in range(self.shards):
                    cmd = (f"python -m pipeline.review --stream {path} --sources {sources} --model {ev['path']} --output {out} --shards {self.shards} --shard {k}"
                           + (f" --max-examples {self.events}" if self.events else ""))
                    jobs.append(Job("verify", f"verify_{m}_{d}_{k}", cmd, done=out / f"shard{k}of{self.shards}.jsonl", group=out, group_size=self.shards,
                                    env={str(x): str(y) for x, y in (ev.get("env_vars") or {}).items()}))
        return jobs

    def team(self) -> list[Job]:
        """The lead's choice of source replayed along each stream (pipeline.team replay, feedback_state.agent_team): by the record
        against success counts and a random choice, with the first report committed as it is, with the lead's check when the
        experiment has the verify step, and with the dataset's check (`team.dataset_check`: the gold answer verifies before the
        commit); a report that fails sends the sub-task to the next source, `team.calls` at most. With own_answer the lead's own answer is the last source: the autonomous option."""
        tc, rec = self.cfg.get("team", {}), self.cfg.get("record", {})
        dim = int(self.cfg.get("smoke", {}).get("dim", 8)) if self.smoke else int(tc.get("dim", 64))
        jobs = []
        for m in self.cfg.get("central", []):
            for d in self.eval_datasets():
                (path, sources), out = self.source(m, d), self.L.eval_dir(m, d, "team")
                cmd = (f"python -m pipeline.team replay --stream {path} --features {self.L.features_dir(m, d)} --sources {sources} "
                       + (f"--own-slot {sources - 1} " if self.L.own else "") + ("--by-task " if self.L.stream(d).get("by_task") else "")
                       + (f"--verdicts {self.L.review_dir(m, d) / 'lead_check' / 'verdicts'} " if "verify" in self.cfg.get("steps", []) else "")
                       + ("--dataset-check " if tc.get("dataset_check") else "") + f"--calls {int(tc.get('calls', 2))} "
                       + (f"--direct {self.L.eval_dir(m, tc['direct'], 'solo')} " if tc.get("direct") and "direct" in self.cfg.get("steps", []) else "")
                       + f"--dim {dim} --lam {rec.get('lam', 100.0)} --orders {' '.join(str(o) for o in tc.get('orders', [0, 1, 2]))}"
                       + (f" --limit {self.events}" if self.events else "") + f" --title {shlex.quote(f'{m} leading {d}')} --out {out}")
                jobs.append(Job("team", f"team_{m}_{d}", cmd, done=out / "team.json"))
        return jobs

    def own(self) -> list[Job]:
        """Every model answers first: the central model's no-consultation answer (waves 0-1, shared with the base stream) joins
        each dataset's stream as its last answer (wave 2)."""
        if not self.L.own:
            raise SystemExit("step own needs own_answer: true")
        jobs = []
        for m in self.cfg.get("central", []):
            spec = self.central(m)
            jobs += self.eval_jobs(m, spec, m, self.eval_datasets(), conditions=["solo"], step="own")
            for d in self.eval_datasets():
                solo = self.L.eval_dir(m, self.L.eval_dataset(d, {"mode": "solo"}), "solo")
                out = self.L.own_stream(m, d)
                jobs.append(Job("own", f"own_{m}_{d}", f"python -m pipeline.streams add --base {self.L.stream(d)['path']} --eval {solo} --out {out}",
                                gpus=0, done=out, wave=2))
        return jobs

    def source(self, model: str, dataset: str) -> tuple[Path, int]:
        """The stream the judge and the record read, and its number of answers: with own_answer, the model's own answer is one more."""
        spec = self.L.stream(dataset)
        return (self.L.own_stream(model, dataset), spec["peers"] + 1) if self.L.own else (spec["path"], spec["peers"])

    def feature_streams(self) -> list[str]:
        return ([self.fit()] if self.fit() != "self" else []) + self.eval_datasets()

    def features(self) -> list[Job]:
        jobs = []
        for m in self.cfg.get("central", []):
            spec = self.central(m)                        # the judge: the central model itself
            for s in self.feature_streams():
                jobs += self.feature_jobs(m, spec, s)
        return jobs

    def feature_jobs(self, m: str, spec: dict, s: str, step: str = "features") -> list[Job]:
        """The judge's features of stream s, one job per shard (skipped once the shards are complete)."""
        fe = self.cfg.get("features", {})
        (path, answers), out = self.source(m, s), self.L.features_dir(m, s)
        jobs = []
        for k in range(self.shards):
            cmd = (f"{self.model_env(spec)}python -m pipeline.features --stream {path} --model {spec['path']} "
                   f"--output {out}/shard{k}of{self.shards}.pt --shards {self.shards} --shard {k} --peers {answers} "
                   f"--max-length {fe.get('max_length', 8192)} --dtype {fe.get('dtype', 'bfloat16')}"
                   + (f" --max-examples {self.events}" if self.events else ""))
            jobs.append(Job(step, f"features_{m}_{s}_{k}", cmd, done=out / f"shard{k}of{self.shards}.pt", group=out, group_size=self.shards))
        return jobs

    def record(self) -> list[Job]:
        rec = self.cfg.get("record", {})
        dim = int(self.cfg.get("smoke", {}).get("dim", 32)) if self.smoke else int(rec.get("dim", 256))
        jobs = []
        for m in self.cfg.get("central", []):
            spec = self.central(m)
            for s in self.eval_datasets():
                (path, answers), out = self.source(m, s), self.L.record_file(m, s)
                fit = ""
                if self.fit() != "self":
                    fit = f" --fit-stream {self.L.stream(self.fit())['path']} --fit-features {self.L.features_dir(m, self.fit())}"
                cmd = (f"{self.model_env(spec)}python -m pipeline.record --stream {path} --features {self.L.features_dir(m, s)}{fit} "
                       f"--peers {answers} --order {rec.get('order', 'shuffled0')} --design {rec.get('design', 'qc')} "
                       f"--dim {dim} --lam {rec.get('lam', 100.0)} --out {out}" + (f" --own-slot {answers - 1}" if self.L.own else "")
                       + (f" --save-addresses {self.L.addresses_file(out)}" if rec.get("save_addresses") else ""))
                jobs.append(Job("record", f"record_{m}_{s}", cmd, done=out))
        return jobs

    def eval_jobs(self, model_key: str, model_spec: dict, record_model: str, streams: list[str],
                  conditions: list[str] | None = None, step: str = "evaluate") -> list[Job]:
        ev, conds = self.cfg.get("evaluation", {}), self.cfg.get("conditions", {})
        jobs, seen = [], set()
        for d in streams:
            for c in conditions if conditions is not None else self.cfg.get("eval_conditions", list(conds)):
                if c not in conds:
                    raise SystemExit(f"condition {c!r} is not defined under conditions:")
                cd = conds[c]
                s = self.L.eval_dataset(d, cd)            # No consultation: the base stream's result, shared by its variants
                out = self.L.eval_dir(model_key, s, c)
                if out in seen:
                    continue
                seen.add(out)
                # the record fixes which events are evaluated, in which order; any record of the same events will do for solo
                stream, record = self.L.stream(s), self.L.record_file(record_model, d)
                engine = model_spec.get("engine", ev.get("engine", "vllm"))
                mode = cd.get("mode", "peers")
                debate = ""
                wave = 0
                if mode == "debate":   # round r reads round r-1's answers, so it runs two waves (shards, merge) later
                    rnd, prev = int(cd.get("round", 1)), cd.get("previous", "solo")
                    if prev not in conds:
                        raise SystemExit(f"condition {c!r}: previous {prev!r} is not defined under conditions:")
                    debate = f" --round {rnd} --previous {self.L.eval_dir(model_key, self.L.eval_dataset(d, conds[prev]), prev)}"
                    wave = 2 * (rnd - 1)
                # with own_answer the record needs the no-consultation answer first, and a no-consultation run needs no record
                source = (f"--order {self.cfg.get('record', {}).get('order', 'shuffled0')}" + (f" --limit {self.events}" if self.events else "")
                          if self.L.own and cd.get("mode") == "solo" else f"--record {record}")
                args = (f"--model {model_spec['path']} {source} --stream {stream['path']} --condition {c} --mode {mode} "
                        f"--gamma {float(cd.get('gamma', 0.0))}{' --swap' if cd.get('swap') else ''} --bias-form {ev.get('bias_form', 'logratio')} "
                        f"--engine {engine} --max-new-tokens {ev.get('max_new_tokens', 768)} "
                        f"--gpu-memory-utilization {ev.get('gpu_memory_utilization', 0.85)}" + debate)
                want = {"mode": cd.get("mode", "peers"), "gamma": float(cd.get("gamma", 0.0)), "swap_record": bool(cd.get("swap", False)),
                        "max_new_tokens": int(ev.get("max_new_tokens", 768))}
                if cd.get("mode", "peers") != "solo":           # the answers never see the record without peers
                    want["record"] = str(record)
                if mode == "debate":
                    want["round"] = int(cd.get("round", 1))
                check = lambda out=out, want=want: stale(out / "eval_metrics.json", want)
                pre = self.model_env(model_spec)
                vllm_shards = 1 if self.smoke else int(ev.get("vllm_shards", 1))   # a long stream: one vLLM engine per GPU, merged after
                merged = out / "eval_metrics.json"           # a merged result, however it was sharded, makes every shard done
                if engine == "vllm" and vllm_shards > 1:
                    for k in range(vllm_shards):
                        jobs.append(Job(step, f"eval_{model_key}_{s}_{c}_{k}", f"{pre}python -m pipeline.evaluate {args} --shard {k}/{vllm_shards} --output {out}/shard{k}",
                                        done=merged if merged.exists() else out / f"shard{k}" / "eval_metrics.json", check=check, wave=wave))
                    jobs.append(Job(step, f"merge_{model_key}_{s}_{c}", f"python -m pipeline.evaluate --merge --output {out}",
                                    gpus=0, done=out / "eval_metrics.json", check=check, wave=wave + 1))
                elif engine == "vllm":
                    jobs.append(Job(step, f"eval_{model_key}_{s}_{c}", f"{pre}python -m pipeline.evaluate {args} --output {out}",
                                    done=out / "eval_metrics.json", check=check, wave=wave))
                else:   # HF engine: one shard per GPU, merged when all are in
                    n = 1 if self.smoke else int(model_spec.get("shards", len(self.gpus)))
                    for k in range(n):
                        jobs.append(Job(step, f"eval_{model_key}_{s}_{c}_{k}",
                                        f"{pre}python -m pipeline.evaluate {args} --batch-size {int(model_spec.get('batch_size', 8))} --shard {k}/{n} --output {out}/shard{k}",
                                        done=merged if merged.exists() else out / f"shard{k}" / "eval_metrics.json", check=check))
                    jobs.append(Job(step, f"merge_{model_key}_{s}_{c}", f"python -m pipeline.evaluate --merge --output {out}",
                                    gpus=0, done=out / "eval_metrics.json", check=check, wave=1))
        return jobs

    def evaluate(self) -> list[Job]:
        jobs = []
        for m in self.cfg.get("central", []):
            spec = self.central(m)
            conditions = [c for c in self.cfg.get("eval_conditions", list(self.cfg.get("conditions", {})))
                          if not (self.L.own and self.cfg.get("conditions", {}).get(c, {}).get("mode") == "solo")]   # own ran those
            jobs += self.eval_jobs(m, spec, m, self.eval_datasets(), conditions=conditions)
        return jobs

    def vote(self) -> list[Job]:
        """Majority votes over the peers' answers, with or without an answer of the central model (pipeline.vote); CPU."""
        conds = self.cfg.get("conditions", {})
        jobs = []
        for m in self.cfg.get("central", []):
            for d in self.eval_datasets():
                for c in self.cfg.get("vote_conditions", []):
                    cd = conds.get(c)
                    if cd is None or cd.get("mode") != "vote":
                        raise SystemExit(f"vote condition {c!r} must be defined under conditions: with mode: vote")
                    own = cd.get("own")
                    if own is not None and own not in conds:
                        raise SystemExit(f"vote condition {c!r}: own {own!r} is not defined under conditions:")
                    out = self.L.eval_dir(m, d, c)
                    cmd = (f"python -m pipeline.vote --record {self.L.record_file(m, d)} --stream {self.L.stream(d)['path']} --condition {c}"
                           + (f" --own {self.L.eval_dir(m, self.L.eval_dataset(d, conds[own]), own)}" if own else "") + f" --output {out}")
                    jobs.append(Job("vote", f"vote_{m}_{d}_{c}", cmd, gpus=0, done=out / "eval_metrics.json"))
        return jobs

    def combination(self) -> list[Job]:
        """BaRe-Mem: per event, Advisors + memory or No consultation, chosen by the central model's reading line (pipeline.combination)."""
        if not self.L.own:
            raise SystemExit("step combination needs own_answer: true (the record must estimate the own answer)")
        cb = self.cfg.get("combination", {})
        prior = ",".join(str(float(x)) for x in cb.get("prior", [0.5, 0.0]))
        jobs = []
        for m in self.cfg.get("central", []):
            for d in self.eval_datasets():
                out = self.L.eval_dir(m, d, "combination")
                cmd = (f"python -m pipeline.combination --record {self.L.record_file(m, d)} "
                       f"--peers-memory {self.L.eval_dir(m, d, cb.get('peers_memory', 'tilt'))} "
                       f"--question-alone {self.L.eval_dir(m, self.L.eval_dataset(d, {'mode': 'solo'}), 'solo')} --prior {prior} "
                       f"--lam {float(cb.get('lam', 1.0))} --output {out}")
                jobs.append(Job("combination", f"combination_{m}_{d}", cmd, gpus=0, done=out / "eval_metrics.json"))
        return jobs

    def table(self) -> list[Job]:
        resolved = self.L.run_dir(self.exp) / "resolved.yaml"
        return [Job("table", "table", f"python -m pipeline.table --resolved {resolved}" + (" --smoke" if self.smoke else ""), gpus=0)]


def streams_command(L: Layout, name: str, limit: int | None = None) -> str:
    """The pipeline.streams command that builds a registered misleading stream."""
    spec = L.stream(name)
    regime = spec["regime"]
    if isinstance(regime, str):
        from feedback_state.adversarial import adhoc_spec

        label, regime = regime, adhoc_spec(regime)
    else:
        label = name
    answers = L.stream(spec["answers"])
    dirs = ",".join(p["name"] for p in L.peer_models(spec["peer_names"]))
    return (f"python -m pipeline.streams replace --base {L.stream(spec['base'])['path']} --answers {answers['path']} --peer-dirs {dirs} "
            f"--regime {label} --regime-spec {shlex.quote(json.dumps(regime))} --order {spec.get('order', 'shuffled0')} --out {spec['path']}"
            + (" --drop-forced" if spec.get("drop_forced") else "") + (f" --limit {limit}" if limit else ""))


def stale(metrics: Path, want: dict) -> str | None:
    if not metrics.exists():
        return None
    have = json.load(open(metrics))
    diff = {k: (have.get(k), v) for k, v in want.items() if k in have and have.get(k) != v}
    for k in ("record", "stream"):   # paths: equal if they name the same place in the project, from any snapshot
        if k in diff and None not in diff[k] and shown(diff[k][0]) == shown(diff[k][1]):
            diff.pop(k)
    return None if not diff else "stored results used other settings: " + ", ".join(f"{k} {a!r} (asked {b!r})" for k, (a, b) in diff.items())


# --- scheduling -----------------------------------------------------------------------------------------------------------
class Scheduler:
    def __init__(self, gpus: list[int], exp: str, layout: Layout, dry: bool, retries: int = 1):
        self.retries = retries             # a failed job runs again this many times (e.g. a killed process) before it counts as failed
        self.free = list(gpus)
        self.total = len(gpus)
        self.cv = threading.Condition()
        self.exp, self.L, self.dry = exp, layout, dry
        self.failed: list[str] = []
        self.commands = layout.run_dir(exp) / "commands.log"

    def finished(self, job: Job) -> bool:
        if job.group is not None and (job.group / "complete.json").exists():
            return True
        return job.done is not None and job.done.exists() and job.group is None

    def run_step(self, jobs: list[Job]) -> None:
        todo = []
        for j in jobs:
            reason = j.check() if j.check else None
            if reason:
                print(f"[{stamp()}] STALE {j.name}: {reason}; rename the condition or remove {j.done.parent}", flush=True)
                self.failed.append(j.name)
            elif self.finished(j) or (j.done is not None and j.done.exists() and j.group is not None):
                print(f"[{stamp()}] skip  {j.name}", flush=True)
            else:
                todo.append(j)
        if self.dry:
            for j in todo:
                print(f"[dry-run] {j.name} ({j.gpus} GPU{'s' if j.gpus != 1 else ''}):\n    {j.cmd}", flush=True)
            return
        with ThreadPoolExecutor(max_workers=max(1, len(todo))) as ex:
            list(ex.map(self.run, todo))
        groups: dict[Path, list[Job]] = {}
        for j in jobs:
            if j.group is not None:
                groups.setdefault(j.group, []).append(j)
        for g, members in groups.items():
            if all(m.done.exists() for m in members) and not (g / "complete.json").exists() and not any(m.name in self.failed for m in members):
                (g / "complete.json").write_text(json.dumps({"shards": len(members), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))

    def run(self, job: Job) -> None:
        need = min(job.gpus, self.total)
        with self.cv:
            while len(self.free) < need:
                self.cv.wait()
            got = [self.free.pop(0) for _ in range(need)]
        try:
            log = self.L.log(self.exp, job.name)
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, **job.env)
            if need:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, got))
            print(f"[{stamp()}] start {job.name}" + (f" on GPU {','.join(map(str, got))}" if need else ""), flush=True)
            with open(self.commands, "a") as c:
                c.write(f"# {stamp()} {job.name}\n{' '.join(f'{k}={v}' for k, v in job.env.items())} {job.cmd}\n")
            for attempt in range(1 + self.retries):
                with open(log, "w" if attempt == 0 else "a") as f:
                    if attempt:
                        f.write(f"\n# ---- attempt {attempt + 1} ----\n")
                        f.flush()
                    rc = subprocess.call(job.cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT, env=env)
                ok = rc == 0 and (job.done is None or job.done.exists())
                if ok or attempt == self.retries:
                    break
                print(f"[{stamp()}] retry {job.name} (exit {rc}{', killed by a signal' if rc > 128 or rc < 0 else ''})", flush=True)
            print(f"[{stamp()}] {'done ' if ok else 'FAILED'} {job.name}" + ("" if ok else f" (exit {rc}; see {log})"), flush=True)
            if not ok:
                self.failed.append(job.name)
        finally:
            with self.cv:
                self.free.extend(got)
                self.cv.notify_all()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--steps", nargs="*", help=f"a subset of {ORDER} (default: the config's steps)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: the config's gpus)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.key=value (YAML value)")
    args = ap.parse_args(argv)
    cfg = load(args.config, args.overrides)
    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else [int(g) for g in cfg.get("gpus", [0])]
    steps = args.steps or cfg.get("steps", [])
    unknown = [s for s in steps if s not in ORDER]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; steps are {ORDER}")
    plan = Plan(cfg, args.config, args.smoke, gpus)
    plan.dry = args.dry_run
    run_dir = plan.L.run_dir(plan.exp)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump({**cfg, "_smoke": args.smoke, "_gpus": gpus}, sort_keys=False))
    sched = Scheduler(gpus, plan.exp, plan.L, args.dry_run, retries=int(cfg.get("retries", 1)))
    print(f"[{stamp()}] {plan.exp}{' (smoke)' if args.smoke else ''}: steps {[s for s in ORDER if s in steps]}, GPUs {gpus}, "
          f"datasets {plan.datasets}", flush=True)
    for step in ORDER:
        if step not in steps:
            continue
        jobs = getattr(plan, step)()
        for wave in sorted({j.wave for j in jobs}):   # e.g. shard merges after their shards
            sched.run_step([j for j in jobs if j.wave == wave])
    if sched.failed:
        print(f"[{stamp()}] FAILED: {sched.failed}", flush=True)
        sys.exit(1)
    print(f"[{stamp()}] RUN_COMPLETE {plan.exp}{' (smoke)' if args.smoke else ''}", flush=True)


if __name__ == "__main__":
    main()
