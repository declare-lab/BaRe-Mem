"""The registries: every dataset, model and peer is one YAML file under configs/, found by scanning the folders.

    configs/datasets/<name>.yaml   a dataset, by kind:
                                     stream       a stream of events with its peers' answers (path; peers: the peers in
                                                  peer_0, peer_1, ... order)
                                     answers      the stream's peers' generated answers (base, mode)
                                     misleading   a stream whose answers are replaced under a regime (base, answers, regime)
                                   or a group: `group: [names]`, which experiments can name instead of its members
    configs/models/<name>.yaml     a model: its directory under paths.models_root and how to run it
    configs/peers/<name>.yaml      a peer: the registered model that answers, and peer-only settings (reasoning; tool: the
                                   view of a stream with `views` it answers from, e.g. a searcher's retrieval)
    configs/tasks/<name>.yaml      a task type: its grading rule (grader: a rule of feedback_state/tasks.py), the central
                                   model's instruction, the peers' answer budget, whether a passage is shown, the vote's
                                   agreement rule

The file name is the registered name. Files starting with "_" are templates, not registered: `include: _misleading.yaml`
merges one under a file, and string values may use {name} and {base}. Experiments refer to registered names only, so
adding a dataset, a model, a peer or a task is adding a file.

    python -m pipeline.registry            list everything and check every reference (exit 1 on a problem)
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from pipeline.config import REPO, deep_merge

KINDS = ("stream", "answers", "misleading")
MODES = ("honest", "misleading")
_REGIME_SHORT = re.compile(r"^(p(\d{3})|k\d+)$")


def _read(path: Path, seen: tuple = ()) -> dict:
    if path in seen:
        raise ValueError(f"include cycle: {' -> '.join(map(str, seen + (path,)))}")
    spec = yaml.safe_load(path.read_text()) or {}
    inc = spec.pop("include", None)
    if inc:
        spec = deep_merge(_read(path.parent / inc, seen + (path,)), spec)
    return spec


def _fill(value, fields: dict):
    if isinstance(value, str):
        for k, v in fields.items():
            value = value.replace("{" + k + "}", str(v))
        return value
    if isinstance(value, dict):
        return {k: _fill(v, fields) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, fields) for v in value]
    return value


def _scan(folder: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(folder.glob("*.yaml")):
        if f.name.startswith("_"):
            continue
        spec = _read(f)
        spec = _fill(spec, {"name": f.stem, "base": spec.get("base", "")})
        spec["name"], spec["file"] = f.stem, str(f.relative_to(REPO) if f.is_relative_to(REPO) else f)
        out[f.stem] = spec
    return out


def _unknown(what: str, name: str, known) -> KeyError:
    close = difflib.get_close_matches(str(name), list(known), n=3)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    return KeyError(f"{what} {name!r} is not registered (no configs/{what}s/{name}.yaml){hint}")


class Registry:
    def __init__(self, root: Path):
        self.root = root
        entries = _scan(root / "datasets")
        self.groups = {k: v for k, v in entries.items() if "group" in v}
        self.datasets = {k: dict(v, kind=v.get("kind", "stream")) for k, v in entries.items() if "group" not in v}
        self.models = _scan(root / "models")
        self.peers = _scan(root / "peers")
        self.tasks = _scan(root / "tasks")

    # --- lookups ------------------------------------------------------------------------------------------------------
    def dataset(self, name: str) -> dict:
        if name in self.groups:
            raise KeyError(f"{name!r} is a group of datasets ({', '.join(self.expand([name]))}), not one dataset")
        if name not in self.datasets:
            raise _unknown("dataset", name, list(self.datasets) + list(self.groups))
        return self.datasets[name]

    def model(self, name: str) -> dict:
        if name not in self.models:
            raise _unknown("model", name, self.models)
        return self.models[name]

    def peer(self, name: str) -> dict:
        if name not in self.peers:
            raise _unknown("peer", name, self.peers)
        return self.peers[name]

    def task(self, name: str) -> dict:
        if name not in self.tasks:
            raise _unknown("task", name, self.tasks)
        return self.tasks[name]

    def expand(self, names) -> list[str]:
        """Dataset names with every group replaced by its members, in order, each once."""
        out: list[str] = []

        def add(n, trail=()):
            if n in trail:
                raise ValueError(f"group cycle: {' -> '.join(trail + (n,))}")
            if n in self.groups:
                for m in self.groups[n]["group"]:
                    add(m, trail + (n,))
            elif n not in out:
                self.dataset(n)
                out.append(n)

        for n in names or []:
            add(n)
        return out

    def stream_of(self, name: str) -> dict:
        """The released stream a dataset is built on (itself for a stream)."""
        d = self.dataset(name)
        return d if d["kind"] == "stream" else self.stream_of(d["base"])

    # --- checks -------------------------------------------------------------------------------------------------------
    def problems(self) -> list[str]:
        bad = []

        def need(cond, msg):
            if not cond:
                bad.append(msg)

        for name, d in self.datasets.items():
            where = d["file"]
            kind = d["kind"]
            need(kind in KINDS, f"{where}: kind {kind!r} is not one of {KINDS}")
            need("task" not in d or d["task"] in self.tasks, f"{where}: task {d.get('task')!r} is not registered (configs/tasks/{d.get('task')}.yaml)")
            if kind == "stream":
                need("path" in d, f"{where}: a stream needs path (under paths.data)")
                listed = d.get("peers")
                need(isinstance(listed, list) and (listed or d.get("solo_only")), f"{where}: peers must list the stream's peers in peer_0, peer_1, ... order "
                     f"(an empty list only with solo_only: true, a stream a model answers alone)")
                for p in listed if isinstance(listed, list) else []:
                    need(p in self.peers, f"{where}: peer {p!r} is not registered (configs/peers/{p}.yaml)")
                    tool = self.peers.get(p, {}).get("tool")
                    views = d.get("views") or self.datasets.get(d.get("base"), {}).get("views") or {}
                    need(tool is None or tool in views, f"{where}: peer {p!r} answers from the view {tool!r}, which the stream (or its base) does not list under views")
                continue
            base = self.datasets.get(d.get("base"))
            need(base is not None and base["kind"] == "stream", f"{where}: base {d.get('base')!r} is not a registered stream")
            if kind == "answers":
                need(d.get("mode") in MODES, f"{where}: mode must be one of {MODES}")
            if kind == "misleading":
                ans = self.datasets.get(d.get("answers"))
                need(ans is not None and ans["kind"] == "answers" and ans.get("base") == d.get("base"),
                     f"{where}: answers {d.get('answers')!r} is not a registered answers dataset on {d.get('base')!r}")
                regime = d.get("regime")
                need(isinstance(regime, dict) and "kind" in regime or isinstance(regime, str) and _REGIME_SHORT.match(regime)
                     and int((_REGIME_SHORT.match(regime).group(2) or 0)) <= 100,
                     f"{where}: regime must be pNNN (a share, at most p100), kN (misleading peers per event) or a mapping with kind")
        for name, g in self.groups.items():
            for m in g["group"]:
                need(m in self.datasets or m in self.groups, f"{g['file']}: member {m!r} is not registered")
        for name, m in self.models.items():
            need("path" in m, f"{m['file']}: a model needs path (under paths.models_root, or absolute)")
        for name, p in self.peers.items():
            need(p.get("model") in self.models, f"{p['file']}: model {p.get('model')!r} is not registered")
        from feedback_state.tasks import GRADERS

        for name, t in self.tasks.items():
            need(t.get("grader") in GRADERS, f"{t['file']}: grader {t.get('grader')!r} is not a grading rule of feedback_state/tasks.py ({', '.join(sorted(GRADERS))})")
            need(isinstance(t.get("instruction"), str) and t["instruction"].strip(), f"{t['file']}: a task needs the central model's instruction")
            need(t.get("agreement", "none") in ("canonical", "math", "qa", "none"), f"{t['file']}: agreement must be canonical, math, qa or none")
        return bad


@lru_cache(maxsize=None)
def load_registry(root: str | Path | None = None) -> Registry:
    root = Path(root) if root else REPO / "configs"
    return Registry(root if root.is_absolute() else REPO / root)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="the configs folder (default: configs/)")
    args = ap.parse_args(argv)
    reg = load_registry(args.root)
    print("datasets")
    for n, d in reg.datasets.items():
        if d["kind"] == "stream":
            what = f"{d.get('path')}  {len(d.get('peers') or [])} peers"
        elif d["kind"] == "answers":
            what = f"{d.get('mode')} answers of the peers of {d.get('base')}"
        else:
            what = f"{d.get('base')} + {d.get('answers')}, regime {d.get('regime')}"
        print(f"  {n:34s} {d['kind']:10s} {what}")
    for n, g in reg.groups.items():
        print(f"  {n:34s} {'group':10s} {', '.join(g['group'])}")
    print("models")
    for n, m in reg.models.items():
        print(f"  {n:34s} {m.get('hf_id', m['path'])}" + "".join(f"  {k}={m[k]}" for k in ("engine", "reasoning", "env_vars") if k in m))
    print("peers")
    for n, p in reg.peers.items():
        print(f"  {n:34s} model {p.get('model')}" + ("  reasoning" if p.get("reasoning") else "") + (f"  tool {p['tool']}" if p.get("tool") else ""))
    print("tasks")
    for n, t in reg.tasks.items():
        print(f"  {n:34s} grader {t.get('grader')}  agreement {t.get('agreement', 'none')}" + (f"  max_tokens {t['max_tokens']}" if "max_tokens" in t else "")
              + ("  context" if t.get("context") else ""))
    bad = reg.problems()
    for b in bad:
        print(f"PROBLEM {b}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
