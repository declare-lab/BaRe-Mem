"""Experiment configs: YAML files with `base:` inheritance and command-line overrides.

    cfg = load("configs/experiments/main.yaml", ["evaluation.max_new_tokens=1024", "gpus=[0,1]"])

A config may name a `base:` file (relative to itself); the base is loaded first and the config is deep-merged over it
(mappings merge key by key, everything else is replaced). Overrides are `dotted.key=value` with the value parsed as
YAML, applied last.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO = Path(__file__).resolve().parents[1]


def shown(path: str | Path | None) -> str | None:
    """A path as written into a manifest or metrics file: relative to the repository when inside it.

    Jobs run from a per-job code snapshot, so an absolute path would name that snapshot rather than the project.
    """
    if path is None:
        return None
    p = str(path)
    if not Path(p).is_absolute():
        return p
    for root in _roots():
        if p.startswith(root + "/"):
            return p[len(root) + 1:]
    return p


def _roots() -> list[str]:
    """The repository, and the live project its outputs/ symlink points to when running from a snapshot."""
    roots = [str(REPO)]
    live = (REPO / "outputs").resolve().parent
    if str(live) not in roots:
        roots.append(str(live))
    return roots


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {dotted}: {k} is not a mapping")
    node[keys[-1]] = value


def load(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO / path
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load(path.parent / base), cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, raw = item.split("=", 1)
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    cfg.setdefault("name", path.stem)
    return cfg
