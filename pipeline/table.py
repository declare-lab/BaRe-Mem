"""Table: the result table of an experiment, read from the shared output layout (pipeline.layout).

    python -m pipeline.table --resolved outputs/runs/<experiment>/resolved.yaml [--smoke]
    (pipeline.run writes the resolved config and calls this as the `table` step)

The experiment's `table:` block chooses the rows: `models` (one row per central model, one column group per dataset),
`regimes` (one row per misleading regime, one column group per base stream, for the experiment's model). `reference:` adds rows for
models evaluated elsewhere on the base streams (the frozen Qwen3-4B of the main experiment). Cells are accuracy in percent
per stream and condition, then the requested differences, then the record's AUC / favourite-right per stream.
With `misleading_probe: true` two more tables follow: what the record makes of the misleading answers, and the peers.
Columns are named by each condition's `label` (Advisors + memory, Question + Peers, No consultation). `extra_columns:
[combination]` adds the column pipeline.combination writes, and with it a table of the reading line per row; `columns:`
sets every column and its order instead (conditions evaluated elsewhere, votes, combination).
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
from pathlib import Path

import yaml

from pipeline.config import deep_merge
from pipeline.layout import Layout


def auc(pairs):
    pos = sorted(p for p, y in pairs if y == 1)
    neg = sorted(p for p, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos) * len(neg))


def load_json(path: Path):
    return json.load(open(path)) if path.exists() else None


def pct(m):
    return "  -  " if m is None else f"{100 * m['accuracy']:5.1f}"


def delta(a, b):
    return "-" if not (a and b) else f"{100 * (a['accuracy'] - b['accuracy']):+.1f}"


def quality_cell(L: Layout, model: str, stream: str) -> str:
    q = load_json(L.quality_file(L.record_file(model, stream)))
    if not q or "memory" not in q:
        return "-"
    return f"{q['memory']['auc']:.2f} / {q['memory']['favourite_acc_mixed']:.0f}%"


def misleading_probe(record: Path, stream: Path) -> dict | None:
    """Mean estimate on honest vs misleading answers, AUC of that separation, how often the favourite is misleading."""
    cache = record.with_name(record.name[: -len(".jsonl")] + ".misleading_probe.json")
    if cache.exists():
        return json.load(open(cache))
    if not (record.exists() and stream.exists()):
        return None
    misled = {}
    for line in stream.open():
        rec = json.loads(line)
        meta = rec.get("peer_metadata", {})
        misled[str(rec["id"])] = [bool(meta.get(k, {}).get("misled")) for k in sorted(rec.get("peer_responses", {}))]
    pairs, sums, fav = [], [[0.0, 0], [0.0, 0]], [0, 0]
    for line in record.open():
        row = json.loads(line)
        flags = misled.get(str(row["id"]))
        if flags is None:
            continue
        probs = [float(p) for p in row["memory_prob"]]
        slots = [bool(flags[int(p)]) for p in row["peer_order"]]
        for p, m in zip(probs, slots):
            pairs.append((p, int(m))); sums[int(m)][0] += p; sums[int(m)][1] += 1
        if any(slots):
            fav[0] += int(slots[max(range(len(probs)), key=lambda s: probs[s])]); fav[1] += 1
    if not pairs:
        return None
    res = {"mean_prob_honest": sums[0][0] / max(1, sums[0][1]), "mean_prob_misleading": sums[1][0] / max(1, sums[1][1]),
           "auc_honest_over_misleading": 1 - auc(pairs), "favourite_misleading_pct": 100 * fav[0] / max(1, fav[1])}
    cache.write_text(json.dumps(res, indent=1))
    return res


def build(cfg: dict, smoke: bool) -> str:
    L = Layout(cfg, smoke)
    tb = cfg.get("table", {})
    conds = list(tb.get("columns") or list(cfg.get("eval_conditions", list(cfg.get("conditions", {})))) + list(tb.get("extra_columns", [])))
    deltas = [tuple(d) for d in tb.get("deltas", [])]
    reg = L.registry
    sm = cfg.get("smoke", {})
    asked = reg.expand(cfg.get("datasets", []))
    datasets = reg.expand(sm["datasets"]) if smoke and "datasets" in sm else asked[:1] if smoke else asked
    datasets = [d for d in datasets if reg.dataset(d)["kind"] != "answers"]
    kind = tb.get("rows", "models")
    by_regime: dict[str, dict[str, str]] = {}   # regime -> {base stream: dataset}
    if kind == "regimes":
        for d in datasets:
            spec = reg.dataset(d)
            label = spec["regime"] if isinstance(spec.get("regime"), str) else d
            by_regime.setdefault(label, {})[spec.get("base", d)] = d
        base_streams = list(dict.fromkeys(b for m in by_regime.values() for b in m))
    else:
        base_streams = datasets
    rows = []   # (label, eval model key, record model, {base stream: evaluated stream}, layout for the record)
    ref_cfg = deep_merge(cfg, {"record": {"fit": tb.get("reference_fit", "address_fit")}, "own_answer": False})
    for m in tb.get("reference", []) if not smoke else []:
        rows.append((f"{m} (reference)", m, m, {s: s for s in base_streams}, Layout(ref_cfg, False)))
    if kind == "models":
        for m in cfg.get("central", []):
            rows.append((m, m, m, {s: s for s in base_streams}, L))
    elif kind == "regimes":   # one block of regime rows per central model
        central = cfg.get("central", [])
        for m in central:
            for r, smap in by_regime.items():
                rows.append((r if len(central) == 1 else f"{m} · {r}", m, m, smap, L))
    cond_label = lambda c: cfg.get("conditions", {}).get(c, {}).get("label", "BaRe-Mem" if c == "combination" else c)
    stream_label = lambda s: reg.dataset(s).get("label", s) if s in reg.datasets else s   # a stream's registered name (label:)
    head = (["row"] + [f"{stream_label(s)}: {cond_label(c)}" for s in base_streams for c in conds]
            + [f"{stream_label(s)}: {cond_label(a)} − {cond_label(b)}" for s in base_streams for a, b in deltas]
            + [f"{stream_label(s)}: record AUC / fav" for s in base_streams])
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for label, key, rec_model, smap, lay in rows:
        m = {(s, c): load_json(lay.eval_dir(key, lay.eval_dataset(smap[s], cfg.get("conditions", {}).get(c, {})), c) / "eval_metrics.json")
             if s in smap else None for s in base_streams for c in conds}
        if all(v is None for v in m.values()):
            continue
        cells = [pct(m[(s, c)]) for s in base_streams for c in conds]
        cells += [delta(m.get((s, a)), m.get((s, b))) for s in base_streams for a, b in deltas]
        cells += [quality_cell(lay, rec_model, smap[s]) if s in smap else "-" for s in base_streams]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    notes = []
    for c in conds:
        if c == "combination":
            notes.append("BaRe-Mem (`combination`): per event Advisors + memory or No consultation, chosen by the reading line")
            continue
        cd = cfg.get("conditions", {}).get(c, {})
        if cd.get("mode") == "vote":
            notes.append(f"{cond_label(c)} (`{c}`): majority vote over the peers' answers" + (f" and the central model's answer in `{cd['own']}`" if cd.get("own") else ""))
            continue
        if cd.get("mode") == "debate":
            notes.append(f"{cond_label(c)} (`{c}`): the central model's answer after {cd.get('round', 1)} debate round(s) with the peers' answers")
            continue
        notes.append(f"{cond_label(c)} (`{c}`): mode {cd.get('mode', 'peers')}" + (f", tilt γ = {cd['gamma']}" if cd.get("gamma") else "") + (", record permuted by rank" if cd.get("swap") else ""))
    text = f"# {cfg['name']}{' (smoke)' if smoke else ''}\n\n" + "\n".join(lines) + "\n\n" + "; ".join(notes) + ".\n"

    if "combination" in conds:
        dlines = ["| row · stream | took Advisors + memory | reading line by task type: ρ̂ / δ̂ / Advisors + memory wins when (at the mean own estimate) |",
                  "|---|---:|---|"]
        for row_label, key, rec_model, smap, lay in rows:
            for s, name in smap.items():
                d = load_json(lay.eval_dir(key, name, "combination") / "eval_metrics.json")
                if not d:
                    continue
                parts = [f"{t} {v['rho']:.2f} / {v['delta']:.2f} / {v['peers_memory_wins_at_mean_own_prob']}" for t, v in d["reading_line"].items()]
                dlines.append(f"| {row_label} · {stream_label(s)} | {100 * d['share_peers_memory']:.0f}% | {'; '.join(parts)} |")
        text += "\n## The reading line\n\n" + "\n".join(dlines) + "\n"

    if tb.get("misleading_probe"):
        plines = ["| regime · stream | honest answers: mean estimate | misleading answers | AUC honest over misleading | favourite is misleading |",
                  "|---|---:|---:|---:|---:|"]
        peer_lines = ["| regime · stream | peer | ratio asked | reached | accuracy honest | in stream | forced | usable misleading answers |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
        seen = set()   # the peers are the same for every central model: listed once per dataset
        for label, key, rec_model, smap, lay in rows:
            if label.endswith("(reference)"):
                continue
            for s, name in smap.items():
                spec = L.stream(name)
                stream = spec["path"]
                probe = misleading_probe(L.record_file(rec_model, name), stream)
                if probe:
                    plines.append(f"| {label} · {s} | {probe['mean_prob_honest']:.2f} | {probe['mean_prob_misleading']:.2f} | "
                                  f"{probe['auc_honest_over_misleading']:.2f} | {probe['favourite_misleading_pct']:.0f}% |")
                if name in seen:
                    continue
                seen.add(name)
                man = load_json(stream.parent / "manifest.json")
                for peer, v in sorted((man or {}).get("peers", {}).items(), key=lambda kv: kv[1]["index"]):
                    answers = L.stream(spec["answers"])["path"] / peer if spec.get("answers") else None
                    sums = [json.load(open(f)) for f in glob.glob(str(answers / "summary.shard*.json"))] if answers else []
                    usable = f"{100 * sum(x.get('accepted', 0) for x in sums) / max(1, sum(x['n'] for x in sums)):.0f}%" if sums else "-"
                    peer_lines.append(f"| {label.split(' · ')[-1]} · {s} | peer_{v['index']} {peer} | {v['requested_ratio']:.0f}% | {v['realised_ratio']:.1f}% | "
                                      f"{v['accuracy_honest']:.1f} | {v['accuracy_in_stream']:.1f} | {v['forced']} | {usable} |")
        text += "\n## What the record makes of the misleading answers\n\n" + "\n".join(plines) + "\n"
        text += "\n## The peers\n\n" + "\n".join(peer_lines) + "\n"
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resolved", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.resolved.read_text())
    text = build(cfg, args.smoke)
    out = Layout(cfg, args.smoke).table_file(cfg["name"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"[table] written to {out}")


if __name__ == "__main__":
    main()
