"""Download the released datasets from the Hugging Face Hub into data/, the layout the pipeline reads.

    python datasets/download.py                    every dataset in datasets/manifest.json
    python datasets/download.py --repo <user>/<dataset>   the same files from another Hugging Face dataset repository
    python datasets/download.py --only capability_supported capability_challenging_misleading_p050

Each file is checked against its sha256 before it is unpacked; an existing file is kept, never overwritten. Streams go
to data/<dir>/<file>.jsonl (a misleading stream with its manifest.json), the peers' answers to data/<dataset>/<peer>/
as the peers step writes them. Needs only huggingface_hub (installed with transformers).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack_stream(entry: dict, archive: Path) -> None:
    target = ROOT / entry["unpacks_to"]
    if target.exists() and target.stat().st_size:
        print(f"keep   {entry['unpacks_to']}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with gzip.open(archive, "rt", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
    tmp.rename(target)
    if entry["kind"] == "misleading":
        (target.parent / "manifest.json").write_text(json.dumps(entry["stream_manifest"], indent=1))
    print(f"wrote  {entry['unpacks_to']} ({entry['rows']:,} events)")


def unpack_answers(entry: dict, archive: Path) -> None:
    target = ROOT / entry["unpacks_to"]
    todo = {p for p in entry["peers"] if not (target / p / "complete.json").exists()}
    if not todo:
        print(f"keep   {entry['unpacks_to']}/ ({len(entry['peers'])} peers)")
        return
    handles, counts = {}, collections.Counter()
    try:
        with gzip.open(archive, "rt", encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                peer = row.pop("peer")
                if peer not in todo:
                    continue
                if peer not in handles:
                    (target / peer).mkdir(parents=True, exist_ok=True)
                    handles[peer] = (target / peer / "shard0of1.jsonl").open("w", encoding="utf-8")
                handles[peer].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[peer] += 1
    finally:
        for h in handles.values():
            h.close()
    for peer in sorted(todo):
        (target / peer / "summary.shard0of1.json").write_text(json.dumps(entry["peers"][peer], indent=1))
        (target / peer / "complete.json").write_text(json.dumps({"shards": 1, "downloaded_from": entry["file"]}))
        print(f"wrote  {entry['unpacks_to']}/{peer}/ ({counts[peer]:,} answers)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="dataset names (default: all)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "datasets" / "manifest.json")
    ap.add_argument("--repo", default=None, help="the Hugging Face dataset repository (default: the manifest's `repo`)")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    man = json.load(open(args.manifest))
    if args.repo:
        man["repo"] = args.repo
    files = [e for e in man["files"] if args.only is None or e["dataset"] in args.only]
    missing = set(args.only or []) - {e["dataset"] for e in files}
    if missing:
        sys.exit(f"not in the release: {sorted(missing)}")
    print(f"https://huggingface.co/datasets/{man['repo']} at {man.get('revision', 'main')}")
    bad = []
    for e in files:
        archive = Path(hf_hub_download(man["repo"], e["file"], repo_type="dataset", revision=man.get("revision")))
        if sha256(archive) != e["sha256"]:
            print(f"MISMATCH {e['file']}")
            bad.append(e["file"])
            continue
        (unpack_answers if e["kind"] == "answers" else unpack_stream)(e, archive)
    if bad:
        sys.exit(f"{len(bad)} file(s) differ from the release: {bad}")
    print(f"ok     {len(files)} datasets")


if __name__ == "__main__":
    main()
