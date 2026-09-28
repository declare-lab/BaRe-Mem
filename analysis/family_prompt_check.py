"""CPU-only check that the memory's tilt maps onto a central model's tokens: for each model, render the
peers prompt of the first K events of a stream with that model's own chat template and tokenizer, locate the six peer
blocks, and report how completely the tilt covers them.  Run it for a new model family before its experiments.

    PYTHONPATH=. python -m analysis.family_prompt_check --models models/phi-4 ... [--k 100]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from feedback_state.attn_bias import FLAT_SPREAD, bias_values, peer_char_spans, prompt_token_bias
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import has_chat_template, render_prompt
from feedback_state.memory_generator import peer_texts_in_prompt_order


def check(model: str, rows: list[dict], records: dict, gamma: float) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    n_tok, cover, tilted, any_tilt, first_bos, missing = [], [], [], 0, 0, 0
    for r in rows:
        rec = records[str(r["id"])]
        texts = peer_texts_in_prompt_order(rec, r["peer_order"])
        prompt = render_prompt(tok, r["messages_peers"])
        probs = [float(p) for p in r["memory_prob"]]
        ids, bias = prompt_token_bias(tok, prompt, texts, probs, gamma)
        n_tok.append(len(ids))
        first_bos += int(tok.bos_token_id is not None and len(ids) > 0 and ids[0] == tok.bos_token_id)
        spans = peer_char_spans(texts, prompt)
        missing += sum(1 for (a, b), t in zip(spans, texts) if b <= a and str(t).strip())
        vals = bias_values(probs, gamma)
        if not any(vals):
            continue
        any_tilt += 1
        tilted.append(int((bias != 0).sum()))
        # coverage: characters of the tilted peers' blocks that lie under a tilted token
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        covered = sum(e - s for (s, e), b in zip(enc["offset_mapping"], bias) if b != 0)
        total = sum(b - a for (a, b), v in zip(spans, vals) if v != 0 and b > a)
        cover.append(covered / total if total else 1.0)
    return {"model": Path(model).name, "tokenizer": type(tok).__name__, "chat_template": has_chat_template(tok),
            "bos_first": f"{first_bos}/{len(rows)}", "tokens": f"{min(n_tok)}-{max(n_tok)} (median {int(statistics.median(n_tok))})",
            "peer_blocks_missing": missing, "prompts_with_tilt": f"{any_tilt}/{len(rows)} (record spread > {FLAT_SPREAD})",
            "tilted_tokens_per_prompt": int(statistics.median(tilted)) if tilted else 0,
            "tilted_block_coverage": f"{100 * statistics.mean(cover):.1f}%" if cover else "-"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", default="outputs/record/q3_4b/capability_supported/shuffled0.fit-address_fit.jsonl")
    ap.add_argument("--records", default="data/capability_supported/test.jsonl")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=3.0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.prompts)][: args.k]
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    ok = True
    for m in args.models:
        res = check(m, rows, records, args.gamma)
        print(json.dumps(res))
        cov = res["tilted_block_coverage"]
        if res["peer_blocks_missing"] or (cov != "-" and float(cov.rstrip("%")) < 95):
            ok = False
    print("PROMPT_CHECK_OK" if ok else "PROMPT_CHECK_FAILED")


if __name__ == "__main__":
    main()
