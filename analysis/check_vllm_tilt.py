"""Does the vLLM bias kernel compute the same tilted attention as the HF mask hook?

Stage ``vllm``: patched vLLM (feedback_state.vllm_attn_bias) decodes greedy continuations of real six-peer prompts with a
non-flat record, once with the tilt and once without, and records the chosen tokens with their log-probs.
Stage ``hf``: HF eager attention + the mask hook re-scores those continuations with and without the tilt, and decodes
its own greedy continuation under the tilt.  The report says how close vLLM's log-probs are to HF's under the same
tilt (should be bf16 noise, ~1e-2) and how far from HF without it (should be clearly larger), plus greedy agreement.

  PYTHONPATH=. python -m analysis.check_vllm_tilt --stage vllm --central_model /models/Qwen3-4B --out outputs/analysis/tilt_check
  PYTHONPATH=. python -m analysis.check_vllm_tilt --stage hf --central_model /models/Qwen3-4B --out outputs/analysis/tilt_check
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from feedback_state.attn_bias import FLAT_SPREAD, prompt_token_bias
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import render_prompt
from feedback_state.memory_generator import peer_texts_in_prompt_order


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["vllm", "hf"], required=True)
    p.add_argument("--central_model", required=True)
    p.add_argument("--prompts", type=Path, default=Path("outputs/record/q3_4b/capability_supported/shuffled0.fit-self.jsonl"))
    p.add_argument("--records", type=Path, default=Path("data/capability_supported/test.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--pos_start", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--form", default="logratio")
    p.add_argument("--max_new_tokens", type=int, default=48)
    return p.parse_args()


def select(args, tok):
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = sorted((json.loads(l) for l in args.prompts.open()), key=lambda r: int(r["pos"]))
    chosen = []
    for r in rows:
        if int(r["pos"]) < args.pos_start:
            continue
        probs = [float(p) for p in r["memory_prob"]]
        if max(probs) - min(probs) <= FLAT_SPREAD:
            continue
        rec = records[str(r["id"])]
        prompt = render_prompt(tok, r["messages_peers"])
        ids, bias = prompt_token_bias(tok, prompt, peer_texts_in_prompt_order(rec, r["peer_order"]), probs, args.gamma, args.form)
        if not bias.any():
            continue
        chosen.append({"id": str(r["id"]), "pos": int(r["pos"]), "task": r["task_type"], "ids": ids, "bias": bias.tolist(), "probs": probs})
        if len(chosen) >= args.n:
            break
    return chosen


def stage_vllm(args):
    from transformers import AutoTokenizer

    from feedback_state import vllm_attn_bias

    vllm_attn_bias.install()
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    items = select(args, tok)
    print(f"[tilt-check] {len(items)} prompts, biased tokens per prompt: {[int(np.count_nonzero(x['bias'])) for x in items]}", flush=True)
    llm = LLM(model=args.central_model, dtype="bfloat16", gpu_memory_utilization=0.8, max_model_len=9216, enforce_eager=True, enable_prefix_caching=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, logprobs=1)
    inputs = [{"prompt_token_ids": x["ids"]} for x in items]
    out = {}
    for name, use_bias in (("tilt", True), ("plain", False)):
        vllm_attn_bias.clear()
        if use_bias:
            for x in items:
                vllm_attn_bias.register(x["ids"], x["bias"])
        t0 = time.time()
        res = llm.generate(inputs, params, use_tqdm=False)
        for x, r in zip(items, res):
            o = r.outputs[0]
            lps = [float(lp[t].logprob) for t, lp in zip(o.token_ids, o.logprobs)]
            x.setdefault(name, {})["tokens"] = list(o.token_ids)
            x[name]["logprobs"] = lps
            x[name]["text"] = o.text
        print(f"[tilt-check] vllm {name}: {time.time() - t0:.1f}s", flush=True)
    vllm_attn_bias.clear()
    differ = sum(x["tilt"]["tokens"] != x["plain"]["tokens"] for x in items)
    print(f"[tilt-check] tilt changes the greedy continuation on {differ}/{len(items)} prompts", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "vllm.json").write_text(json.dumps(items))


def stage_hf(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from feedback_state.attn_bias import install_hf_hooks

    items = json.loads((args.out / "vllm.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.central_model, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).cuda().eval()
    steer = install_hf_hooks(model)
    summary = {"tilt_vs_hf_tilt": [], "tilt_vs_hf_plain": [], "plain_vs_hf_plain": [], "plain_vs_hf_tilt": [], "greedy_match_tilt": [], "greedy_match_plain": [],
               "argmax_agree_tilt": [], "argmax_agree_plain": []}

    def score(ids, gen, bias):
        full = torch.tensor([ids + gen], device="cuda")
        steer.bias = None if bias is None else torch.tensor([list(bias) + [0.0] * len(gen)], device="cuda")
        with torch.no_grad():
            logits = model(full).logits[0, len(ids) - 1: len(ids) + len(gen) - 1].float()
        lp = torch.log_softmax(logits, -1)
        chosen = lp[torch.arange(len(gen)), torch.tensor(gen, device="cuda")]
        return chosen.cpu().numpy(), lp.argmax(-1).cpu().numpy()

    def greedy(ids, bias):
        inp = torch.tensor([ids], device="cuda")
        steer.bias = None if bias is None else torch.tensor([list(bias) + [0.0] * (args.max_new_tokens + 1)], device="cuda")
        with torch.no_grad():
            g = model.generate(inp, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return g[0, len(ids):].tolist()

    t0 = time.time()
    for x in items:
        ids = x["ids"]
        for name in ("tilt", "plain"):
            gen = x[name]["tokens"]
            v = np.array(x[name]["logprobs"])
            h_t, am_t = score(ids, gen, x["bias"])
            h_p, am_p = score(ids, gen, None)
            summary[f"{name}_vs_hf_tilt"].append(float(np.mean(np.abs(v - h_t))))
            summary[f"{name}_vs_hf_plain"].append(float(np.mean(np.abs(v - h_p))))
            summary[f"argmax_agree_{name}"].append(float(np.mean(am_t == np.array(gen)) if name == "tilt" else np.mean(am_p == np.array(gen))))
            hg = greedy(ids, x["bias"] if name == "tilt" else None)
            summary[f"greedy_match_{name}"].append(float(np.mean([a == b for a, b in zip(hg, gen)])))
            x[name]["hf_greedy_text"] = tok.decode(hg)
    steer.bias = None
    rep = {k: float(np.mean(v)) for k, v in summary.items()}
    rep["n"] = len(items); rep["seconds"] = time.time() - t0
    print("[tilt-check] mean |logprob difference| of vLLM's own greedy tokens:")
    print(f"   vLLM tilt  vs HF tilt  {rep['tilt_vs_hf_tilt']:.4f}   vs HF plain {rep['tilt_vs_hf_plain']:.4f}")
    print(f"   vLLM plain vs HF plain {rep['plain_vs_hf_plain']:.4f}   vs HF tilt  {rep['plain_vs_hf_tilt']:.4f}")
    print(f"   HF argmax agrees with vLLM's token: tilt {100 * rep['argmax_agree_tilt']:.1f}%  plain {100 * rep['argmax_agree_plain']:.1f}%")
    print(f"   greedy continuation token match (HF vs vLLM, same condition): tilt {100 * rep['greedy_match_tilt']:.1f}%  plain {100 * rep['greedy_match_plain']:.1f}%")
    (args.out / "report.json").write_text(json.dumps(rep, indent=1))
    (args.out / "hf.json").write_text(json.dumps(items))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if args.stage == "vllm":
        stage_vllm(args)
    else:
        stage_hf(args)


if __name__ == "__main__":
    main()
