# families: other central models, each with its own record

```bash
bash run.sh configs/experiments/families.yaml --smoke --set central=[qwen25]   # one model, 48 events
bash run.sh configs/experiments/families.yaml                                  # Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, phi-4
```

Does the record help central models other than Qwen3-4B? Each model takes both roles itself: it is the frozen judge whose
hidden states address its own record, and it is the model that answers. Nothing Qwen is involved in a family's rows.
Everything else is identical to the main experiment (the six peers and their answers, the streams and their order, the
record's equations, the tilt, the grading), so a family's rows are the counterpart of the frozen Qwen3-4B rows, which
the table adds as the reference.

Add a model: register it as `configs/models/<name>.yaml` (`hf_id`, `path` under `paths.models_root`, `engine`) and put
the name in `central:` of this file. The tilt is exact reweighting of
softmax attention, `softmax(s + b) = Norm(A ⊙ c)`, so it applies to any softmax-attention model (MHA or GQA, RoPE, any head
size); the only model-specific work, locating each peer block's tokens under the model's own tokenizer and chat template,
is automatic (`feedback_state/attn_bias.py`). `python -m analysis.family_prompt_check --models <dir>` checks it on CPU. Not covered: linear-attention layers (no softmax over keys) and ALiBi / iRoPE.

A model vLLM 0.8.5 cannot run takes the HF engine: in its model file write `engine: hf` (and `conda_env: <env>` when it
needs its own environment).

Notes:
- Ministral's 32k sliding window is switched off inside the engine; the prompts are under 4k tokens, so nothing changes.
- Every condition hands vLLM the same token ids (tokenised without added special tokens), so tilt and no-tilt see
  identical sequences.
- The base Meta-Llama-3-8B was tried and dropped: without a chat template it does not follow the answer format.
- The 2026-09-11 smoke run (48 events) confirmed every family end to end: prompts tilted 43-45/48, bias kernels taken,
  generations changed by the tilt 18-44/48. No full family run exists yet.
