# misleading_families: how robust are other central models to misleading peers?

This guide runs one experiment: five central models answer the ten misleading datasets (0, 25, 50, 75 and 100% of the
six peers' answers misleading, capability-supported and capability-challenging), each with its own BaRe-Mem record. It is the same protocol
that produced the Qwen3-4B result (`docs/experiments/misleading.md`). You only evaluate; the misleading answers and
datasets are released on Hugging Face, and nothing is generated.

| registered name | model | Hugging Face |
|---|---|---|
| `llama31` | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (gated: accept the licence on its page first) |
| `ministral` | Ministral-8B-Instruct-2410 | `mistralai/Ministral-8B-Instruct-2410` (gated) |
| `qwen25` | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| `phi4` | phi-4 (14B) | `microsoft/phi-4` |
| `qwen3_14b` | Qwen3-14B, thinking off | `Qwen/Qwen3-14B` |

For every model and dataset, three conditions are measured:

- `tilt`: the six peer answers in the prompt, with the record's attention tilt (Advisors + memory).
- `peers`: the same prompt, no tilt (Question + Peers).
- `solo`: the question only. It contains no peer answers, so it is run once per stream and shared by all rates.

## 1. Set up (once)

```bash
cd BaRe-Mem                                     # this repository
conda create -n bare-mem python=3.12 && conda activate bare-mem
pip install -r requirements_qwen3.txt           # torch 2.6.0, transformers 4.56.2, vLLM 0.8.5 (exactly this version)
python datasets/download.py                     # the datasets into data/
```

It must end with `ok     15 datasets` (the three streams, the peers' misleading answers on two of them, and the ten
misleading datasets). Every file is checked against its sha256; a `MISMATCH` means the download is corrupt or not the
release: run it again. Where the data is hosted is set in `datasets/manifest.json` (`repo`).

If the five models are not on the machine yet, download them into one directory (each in its own sub-directory, named
as below), for example:

```bash
export MODELS=/path/to/models
huggingface-cli login                           # for the two gated models
for m in meta-llama/Llama-3.1-8B-Instruct:Meta-Llama-3.1-8B-Instruct mistralai/Ministral-8B-Instruct-2410:Ministral-8B-Instruct-2410 \
         Qwen/Qwen2.5-7B-Instruct:Qwen2.5-7B-Instruct microsoft/phi-4:phi-4 Qwen/Qwen3-14B:Qwen3-14B; do
  huggingface-cli download "${m%%:*}" --local-dir "$MODELS/${m##*:}"
done
```

Set `paths.models_root` in `configs/base.yaml` to the directory holding the models, or add
`--set paths.models_root=$MODELS` to every command below. Each model's sub-directory name must match its `path:` in
`configs/models/<name>.yaml` (edit `path:` if your copy is named differently, or give an absolute path there).

## 2. Smoke test

48 events of one dataset, every step, written to `outputs/smoke/` so that it never mixes with real results:

```bash
bash run.sh configs/experiments/misleading_families.yaml --smoke --set central=[qwen25]
```

It must end with `RUN_COMPLETE misleading_families (smoke)` and print a small table. Do this once for each model before
its full run (replace `qwen25`); it catches a missing model directory or a licence problem early.

## 3. Full run

One model at a time (every command is resumable: run it again after an interruption and finished
work is skipped; a failed job is retried once):

```bash
bash run.sh configs/experiments/misleading_families.yaml --set central=[llama31]
bash run.sh configs/experiments/misleading_families.yaml --set central=[ministral]
bash run.sh configs/experiments/misleading_families.yaml --set central=[qwen25]
bash run.sh configs/experiments/misleading_families.yaml --set central=[phi4]
bash run.sh configs/experiments/misleading_families.yaml --set central=[qwen3_14b]
```

Per-job logs are in `logs/misleading_families/`.

Each run does, per model: the model's features on the ten datasets, its record on each, then 22
evaluations (10 datasets × `tilt`, `peers`, plus `solo` on the two base streams), then the table.

## 4. Results

When all five models are done, build one table with every model:

```bash
bash run.sh configs/experiments/misleading_families.yaml --steps table
```

`outputs/tables/misleading_families.md` has one block of rows per model (`llama31 · p000` ... `qwen3_14b · p100`),
with accuracy in `tilt`, `peers` and `solo` for capability-supported and capability-challenging, plus the record's quality.

## Troubleshooting

- **`STALE ... stored results used other settings`**: a result from a run with different settings is in the way. Nothing
  is overwritten; check before deleting anything.
- **`this needs vLLM 0.8.5`**: the tilt patches vLLM 0.8.5's attention kernels; other versions are refused on purpose.
- **A model directory is not found**: check `paths.models_root` and that the sub-directory name matches `path:` in
  `configs/models/<name>.yaml`.
- **Ministral**: its 32k sliding window is switched off inside the engine; the prompts are under 4k tokens, so nothing changes.
- **Qwen3-14B** runs with thinking off, like Qwen3-4B.

## Result (2026-09-14)

Accuracy (%) with memory / without memory (`tilt` / `peers`) at each misleading information ratio, and No consultation (`solo`), each
model with its own record.

Capability-challenging:

| model | 0% | 25% | 50% | 75% | 100% | solo |
|---|---|---|---|---|---|---:|
| Qwen3-14B | 75.6 / 69.8 | 72.2 / 65.6 | 69.6 / 60.4 | 65.7 / 55.2 | 54.8 / 50.6 | 66.5 |
| Qwen3-4B (ours) | 74.0 / 69.2 | 70.3 / 64.0 | 67.9 / 58.1 | 63.5 / 51.4 | 50.2 / 45.8 | 67.8 |
| Qwen2.5-7B | 71.5 / 65.9 | 67.9 / 59.3 | 65.2 / 52.9 | 59.7 / 47.3 | 48.3 / 43.3 | 59.1 |
| Llama-3.1-8B | 71.3 / 68.3 | 67.9 / 61.2 | 65.1 / 54.5 | 60.2 / 48.2 | 49.4 / 45.5 | 67.4 |
| phi-4 | 71.1 / 65.8 | 68.8 / 62.4 | 66.9 / 57.9 | 62.8 / 52.5 | 52.2 / 48.3 | 63.4 |
| Ministral-8B | 70.9 / 64.0 | 66.1 / 54.8 | 62.2 / 41.6 | 53.7 / 27.4 | 26.1 / 18.0 | 58.6 |

Capability-supported, with reading answers graded by exact match (an earlier rule; the pipeline now grades reading by token-F1
≥ 0.5; Qwen3-4B is shown under the same rule here):

| model | 0% | 25% | 50% | 75% | 100% | solo |
|---|---|---|---|---|---|---:|
| Qwen3-14B | 68.4 / 67.0 | 67.9 / 66.7 | 67.4 / 66.2 | 66.0 / 64.9 | 65.2 / 64.6 | 63.6 |
| Qwen3-4B (ours) | 67.3 / 64.8 | 66.0 / 64.9 | 65.4 / 64.5 | 64.4 / 63.4 | 63.8 / 62.9 | 60.5 |
| Qwen2.5-7B | 66.7 / 62.8 | 65.5 / 61.9 | 64.4 / 61.0 | 62.8 / 59.6 | 61.4 / 58.9 | 63.6 |
| phi-4 | 66.4 / 62.0 | 64.8 / 62.2 | 63.7 / 61.9 | 62.5 / 61.3 | 62.2 / 60.6 | 59.4 |
| Llama-3.1-8B | 63.3 / 57.8 | 62.1 / 57.1 | 59.5 / 55.4 | 55.2 / 52.3 | 52.7 / 50.4 | 57.7 |
| Ministral-8B | 60.1 / 53.5 | 57.7 / 52.9 | 54.3 / 51.1 | 51.6 / 49.0 | 51.3 / 47.3 | 47.0 |

Reading: with memory every model is above its no-memory accuracy at every rate; a model stays above its no-consultation
accuracy on capability-challenging up to 25% misleading (Llama-3.1-8B), 50% (Qwen3-4B, Qwen3-14B, phi-4, Ministral-8B) or 75%
(Qwen2.5-7B). Ministral-8B is the most swayed without memory (64.0 → 18.0 on capability-challenging); Qwen3-14B is the highest with memory
at every rate on both streams. The record built from each model's own features separates honest from misleading answers
alike (AUC 0.68–0.73 capability-supported, 0.77–0.87 capability-challenging).
