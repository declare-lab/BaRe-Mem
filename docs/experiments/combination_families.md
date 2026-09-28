# combination_families: BaRe-Mem for other central models

This guide runs one experiment: five central models, each on the ten misleading datasets (0, 25, 50, 75 and 100% of
the six peers' answers misleading, capability-supported and capability-challenging). It is the protocol that produced the Qwen3-4B and Qwen3-8B
results (`docs/experiments/combination.md`). Peers' answers and datasets are released on Hugging Face; the models only
answer.

| registered name | model | Hugging Face |
|---|---|---|
| `llama31` | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (gated) |
| `ministral` | Ministral-8B-Instruct-2410 | `mistralai/Ministral-8B-Instruct-2410` (gated) |
| `qwen25` | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| `phi4` | phi-4 (14B) | `microsoft/phi-4` |
| `qwen3_14b` | Qwen3-14B, thinking off | `Qwen/Qwen3-14B` |

For every model and dataset, four results are produced:

- **Advisors + memory** (`tilt`): the question and the six peers' answers, attention tilted by the model's record.
- **Question + Peers** (`peers`): the same prompt, no tilt.
- **No consultation** (`solo`): the question only. It contains no peer answers, so it is run once per stream and shared by
  all rates.
- **BaRe-Mem**: per question, the final answer is Advisors + memory or No consultation, chosen from the record and the
  model's reading line before the answer is graded.

How it works: every model answers alone first, and its own answer joins the six peers as a seventh answer. One record,
built from the model's own features (each model is its own frozen judge), estimates all seven answers. Advisors + memory
reads the six peers with the tilt from that record. BaRe-Mem then compares T·ρ + (1 − T)(κ − δ) with κ, where T is
the record's top trust among the peers, κ its estimate of the no-consultation answer, and ρ, δ the reading line learned
from earlier questions (`feedback_state/reading_line.py`).

## 1. Set up (once)

Follow section 1 of `docs/experiments/misleading_families.md` (environment, `python datasets/download.py`, models,
`paths.models_root`).

## 2. Smoke test

48 questions of one dataset, every step, written to `outputs/smoke/`:

```bash
bash run.sh configs/experiments/combination_families.yaml --smoke --set central=[qwen25]
```

It must end with `RUN_COMPLETE combination_families (smoke)` and print a table with a BaRe-Mem column and a
"reading line" table. Run it once per model before its full run.

## 3. Full run

One model at a time. Every command is resumable: run it again after an interruption and finished work is skipped; a
failed job is retried once.

```bash
bash run.sh configs/experiments/combination_families.yaml --set central=[llama31]
bash run.sh configs/experiments/combination_families.yaml --set central=[ministral]
bash run.sh configs/experiments/combination_families.yaml --set central=[qwen25]
bash run.sh configs/experiments/combination_families.yaml --set central=[phi4]
bash run.sh configs/experiments/combination_families.yaml --set central=[qwen3_14b]
```

Per-job logs are in `logs/combination_families/`.

What each run does, per model:

1. **own**: its no-consultation answers join each of the ten datasets as a seventh
   answer (`data/<dataset>+<model>/`).
2. **features**: the model reads the seven answers of every question.
3. **record**: one record per dataset over the seven answers.
4. **evaluate**: 20 evaluations, Advisors + memory and Question + Peers on each dataset.
5. **BaRe-Mem**: per question, Advisors + memory or No consultation.
6. **table**.

The six-answer
features of `misleading_families` (`outputs/features/<model>/<dataset>/`, without `+own`) are not read by this
experiment.

## 4. Results

When all five models are done, build one table:

```bash
bash run.sh configs/experiments/combination_families.yaml --steps table
```

`outputs/tables/combination_families.md` has one block of rows per model (`llama31 · p000` … `qwen3_14b · p100`). It
shows Advisors + memory, Question + Peers, No consultation and BaRe-Mem for capability-supported and capability-challenging, then the reading
line per task type (the share of questions that took Advisors + memory, ρ̂, δ̂).

## Troubleshooting

- **`STALE ... stored results used other settings`**: a stored result was made with other settings. Nothing is
  overwritten; check before deleting anything.
- **`own_<model>_... FAILED`**: the no-consultation run is missing. Check `outputs/eval/<model>/capability_supported/solo/` and
  `capability_challenging/solo/` (`generations.jsonl` with 4,319 and 17,403 lines). After the own step, every
  `data/<dataset>+<model>/manifest.json` must show `"dropped_for_missing_answers": {}`; anything else means an
  incomplete no-consultation run, and the record would cover fewer questions.
- **`this needs vLLM 0.8.5`**: the tilt patches vLLM 0.8.5's attention kernels; other versions are refused on purpose.
- **A model directory is not found**: check `paths.models_root` and the `path:` in `configs/models/<name>.yaml`.

## Our results, for comparison

Accuracy %, each model with its own features, record and answers (`docs/experiments/combination.md`).

Capability-challenging:

| model · condition | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B · Advisors + memory | 73.8 | 70.4 | 67.5 | 62.5 | 50.7 |
| Qwen3-4B · Question + Peers | 69.2 | 64.0 | 58.1 | 51.4 | 45.8 |
| Qwen3-4B · No consultation | 67.8 | 67.8 | 67.8 | 67.8 | 67.8 |
| Qwen3-4B · **BaRe-Mem** | **75.0** | **72.4** | **70.8** | **70.2** | **69.3** |
| Qwen3-8B · Advisors + memory | 74.2 | 70.1 | 66.9 | 61.0 | 46.2 |
| Qwen3-8B · Question + Peers | 67.9 | 61.6 | 54.3 | 46.7 | 40.0 |
| Qwen3-8B · No consultation | 65.6 | 65.6 | 65.6 | 65.6 | 65.6 |
| Qwen3-8B · **BaRe-Mem** | **74.9** | **71.4** | **70.0** | **68.7** | **67.7** |

Capability-supported:

| model · condition | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B · Advisors + memory | 76.7 | 75.5 | 74.3 | 73.1 | 72.4 |
| Qwen3-4B · Question + Peers | 74.6 | 74.2 | 73.7 | 72.4 | 71.3 |
| Qwen3-4B · No consultation | 69.2 | 69.2 | 69.2 | 69.2 | 69.2 |
| Qwen3-4B · **BaRe-Mem** | **76.2** | **75.4** | **74.1** | **72.7** | **72.0** |
| Qwen3-8B · Advisors + memory | 78.1 | 77.0 | 76.3 | 74.8 | 74.0 |
| Qwen3-8B · Question + Peers | 75.7 | 75.1 | 74.3 | 73.6 | 72.3 |
| Qwen3-8B · No consultation | 72.9 | 72.9 | 72.9 | 72.9 | 72.9 |
| Qwen3-8B · **BaRe-Mem** | **77.4** | **76.6** | **76.1** | **74.6** | **73.7** |

On capability-challenging, BaRe-Mem is above both Advisors + memory and No consultation at every rate for both models. On capability-supported, it
is within 0.7 points of Advisors + memory.
