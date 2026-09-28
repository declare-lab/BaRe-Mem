# baselines_families: BaRe-Mem against multi-agent baselines, for six central models

This guide runs one experiment: six central models, each on the ten misleading datasets (0, 25, 50, 75 and 100% of the
six peers' answers misleading, capability-supported and capability-challenging). It is the protocol that produced the Qwen3-4B comparison
(`docs/experiments/baselines.md`). Peers' answers and datasets are released on Hugging Face; the models only answer.

| registered name | model | Hugging Face |
|---|---|---|
| `qwen3_8b` | Qwen3-8B, thinking off | `Qwen/Qwen3-8B` |
| `qwen3_14b` | Qwen3-14B, thinking off | `Qwen/Qwen3-14B` |
| `llama31` | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (gated) |
| `ministral` | Ministral-8B-Instruct-2410 | `mistralai/Ministral-8B-Instruct-2410` (gated) |
| `phi4` | phi-4 (14B; not phi-4-mini) | `microsoft/phi-4` |
| `qwen25` | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |

For every model and dataset, seven results are recorded:

- **BaRe-Mem**: per question, Advisors + memory or No consultation, chosen from the record and the model's reading line
  before the answer is graded.
- **Advisors + memory** (`tilt`): the question and the six peers' answers, attention tilted by the model's record.
- **Question + Peers** (`peers`): the same prompt, no tilt.
- **Debate (2 rounds)** (`debate2`): multi-agent debate. The model starts from its no-consultation
  answer, reads the six peers' answers and updates its answer, twice. The peers' answers stay as released, so a
  misleading answer stays misleading. Round 1 (`debate1`) is run too, because round 2 continues it, but it is not in
  the table.
- **Majority vote (advisors + own)** (`vote_all`): the answer most of the six peers and the model's no-consultation answer
  give. No model runs; a tie counts as an even split.
- **Majority vote (advisors)** (`vote_peers`): the answer most of the six peers give.
- **No consultation** (`solo`): the question only; shared by all rates of a stream.

One command per model runs whatever is missing, in order: the `combination` pipeline (no-consultation answers, features,
record, Advisors + memory, Question + Peers, BaRe-Mem) and then the baselines. Finished work is skipped: if you ran
`combination_families` for a model, its results are reused and only the debate and the votes run.

## 1. Set up (once)

```bash
cd BaRe-Mem
```

If Qwen3-8B or Qwen3-14B is not on the machine yet, download it next to the other models (the sub-directory name must
match `path:` in `configs/models/<name>.yaml`):

```bash
huggingface-cli download Qwen/Qwen3-8B  --local-dir "$MODELS/Qwen3-8B"
huggingface-cli download Qwen/Qwen3-14B --local-dir "$MODELS/Qwen3-14B"
```

On a new machine, follow section 1 of `docs/experiments/misleading_families.md` first (environment,
`python datasets/download.py`, models, `paths.models_root`).

## 2. Smoke test

48 questions of one dataset, every step, written to `outputs/smoke/`:

```bash
bash run.sh configs/experiments/baselines_families.yaml --smoke --set central=[qwen25]
```

It must end with `RUN_COMPLETE baselines_families (smoke)` and print a table with the seven columns. Run it once per
model before its full run.

## 3. Full run

One model at a time. Every command is resumable: run it again after an interruption and finished work is skipped; a
failed job is retried once.

```bash
bash run.sh configs/experiments/baselines_families.yaml --set central=[llama31]
bash run.sh configs/experiments/baselines_families.yaml --set central=[ministral]
bash run.sh configs/experiments/baselines_families.yaml --set central=[phi4]
bash run.sh configs/experiments/baselines_families.yaml --set central=[qwen25]
bash run.sh configs/experiments/baselines_families.yaml --set central=[qwen3_8b]
bash run.sh configs/experiments/baselines_families.yaml --set central=[qwen3_14b]
```

Per-job logs are in `logs/baselines_families/`.

What each run does, per model:

1. **own**: its no-consultation answers join each dataset as a seventh answer (skipped if done).
2. **features**, **record**: the model reads the seven answers of every question; one record per dataset (skipped if done).
3. **evaluate**: Advisors + memory and Question + Peers (skipped if done), then debate round 1 on all ten datasets, then
   round 2.
4. **vote**: the two majority votes.
5. **BaRe-Mem** (skipped if done) and **table**.

Debate prompts are longer than Advisors + memory's: the peers'
answers are sent in each round, so round 2 prompts are up to twice as long.

## 4. Results

When all six models are done, build one table:

```bash
bash run.sh configs/experiments/baselines_families.yaml --steps table
```

`outputs/tables/baselines_families.md` has one block of rows per model (`qwen3_8b · p000` … `qwen25 · p100`) with the
seven columns for capability-supported and capability-challenging.

## Troubleshooting

- **`STALE ... stored results used other settings`**: a stored result was made with other settings. Nothing is
  overwritten; check before deleting anything.
- **`... holds 1 answers per event; debate round 2 needs 2`** or **`event ... has no answer in ...`**: round 2 found an
  incomplete round 1, or round 1 an incomplete no-consultation run. Run the same command again (round 1 finishes first);
  if it persists, check that `outputs/eval/<model>/<dataset>+own/debate1/generations.jsonl` has 4,319 lines
  (capability-supported) or 17,403 (capability-challenging).
- **`this needs vLLM 0.8.5`**: the tilt patches vLLM 0.8.5's attention kernels; other versions are refused on purpose.
- **A model directory is not found**: check `paths.models_root` and the `path:` in `configs/models/<name>.yaml`.

## Our results, for comparison

Qwen3-4B, accuracy % (`docs/experiments/baselines.md`).

Capability-challenging:

| misleading | BaRe-Mem | Advisors + memory | Question + Peers | Debate (2 rounds) | Majority vote (advisors + own) | Majority vote (advisors) | No consultation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | **75.0** | 73.8 | 69.2 | 71.4 | 68.2 | 64.1 | 67.8 |
| 25% | **72.4** | 70.4 | 64.0 | 68.6 | 56.8 | 48.7 | 67.8 |
| 50% | **70.8** | 67.5 | 58.1 | 64.6 | 34.3 | 24.5 | 67.8 |
| 75% | **70.2** | 62.5 | 51.4 | 60.0 | 10.4 | 5.9 | 67.8 |
| 100% | **69.3** | 50.7 | 45.8 | 56.1 | 3.0 | 1.2 | 67.8 |

Capability-supported:

| misleading | BaRe-Mem | Advisors + memory | Question + Peers | Debate (2 rounds) | Majority vote (advisors + own) | Majority vote (advisors) | No consultation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | 76.2 | **76.7** | 74.6 | 73.5 | 68.0 | 64.3 | 69.2 |
| 25% | 75.4 | **75.5** | 74.2 | 73.2 | 61.8 | 53.0 | 69.2 |
| 50% | 74.1 | **74.3** | 73.7 | 73.0 | 46.9 | 32.5 | 69.2 |
| 75% | 72.7 | **73.1** | 72.4 | 72.3 | 27.3 | 14.6 | 69.2 |
| 100% | 72.0 | **72.4** | 71.3 | 72.1 | 18.8 | 9.4 | 69.2 |
