# agent_team: the record inside a lead-worker team

Question: what does the reliability record do in a real agent team, where a lead splits a task, gives each sub-task to one
worker, verifies what comes back, replans what fails and synthesises the rest?

A team differs from the QA streams in three ways, and each one is a decision the record can serve:

| in the QA streams | in a team | the record's part |
|---|---|---|
| every peer answers every event | a sub-task goes to **one** worker | **assign**: P(right) of every source from the sub-task alone, before any report exists |
| the central model weighs six answers itself | the lead checks the one report against what it expected; a report that does not meet it goes to the next source, or the lead does the sub-task itself | the check is the lead's, not the record's; the record's ranking is the order of reassignment |
| every answer is verified after the event | only the reports that were produced can be verified | **write**: the gold label of every report that was produced |

```
                 Lead agent (Qwen3-4B)
                        |
                 task -> sub-tasks
        assign: record(sub-task) ranks the sources (the workers, and the lead itself)
                        v
        first source (one model, one tool) -> report
                        v
        verification, never by the record:
            the lead's check     does the report meet what I expected (a plain, specific answer of the right kind)? no evidence
            the dataset's check  the benchmark's evaluator (the sub-task's gold answer): exact; the second illustration
            fail -> replan: the next source in the record's order (consultation, or the lead itself: autonomy), two at most
            pass -> commit -> synthesis
                        v
        after the commit: the gold answer labels EVERY report that was produced, the failed ones included -> written into the record
```

## What had to change in the method

- **The record is read before any result exists, from the sub-task alone.** In the QA streams a peer's address comes from a judge
  prompt that lists all six answers. A lead that is choosing a source has no answers yet. Source s's row is the sub-task's
  address in s's block, `[e_s ⊗ [ψ_q ; 1]]`: ψ_q is the lead model's question-only features (`q_mean`, label-free PCA, 64
  dimensions), the constant the source's overall track record. No report enters an address. (With the QA streams' design `qc`,
  whose rows also carry the report's features, the first choice is worse: 65.5% of sub-tasks against 67.2%.)
- **Feedback only for the source that was chosen.** A worker that was not asked produced nothing to verify, so nothing is written
  for it. One source per sub-task, from a cold start.
- **Verification is not the record.** Whether a report is accepted is never decided by the record, and the record is written with
  gold labels only. Two verifications are replayed: the lead's own check (`pipeline.review`, `team.checker`; the task's `lead_check`
  instruction: does the report meet what I expected? it sees the sub-task and the report, no evidence and not the whole task), which
  a deployed team can run; and the dataset's check (`team.dataset_check`: the gold answer before the commit), which is exact and
  shows how fast the record adapts when every failure is known at once.
- **Every report is written, not only the committed one.** A report that failed is what says its source was unreliable on this kind
  of sub-task. `write="final"` in `feedback_state.agent_team.replay` is the ablation that throws the failed reports away.

## The research team (`agent_team.yaml`)

MuSiQue (the answerable dev split, 2,417 questions) gives every multi-hop question its own
decomposition: 2 to 4 sub-questions with gold answers, later ones referring to earlier answers. `pipeline.team build` turns every
hop into one `subqa` event (6,404 sub-tasks), teacher-forced: references are filled with the gold answers, so a report is
verified against the hop's gold answer whatever was committed before. A task is solved when every sub-task's committed report
is right.

The pool is the six peers of the QA streams, each a worker with a tool (`configs/peers/searcher_*.yaml`, `tool:`), which is a
view of the stream (`configs/datasets/musique_hops_q.yaml`, `views:`):

| worker | tool | the supporting paragraph is retrieved |
|---|---|---:|
| Gemma-3-4B | closed book | – |
| Phi-4-mini, DeepSeek-Coder-V2-Lite | `local3`: BM25 top-3 over the question's own 20 paragraphs | 89.0% |
| Qwen2.5-Coder-7B | `local1`: BM25 top-1 over them | 70.6% |
| Llama-3.1-8B, R1-Distill-Qwen-7B (thinking) | `global3`: BM25 top-3 over all 21,100 paragraphs of the split | 76.2% |
| Qwen3-4B, the lead itself (`own_answer: true`) | `local3`, its own view of the stream | 89.0% |

A report is graded by the `subqa` rule (`configs/tasks/subqa.yaml`): token-F1 ≥ 0.5 against the gold answer, or a short report
(at most 20 words) that contains it; a worker answers in a sentence, and the lead needs the entity in it.

**The full table is the evaluation device, not the protocol.** Every worker reports on every sub-task once, offline, so that
any policy can be replayed exactly on the same events (`pipeline.team replay`). A policy only ever sees, and the record is only
ever written with, the reports of the sources it called; the calls per sub-task are what the team would pay.

Policies, who gets the sub-task: `memory` (the record's highest P(right), read from the sub-task), `success_counts` (Thompson
sampling on Beta success counts: a track record that ignores the sub-task), `random`. Each is replayed with the first report
committed as it is, with the lead's check and with the dataset's check (a report that fails sends the sub-task to the next source in the
same order, `team.calls` at most), and with the dataset's check when only the committed report is written.

## Run

```bash
bash run.sh configs/experiments/agent_team.yaml --smoke      # 16 tasks end to end, under outputs/smoke/
bash run.sh configs/experiments/agent_team.yaml              # the 2,417 tasks
bash run.sh configs/experiments/agent_team_<lead>.yaml        # another lead (the workers' reports are reused): qwen3_8b, qwen3_14b, qwen25, llama31, ministral, phi4
```

Steps: `questions` (the sub-task stream and its views; the benchmark's file is read from `data/musique_src/`, fetched from
Hugging Face `dgslibisey/MuSiQue` when missing), `peers` (each worker reports from its view), `streams`, `own` (the lead's own
answer joins as the last source), `direct` (the baseline without a team: the lead answers every whole question of `musique_tasks` directly), `verify` (the lead's
check of every report), `features`, `team`.

## Outputs

```
data/musique_hops_q/{test.jsonl, agents/<view>.jsonl, manifest.json}   the sub-tasks: the lead's view and the tools' views
data/musique_hops_answers/<worker>/                                     the workers' reports
data/musique_team/test.jsonl, data/musique_team+q3_4b/test.jsonl        the stream, and with the lead's own answer
outputs/review/q3_4b/musique_team+own/lead_check/verdicts/              the lead's verdict and critique on every report
outputs/features/q3_4b/musique_team+own/                                the lead model's features (the question-only part is the address)
outputs/eval/q3_4b/musique_team+own/team/{team.json, team.md}           every choice, with and without the lead's check; the check's quality per source
```

## Results

Qwen3-4B leading the MuSiQue team, 2026-09-19 (`outputs/eval/q3_4b/musique_team+own/team/team.md`; mean of three stream orders, every choice from a cold
start; two sources per sub-task at most).

| who gets the sub-task | verification before the commit | worker calls | sub-task accuracy | task accuracy | first source right | first source right along the stream (eighths) |
|---|---|---:|---:|---:|---:|---|
| memory | nothing: the first report is committed | 1.00 | 67.2 | 36.9 | 67.2 | 62 65 67 68 69 69 69 69 |
| memory | the lead's check; rejected -> the next source | 1.16 | 69.1 | 39.9 | 68.3 | 61 66 70 69 70 70 69 70 |
| memory | the dataset's check; failed -> the next source | 1.30 | 78.0 | 53.8 | 69.8 | 63 68 71 70 72 71 72 71 |
| memory | the dataset's check; only the committed report is written | 1.38 | 76.7 | 51.8 | 61.8 | 59 63 63 62 62 62 62 61 |
| success_counts | nothing: the first report is committed | 1.00 | 62.2 | 32.1 | 62.2 | 60 62 62 61 64 62 64 62 |
| success_counts | the lead's check; rejected -> the next source | 1.12 | 64.5 | 35.3 | 62.8 | 61 64 64 62 64 62 64 62 |
| success_counts | the dataset's check; failed -> the next source | 1.38 | 74.5 | 48.9 | 62.0 | 62 63 62 61 62 62 63 61 |
| success_counts | the dataset's check; only the committed report is written | 1.38 | 74.1 | 48.1 | 61.9 | 61 63 63 60 63 62 63 61 |
| random | nothing: the first report is committed | 1.00 | 54.0 | 23.5 | 54.0 | 53 56 54 54 55 52 54 53 |
| random | the lead's check; rejected -> the next source | 1.17 | 56.9 | 26.4 | 54.0 | 53 56 54 54 55 52 54 53 |
| random | the dataset's check; failed -> the next source | 1.46 | 70.2 | 42.6 | 54.0 | 53 56 54 54 55 52 54 53 |
| random | the dataset's check; only the committed report is written | 1.46 | 70.2 | 42.6 | 54.0 | 53 56 54 54 55 52 54 53 |

The lead's check, over every source's report: rejects 17.3%, catches 30.3% of the wrong reports, rejects 6.2% of the right ones (per source in team.md).

- The choice of source is where the record pays: 36.9% of tasks with its choice, 32.1% with success counts, 23.5% at random (one call, first report committed).
- The lead's own check adds to it cheaply (39.9% at 1.16 worker calls): conservative, and the sub-task goes to a source with another tool.
- With the dataset's check the same loop reaches 53.8% at 1.30 calls (success counts 48.9% at 1.38, random 42.6% at 1.46).
- How fast it adapts: the record's first source is right on 63% of the first eighth of the stream and 72% from the third on; success counts stay near 62%.
- The failed reports are what it learns from: written with the committed report only, the record's first source is right on 61.8% instead of 69.8%
  (no better than success counts), and the tasks solved fall to 51.8%.

The dataset's check against the sources a sub-task may go to: sub-task / task accuracy (worker calls used). Every order ends at "any source right"; the record's
order gets there with fewer sources and fewer calls.

| sources allowed | memory (every report written) | memory (committed report only) | success counts | random |
|---:|---:|---:|---:|---:|
| 1 | 67.2 / 36.9 (1.00) | 67.2 / 36.9 (1.00) | 62.2 / 32.1 (1.00) | 54.0 / 23.5 (1.00) |
| 2 | 78.0 / 53.8 (1.30) | 76.7 / 51.8 (1.38) | 74.5 / 48.9 (1.38) | 70.2 / 42.6 (1.46) |
| 3 | 81.8 / 60.4 (1.52) | 80.4 / 57.3 (1.65) | 79.8 / 57.0 (1.62) | 77.2 / 53.0 (1.76) |
| 4 | 83.8 / 64.0 (1.69) | 83.3 / 63.0 (1.81) | 82.4 / 61.5 (1.82) | 81.4 / 59.9 (1.99) |
| 5 | 85.3 / 66.9 (1.85) | 85.0 / 66.4 (1.96) | 84.7 / 65.7 (2.00) | 83.9 / 64.1 (2.17) |
| 6 | 86.2 / 68.6 (1.98) | 86.1 / 68.4 (2.09) | 86.1 / 68.3 (2.14) | 85.7 / 67.4 (2.33) |
| 7 | 86.9 / 69.9 (2.12) | 86.9 / 69.9 (2.19) | 86.9 / 69.9 (2.31) | 86.9 / 69.9 (2.48) |

- Tried and removed (2026-09-19): the record as the verifier (verification should not rely on the memory); a fact-checking 4B evaluator with one revision by the same
  worker (it rejected 25% of the right reports, a revision broke 39% of those, and the record's result fell to 34.3%).
- The first version of the lead's check also showed it the whole task; the 4B model then held right sub-answers against the task's question and rejected 27% of them.

### Qwen3-8B as the lead (`agent_team_qwen3_8b.yaml`)

The same workers and reports; the lead's own answers, its check and the record's addresses are Qwen3-8B's (`outputs/eval/qwen3_8b/musique_team+own/team/team.md`).
Qwen3-8B alone is right on 68.4% of sub-tasks and solves 41.9% of the tasks, more than any worker (32.9%).

| who gets the sub-task | verification before the commit | worker calls | sub-task accuracy | task accuracy | first source right | committed from the lead itself |
|---|---|---:|---:|---:|---:|---:|
| memory | nothing: the first report is committed | 1.00 | 68.8 | 39.3 | 68.8 | 36% |
| memory | the lead's check; rejected -> the next source | 1.13 | 69.7 | 40.8 | 68.7 | 32% |
| memory | the dataset's check; failed -> the next source | 1.29 | 78.6 | 54.8 | 70.7 | 31% |
| memory | the dataset's check; only the committed report is written | 1.36 | 77.9 | 53.0 | 63.6 | 20% |
| success_counts | nothing: the first report is committed | 1.00 | 67.4 | 40.6 | 67.4 | 90% |
| success_counts | the lead's check; rejected -> the next source | 1.08 | 68.6 | 41.8 | 67.8 | 89% |
| success_counts | the dataset's check; failed -> the next source | 1.32 | 76.3 | 52.5 | 68.0 | 67% |
| success_counts | the dataset's check; only the committed report is written | 1.39 | 74.6 | 49.1 | 61.2 | 37% |
| random | nothing: the first report is committed | 1.00 | 54.6 | 24.2 | 54.6 | 14% |
| random | the lead's check; rejected -> the next source | 1.16 | 57.3 | 26.9 | 54.6 | 16% |
| random | the dataset's check; failed -> the next source | 1.45 | 70.8 | 43.4 | 54.6 | 16% |
| random | the dataset's check; only the committed report is written | 1.45 | 70.8 | 43.4 | 54.6 | 16% |

| sources allowed | memory (every report written) | memory (committed report only) | success counts | random |
|---:|---:|---:|---:|---:|
| 1 | 68.8 / 39.3 (1.00) | 68.8 / 39.3 (1.00) | 67.4 / 40.6 (1.00) | 54.6 / 24.2 (1.00) |
| 2 | 78.6 / 54.8 (1.29) | 77.9 / 53.0 (1.36) | 76.3 / 52.5 (1.32) | 70.8 / 43.4 (1.45) |
| 3 | 82.1 / 61.1 (1.50) | 80.5 / 57.7 (1.63) | 80.5 / 58.8 (1.55) | 77.8 / 53.8 (1.75) |
| 4 | 84.1 / 64.6 (1.67) | 83.5 / 63.9 (1.77) | 82.7 / 62.1 (1.75) | 81.8 / 60.4 (1.97) |
| 5 | 85.7 / 67.5 (1.81) | 85.0 / 66.2 (1.94) | 84.8 / 66.1 (1.93) | 84.2 / 64.6 (2.15) |
| 6 | 86.3 / 68.7 (1.95) | 86.2 / 68.8 (2.08) | 85.9 / 67.9 (2.09) | 85.8 / 67.7 (2.31) |
| 7 | 87.0 / 70.0 (2.08) | 87.0 / 70.0 (2.18) | 87.0 / 70.0 (2.23) | 87.0 / 70.0 (2.45) |

- With this lead a single call gains nothing at the task level: the lead itself is the strongest source, success counts learn to keep 90% of the sub-tasks for it (40.6% of tasks),
  and the record's choice is right on slightly more sub-tasks (68.8% against 67.4%) but solves fewer tasks (39.3%): it spreads sub-tasks over more sources, so its errors fall into more tasks.
- With the dataset's check the record's order is ahead again: 54.8% of tasks with two sources against 52.5% for success counts and 43.4% at random, and it converges with fewer sources
  (82.1% of sub-tasks with three against 80.5% and 77.8%). Writing every report matters here too (70.7% against 63.6% for the first source).
- The lead's check (Qwen3-8B): rejects 16.5% of reports, catches 29.7% of the wrong ones, rejects 5.5% of the right ones.

<!-- leads:begin -->
### Every lead

The same team with each lead (the workers' reports are shared; the lead's own answers, its check and the record's addresses are its own). Tasks solved, %; two sources per sub-task at most.
Baselines without a team: *direct* = the lead answers the whole question in one go (the question's paragraphs ranked by BM25, the first 8,000 characters; the final answer is graded);
*every sub-task itself* = the lead works through the benchmark's sub-tasks alone with its search tool (every sub-task must be right, as for the team).

| lead | direct | every sub-task itself | memory, first report | success counts, first report | memory + lead's check | success counts + lead's check | memory + dataset's check | success counts + dataset's check | random + dataset's check | memory + dataset's check, committed report only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B (`agent_team.yaml`) | 35.7 | 35.3 | 36.9 | 32.1 | 39.9 | 35.3 | 53.8 | 48.9 | 42.6 | 51.8 |
| Qwen3-8B (`agent_team_qwen3_8b.yaml`) | 42.3 | 41.9 | 39.3 | 40.6 | 40.8 | 41.8 | 54.8 | 52.5 | 43.4 | 53.0 |
| Qwen3-14B (`agent_team_qwen3_14b.yaml`) | 39.6 | 37.7 | 38.9 | 35.5 | 41.4 | 39.5 | 54.3 | 50.4 | 43.0 | 52.9 |
| Qwen2.5-7B (`agent_team_qwen25.yaml`) | 37.4 | 33.5 | 37.5 | 32.0 | 39.4 | 35.3 | 51.9 | 49.5 | 42.8 | 51.5 |
| Llama-3.1-8B (`agent_team_llama31.yaml`) | 35.4 | 36.0 | 38.9 | 33.4 | 39.3 | 38.9 | 55.7 | 50.7 | 43.4 | 53.1 |
| Ministral-8B (`agent_team_ministral.yaml`) | 34.8 | 41.8 | 41.6 | 40.2 | 40.8 | 42.9 | 55.0 | 53.2 | 43.7 | 52.9 |
| phi-4 (`agent_team_phi4.yaml`) | 40.3 | 34.9 | 40.0 | 31.8 | 42.6 | 37.0 | 54.2 | 50.5 | 42.8 | 52.7 |

Convergence with the dataset's check: sub-task accuracy of memory / success counts / random against the sources a sub-task may go to; every order ends at the ceiling.

| lead | 1 source | 2 | 3 | 4 | ceiling (all 7) |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 67.2 / 62.2 / 54.0 | 78.0 / 74.5 / 70.2 | 81.8 / 79.8 / 77.2 | 83.8 / 82.4 / 81.4 | 86.9 |
| Qwen3-8B | 68.8 / 67.4 / 54.6 | 78.6 / 76.3 / 70.8 | 82.1 / 80.5 / 77.8 | 84.1 / 82.7 / 81.8 | 87.0 |
| Qwen3-14B | 68.9 / 65.1 / 54.3 | 78.2 / 75.7 / 70.8 | 82.4 / 80.4 / 77.8 | 84.1 / 82.9 / 81.8 | 87.1 |
| Qwen2.5-7B | 67.5 / 61.9 / 54.0 | 76.9 / 74.7 / 70.4 | 82.0 / 79.6 / 77.5 | 84.1 / 82.5 / 81.6 | 86.9 |
| Llama-3.1-8B | 68.7 / 63.1 / 54.3 | 79.1 / 75.4 / 70.8 | 82.6 / 80.8 / 77.9 | 84.4 / 83.2 / 82.0 | 87.2 |
| Ministral-8B | 69.8 / 67.7 / 54.5 | 78.8 / 77.1 / 71.0 | 82.4 / 81.3 / 78.0 | 84.1 / 83.4 / 82.1 | 87.1 |
| phi-4 | 69.1 / 62.2 / 53.9 | 78.3 / 75.4 / 70.5 | 82.1 / 80.4 / 77.7 | 84.6 / 83.3 / 81.7 | 87.2 |

| lead | its check rejects | wrong reports caught | right reports rejected | first source right: memory, every report written | memory, committed report only | success counts, every report written | success counts, committed report only | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 17.3 | 30.3 | 6.2 | 69.8 | 61.8 | 62.0 | 61.9 | 54.0 |
| Qwen3-8B | 16.5 | 29.7 | 5.5 | 70.7 | 63.6 | 68.0 | 61.2 | 54.6 |
| Qwen3-14B | 22.9 | 39.6 | 8.8 | 70.3 | 64.1 | 65.2 | 59.6 | 54.3 |
| Qwen2.5-7B | 32.0 | 49.7 | 16.8 | 68.1 | 62.1 | 62.4 | 61.7 | 54.0 |
| Llama-3.1-8B | 38.0 | 53.6 | 24.9 | 70.7 | 63.5 | 63.2 | 61.9 | 54.3 |
| Ministral-8B | 36.3 | 51.3 | 23.9 | 71.7 | 63.8 | 66.8 | 61.2 | 54.5 |
| phi-4 | 22.1 | 39.1 | 7.5 | 70.9 | 62.7 | 63.3 | 60.5 | 53.9 |

<!-- leads:end -->

## What this is not yet

- The lead's check only catches reports that visibly fail (no answer, the wrong kind of thing); a plausible wrong name passes.
- The plan is the benchmark's (gold decomposition) and the chain is teacher-forced. Online, the lead's committed result feeds the
  next sub-question. Not built.
- The record is written from gold labels per sub-task. Where only a task's final outcome can be checked, the labels would have to
  be shared out over its sub-tasks. Not built.
- One benchmark, and a pool whose sources are close to one another on every kind of sub-question.
