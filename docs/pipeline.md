# The pipeline

Reference for running and extending the code: the experiments, the stages behind `run.sh`, the registries, what an
experiment file can set, where results go, and the implementation checks. Setup and the main commands are in the
[README](../README.md).

## Experiments

| experiment | question | guide |
|---|---|---|
| `main` | the frozen Qwen3-4B with its record on the capability-supported and capability-challenging streams (the reference rows) | [experiments/main.md](experiments/main.md) |
| `families` | the record for other central models (Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, phi-4), each with its own record | [experiments/families.md](experiments/families.md) |
| `misleading` | peers that are misleading on purpose, at any misleading information ratio | [experiments/misleading.md](experiments/misleading.md) |
| `misleading_families` | the same misleading datasets for Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, phi-4 and Qwen3-14B | [experiments/misleading_families.md](experiments/misleading_families.md) |
| `combination` | BaRe-Mem: per question, Advisors + memory or No consultation, chosen by the reading line (ρ, δ); 0–100% misleading | [experiments/combination.md](experiments/combination.md) |
| `combination_families` | BaRe-Mem for Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, phi-4 and Qwen3-14B, each its own judge | [experiments/combination_families.md](experiments/combination_families.md) |
| `baselines` | multi-agent baselines on the same streams: Majority vote (advisors; advisors + own answer) and multi-agent debate (1 and 2 rounds; Debate + vote) | [experiments/baselines.md](experiments/baselines.md) |
| `baselines_families` | the baselines for Qwen3-8B, Qwen3-14B, Llama-3.1-8B, Ministral-8B, phi-4 and Qwen2.5-7B, each its own judge | [experiments/baselines_families.md](experiments/baselines_families.md) |
| `agent_team` | the record inside a lead-worker team on MuSiQue: the lead gives every sub-task to one source, in the order the record gives from the sub-task alone; no check, the lead's own check, or the dataset's check before the commit; against success counts and a random order | [experiments/agent_team.md](experiments/agent_team.md) |
| `agent_team_qwen3_8b`, `agent_team_qwen3_14b`, `agent_team_qwen25`, `agent_team_llama31`, `agent_team_ministral`, `agent_team_phi4` | the same team with another lead: its own answers, its check of the reports and its record; the workers' reports are reused | [experiments/agent_team.md](experiments/agent_team.md) |

## The stages

| stage | entry point | does |
|---|---|---|
| questions | `pipeline/team.py build` | a benchmark's own sub-tasks as a stream with one view per tool (`built: musique`: MuSiQue's hops, teacher-forced) |
| peers | `pipeline/peers.py` | a peer model answers every event of a stream, honestly or with verified misleading answers |
| streams | `pipeline/streams.py` | a derived stream: peers added, or answers replaced under a regime |
| own | `pipeline/evaluate.py` + `pipeline/streams.py add` | the central model's no-consultation answer joins the stream as one more answer |
| direct | `pipeline/evaluate.py` (No consultation) | the team baseline: the lead answers every whole task directly, alone |
| verify | `pipeline/review.py` | the lead's check of every report of a team's stream: does it meet what the lead expected of it? |
| features | `pipeline/features.py` | the frozen judge reads question + answers; its hidden states address the record |
| record | `pipeline/record.py` | the Bayesian record along the stream, read before write, and its quality |
| evaluate | `pipeline/evaluate.py` | the central model answers each event under a condition; answers graded |
| vote | `pipeline/vote.py` | majority votes over the peers' answers, with or without the central model's answer |
| combination | `pipeline/combination.py` | per event, Advisors + memory or No consultation, chosen by the reading line |
| team | `pipeline/team.py replay` | the lead's choice of source per sub-task (`feedback_state/agent_team.py`) replayed along a stream |
| table | `pipeline/table.py` | the experiment's result table |

Each entry point is a plain command with explicit paths (`python -m pipeline.<stage> --help`); `pipeline/run.py` derives
the paths from the config and schedules the jobs (the devices are `gpus:` in `configs/base.yaml`).

## Registering datasets, models, peers and tasks

Datasets, models, peers and task types are registered one YAML file each, and experiments refer to them by file name, so
adding one is adding a file (`python -m pipeline.registry` lists everything and checks every reference):

```
configs/datasets/<name>.yaml   kind stream: {path, peers: [peer names, in peer_0 ... order]}; kind answers: {base, mode};
                               kind misleading: {base, answers, regime};
                               or a group: {group: [names]}. `include: _misleading.yaml` pulls in a template.
configs/models/<name>.yaml     {hf_id, path (under paths.models_root), engine, env_vars, prefix_caching, ...}
configs/peers/<name>.yaml      {model: a registered model, reasoning, tool, ...}: one peer; a new peer is a new file. `tool:` names
                               the view of the stream the peer answers from (a stream's `views:`, e.g. what a searcher retrieves)
configs/tasks/<name>.yaml      {grader, agreement, context, instruction, max_tokens}: one task type. `grader:` names a grading rule
                               in feedback_state/tasks.py (GRADERS); the rest is what is not code: the central model's instruction,
                               the peers' answer budget, whether the passage is shown, when two answers agree in a vote. A new task
                               over an existing rule is a new file; a new rule is one function plus its file
```

For example a new regime, two peers that always lie, is `configs/datasets/capability_supported_saboteurs.yaml`:

```yaml
include: _misleading.yaml
base: capability_supported
regime: {kind: fraction, rate: 1.0, peers: [1, 4]}
```

and `--set "datasets=[capability_supported_saboteurs]"` runs it (the streams step builds it from the answers already generated).

## Adding an experiment

Copy the closest file in `configs/experiments/` and change what differs. What a file can set:

- `steps`: any of questions, peers, streams, own, direct, verify, features, record, evaluate, vote, combination, team, table.
- `central`: the models that answer (registered names).
- `datasets`: registered datasets or groups. The steps follow from their kinds: `peers` generates the answers datasets
  the named misleading datasets need, `streams` builds those, and features, record and evaluate run on each.
- `eval_conditions` (conditions are defined once under `conditions:`; a new setting, e.g. another γ, gets a new name,
  because results are stored by condition name). A condition with `mode: debate` answers after `round` debate rounds and
  reads its `previous` condition's answers; `vote_conditions` (`mode: vote`, optional `own`) are majority votes, step `vote`.
- `record`: design, dim, lam, order, fit (a dataset name, or `self`).
- `team`: dim, orders, checker, calls, dataset_check, direct (the lead's choice of source and the verification before the commit).
- `own_answer: true`: the central model's no-consultation answer joins every stream as one more answer, recorded like the
  peers' but kept out of the prompt (step `own`); `combination` then picks Advisors + memory or No consultation per event.
- `table`: rows (`models` or `regimes`), `reference` models, `deltas`, and `columns` (every column, in order).

A result is reused only if it was produced with the same settings: an evaluation stores its full condition, and the
runner stops with a clear message instead of silently reusing a result made with other settings.

## Where results go

```
data/<dataset>/                                          registered datasets: downloaded, generated answers (<peer>/), built streams
outputs/features/<model>/<stream>/                       the judge's features
outputs/record/<model>/<stream>/<order>.fit-<fit>.jsonl  the record (+ .quality.json)
outputs/eval/<model>/<stream>/<condition>/               generations.jsonl + eval_metrics.json
<stream>+own                                             with own_answer: the features, record and evaluations of that stream
                                                         plus the central model's own answer (No consultation stays on <stream>)
outputs/tables/<experiment>.md                           result tables
outputs/runs/<experiment>/                               the resolved config and every command
logs/<experiment>/<job>.log
```

Experiments share this layout, so a result computed once (the main experiment's rows) is read by every other experiment
that needs it.

## Checks

```bash
python -m pipeline.registry                                       # lists every registered dataset, model, peer and task and checks every reference
PYTHONPATH=. python -m analysis.check_vllm_tilt --help            # the vLLM tilt kernel against the HF attention reference
PYTHONPATH=. python -m analysis.check_kalman_numerics --help      # the record's covariance over a whole stream: symmetric, positive definite
PYTHONPATH=. python -m analysis.family_prompt_check --help        # CPU: the tilt covers the peer blocks under a model's chat template
```

## Repository layout

```
run.sh               the one command
configs/             base.yaml (paths, defaults, conditions), datasets/, models/, peers/, tasks/ (the registries), experiments/
pipeline/            the stages, the runner (run.py), config loading, the registry and the output layout
feedback_state/      the method: kalman_memory.py (the record), addresses.py, memory_runtime.py, reading_line.py (consult or
                     answer alone), attn_bias.py + vllm_attn_bias.py (the tilt), judge_prompt.py, judge_features.py,
                     memory_generator.py (prompts, grading), tasks.py, adversarial.py (misleading answers and regimes),
                     baselines.py (debate), answer_groups.py (votes), agent_team.py (the lead-worker loop)
analysis/            the sparse-feedback analysis, its summary and the implementation checks
data/builders/       code grading (executing programs against their tests)
datasets/            download.py and the manifest of the released files
docs/                pipeline.md (this file) and experiments/, one guide per experiment
tests/unit/          CPU tests
```
