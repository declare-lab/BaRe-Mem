# misleading: the six peers answer with misleading but relevant solutions

```bash
python datasets/download.py                                                      # the released answers and streams -> data/
bash run.sh configs/experiments/misleading.yaml --smoke                          # 48 events of capability_supported_misleading_p050, every step
bash run.sh configs/experiments/misleading.yaml                                  # the sweep: datasets [misleading_rates]
bash run.sh configs/experiments/misleading.yaml --set "datasets=[capability_challenging_misleading_p100]"   # one dataset
```

Does the record survive peers that are wrong on purpose? Every peer answers every event with a confident, on-topic,
verified-wrong solution once: the answers datasets `capability_supported_misleading` and `capability_challenging_misleading` (`peers` step; released with the
datasets, so it is skipped after downloading). Each misleading dataset (`configs/datasets/`) names a
base stream and a regime that chooses which of those answers replace the honest ones (`streams` step), and the
main pipeline runs unchanged on the result.

## How a misleading answer is produced (`pipeline/peers.py --mode misleading`)

Asking once is not enough: in a pilot, 44% of the answers a peer gave when told to be wrong were correct anyway, some
announced the trick, some refused, some programs were tables of hard-coded outputs. So every event is a small search
(rules in `feedback_state/adversarial.py`):

| step | what happens |
|---|---|
| ask | the task's peer prompt plus the instruction to be plausibly wrong; the gold answer is given only so it can be avoided |
| grade | the streams' own rule: token-F1 ≥ 0.5 for reading, exact match for math, hidden tests for code |
| check | kept only if graded wrong, in the task's answer format, free of meta-commentary, refusals and repetition loops, not naming the gold, a reading answer taken from the passage, a real program, a closed think block |
| retry | the failures again at temperature 0.7 then 1.0, told what was wrong; on the last attempt a math peer that keeps reaching the gold is given a wrong value (one of its own intermediate results) to arrive at |
| rewrite | only if all that fails, for math / multiple choice / yes-no: the conclusion is replaced and re-graded, flagged `forced`; never for an answer with no argument besides its final line (at least 20 characters) or a looping one, which would turn an empty reply into a bare wrong label |

Budgets: misleading answers get the honest budgets (math 512, reading 256, code 768 tokens) except on the short-answer
capability-challenging tasks, where 96 tokens cut most arguments off before the final line; there they get 256 and are asked to argue in
one or two sentences. They still come out longer than the honest answers (on one capability-challenging shard of gemma-3-4b-it: 330 against
170 characters on yes/no, 429 against 160 on multiple choice, 379 against 137 on short answer), a difference the central
model could in principle pick up on.

An event with no usable answer keeps the honest answer in every stream. `data/<stream>_misleading/<peer>/summary.shard*.json`
(and the dataset card on Hugging Face) reports each peer's acceptance rate, forced count, attempts and why the rest were
unusable.

Two generation faults were found and fixed on 2026-09-13/14, and the answers of the affected peers regenerated: vLLM
0.8.5's V0 prefix caching corrupts DeepSeek-Coder-V2-Lite's MLA attention (about 30% of its answers, honest ones
included, came out as unrelated text or symbol runs; its model file now sets `prefix_caching: false`), and the engine
added a second start token to the rendered prompt of gemma-3, Llama-3.1 and both DeepSeek peers (peers are now fed the
template's token ids). The acceptance rules also reject answers that are not about the question (`off_topic`).

## Regimes and the ratio

A misleading dataset is a file in `configs/datasets/` that includes the template `_misleading.yaml` and sets `base` and
`regime`. The regime is one of two short forms, where the number is what the stream ends up with (only usable answers
count; shares are rounded to whole events):

- `pNNN`: that share of every peer's answers is misleading, a different set of events per peer, nested across rates
  (p050 poisons p025's events plus more). `p000` is the honest stream rebuilt through the same steps.
- `kN`: exactly N of the six peers are misleading on every event, a different set each time (a lying minority k1-k2
  against a lying majority k4-k5).

or a mapping: `{kind: fraction, rate: 1.0, peers: [1, 4]}` (Phi-4-mini and DeepSeek-Coder always lie, four peers honest),
`{kind: flip, at: 0.5, peers: [1, 4]}` (they turn misleading halfway through the record's order), `{kind: targeted, rate:
1.0}` (every peer misleading exactly where it was right). `drop_forced: true` leaves out answers whose conclusion was
rewritten. The released ones are the rates `p000`, `p025`, `p050`, `p075`, `p100` on both streams (group
`misleading_rates`). Each built stream has a `manifest.json` with the requested and realised ratio per peer, events by
number of misleading peers, forced and unavailable counts, and accuracy before and after.

A peer cannot go beyond its usable share, so the top rates reach less than asked. The released answers (2026-09-14):

| peer | usable, capability_supported | usable, capability_challenging | rewritten (of usable), capability_supported / capability_challenging |
|---|---:|---:|---:|
| gemma-3-4b-it | 71.3% | 92.0% | 21.8% / 8.4% |
| Phi-4-mini-instruct | 90.3% | 95.6% | 7.0% / 4.8% |
| Qwen2.5-Coder-7B-Instruct | 89.2% | 98.3% | 2.4% / 4.0% |
| Meta-Llama-3.1-8B-Instruct | 93.6% | 97.8% | 0.7% / 0.6% |
| DeepSeek-Coder-V2-Lite-Instruct | 68.6% | 94.5% | 16.0% / 6.3% |
| DeepSeek-R1-Distill-Qwen-7B | 66.5% | 90.9% | 18.7% / 11.5% |

So p025 and p050 are exact on both streams and p075 on capability_challenging. p075 on capability_supported reaches 71.9% overall because three peers
stop at their usable share. p100 reaches 79.9% on capability_supported and 94.8% on capability_challenging, the most these answers allow.

## Result (2026-09-14): frozen Qwen3-4B on the rate datasets

Accuracy (%) of the central model with the six answers in the prompt, with (`tilt`) and without (`peers`) the record's
attention tilt, and with No consultation (`solo`); the record is fit on each stream itself. A question-only prompt holds
no peer answers, so `solo` is the same for every misleading share: it is read from the base stream's result
(`Layout.eval_dataset`), not re-run.

| misleading share asked (reached: capability_supported / capability_challenging) | capability_supported tilt | capability_supported peers | capability_supported solo | capability_challenging tilt | capability_challenging peers | capability_challenging solo |
|---|---:|---:|---:|---:|---:|---:|
| honest (main experiment) | 76.6 | 74.6 | 69.2 | 74.0 | 69.2 | 67.8 |
| 0% | 76.8 | 74.6 | 69.2 | 74.0 | 69.2 | 67.8 |
| 25% | 75.3 | 74.3 | 69.2 | 70.3 | 64.0 | 67.8 |
| 50% | 74.3 | 73.6 | 69.2 | 67.9 | 58.1 | 67.8 |
| 75% (71.9 / 75.0) | 73.1 | 72.3 | 69.2 | 63.5 | 51.4 | 67.8 |
| 100% (79.9 / 94.8) | 72.2 | 71.5 | 69.2 | 50.2 | 45.8 | 67.8 |

The record gives misleading answers a lower estimate than honest ones (AUC 0.70-0.71 capability-supported, 0.77-0.86 capability-challenging),
without being told which answers are misleading. Full table: `outputs/tables/misleading.md`.

## Reading the table

`outputs/tables/misleading.md`: accuracy per regime under `tilt` (Advisors + memory), `peers` (Question + Peers) and `solo`
(No consultation, shared with the base stream), against the honest rows (`swap` is not run by
default: add it to `eval_conditions`); what the record makes of the misleading answers (mean estimate on honest vs misleading answers, the AUC of that
separation, how often its favourite is misleading); and the peers (ratio asked and reached, accuracy before and after,
forced count, usable share).

## Notes

- The record is fit on each misleading stream itself (`record.fit: self`), label-free, as it would be in deployment.
- DeepSeek-Coder-V2-Lite-Instruct runs with `VLLM_USE_V1=0` (set in `configs/models/deepseek_coder_v2_lite.yaml`): its MLA attention has no working
  chunked-prefill path in vLLM 0.8.5's V1 engine.
- Some misleading reading answers are partly right: the grader counts a correct fragment of a long gold answer as wrong
  when its word overlap is under 0.5.
