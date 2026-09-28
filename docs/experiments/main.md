# main: the frozen Qwen3-4B reference rows

```bash
bash run.sh configs/experiments/main.yaml --smoke     # 48 events, every step
bash run.sh configs/experiments/main.yaml             # the whole capability-supported and capability-challenging streams
```

Qwen3-4B is both the frozen judge whose hidden states address the record and the central model that answers. The record
is fit on `address_fit` (PCA addresses, label-free) and run read-before-write along `capability_supported` and `capability_challenging` in the order
`shuffled0`. The model answers every event in four conditions.

| step | what runs | where it lands |
|---|---|---|
| features | `pipeline.features` on address_fit, capability_supported, capability_challenging | `outputs/features/q3_4b/<stream>/` |
| record | `pipeline.record` per evaluated stream | `outputs/record/q3_4b/<stream>/shuffled0.fit-address_fit.jsonl` + `.quality.json` |
| evaluate | `pipeline.evaluate` per stream and condition, vLLM | `outputs/eval/q3_4b/<stream>/<condition>/` |
| table | `pipeline.table` | `outputs/tables/main.md` |

Results (whole streams, thinking off, 768 new tokens, greedy):

| stream | tilt | peers | solo | swap | record AUC / favourite right on mixed events |
|---|---:|---:|---:|---:|---|
| capability_supported (4,319 events) | 76.6 | 74.6 | 69.2 | 69.3 | 0.92 / 91% |
| capability_challenging (17,403 events) | 74.0 | 69.2 | 67.8 | 64.2 | 0.92 / 83% |

`swap` gives the record's estimates to the wrong peers (by rank); it falls below `peers`, so the gain of `tilt` comes from
what the record knows, not from tilting attention as such.

One note on the conditions: `peers` and `solo` run on vLLM's default attention backend, `tilt` and `swap` on the patched
Triton kernels, so `tilt − peers` also contains the (small) numerical difference between the two kernels.

## Reproducibility (checked 2026-09-13)

- **Record.** The PCA addresses use a randomized low-rank SVD. Since 2026-09-13 it is seeded, so a record rebuilt on the
  same kind of device is identical row for row (checked: two rebuilds of capability_supported, 4,319 of 4,319 rows identical). The
  stored records were built unseeded and cannot be regenerated bit for bit; a rebuild differs in the estimates by 0.009 on
  average and matches in quality (AUC 0.9233 against 0.9232, favourite right on mixed events 90.8% against 91%) and in every
  prompt.
- **Prompts.** The central model's system prompts are exactly those of the stored records. A sentence against long
  reasoning that was added on 2026-09-09 for thinking mode is no longer part of them.
- **Grading.** Every answer is graded by its stream's rule, the central model's included: exact match for math, token-F1
  ≥ 0.5 for reading, the hidden tests for code, option match on the capability-challenging tasks. Until 2026-09-14 the central model's
  reading answers were graded by exact match while the peers' labels used F1 ≥ 0.5; the stored evaluations were regraded
  with `python -m pipeline.evaluate --regrade --output <dir>` (the generations are unchanged), which raised the
  capability-supported rows by 8–10 points (Advisors + memory 67.2 → 76.6, Question + Peers 64.8 → 74.6, No consultation 60.5 → 69.2) and left capability-challenging as it was.
- **Evaluation.** Re-evaluating the stored capability_supported record in the tilt condition gave 67.08 against the stored 67.19 (both
  under the exact-match rule of the time): the same verdict on 4,288 of 4,319 events. vLLM's batched greedy decoding is not bit-deterministic, so expect differences of
  about 0.1 point between runs of the same condition; compare conditions within one run of an experiment where it matters.
