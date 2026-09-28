# BaRe-Mem: Advisors + memory or No consultation, chosen per event

Question: with a fixed central model (Qwen3-4B, then Qwen3-8B), can it keep accuracy up as the peers turn misleading, by
choosing per event between Advisors + memory and No consultation?

## Setup

- **Seven answers per event.** The six peers of the stream, plus the central model's no-consultation answer. One record
  estimates all seven alike, addressed by question and answer; the central model's own answer is recorded but never shown
  in the prompt. Each central model is its own judge: its features, record and answers are its own.
- **Advisors + memory.** The central model answers with the question and the six peers in the prompt, attention tilted
  (γ = 3) by that same record. **Question + Peers** is the same prompt without the tilt.
- **The reading line.** On each event, T is the record's top estimate among the six peers and κ its estimate of the
  no-consultation answer. Advisors + memory is worth A(T) = T·ρ + (1 − T)(κ − δ), No consultation κ. With u = [T, T − 1] and
  z = y − (1 − T)κ (y: whether Advisors + memory was right), (ρ, δ) = P⁻¹q with P = I + Σuuᵀ and q = (0.5, 0) + Σu·z: a 2×2
  state per task type, separate from the record's Λ, read before write.
- **BaRe-Mem.** The final answer is Advisors + memory when A(T) ≥ κ, otherwise No consultation. The lines cross at
  T* = δ / (ρ + δ − κ): with δ > 0 and ρ > κ Advisors + memory wins for T ≥ T*; the table says where it wins in every case.
- **Datasets.** `misleading_rates`: capability-supported and capability-challenging at 0, 25, 50, 75 and 100% misleading answers.

The reading line and its update, (ρ, δ) = P⁻¹q, are implemented in `feedback_state/reading_line.py`.

## Run

```bash
bash run.sh configs/experiments/combination.yaml --smoke     # 48 events of capability_supported_misleading_p050, into outputs/smoke/
bash run.sh configs/experiments/combination.yaml             # every rate; finished work is skipped
```

Steps: `own` (each central model's no-consultation answers, shared with the base streams, join each stream as
`data/<dataset>+<model>/`), `features` and `record` on the seven answers, `evaluate` (Advisors + memory, Question + Peers),
`combination`, `table`.

## Outputs

```
outputs/features/<model>/<dataset>+own/
outputs/record/<model>/<dataset>+own/shuffled0.fit-self.jsonl     rows carry own_prob and own_correct
outputs/eval/<model>/<dataset>+own/tilt/                          Advisors + memory
outputs/eval/<model>/<dataset>+own/peers/                         Question + Peers
outputs/eval/<model>/<dataset>+own/combination/                   BaRe-Mem: the choice per event and eval_metrics.json: accuracy,
                                                                  share_peers_memory, peers_memory_accuracy,
                                                                  question_alone_accuracy, reading_line (ρ, δ per task
                                                                  type), by_trust
outputs/eval/<model>/<base stream>/solo/                          No consultation
outputs/tables/combination.md
```

## Results (2026-09-15)

Qwen3-4B, accuracy % (job 20260915-021113; `outputs/tables/combination.md`, per event and per trust band in
`outputs/eval/q3_4b/<dataset>+own/combination/`). "Took Advisors + memory" is the share of events where BaRe-Mem chose it.

| misleading | Capability-supported: Advisors + memory | Question + Peers | No consultation | BaRe-Mem | took Advisors + memory | Capability-challenging: Advisors + memory | Question + Peers | No consultation | BaRe-Mem | took Advisors + memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 76.7 | 74.6 | 69.2 | 76.2 | 86% | 73.8 | 69.2 | 67.8 | 75.0 | 86% |
| 25% | 75.5 | 74.2 | 69.2 | 75.4 | 85% | 70.4 | 64.0 | 67.8 | 72.4 | 78% |
| 50% | 74.3 | 73.7 | 69.2 | 74.1 | 85% | 67.5 | 58.1 | 67.8 | 70.8 | 69% |
| 75% | 73.1 | 72.4 | 69.2 | 72.7 | 86% | 62.5 | 51.4 | 67.8 | 70.2 | 53% |
| 100% | 72.4 | 71.3 | 69.2 | 72.0 | 85% | 50.7 | 45.8 | 67.8 | 69.3 | 15% |

Qwen3-8B, its own features, record and answers (same layout under `outputs/eval/qwen3_8b/`):

| misleading | Capability-supported: Advisors + memory | Question + Peers | No consultation | BaRe-Mem | took Advisors + memory | Capability-challenging: Advisors + memory | Question + Peers | No consultation | BaRe-Mem | took Advisors + memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 78.1 | 75.7 | 72.9 | 77.4 | 86% | 74.2 | 67.9 | 65.6 | 74.9 | 89% |
| 25% | 77.0 | 75.1 | 72.9 | 76.6 | 85% | 70.1 | 61.6 | 65.6 | 71.4 | 80% |
| 50% | 76.3 | 74.3 | 72.9 | 76.1 | 85% | 66.9 | 54.3 | 65.6 | 70.0 | 70% |
| 75% | 74.8 | 73.6 | 72.9 | 74.6 | 84% | 61.0 | 46.7 | 65.6 | 68.7 | 52% |
| 100% | 74.0 | 72.3 | 72.9 | 73.7 | 83% | 46.2 | 40.0 | 65.6 | 67.7 | 14% |

- Capability-challenging: BaRe-Mem is above Advisors + memory and No consultation at every rate, for both models. From 0% to 100% it falls
  5.7 points (Qwen3-4B) and 7.2 (Qwen3-8B); Advisors + memory falls 23.1 and 28.0.
- Capability-supported: Advisors + memory stays above No consultation at every rate, and BaRe-Mem is within 0.5 (Qwen3-4B)
  and 0.7 (Qwen3-8B) points of it.
- The memory's tilt carries the reading: Question + Peers falls to 45.8 on capability-challenging at 100%; Advisors + memory is 4.6 points
  above it with honest peers and 9.4–11.1 points at 50–75%.
- The straight line over-states Advisors + memory at high trust: ρ̂ is above 1 for most task types (up to 1.48).
  Capability-supported at low trust BaRe-Mem keeps No consultation where Advisors + memory was a little better (0%, T 0.2–0.4:
  Advisors + memory right on 15.8%, No consultation on 10.0%, BaRe-Mem took Advisors + memory on 11% of those events).
