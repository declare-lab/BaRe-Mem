"""The central model's reading line and the pieces around it: the estimate, the decision, the own answer in the stream."""
import json

import numpy as np

from feedback_state.permutations import random_order
from feedback_state.reading_line import ReadingLine
from pipeline import streams
from pipeline.combination import PEERS_MEMORY, QUESTION_ALONE, combine
from pipeline.evaluate import question_only_rows
from pipeline.record import order_of, prompt_slots


def test_the_estimate_is_the_least_squares_solution_with_the_prior():
    line = ReadingLine(prior=(0.5, 0.0), lam=1.0)
    for trust, own, correct in ((1.0, 0.3, 1), (1.0, 0.3, 0), (0.0, 0.8, 0)):
        line.write(trust, own, correct)
    assert np.allclose(line.estimate(), (0.5, 0.4))                              # a worked check, by hand

    rng = np.random.default_rng(0)
    events = [(float(rng.uniform()), float(rng.uniform()), int(rng.integers(2))) for _ in range(200)]
    line = ReadingLine(prior=(0.5, 0.0), lam=1.0)
    for trust, own, correct in events:
        line.write(trust, own, correct)
    U = np.array([[t, t - 1.0] for t, _, _ in events])
    z = np.array([y - (1.0 - t) * k for t, k, y in events])
    batch = np.linalg.solve(np.eye(2) + U.T @ U, np.array([0.5, 0.0]) + U.T @ z)
    assert np.allclose(line.estimate(), batch)                                   # updating event by event = solving once over all


def test_the_switch_is_where_reading_and_the_own_answer_are_worth_the_same():
    line = ReadingLine(prior=(0.9, 0.3))                                         # no events: the estimate is the prior
    assert np.isclose(line.switch(0.6), 0.5) and np.isclose(line.value(0.5, 0.6), 0.6)
    assert not line.reads(0.49, 0.6) and line.reads(0.51, 0.6)
    assert line.switch(0.4) < line.switch(0.6) < line.switch(0.8)                # a more capable own answer needs more trust
    assert ReadingLine(prior=(0.9, 0.3)).reads_when(0.6) == "T >= 0.50"          # gains with good evidence, loses with bad
    assert ReadingLine(prior=(0.5, 0.1)).reads_when(0.7) == "never"              # worse at every trust
    assert ReadingLine(prior=(0.9, -0.2)).reads_when(0.6) == "always"            # even bad evidence helps
    assert ReadingLine(prior=(0.4, -0.2)).reads_when(0.6) == "T <= 0.50"         # helps only while the evidence is doubtful


def _row(pos, rid, trust, own_prob, own_correct, task="mcqa"):
    return {"pos": pos, "id": rid, "task_type": task, "source": "s", "memory_prob": [trust, trust / 2], "peer_correct": [1, 0],
            "own_prob": own_prob, "own_correct": own_correct}


def test_combination_reads_before_it_writes():
    rows = [_row(1, "b", 0.9, 0.6, 0), _row(0, "a", 0.9, 0.6, 0)]               # out of order on purpose
    peers_memory = {rid: {"correct": 1, "generation": f"advisors+memory {rid}"} for rid in "ab"}
    question_alone = {rid: {"correct": 0, "generation": f"alone {rid}"} for rid in "ab"}
    out, summary = combine(rows, peers_memory, question_alone, prior=(0.5, 0.0), lam=1.0)

    assert [o["id"] for o in out] == ["a", "b"]                                  # the record's order
    assert (out[0]["rho"], out[0]["delta"], out[0]["choice"]) == (0.5, 0.0, QUESTION_ALONE)   # the first event sees only the prior
    assert out[1]["choice"] == PEERS_MEMORY and out[1]["generation"] == "advisors+memory b" and out[1]["correct"] == 1
    assert summary["own_labels_mismatched"] == 0 and summary["reading_line"]["mcqa"]["events"] == 2


def test_low_trust_keeps_question_alone():
    rows = [_row(0, "a", 0.1, 0.6, 1)]
    out, _ = combine(rows, {"a": {"correct": 0, "generation": "p"}}, {"a": {"correct": 1, "generation": "q"}}, prior=(0.5, 0.0), lam=1.0)
    assert out[0]["choice"] == QUESTION_ALONE and out[0]["correct"] == 1


def test_a_question_only_evaluation_joins_a_stream_as_its_last_answer(tmp_path):
    base = tmp_path / "base.jsonl"
    base.write_text("".join(json.dumps({"id": i, "peer_responses": {"peer_0": "x", "peer_1": "y"}, "peer_correct": {"peer_0": 1.0, "peer_1": 0.0}}) + "\n"
                            for i in ("a", "b")))
    ev = tmp_path / "solo"
    ev.mkdir()
    (ev / "eval_metrics.json").write_text(json.dumps({"central_model": "/models/Qwen3-4B", "condition": "solo", "mode": "solo", "max_new_tokens": 768}))
    (ev / "generations.jsonl").write_text(json.dumps({"id": "a", "generation": "own answer", "correct": 1}) + "\n")
    out = tmp_path / "with_own/test.jsonl"
    streams.main(["add", "--base", str(base), "--eval", str(ev), "--out", str(out)])

    rows = [json.loads(line) for line in out.open()]
    assert [r["id"] for r in rows] == ["a"]                                      # an event without the own answer is dropped
    assert rows[0]["peer_responses"]["peer_2"] == "own answer" and rows[0]["peer_correct"]["peer_2"] == 1.0
    assert rows[0]["peer_metadata"]["peer_2"]["model"] == "Qwen3-4B"
    assert json.load(open(out.parent / "manifest.json"))["dropped_for_missing_answers"] == {"Qwen3-4B": 1}


def test_the_own_answer_is_recorded_but_never_shown():
    for seed in (1, 7, 12345):
        assert prompt_slots(7, 6, seed) == random_order(6, seed)                 # the six peers, permuted as a six-answer record would
        assert prompt_slots(6, None, seed) == random_order(6, seed)
    assert sorted(prompt_slots(7, 2, 3)) == [0, 1, 3, 4, 5, 6]


def test_a_question_only_run_needs_no_record():
    records = [{"id": f"e{i}", "task_type": "math", "source": "s", "problem": f"q{i}", "peer_correct": {"peer_0": 1.0, "peer_1": 0.0}}
               for i in range(5)]
    rows = question_only_rows(records, "shuffled0", limit=3)
    assert [r["pos"] for r in rows] == [0, 1, 2]
    assert [r["id"] for r in rows] == [f"e{t}" for t in order_of(3, "shuffled0")]   # the events a 3-event record holds, in its order
    assert all("Peer answers" not in r["messages_solo"][1]["content"] and r["peer_correct"] == [1, 0] for r in rows)
