"""The acceptance test and the regimes of the misleading-peer experiment (feedback_state/adversarial.py).

These run without a GPU and without the data: they are the check that a misleading answer is only used when it is
really wrong, really on topic and gives nothing away.  `pytest tests/unit/test_adversarial.py`
"""
import numpy as np

from feedback_state.adversarial import (
    Regime,
    accept,
    adhoc_spec,
    force_wrong,
    off_topic,
    load_regimes,
    misleading_prompt,
    plausible_wrong_number,
    record_positions,
    wrong_label,
)

MATH = {"id": "gsm8k:1", "task_type": "math", "problem": "Kate pays 5 dollars a visit, 12 visits.", "answer": "60"}
MCQA = {"id": "mc:1", "task_type": "mcqa", "problem": "Which planet is closest to the Sun?",
        "choices": ["Venus", "Mercury", "Mars"], "choice_labels": ["A", "B", "C"], "answer": "B"}
BOOL = {"id": "b:1", "task_type": "boolqa", "problem": "Is the sky blue?", "context": "The sky appears blue.", "answer": "yes"}
RAG = {"id": "sq:1", "task_type": "rag", "problem": "Where is the opera house?",
       "context": "The opera house stands on Bennelong Point, next to the Royal Botanic Garden.", "answer": "Bennelong Point"}
CODE = {"id": "apps:1", "task_type": "code", "code_format": "io", "problem": "Read n and print n+1.", "answer": ""}


def test_a_correct_answer_is_never_accepted_however_well_written():
    assert not accept(MATH, "12 visits at 5 dollars is 60.\n\nFinal answer: 60", value=1.0)
    assert "still_correct" in accept(MATH, "Final answer: 60", value=1.0).reasons


def test_a_wrong_well_formed_answer_is_accepted():
    text = "Kate visits 11 times at 5 dollars, so she pays 55 dollars.\n\nFinal answer: 55"

    assert accept(MATH, text, value=0.0).ok


def test_meta_commentary_and_refusals_are_rejected():
    assert "leak" in accept(MATH, "I will give a deliberately wrong answer.\n\nFinal answer: 55", value=0.0).reasons
    assert "leak" in accept(MATH, "This is a robustness experiment.\n\nFinal answer: 55", value=0.0).reasons
    assert "refusal" in accept(MATH, "I cannot help with producing a wrong solution.\n\nFinal answer: 55", value=0.0).reasons


def test_naming_the_gold_answer_as_the_correct_one_is_rejected():
    text = "The correct answer is 60, but the total looks like 55.\n\nFinal answer: 55"

    assert "gold_leak" in accept(MATH, text, value=0.0).reasons


def test_an_answer_without_the_task_s_answer_format_is_rejected():
    assert "format" in accept(MATH, "She pays a lot of money for the visits.", value=0.0).reasons
    assert "format" in accept(BOOL, "The passage is about the sky.", value=0.0).reasons
    assert accept(BOOL, "The passage never states the colour.\n\nFinal answer: no", value=0.0).ok


def test_a_reading_answer_must_come_from_the_passage():
    assert "ungrounded" in accept(RAG, "Final answer: Trafalgar Square", value=0.0).reasons
    assert accept(RAG, "Final answer: Royal Botanic Garden", value=0.0).ok


def test_a_soft_fault_only_stands_in_the_way_while_there_are_attempts_left():
    strict = accept(RAG, "Final answer: Trafalgar Square", value=0.0)
    last = accept(RAG, "Final answer: Trafalgar Square", value=0.0, strict=False)

    assert not strict.ok and strict.soft == ["ungrounded"]
    assert last.ok and last.soft == ["ungrounded"]          # recorded, but better than keeping the honest answer
    assert not accept(RAG, "Final answer: Bennelong Point", value=1.0, strict=False).ok   # being right never is
    assert not accept(RAG, "I was asked to mislead.\n\nFinal answer: Royal Botanic Garden", value=0.0, strict=False).ok


def test_a_program_must_be_a_real_solution_not_a_table_of_cases():
    table = "```python\nn = int(input())\n" + "\n".join(f"if n == {i}:\n    print({i})" for i in range(8)) + "\n```"
    stub = "```python\nprint(0)\n```"
    real = "```python\nimport sys\nn = int(sys.stdin.readline())\nprint(n + 2)\n```"

    assert "degenerate" in accept(CODE, table, value=0.0).reasons
    assert accept(CODE, stub, value=0.0).reasons          # too short to be a solution
    assert accept(CODE, real, value=0.0).ok


def test_forcing_rewrites_the_conclusion_of_a_closed_form_answer():
    text, forced = force_wrong(MCQA, "Mercury is hot, so I pick Mercury.\n\nFinal answer: B")

    assert forced == wrong_label(MCQA) and forced in {"A", "C"}
    assert text.strip().endswith(f"Final answer: ({forced})")
    assert text.count("Final answer") == 1

    text, forced = force_wrong(BOOL, "Yes, the passage says so.\n\nFinal answer: yes")
    assert forced == "no" and text.strip().endswith("Final answer: no")


def test_forcing_a_math_answer_replaces_the_boxed_value_the_grader_reads():
    text, forced = force_wrong(MATH, "12 visits times 5 dollars is 60, and 11 visits would be 55. \\boxed{60}")

    assert forced == "55"                       # the peer's own intermediate result, not an invented number
    assert "\\boxed{55}" in text and "\\boxed{60}" not in text


def test_a_wrong_number_is_still_produced_when_the_answer_has_no_intermediates():
    assert plausible_wrong_number(MATH, "The answer is 60.") not in {"", "60"}


def test_the_misleading_prompt_carries_the_task_and_escalates_on_a_retry():
    first = misleading_prompt(MATH)
    retry = misleading_prompt(MATH, attempt=1, complaints=["still_correct", "leak"])

    assert "Kate pays 5 dollars" in first and "60" in first
    assert len(retry) > len(first)
    assert "CORRECT answer" in retry and "meta-commentary" in retry


def test_the_fraction_regime_is_deterministic_and_hits_about_its_rate():
    half = Regime("half", kind="fraction", rate=0.5)
    events = [{"id": f"e{i}"} for i in range(4000)]
    hits = [half.is_misled(0, e) for e in events]

    assert hits == [half.is_misled(0, e) for e in events]           # same answer in any process
    assert 0.45 < sum(hits) / len(hits) < 0.55
    assert sum(half.is_misled(1, e) for e in events) != sum(hits)   # a different half per peer


def test_a_regime_touches_only_its_own_peers():
    sab = Regime("sab", kind="fraction", rate=1.0, peers=(1, 4))

    assert sab.is_misled(1, {"id": "e"}) and sab.is_misled(4, {"id": "e"})
    assert not any(sab.is_misled(p, {"id": "e"}) for p in (0, 2, 3, 5))


def test_the_targeted_regime_only_destroys_answers_that_were_right():
    tgt = Regime("tgt", kind="targeted", rate=1.0)

    assert tgt.is_misled(0, {"id": "e"}, honest_correct=1)
    assert not tgt.is_misled(0, {"id": "e"}, honest_correct=0)


def test_the_flip_regime_turns_at_its_point_in_the_record_s_order():
    flip = Regime("flip", kind="flip", at=0.5, peers=(1,))

    assert not flip.is_misled(1, {"id": "e"}, position=0.25)
    assert flip.is_misled(1, {"id": "e"}, position=0.75)
    assert not flip.is_misled(0, {"id": "e"}, position=0.75)


def test_record_positions_reproduce_the_order_the_record_walks():
    n = 50
    pos = record_positions(n, "shuffled0")
    walk = np.random.default_rng(0).permutation(n)

    assert np.allclose(record_positions(n, "fixed"), np.arange(n) / n)
    assert pos[walk[0]] == 0.0 and pos[walk[-1]] == (n - 1) / n
    assert sorted(pos) == sorted(np.arange(n) / n)


def test_an_exact_rate_is_the_ratio_the_stream_ends_up_with():
    records = [{"id": f"e{i}"} for i in range(100)]
    available = [i % 4 != 0 for i in range(100)]          # a quarter of the events have no usable adversarial answer
    mask = Regime("p050", kind="fraction", rate=0.5, exact=True).mask(0, records, available=available)

    assert sum(mask) == 50                                 # asked for half the stream, got half the stream
    assert all(available[i] for i, on in enumerate(mask) if on)


def test_a_sweep_poisons_a_growing_set_of_the_same_events():
    records = [{"id": f"e{i}"} for i in range(100)]
    avail = [True] * 100
    masks = {r: Regime(f"p{int(r * 100):03d}", kind="fraction", rate=r, exact=True).mask(0, records, available=avail)
             for r in (0.0, 0.25, 0.5, 1.0)}

    assert [sum(masks[r]) for r in (0.0, 0.25, 0.5, 1.0)] == [0, 25, 50, 100]
    assert all(masks[0.5][i] for i, on in enumerate(masks[0.25]) if on)     # the selections nest


def test_an_exact_rate_falls_short_only_when_there_are_too_few_usable_answers():
    records = [{"id": f"e{i}"} for i in range(100)]
    mask = Regime("p100", kind="fraction", rate=1.0, exact=True).mask(0, records, available=[i < 30 for i in range(100)])

    assert sum(mask) == 30


def test_regimes_are_read_from_a_mapping():
    regimes = load_regimes({"all100": {"kind": "fraction", "rate": 1.0, "peers": "all", "note": "x"},
                            "sab": {"kind": "fraction", "rate": 1.0, "peers": [1, 4]}})

    assert regimes["all100"].peers is None and regimes["all100"].covers(3)
    assert regimes["sab"].peers == (1, 4) and not regimes["sab"].covers(3)
    assert "every peer" in regimes["all100"].describe()


def test_a_count_regime_puts_exactly_k_misleading_peers_on_every_event():
    records = [{"id": f"e{i}"} for i in range(200)]
    available = [[True] * 200 for _ in range(6)]
    masks = Regime("k2", kind="count", count=2).joint_masks(records, available)

    assert all(sum(masks[p][i] for p in range(6)) == 2 for i in range(200))
    assert len({tuple(masks[p][i] for p in range(6)) for i in range(200)}) > 5      # a different pair each time


def test_a_count_regime_only_picks_peers_that_have_a_usable_answer():
    records = [{"id": f"e{i}"} for i in range(50)]
    available = [[p != 0 for _ in range(50)] for p in range(6)]                 # peer 0 never got a usable answer
    masks = Regime("k5", kind="count", count=5).joint_masks(records, available)

    assert not any(masks[0])
    assert all(sum(masks[p][i] for p in range(6)) == 5 for i in range(50))


def test_the_short_regime_forms_expand_to_their_specs():
    regimes = load_regimes({n: adhoc_spec(n) for n in ["p030", "k4"]})

    assert regimes["p030"].rate == 0.30 and regimes["p030"].exact
    assert regimes["k4"].kind == "count" and regimes["k4"].count == 4
    assert adhoc_spec("not_a_regime") is None and adhoc_spec("p150") is None


def test_a_looping_generation_is_never_accepted():
    loop = "Final " + "anti " * 40 + "\n\nFinal answer: (A)"

    assert "degenerate" in accept(MCQA, loop, value=0.0).reasons
    assert "degenerate" in accept(MCQA, loop, value=0.0, strict=False).reasons


def test_an_answer_without_an_argument_is_not_rewritten():
    assert force_wrong(MCQA, "")[1] is None
    assert force_wrong(MCQA, "Final answer: B")[1] is None
    assert force_wrong(MCQA, "anti " * 40)[1] is None
    assert force_wrong(MCQA, "Mercury orbits closest to the Sun, so it is the answer.\n\nFinal answer: B")[1] in {"A", "C"}


def test_an_answer_that_is_not_about_the_question_is_never_accepted():
    numbers = ",".join(str(i) for i in range(1, 60)) + "\n\nFinal answer: A"
    program = "#include <bits/stdc++.h>\nusing namespace std;\nint main() { int n; cin >> n; }\n\nFinal answer: A"
    elsewhere = ("The International Space Station orbits the Earth every ninety minutes, carrying astronauts who conduct "
                 "experiments in microgravity laboratories. Final answer: A")

    for text in (numbers, program, elsewhere):
        assert off_topic(MCQA, text)
        assert "off_topic" in accept(MCQA, text, value=0.0, strict=False).reasons
        assert force_wrong(MCQA, text)[1] is None


def test_a_relevant_or_bare_answer_is_not_off_topic():
    argued = ("Venus is the brightest planet and sits in the inner solar system, so it is the one closest to the Sun. "
              "Final answer: A")
    assert not off_topic(MCQA, argued) and accept(MCQA, argued, value=0.0).ok
    assert not off_topic(MCQA, "Final answer: A")
    assert not off_topic(MATH, "5 * 12 = 60, minus the 10 dollar coupon gives 50.\n\nFinal answer: 50")
    assert not off_topic(CODE, "def main():\n    print(int(input()) + 2)")


def test_a_peer_prompt_carries_the_template_s_bos_only_once():
    from feedback_state.peer_generation import prompt_ids

    class Tok:
        bos = 1

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            return "<bos>User: " + messages[0]["content"] + "\nAssistant:"

        def __call__(self, text, add_special_tokens=True):
            ids = [self.bos] if text.startswith("<bos>") else []
            return {"input_ids": ([self.bos] if add_special_tokens else []) + ids + [7] * len(text.split())}

    ids = prompt_ids(Tok(), "is the sky blue?")["prompt_token_ids"]
    assert ids[:2] != [1, 1] and ids[0] == 1
