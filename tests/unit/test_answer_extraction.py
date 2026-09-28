from feedback_state.tasks import code_extract_answer, peer_target_value, qa_extract_answer
from feedback_state.utils import extract_final_answer, normalize_answer


def test_code_extractor_prefers_later_tagged_block_after_closing_fence():
    response = "old fragment\n```\n```python\ndef solve():\n    return 42\n```"

    assert code_extract_answer(response) == "def solve():\n    return 42"


def test_code_extractor_never_turns_nonempty_raw_text_into_empty_code():
    response = "explanation\n```python\n```"

    assert code_extract_answer(response) == response
    assert code_extract_answer("   ") == ""


def test_code_extractor_handles_same_line_closer_and_tagged_opener():
    response = (
        "pass\n``` ```python\ndef solve():\n    return 1\n"
        "``` ```python\ndef solve():\n    return 2\n```"
    )

    assert code_extract_answer(response) == "def solve():\n    return 2"


def test_qa_extractor_skips_trailing_formatting_fragments():
    assert qa_extract_answer("Rowan Blanchard\n:") == "Rowan Blanchard"
    assert qa_extract_answer("Europe\n.") == "Europe"
    assert qa_extract_answer("Therefore, the answer is:\n\nPolypodium glycyrrhiza") == "Polypodium glycyrrhiza"


def test_math_extractor_preserves_textual_and_display_answers():
    assert extract_final_answer(r"Final answer: \boxed{\text{No solution}}") == "nosolution"
    assert normalize_answer(r"5.4 \text{ cents}") == "5.4"
    assert extract_final_answer("Final answer:\n$$\n\\frac{1}{2}\n$$") == "1/2"
    assert extract_final_answer("Final Answer:\n$x^3+3x-6$") == "x^3+3x-6"
    assert extract_final_answer("Final Answer:\n\\(6+9i\\)") == "6+9i"
    assert extract_final_answer(
        "Earlier: \\frac{(a+b+c)^2}{a+b+c}.\nFinal Answer:\n\n60\n"
    ) == "60"
    assert extract_final_answer(
        "### Final Answer\nThe critical points are:\n"
        "- \\(x=0\\) (neither)\n- \\(x=1\\) (local minimum)"
    ) == "x=1"
    assert extract_final_answer(
        "### Final Answer\nThe solution in interval notation is:\n\\[(-1,1]\\]"
    ) == "(-1,1]"


def test_precomputed_peer_correct_still_has_priority():
    record = {
        "task_type": "math",
        "answer": "42",
        "peer_correct": {"peer_0": 0.25},
    }

    assert peer_target_value(record, "peer_0", "Final answer: 42") == 0.25
