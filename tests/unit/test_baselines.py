"""The multi-agent baselines: the debate conversation and the majority vote (feedback_state.baselines, pipeline.vote)."""
import pytest

from feedback_state.baselines import debate_messages, plurality
from pipeline.vote import vote_rows


def test_a_debate_round_continues_the_central_models_conversation_with_the_fixed_peer_answers():
    record = {"task_type": "math", "problem": "What is 2 + 3?"}
    peers = ["It is 5. Final answer: 5", "It is 6. Final answer: 6"]

    first = debate_messages(record, peers, ["Final answer: 5"])
    assert [m["role"] for m in first] == ["system", "user", "assistant", "user"]
    assert "What is 2 + 3?" in first[1]["content"] and "Peer" not in first[1]["content"]      # round 0 saw the question alone
    ask = first[-1]["content"]
    assert ask.startswith("These are the solutions to the problem from other agents: ")
    assert "\n\n One agent solution: ```It is 5. Final answer: 5```" in ask and "```It is 6. Final answer: 6```" in ask
    assert "can you give an updated answer? Examine your solution and that other agents step by step." in ask
    assert ask.endswith("'Final answer: <number>'.")                                         # the task's answer format

    second = debate_messages(record, peers, ["Final answer: 5", "Final answer: 6"])
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert second[4]["content"] == "Final answer: 6" and second[5]["content"] == ask             # the same peers every round
    with pytest.raises(ValueError):
        debate_messages(record, peers, [])


def test_the_vote_takes_the_largest_group_and_scores_a_tie_in_expectation():
    record = {"task_type": "boolqa", "answer": "yes"}
    three_to_two = plurality(record, ["Final answer: yes"] * 3 + ["Final answer: no"] * 2 + [""], [1, 1, 1, 0, 0, 0])
    assert three_to_two == {"correct": 1.0, "votes": 3, "tied": 1}

    tie = plurality(record, ["Final answer: yes"] * 3 + ["Final answer: no"] * 3, [1, 1, 1, 0, 0, 0])
    assert tie == {"correct": 0.5, "votes": 3, "tied": 2}

    code = plurality({"task_type": "code"}, ["print(1)", "print(1)", "print(2)"], [1, 1, 0])   # no agreement is measured for code
    assert code["correct"] == pytest.approx(2 / 3) and code["tied"] == 3


def test_the_central_models_answer_can_join_the_vote_and_break_a_tie():
    event = {"id": "q1", "task_type": "boolqa", "answer": "no",
             "peer_responses": {"peer_0": "Final answer: yes", "peer_1": "Final answer: no"}, "peer_correct": {"peer_0": 0, "peer_1": 1}}
    rows = [{"pos": 0, "id": "q1"}]

    peers_only = vote_rows(rows, {"q1": event}, None)
    assert peers_only[0]["correct"] == 0.5 and peers_only[0]["candidates"] == 2
    with_own = vote_rows(rows, {"q1": event}, {"q1": {"generation": "Final answer: no", "correct": 1}})
    assert with_own[0]["correct"] == 1.0 and with_own[0]["votes"] == 2 and with_own[0]["candidates"] == 3
    with pytest.raises(SystemExit, match="no answer"):
        vote_rows(rows, {"q1": event}, {})
