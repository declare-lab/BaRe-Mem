"""One correctness rule per task for every answer, the central model's included."""
from feedback_state.memory_generator import grade
from feedback_state.tasks import peer_target_value

RAG = {"id": "sq:1", "task_type": "rag", "problem": "Which river is the centre near?", "context": "The centre is near the Seine.",
       "answer": "Seine", "answer_aliases": ["the Seine"]}


def test_the_central_model_is_graded_by_the_peers_rule_on_reading():
    for text, right in (("Final answer: Seine River.", True), ("Final answer: the Seine", True), ("Final answer: Loire", False),
                        ("Final answer: near the Seine in central Paris today", False)):
        assert grade(RAG, text) is right
        assert (peer_target_value(RAG, None, text) >= 0.5) is right      # token-F1 >= 0.5, as the streams' peer labels
