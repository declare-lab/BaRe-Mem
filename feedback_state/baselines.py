"""Multi-agent baselines on the same candidates: majority vote and multi-agent debate.

Vote (no model runs): the candidates of an event are grouped by the answer they give (feedback_state.answer_groups) and
the largest group wins. A tie is broken uniformly at random and scored in expectation, so the result is deterministic.
Code has no measurable agreement (every program is its own group), so its vote is a uniformly random candidate.

Multi-agent debate: every agent
answers, then in each round reads the other agents' latest answers and updates its own, keeping its conversation. The
peers' answers here are the released ones and do not change (a misleading answer stays misleading), so only the central
model updates: its round-0 answer is its no-consultation answer, and round r is its answer after reading the peers r times.
The round prompt is the standard debate prompt, with its answer-format sentence replaced by the task's instruction.
"""
from __future__ import annotations

from typing import Any, Sequence

from feedback_state.answer_groups import answer_groups
from feedback_state.memory_generator import INSTRUCTIONS, _clip, build_messages
from feedback_state.tasks import task_type_of

DEBATE_OTHERS = "These are the solutions to the problem from other agents: "
DEBATE_AGENT = "\n\n One agent solution: ```{}```"
DEBATE_UPDATE = ("\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? "
                 "Examine your solution and that other agents step by step. ")


def debate_messages(record: dict[str, Any], peer_texts: Sequence[str], history: Sequence[str], *, char_limit: int = 3000) -> list[dict]:
    """The central model's conversation before its next debate answer.

    It opens with the no-consultation turn; each answer in ``history`` (round 0 first) is followed by the other agents'
    answers and the request to update, so ``len(history)`` is the round being asked for.
    """
    if not history:
        raise ValueError("a debate round needs the central model's earlier answers (round 0: its no-consultation answer)")
    task = task_type_of(record)
    update = (DEBATE_OTHERS + "".join(DEBATE_AGENT.format(_clip(t, char_limit)) for t in peer_texts) + DEBATE_UPDATE
              + INSTRUCTIONS.get(task, INSTRUCTIONS["shortqa"]))
    messages = build_messages(record, [], mode="solo")
    for answer in history:
        messages += [{"role": "assistant", "content": str(answer)}, {"role": "user", "content": update}]
    return messages


def plurality(record: dict[str, Any], texts: Sequence[str], labels: Sequence[float]) -> dict:
    """A plurality vote over ``texts`` with 0/1 ``labels``: its expected correctness under uniform tie-breaking, the size
    of the winning group and how many groups tie for it."""
    if len(texts) != len(labels) or not texts:
        raise ValueError(f"{len(texts)} candidates with {len(labels)} labels")
    groups: dict[int, list[float]] = {}
    for g, y in zip(answer_groups(record, list(texts)), labels):
        groups.setdefault(g, []).append(float(y))
    top = max(len(v) for v in groups.values())
    winners = [v for v in groups.values() if len(v) == top]
    return {"correct": sum(sum(v) / len(v) for v in winners) / len(winners), "votes": top, "tied": len(winners)}
