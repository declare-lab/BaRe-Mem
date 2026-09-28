"""The central model's prompts and the grading of its answers.

Prompt modes (the record never enters a prompt as text; it enters the attention, feedback_state.attn_bias):
  peers   the question (options / passage), the peers' answers as ``Peer 1 ... Peer k``, the task instruction
  solo    the question and the task instruction
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from feedback_state.tasks import TASK_CONFIGS, _choice_labels, _rag_context_text, code_extract_answer, peer_is_correct, task_type_of

# The system prompts of every stored result (records built 2026-09-07). A reminder against long reasoning was appended
# to both on 2026-09-09 for thinking mode, which was dropped; it is not part of them, so rebuilt records match the stored ones.
SYSTEM_PEERS = (
    "You are the central model of a multi-agent system. Several peer models answered the same question. "
    "Treat their answers as evidence, verify them yourself, and produce your own final answer."
)
SYSTEM_SOLO = "Answer the question."

# The central model's instruction line per task type: each registered task carries its own (configs/tasks/<name>.yaml).
INSTRUCTIONS = {name: str(cfg["instruction"]) for name, cfg in TASK_CONFIGS.items() if cfg.get("instruction")}


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def peer_block(index: int, text: str, *, char_limit: int) -> str:
    return f"Peer {index + 1}:\n{_clip(text, char_limit)}"


def build_messages(record: dict[str, Any], texts: Sequence[str], *, mode: str, include_context: bool = True, char_limit: int = 3000) -> list[dict]:
    """System + user messages of one event; ``texts`` are the peers' answers in prompt (slot) order."""
    if mode not in ("peers", "solo"):
        raise ValueError(f"unknown prompt mode {mode!r}")
    task = task_type_of(record)
    parts = [f"Question:\n{str(record.get('problem', record.get('question', ''))).strip()}"]
    if task == "mcqa":
        labels = _choice_labels(record)
        choices = record.get("choices") or []
        if choices:
            parts.append("Options:\n" + "\n".join(f"({labels[i] if i < len(labels) else chr(65 + i)}) {c}" for i, c in enumerate(choices)))
    if include_context and TASK_CONFIGS.get(task, {}).get("context"):   # the task shows its passage (configs/tasks/<task>.yaml)
        ctx = _rag_context_text(record)
        if ctx:
            parts.append(f"Context / Evidence:\n{ctx}")
    if mode == "peers":
        parts.append("Peer answers:\n\n" + "\n\n".join(peer_block(i, t, char_limit=char_limit) for i, t in enumerate(texts)))
    parts.append("Instruction: " + INSTRUCTIONS.get(task, INSTRUCTIONS["shortqa"]))
    system = SYSTEM_PEERS if mode == "peers" else SYSTEM_SOLO
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def peer_texts_in_prompt_order(record: dict[str, Any], peer_order: Sequence[int]) -> list[str]:
    """The peers' answers in the order a record row shows them (slot -> canonical peer id through peer_order)."""
    keys = sorted(record.get("peer_responses", {}))
    return [str(record["peer_responses"][keys[int(p)]]) for p in peer_order]


def has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Chat template with the generation prompt; thinking is always off (enable_thinking=False for templates that know it).

    A tokenizer without a chat template (a base model) gets a plain layout: BOS, the system text, the user turn, then
    ``Answer:``; the peer blocks are the same text, so the tilt's character spans are unchanged.
    """
    if not has_chat_template(tokenizer):
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
        return f"{tokenizer.bos_token or ''}{system}\n\n{user}\n\nAnswer:"
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


THINK_END = "</think>"


def strip_thinking(text: str) -> str:
    """The answer part of a generation: everything after the last closing think tag (the whole text when there is none)."""
    return text.rsplit(THINK_END, 1)[1] if THINK_END in text else text


def grade(record: dict[str, Any], text: str, *, code_timeout: float = 10.0) -> bool:
    task = task_type_of(record)
    text = strip_thinking(text)   # never grade the reasoning trace, only the answer after it
    if task == "code":
        program = code_extract_answer(text)
        if not program.strip():
            return False
        from data.builders.common.code_grading import score_code_record  # sandboxed subprocess execution
        return bool(score_code_record(record, program, timeout=code_timeout).passed)
    return bool(peer_is_correct(record, None, text))


def prob_from_logit(ell: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(ell)))
