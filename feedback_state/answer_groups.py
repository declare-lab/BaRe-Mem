"""Group the candidates of one event by the answer they give.

Reliability-weighted voting (Nitzan-Paroush) needs to know which candidates
agree.  Agreement is defined per task type with the repository's own graders so
that a vote never uses information the evaluator does not have:

Each registered task names its rule (``agreement:`` in configs/tasks/<name>.yaml):

* canonical (mcqa / boolqa / shortqa): the frozen canonicalisation of short answers (``canonical_answer`` below);
* math: pairwise ``math_equal`` on the extracted final answers;
* qa (rag, subqa): SQuAD-normalised extracted short answers;
* none (code): no agreement is measurable (every answer is its own group).

Correctness labels are never consulted here.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from feedback_state.tasks import (
    TASK_CONFIGS,
    _mcqa_pred_label,
    _normalise_label,
    _normalize_bool_label,
    _normalize_qa,
    _shortqa_norm,
    boolqa_extract_answer,
    qa_extract_answer,
    shortqa_extract_answer,
    task_type_of,
)
from feedback_state.utils import extract_final_answer, math_equal

INVALID_ANSWER = "<invalid>"


def _shortqa_declared_labels(record: Mapping[str, Any]) -> list[str]:
    """Read BBH-style option labels without applying QA article removal.

    The historical short-answer normalizer maps the standalone label ``A`` to an
    empty string because it removes English articles.  Grouping still needs ``(A)``
    to be the same option as ``A``.
    """
    labels = record.get("choice_labels") or []
    if labels:
        return [_normalise_label(label) for label in labels]
    problem = str(record.get("problem", ""))
    found = re.findall(r"\(([A-Z0-9])\)", problem)
    return list(dict.fromkeys(_normalise_label(label) for label in found))


def _shortqa_option_label(record: Mapping[str, Any], answer: Any) -> str:
    raw = str(answer or "").strip().strip("`").strip().rstrip(".").strip()
    match = re.fullmatch(r"\(?\s*([A-Za-z0-9]+)\s*\)?", raw)
    if not match:
        return ""
    candidate = _normalise_label(match.group(1))
    return f"opt:{candidate}" if candidate in _shortqa_declared_labels(record) else ""


def canonical_answer(record: Mapping[str, Any], answer: Any) -> str:
    """The task-aware canonical answer label of a response; unparsable responses map to INVALID_ANSWER."""
    rec = dict(record)
    task_type = task_type_of(rec)
    text = str(answer or "")
    if task_type == "mcqa":
        value = _normalise_label(_mcqa_pred_label(text, rec))
    elif task_type == "boolqa":
        value = _normalize_bool_label(boolqa_extract_answer(text))
    elif task_type == "shortqa":
        extracted = shortqa_extract_answer(text)
        value = _shortqa_option_label(rec, extracted) or _shortqa_norm(extracted)
    else:
        raise ValueError(f"canonical answers exist for mcqa, boolqa and shortqa records; got task_type={task_type!r}")
    return value or INVALID_ANSWER


def answer_groups(record: dict[str, Any], texts: Sequence[str]) -> list[int]:
    """Return group ids (0..k-1) per candidate; equal ids = same answer."""
    n = len(texts)
    task = task_type_of(record)
    rule = TASK_CONFIGS.get(task, {}).get("agreement", "none")     # the task's agreement rule (configs/tasks/<task>.yaml)
    if rule == "canonical":
        keys = [canonical_answer(record, t) for t in texts]
        return _ids([None if k == INVALID_ANSWER else k for k in keys])
    if rule == "math":
        finals = [extract_final_answer(t) for t in texts]
        ids = [-1] * n
        nxt = 0
        for i in range(n):
            if ids[i] >= 0:
                continue
            ids[i] = nxt
            if finals[i]:
                for j in range(i + 1, n):
                    if ids[j] < 0 and finals[j] and math_equal(finals[i], finals[j]):
                        ids[j] = nxt
            nxt += 1
        return ids
    if rule == "qa":
        keys = [_normalize_qa(qa_extract_answer(t)) for t in texts]
        return _ids([k or None for k in keys])
    return list(range(n))  # code and unknown task types: no measurable agreement


def _ids(keys: list) -> list[int]:
    """Dense group ids; ``None`` (unparsable) answers never share a group."""
    marked = [("__unique__", i) if k is None else k for i, k in enumerate(keys)]
    dense: dict = {}
    return [dense.setdefault(x, len(dense)) for x in marked]
