"""Task registry: a modular abstraction over heterogeneous task types.

The trust-state pipeline only ever needs a per-peer correctness label
``c_j ∈ [0, 1]`` per record. Math equivalence, multiple-choice accuracy,
open-domain QA (EM/F1) and code pass@1 all reduce to that scalar. This module
is the single place that knows how
to turn a (record, peer_response) into:

  * a *soft* target ``target(...) -> [0,1]`` used for correctness supervision and
    selection floors/ceiling, and
  * a *binary* ``is_correct(...) -> bool`` used for reported accuracy, and
  * ``extract_answer(...)`` for human-readable predictions, and
  * ``build_peer_prompt(...)`` for the offline peer-generation step.

Every task standardises its question under ``record["problem"]`` and its gold
under ``record["answer"]`` so the encoder + selection model are task-agnostic;
only correctness scoring dispatches on ``record["task_type"]``.

Adding a new task = register one ``TaskSpec``. Adding a new task *mixture* = just
mix records with different ``task_type`` values in one JSONL; nothing else changes.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from feedback_state.utils import extract_final_answer, math_equal

DEFAULT_TASK_TYPE = "math"


def task_type_of(record: dict[str, Any]) -> str:
    return str(record.get("task_type") or DEFAULT_TASK_TYPE).lower()


# ---------------------------------------------------------------------------
# Open-domain QA metrics (SQuAD / HotpotQA / TriviaQA style)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)


def _normalize_qa(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD normalisation)."""
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _qa_gold_answers(record: dict[str, Any]) -> list[str]:
    golds = [str(record.get("answer", ""))]
    aliases = record.get("answer_aliases") or record.get("aliases") or []
    if isinstance(aliases, (list, tuple)):
        golds.extend(str(a) for a in aliases)
    return [g for g in golds if g.strip()]


def qa_extract_answer(text: str) -> str:
    """Pull a short answer span out of a peer's free-form QA response."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    # Prefer an explicit "Answer: X" tail. Horizontal whitespace is deliberate:
    # \s would cross a newline and turn a marker-only line into a bogus answer.
    matches = list(
        re.finditer(
            r"(?:final\s+answer|answer)[ \t]*(?:(?:is)[ \t]*:?[ \t]*|:[ \t]*)([^\n\r]+)",
            raw,
            flags=re.IGNORECASE,
        )
    )
    for match in reversed(matches):
        candidate = match.group(1).strip().strip(".")
        if any(ch.isalnum() for ch in candidate):
            return candidate
    # Ignore trailing formatting fragments such as ":", "[", or ".".
    lines = [ln.strip().strip(".") for ln in raw.splitlines() if ln.strip()]
    for candidate in reversed(lines):
        if any(ch.isalnum() for ch in candidate):
            return candidate
    return raw.strip().strip(".")


def qa_f1(pred: str, golds: list[str]) -> float:
    pred_tokens = _normalize_qa(pred).split()
    best = 0.0
    for gold in golds:
        gold_tokens = _normalize_qa(gold).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def qa_exact_match(pred: str, golds: list[str]) -> bool:
    npred = _normalize_qa(pred)
    return any(npred == _normalize_qa(g) for g in golds)


# ---------------------------------------------------------------------------
# Code extraction (used for reporting + by the offline scorer)
# ---------------------------------------------------------------------------

_TAGGED_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)[^\S\r\n]*\r?\n(.*?)(?=```)",
    flags=re.DOTALL | re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```")


def code_extract_answer(text: str) -> str:
    """Extract a code block from a peer response (fenced block, else raw text)."""
    raw = str(text or "")
    # The closer is a lookahead so a same-line sequence such as
    # ````` ```python`` still exposes the second tagged opener to findall().
    tagged = [block.strip() for block in _TAGGED_CODE_FENCE_RE.findall(raw) if block.strip()]
    if tagged:
        return tagged[-1]

    # For untagged or malformed fences, inspect every adjacent pair. This also
    # recovers a real block after a stray closer without pairing that closer with
    # the next opener. A language-only gap is not code.
    fences = list(_CODE_FENCE_RE.finditer(raw))
    blocks = []
    for left, right in zip(fences, fences[1:]):
        block = raw[left.end():right.start()].strip()
        lines = block.splitlines()
        if lines and lines[0].strip().lower() in {"python", "py"}:
            block = "\n".join(lines[1:]).strip()
        if block:
            blocks.append(block)
    if blocks:
        return blocks[-1]
    return raw.strip()


# ---------------------------------------------------------------------------
# TaskSpec + registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    name: str
    # (peer_text, record) -> soft correctness in [0, 1]
    target_fn: Callable[[str, dict[str, Any]], float]
    # (peer_text, record) -> binary correctness (reported accuracy)
    correct_fn: Callable[[str, dict[str, Any]], bool]
    # peer_text -> human-readable extracted answer
    extract_fn: Callable[[str], str]
    # (record, with_context) -> prompt string for offline peer generation
    prompt_fn: Callable[[dict[str, Any], bool], str]
    # True if correctness is precomputed per-peer (read from record["peer_correct"])
    precomputed: bool = False
    # (record, answer) -> (passed, message): what an agent observes after acting (a visible check, never the label); with
    # it, pipeline.peers --turns N lets a peer revise on the message. None: the task has no tool feedback
    feedback_fn: Callable[[dict[str, Any], str], tuple[bool, str]] | None = None
    # (record, response) -> the answer text the stream shows the judge and the central model; None: the response as it is
    display_fn: Callable[[dict[str, Any], str], str] | None = None


def _math_target(text: str, record: dict[str, Any]) -> float:
    pred = extract_final_answer(str(text))
    return 1.0 if (pred and math_equal(pred, str(record.get("answer", "")))) else 0.0


def _math_correct(text: str, record: dict[str, Any]) -> bool:
    return _math_target(text, record) >= 0.5


def _math_prompt(record: dict[str, Any], with_context: bool) -> str:
    return (
        "Solve this math problem. Give a concise solution and end with the final answer.\n\n"
        f"Problem:\n{record.get('problem', '')}"
    )


def _rag_context_text(record: dict[str, Any]) -> str:
    # Cap retrieved context to keep the peer prompt within model length. TriviaQA
    # ships whole wiki/web docs (median ~54k chars, max ~514k) which blow past the
    # context window; HotpotQA passages are short so this never affects them.
    max_chars = 8000
    ctx = record.get("context", "")
    if isinstance(ctx, (list, tuple)):
        parts = []
        for i, passage in enumerate(ctx, start=1):
            if isinstance(passage, (list, tuple)):  # HotpotQA [title, [sentences]]
                title = str(passage[0]) if passage else ""
                body = " ".join(str(s) for s in passage[1]) if len(passage) > 1 else ""
                parts.append(f"[{i}] {title}: {body}".strip())
            else:
                parts.append(f"[{i}] {str(passage)}")
        text = "\n".join(parts)
    else:
        text = str(ctx)
    return text[:max_chars]


def _rag_target(text: str, record: dict[str, Any]) -> float:
    # Soft graded trust via token-F1 against gold (and aliases).
    return qa_f1(qa_extract_answer(text), _qa_gold_answers(record))


def _rag_correct(text: str, record: dict[str, Any]) -> bool:
    # The streams' rule, the same one the peers' labels carry: token-F1 >= 0.5 against the gold answer and its aliases.
    # Until 2026-09-14 this was exact match, so the central model was held to a stricter rule than its peers.
    return _rag_target(text, record) >= 0.5


def _rag_prompt(record: dict[str, Any], with_context: bool) -> str:
    question = record.get("problem", "")
    if with_context:
        context = _rag_context_text(record)
        return (
            "Answer the question using the retrieved context. Respond with a short answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    # Deprived peer (e.g. Gemma for RAG): no retrieved documents.
    return (
        "Answer the question with a short answer.\n\n"
        f"Question: {question}\nAnswer:"
    )


_YES_NO_RE = re.compile(r"\b(yes|no|true|false)\b", flags=re.IGNORECASE)


def _normalize_bool_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return "yes"
    if text in {"0", "false", "no"}:
        return "no"
    match = _YES_NO_RE.search(text)
    if not match:
        return ""
    token = match.group(1).lower()
    return "yes" if token in {"yes", "true"} else "no"


def boolqa_extract_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    explicit = list(
        re.finditer(
            r"(?:final\s+answer|answer)\s*(?:is|:)?\s*(yes|no|true|false)\b",
            raw,
            flags=re.IGNORECASE,
        )
    )
    if explicit:
        return _normalize_bool_label(explicit[-1].group(1))
    # Fall back to the last yes/no-looking token so explanatory answers like
    # "... therefore, no" are scored as intended.
    matches = list(_YES_NO_RE.finditer(raw))
    return _normalize_bool_label(matches[-1].group(1)) if matches else ""


def _boolqa_target(text: str, record: dict[str, Any]) -> float:
    pred = boolqa_extract_answer(text)
    gold = _normalize_bool_label(record.get("answer", ""))
    return 1.0 if pred and gold and pred == gold else 0.0


def _boolqa_correct(text: str, record: dict[str, Any]) -> bool:
    return _boolqa_target(text, record) >= 0.5


def _boolqa_prompt(record: dict[str, Any], with_context: bool) -> str:
    question = str(record.get("problem", "")).strip()
    context = _rag_context_text(record) if with_context else ""
    prefix = (
        "Read the passage and answer the yes/no question. "
        "End with exactly 'Final answer: yes' or 'Final answer: no'."
    )
    if context:
        return f"{prefix}\n\nPassage:\n{context}\n\nQuestion: {question}"
    return f"{prefix}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Multiple-choice and short-answer tasks
# ---------------------------------------------------------------------------

_PARENS_LABEL_RE = re.compile(r"\(([A-Za-z0-9]+)\)")


def _normalise_label(value: Any) -> str:
    text = str(value or "").strip()
    text = text.strip().strip("()[]{}").strip()
    return text.lower()


def _choice_labels(record: dict[str, Any]) -> list[str]:
    labels = record.get("choice_labels") or []
    if isinstance(labels, (list, tuple)) and labels:
        return [str(x) for x in labels]
    choices = record.get("choices") or []
    return [chr(ord("A") + i) for i in range(len(choices))]


def _mcqa_pred_label(text: str, record: dict[str, Any]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    labels = _choice_labels(record)
    norm_to_label = {_normalise_label(label): label for label in labels}

    def pick(segment: str) -> str:
        segment = str(segment or "").strip()
        if not segment:
            return ""
        # Prefer explicit parenthesized options, e.g. "(C)".
        for match in reversed(list(_PARENS_LABEL_RE.finditer(segment))):
            key = _normalise_label(match.group(1))
            if key in norm_to_label:
                return norm_to_label[key]
        # Then phrases like "option C", "choice C", or a bare final label.
        for label in labels:
            escaped = re.escape(str(label).strip())
            if re.fullmatch(rf"\(?\s*{escaped}\s*\)?\.?", segment, flags=re.IGNORECASE):
                return label
            if re.search(
                rf"\b(?:option|choice|answer|letter)\s*[:\-]?\s*\(?{escaped}\)?\b",
                segment,
                flags=re.IGNORECASE,
            ):
                return label
            if re.search(
                rf"\b(?:choose|select|is)\s+\(?{escaped}\)?\b",
                segment,
                flags=re.IGNORECASE,
            ):
                return label
        key = _normalise_label(segment)
        return norm_to_label.get(key, "")

    explicit = list(
        re.finditer(
            r"(?:final\s+answer|answer)\s*(?:is|:)?\s*([^\n\r]+)",
            raw,
            flags=re.IGNORECASE,
        )
    )
    if explicit:
        found = pick(explicit[-1].group(1))
        if found:
            return found
    for line in reversed([ln.strip() for ln in raw.splitlines() if ln.strip()]):
        found = pick(line)
        if found:
            return found
    return pick(raw)


def mcqa_extract_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    explicit = list(
        re.finditer(
            r"(?:final\s+answer|answer)\s*(?:is|:)?\s*([^\n\r]+)",
            raw,
            flags=re.IGNORECASE,
        )
    )
    if explicit:
        return explicit[-1].group(1).strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-1] if lines else raw


def _mcqa_target(text: str, record: dict[str, Any]) -> float:
    pred = _normalise_label(_mcqa_pred_label(text, record))
    gold = _normalise_label(record.get("answer", ""))
    return 1.0 if pred and gold and pred == gold else 0.0


def _mcqa_correct(text: str, record: dict[str, Any]) -> bool:
    return _mcqa_target(text, record) >= 0.5


def _mcqa_prompt(record: dict[str, Any], with_context: bool) -> str:
    labels = _choice_labels(record)
    choices = [str(x) for x in (record.get("choices") or [])]
    options = "\n".join(
        f"({label}) {choice}" for label, choice in zip(labels, choices)
    )
    return (
        "Answer the multiple-choice question. Choose exactly one option. "
        "End with exactly 'Final answer: <option label>'.\n\n"
        f"Question:\n{record.get('problem', '')}\n\nOptions:\n{options}"
    )


def shortqa_extract_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    explicit = list(
        re.finditer(
            r"(?:final\s+answer|answer)\s*(?:is|:)?\s*([^\n\r]+)",
            raw,
            flags=re.IGNORECASE,
        )
    )
    if explicit:
        return explicit[-1].group(1).strip().strip(".")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return (lines[-1] if lines else raw).strip().strip(".")


def _shortqa_norm(text: Any) -> str:
    # BBH targets mix "(B)", "False", "invalid", short phrases and numbers.
    # Reuse QA punctuation/whitespace normalization, with label parentheses removed.
    return _normalize_qa(str(text or "").strip().strip("()[]{}"))


def _shortqa_target(text: str, record: dict[str, Any]) -> float:
    pred = _shortqa_norm(shortqa_extract_answer(text))
    gold = _shortqa_norm(record.get("answer", ""))
    return 1.0 if pred and gold and pred == gold else 0.0


def _shortqa_correct(text: str, record: dict[str, Any]) -> bool:
    return _shortqa_target(text, record) >= 0.5


def _shortqa_prompt(record: dict[str, Any], with_context: bool) -> str:
    return (
        "Answer the question. Give a concise answer and end with "
        "'Final answer: <answer>'.\n\n"
        f"Question:\n{record.get('problem', '')}"
    )


def _code_precomputed(record: dict[str, Any], peer_key: str | None) -> float | None:
    table = record.get("peer_correct")
    if isinstance(table, dict) and peer_key is not None and peer_key in table:
        return float(table[peer_key])
    return None


def _code_target(text: str, record: dict[str, Any]) -> float:
    # Code correctness is precomputed by data.builders.common.score_code_peers.
    # and looked up by peer_key in peer_target_value; this text-only fallback
    # cannot execute, so it returns 0.0 (unknown == untrusted).
    return 0.0


def _code_correct(text: str, record: dict[str, Any]) -> bool:
    return False


def _code_prompt(record: dict[str, Any], with_context: bool) -> str:
    # APPS-style problems are graded by running the program against stdin/stdout,
    # so the peer must read input() and print the answer -- NOT define a function.
    if str(record.get("code_format", "")) == "io":
        return (
            "Write a complete Python program that reads from standard input and "
            "writes the answer to standard output. Use input()/sys.stdin to read and "
            "print() to write. Return only the program inside a ```python code block.\n\n"
            f"{record.get('problem', '')}"
        )
    return (
        "Complete the following Python function. Return only the full function "
        "implementation inside a ```python code block.\n\n"
        f"{record.get('problem', '')}"
    )


# A worker's report on a sub-question (pipeline.team): a sentence that carries the gold entity is right.
def _subqa_target(text: str, record: dict[str, Any]) -> float:
    golds = _qa_gold_answers(record)
    f1 = qa_f1(qa_extract_answer(text), golds)
    if f1 >= 0.5:
        return f1
    words = f" {_normalize_qa(text)} "
    held = len(words.split()) <= 20 and any(f" {_normalize_qa(g)} " in words for g in golds if _normalize_qa(g))
    return 1.0 if held else f1


def _subqa_correct(text: str, record: dict[str, Any]) -> bool:
    return _subqa_target(text, record) >= 0.5


# The grading rules. A task type is registered by a file, configs/tasks/<name>.yaml, that names its rule here (`grader:`) and
# carries what is not code: the central model's instruction, the peers' answer budget, whether a passage is shown, the vote's
# agreement rule. A new task over an existing rule is a new file; a new rule is a function here plus its file.
GRADERS: dict[str, TaskSpec] = {
    "math": TaskSpec("math", _math_target, _math_correct, extract_final_answer, _math_prompt),
    "rag": TaskSpec("rag", _rag_target, _rag_correct, qa_extract_answer, _rag_prompt),
    "boolqa": TaskSpec(
        "boolqa", _boolqa_target, _boolqa_correct, boolqa_extract_answer, _boolqa_prompt
    ),
    "mcqa": TaskSpec("mcqa", _mcqa_target, _mcqa_correct, mcqa_extract_answer, _mcqa_prompt),
    "shortqa": TaskSpec(
        "shortqa", _shortqa_target, _shortqa_correct, shortqa_extract_answer, _shortqa_prompt
    ),
    "code": TaskSpec(
        "code", _code_target, _code_correct, code_extract_answer, _code_prompt, precomputed=True
    ),
    "subqa": TaskSpec("subqa", _subqa_target, _subqa_correct, qa_extract_answer, _rag_prompt),
}

TASKS_DIR = Path(__file__).resolve().parents[1] / "configs" / "tasks"


def load_task_configs(folder: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every registered task type: configs/tasks/<name>.yaml -> {grader, instruction, max_tokens, ...}, by file name."""
    import yaml

    out = {}
    for f in sorted(Path(folder or TASKS_DIR).glob("*.yaml")):
        if not f.name.startswith("_"):
            out[f.stem.lower()] = dict(yaml.safe_load(f.read_text()) or {}, name=f.stem.lower(), file=str(f))
    return out


TASK_CONFIGS: dict[str, dict[str, Any]] = load_task_configs()
unknown = {n: c.get("grader") for n, c in TASK_CONFIGS.items() if c.get("grader") not in GRADERS}
if unknown:
    raise KeyError(f"configs/tasks: no grading rule for {unknown}; rules: {sorted(GRADERS)}")
REGISTRY: dict[str, TaskSpec] = {n: replace(GRADERS[c["grader"]], name=n) for n, c in TASK_CONFIGS.items()}


def task_config(name: str) -> dict[str, Any]:
    """A registered task's settings (configs/tasks/<name>.yaml)."""
    key = str(name or DEFAULT_TASK_TYPE).lower()
    if key not in TASK_CONFIGS:
        raise KeyError(f"Unknown task_type {name!r}. Registered (configs/tasks/): {sorted(TASK_CONFIGS)}")
    return TASK_CONFIGS[key]


def get_task(name: str) -> TaskSpec:
    key = str(name or DEFAULT_TASK_TYPE).lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown task_type {name!r}. Registered (configs/tasks/): {sorted(REGISTRY)}")
    return REGISTRY[key]


def register_task(spec: TaskSpec) -> None:
    REGISTRY[spec.name.lower()] = spec


# ---------------------------------------------------------------------------
# Dispatching helpers used by the data collator and the evaluator
# ---------------------------------------------------------------------------

def peer_target_value(record: dict[str, Any], peer_key: str | None, peer_text: str) -> float:
    """Soft per-peer correctness in [0,1]."""
    # Precomputed labels (any task): if the record carries a peer_correct map, trust
    # it. Lets us pre-grade slow math (sympy) ONCE offline and have eval read it
    # instead of re-running the grader per eval job (which can hang on pathological
    # sympy exprs). Code always uses this path (offline pass@1).
    if peer_key is not None:
        pre = _code_precomputed(record, peer_key)
        if pre is not None:
            return max(0.0, min(1.0, pre))
    task = get_task(task_type_of(record))
    if task.precomputed:
        pre = _code_precomputed(record, peer_key)
        if pre is not None:
            return max(0.0, min(1.0, pre))
    return float(task.target_fn(peer_text, record))


def peer_is_correct(record: dict[str, Any], peer_key: str | None, peer_text: str) -> bool:
    """Binary per-peer correctness used for reported accuracy."""
    task = get_task(task_type_of(record))
    if task.precomputed:
        pre = _code_precomputed(record, peer_key)
        if pre is not None:
            return pre >= 0.5
    return bool(task.correct_fn(peer_text, record))


def extract_answer(record: dict[str, Any], peer_text: str) -> str:
    return get_task(task_type_of(record)).extract_fn(peer_text)


def build_peer_prompt(record: dict[str, Any], *, with_context: bool = True) -> str:
    return get_task(task_type_of(record)).prompt_fn(record, with_context)
