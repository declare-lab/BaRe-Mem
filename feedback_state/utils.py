from __future__ import annotations

import argparse
import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

try:
    import hjson
except ImportError:  # pragma: no cover
    hjson = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# math-verify-backed symbolic grader (optional). math_grader imports only the
# math_verify lib, not this module, so there is no import cycle.
try:
    from feedback_state.math_grader import MATH_VERIFY_AVAILABLE as _MATH_VERIFY_AVAILABLE
    from feedback_state.math_grader import verify_equiv as _verify_equiv
except Exception:  # pragma: no cover
    _MATH_VERIFY_AVAILABLE = False

    def _verify_equiv(pred, gold):  # type: ignore[misc]
        return False


_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)[ \t]*(?:is[ \t]*:?|:)[ \t]*([^\n\r]+)",
    flags=re.IGNORECASE,
)
_FINAL_DISPLAY_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)[ \t]*(?:is[ \t]*:?[ \t]*|:[ \t]*)?"
    r"\r?\n[ \t]*(?:\$\$|\$|\\\[|\\\()[ \t]*(.*?)[ \t]*"
    r"(?:\$\$|\$|\\\]|\\\))(?=[ \t]*(?:\r?\n|$))",
    flags=re.IGNORECASE | re.DOTALL,
)
_FINAL_BARE_NEXT_LINE_RE = re.compile(
    r"final[ \t]+answer[ \t]*(?:is[ \t]*:?[ \t]*|:[ \t]*)?"
    r"\r?\n(?:[ \t]*\r?\n)*[ \t]*([^\n\r]+)",
    flags=re.IGNORECASE,
)
_FINAL_SECTION_MARKER_RE = re.compile(
    r"^[ \t]*(?:\#{1,6}[ \t]*)?final[ \t]+answer[ \t]*:?[ \t]*\r?$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_DISPLAY_MATH_PATTERNS = (
    re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL),
    re.compile(r"\\\[(.*?)\\\]", flags=re.DOTALL),
    re.compile(r"\\\((.*?)\\\)", flags=re.DOTALL),
    re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", flags=re.DOTALL),
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?")
_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser for the simple configs in this repo."""
    root: dict[str, Any] = {}
    current_key: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            root.setdefault(current_key, []).append(_parse_scalar(line[4:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        if value == "":
            root[key] = []
        else:
            root[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    ids: set[str] = set()
    with output.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in record:
                ids.add(str(record["id"]))
    return ids


def _locate_final_answer(text: Any) -> str:
    r"""Locate the final answer in `text` and return it WITHOUT normalization.

    Single source of truth for "where is the answer": last \boxed{...}, then
    `#### x`, then "answer: x", then a short bare answer, then frac/number
    fallbacks. `extract_final_answer` wraps this with `normalize_answer` (the
    lossy string form used by the old scorer); the math-verify path uses the raw
    form so LaTeX structure (\frac, \text{ or }, commas) survives.
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    boxed = _last_boxed_content(raw)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    hashed = _last_regex_group(_HASH_ANSWER_RE, raw)
    if hashed:
        return hashed.strip()
    section_math = _last_final_section_math(raw)
    if section_math:
        return section_math
    display = _last_regex_group(_FINAL_DISPLAY_ANSWER_RE, raw)
    if display and display.strip():
        return display.strip()
    next_line = _last_regex_group(_FINAL_BARE_NEXT_LINE_RE, raw)
    if next_line and next_line.strip():
        return next_line.strip()
    final = _last_regex_group(_FINAL_ANSWER_RE, raw)
    if final:
        return final.strip()
    # Bare answer (e.g. dataset gold "(-1,6)" or "137 \frac{1}{2}"): a short,
    # single-line string with no prose is already the answer. Prose ("I get 42
    # after simplification.") must still fall through to the numeric fallback, so
    # require few alphabetic words once LaTeX commands (\frac, \cup, ...) and
    # single-letter variables (x, i) are discounted.
    if "\n" not in raw and len(raw) <= 64:
        prose_words = re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[a-zA-Z]+", " ", raw))
        if len(prose_words) < 2:
            return raw
    frac_matches = list(_FRAC_RE.finditer(raw))
    if frac_matches:
        match = frac_matches[-1]
        return f"{match.group(1)}/{match.group(2)}"
    numeric_matches = _NUMBER_RE.findall(raw)
    if numeric_matches:
        return numeric_matches[-1]
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else raw


def extract_final_answer(text: Any) -> str:
    """Locate the final answer and normalize it to the canonical string form."""
    return normalize_answer(_locate_final_answer(text))


def _last_boxed_content(text: str) -> str | None:
    r"""Return the brace-balanced content of the LAST \boxed{...} (backslash optional).

    Replaces a regex that only handled one level of nesting, so answers like
    \boxed{\frac{\sqrt{3}}{3}} are extracted whole instead of failing and falling
    through to the numeric fallback. Returns None when no (closed) \boxed{} exists,
    including the truncated case where the closing brace was never generated.
    """
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        marker = "boxed{"
        idx = text.rfind(marker)
        if idx < 0:
            return None
    depth = 1
    out: list[str] = []
    for ch in text[idx + len(marker):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None  # unbalanced (e.g. truncated output) — let other extractors try


def _last_regex_group(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)


def _last_final_section_math(text: str) -> str | None:
    """Return the last math span after a marker-only ``Final Answer`` heading."""
    markers = list(_FINAL_SECTION_MARKER_RE.finditer(text))
    if not markers:
        return None
    suffix = text[markers[-1].end():]
    spans: list[tuple[int, str]] = []
    for pattern in _DISPLAY_MATH_PATTERNS:
        spans.extend(
            (match.start(), match.group(1).strip())
            for match in pattern.finditer(suffix)
            if match.group(1).strip()
        )
    return max(spans, key=lambda item: item[0])[1] if spans else None


def normalize_answer(ans: Any) -> str:
    text = str(ans or "").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\$", "").replace("$", "")
    # Drop digit-grouping commas only (1,000 -> 1000); keep structural commas so
    # tuples/intervals like (-1, 6) don't collapse to (-16).
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.strip().rstrip(".")
    text = re.sub(r"^\\boxed\s*\{(.+)\}$", r"\1", text)
    # Display/text fraction variants are equivalent to \frac.
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    # Strip units/decorations so "30^\circ" == "30" and "5.4 \text{ cents}" == "5.4".
    # Dataset golds often carry a trailing unit the model omits; this normalizes
    # both sides. Done after \frac so it doesn't eat fraction braces.
    text = re.sub(r"\^\s*\\circ", "", text)            # degree symbol
    sole_text = re.fullmatch(r"\\text\s*\{([^{}]*)\}", text)
    if sole_text:
        text = sole_text.group(1)
    else:
        text = re.sub(r"\\text\s*\{[^{}]*\}", "", text)  # \text{ cents}, \text{ m}, ...
    text = text.replace("\\%", "").replace("%", "")    # percent sign
    text = re.sub(r"\\[\s!,;:]", "", text)             # spacing commands incl. "\ "
    text = re.sub(r"\\(?:quad|qquad)", "", text)
    # Unwrap single-token sub/superscript braces: \sqrt{2} -> \sqrt2, x_{5} -> x_5,
    # so they match the unbraced form datasets often use.
    text = re.sub(r"([_^])\{([^{}])\}", r"\1\2", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"\\sqrt\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def math_equal(pred: Any, gold: Any, *, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> bool:
    # Fast path: the original string-normalize + numeric scorer. If it already
    # matches we're done (cheap, and guarantees no regression vs the old behavior).
    if _old_math_equal(pred, gold, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    # Otherwise defer to math-verify (symbolic equivalence: algebra, sets,
    # intervals, matrices, sig-figs). Union semantics — we only ever rescue more
    # matches, never lose one the old scorer accepted. No-op if the lib is absent.
    # Use _locate_final_answer (raw, NOT normalized) so LaTeX structure survives;
    # pred may be a full multi-paragraph solution, so we locate the answer first.
    if _MATH_VERIFY_AVAILABLE:
        return _verify_equiv(_locate_final_answer(pred), _locate_final_answer(gold))
    return False


def _old_math_equal(pred: Any, gold: Any, *, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> bool:
    pred_norm = normalize_answer(extract_final_answer(pred))
    gold_norm = normalize_answer(extract_final_answer(gold))
    if not pred_norm or not gold_norm:
        return pred_norm == gold_norm
    if pred_norm == gold_norm:
        return True
    pred_num = _to_number(pred_norm)
    gold_num = _to_number(gold_norm)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=rel_tol, abs_tol=abs_tol)
    return pred_norm == gold_norm


def _to_number(text: str) -> float | None:
    try:
        return float(Fraction(text))
    except Exception:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def infer_deepmath_fields(example: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    problem_keys = (
        "problem",
        "question",
        "input",
        "prompt",
        "query",
        "instruction",
    )
    solution_keys = ("solution", "reasoning", "response", "output", "cot", "answer")
    answer_keys = ("final_answer", "final", "answer", "target", "label")
    problem_key = next((key for key in problem_keys if key in example and example[key]), None)
    solution_key = next((key for key in solution_keys if key in example and example[key]), None)
    answer_key = next((key for key in answer_keys if key in example and example[key]), None)
    return problem_key, solution_key, answer_key

