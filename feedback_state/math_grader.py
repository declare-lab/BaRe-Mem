"""Math answer grading via HuggingFace `math-verify` (symbolic equivalence).

The repo's original `math_equal` (string-normalize + numeric isclose) under-counts
on algebraic / multi-value / measurement answers (e.g. `10-4n` vs `-4n+10`,
`3^6` vs `729`, gold `4,7` vs `x=4 or x=7`). `math-verify` is the community
standard grader (HF LightEval / Open LLM Leaderboard): it parses LaTeX to sympy
and compares symbolically, covering sets/intervals/matrices/percent/units.

This module is a thin, fail-closed wrapper:
- import is optional — `MATH_VERIFY_AVAILABLE` is False if the lib is missing, so
  offline test runs without it still work (callers fall back to the old scorer).
- `verify_equiv` never raises into callers (any parse/verify error -> False).
- math-verify's default timeout is SIGALRM-based (main-thread only); since
  `math_equal` can be reached from inside asynchronous grading tasks,
  we DISABLE the signal timeout (`timeout_seconds=None`) and keep calls cheap.
"""
from __future__ import annotations

import logging
import re
import signal
import threading
from contextlib import contextmanager
from fractions import Fraction
from math import floor, log10

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    MATH_VERIFY_AVAILABLE = True
    _EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]
    # We pass timeout_seconds=None on purpose (asyncio-safe); silence the per-call
    # "Timeout is disabled" warning it emits, else it floods a 1000-problem eval.
    for _name in ("math_verify", "math_verify.grader", "latex2sympy2_extended"):
        logging.getLogger(_name).setLevel(logging.ERROR)
except Exception:  # pragma: no cover - exercised only when lib is absent
    MATH_VERIFY_AVAILABLE = False
    _EXTRACTION = []


def _wrap_latex(s: str) -> str:
    r"""Light-clean an answer and give math-verify a LaTeX anchor.

    Without `$...$` or `\boxed`, parse() falls back to bare-number extraction and
    mangles expressions (`10-4 n` -> 6); wrapping in `$...$` uses the LaTeX path.
    We also drop the space in "<digit> <letter>" (e.g. gold `13-8 i`), which
    parses fine but makes verify() spuriously return False for an otherwise-equal
    complex number — a known math-verify normalization quirk. This is whitespace
    only; it never changes the math (unlike normalize_answer).
    """
    s = str(s).strip()
    s = re.sub(r"(\d)\s+([A-Za-z])", r"\1\2", s)
    if "\\boxed" in s or (s.startswith("$") and s.endswith("$")):
        return s
    return f"${s}$"


def _has_factorial_bomb(parsed) -> bool:
    r"""True if a parsed expr would make sympy try to evaluate an astronomically
    large number, hanging verify() in un-interruptible C recursion.

    Three known degenerate-peer forms:
    - factorial of a huge/non-numeric arg, e.g. ``(3^{100}-1)!``;
    - a power with a huge exponent (power tower), e.g. ``5^{2^{303}-1}-1`` — the
      exponent ``2^303`` is ~10^91, so evaluating the Pow blows up.
    - an unevaluated ``Sum``/``Product``/``Integral`` (e.g. a peer answering with
      ``\sum_{k=0}^{20} \binom{199}{k}(-1)^k\binom{301-5k}{103-5k}``): verify()
      calls ``simplify()`` on it, which sends ``_expandsums``/``_eval_simplify``
      into a multi-minute symbolic expansion that SIGALRM can't interrupt.
    Inspect the expression TREE (cheap) and bail before verify. Real answers use
    tiny factorials/exponents and never carry an unevaluated Sum/Product/Integral,
    so this never rejects a valid match.
    """
    try:
        from sympy import factorial, Pow
        from sympy.concrete.summations import Sum
        from sympy.concrete.products import Product
        from sympy.integrals.integrals import Integral
    except Exception:
        return False
    expr = parsed[0] if isinstance(parsed, (list, tuple)) and parsed else parsed
    if not hasattr(expr, "atoms"):
        return False
    # Unevaluated Sum/Product/Integral: simplify() on these can hang for minutes.
    try:
        if expr.atoms(Sum, Product, Integral):
            return True
    except Exception:
        return True
    # factorial(x): huge or non-numeric arg
    try:
        for f in expr.atoms(factorial):
            arg = f.args[0]
            if arg.is_number:
                try:
                    if abs(int(arg)) > 10000:
                        return True
                except Exception:
                    return True
            else:
                return True
    except Exception:
        return True
    # Pow with a huge exponent (power tower)
    try:
        for p in expr.atoms(Pow):
            exp = p.exp
            if exp.is_number:
                try:
                    if abs(float(exp)) > 1000:
                        return True
                except (TypeError, ValueError, OverflowError):
                    return True  # exponent itself isn't a plain float -> nested/huge
    except Exception:
        return True
    return False


def _verify_one_way(gold: str, pred: str) -> bool:
    # math-verify signature is verify(gold, target) — gold first.
    gold_parsed = parse(_wrap_latex(gold), extraction_config=_EXTRACTION)
    pred_parsed = parse(_wrap_latex(pred), extraction_config=_EXTRACTION)
    if not gold_parsed or not pred_parsed:
        return False
    # parse() is cheap+safe; verify() can hang forever evaluating a huge factorial
    # (un-interruptible C recursion). Skip such pathological inputs (fail-closed).
    if _has_factorial_bomb(pred_parsed) or _has_factorial_bomb(gold_parsed):
        return False
    return bool(verify(gold_parsed, pred_parsed, timeout_seconds=None))


class _GraderTimeout(Exception):
    pass


# LaTeX constructs that math-verify maps to sympy objects whose simplify()/evalf can
# hang in un-interruptible C recursion (SIGALRM can't preempt it). Matched at the
# STRING level (before parse()) so we never even hand these to sympy:
#   - Sum/Product/Integral/Limit/oo  (euler_maclaurin, etc.)
#   - transcendental fns log/ln/exp/trig  (logcombine/_eval_power on a \log vs a huge
#     fraction sent simplify into an infinite loop on a real OlympiadBench answer
#     gold=\log2/\log2-\log3 vs pred=1162589235/3777893186208).
# Closed-form numeric/algebraic final answers essentially never contain a bare
# \log/\exp/\sin..., so this only skips garbage; the sig-fig fallback still runs.
_HANG_PRONE_RE = re.compile(
    r"\\(sum|prod|int|iint|iiint|oint|lim|infty|nabla|partial)\b"
    # transcendental fns: NO \b — a real answer writes \log2 (fn name glued to its
    # arg), and \b fails between 'g' and '2'. Match the LaTeX command directly.
    r"|\\(log|ln|exp|sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|arcsin|arccos|arctan)"
    r"|\\frac\{d\}|∞"
)


# ---------------------------------------------------------------------------
# Persistent-subprocess HARD timeout for verify().
#
# SIGALRM (the _hard_timeout below) cannot preempt sympy when it is deep in C-level
# recursion (dmp_prem polynomial division, logcombine/_eval_power, euler_maclaurin,
# ...). College-math ODE answers (trig+exp symbolic) trigger several such paths the
# string pre-filter can't fully enumerate. To bound EVERY verify call regardless of
# which C path it takes, run the symbolic compare in a daemon child process and
# SIGKILL it on timeout (a real kill works even on un-interruptible C loops). A
# single long-lived worker is reused; respawned only after a timeout kill, so the
# per-call cost is just a pipe round-trip in the common (fast) case.
# ---------------------------------------------------------------------------
import multiprocessing as _mp

_WORKER = None
_WORKER_LOCK = None


def _grader_worker_loop(conn):  # pragma: no cover - runs in child process
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        gold, pred = msg
        try:
            res = bool(_verify_one_way(gold, pred) or _verify_one_way(pred, gold))
        except Exception:
            res = False
        try:
            conn.send(res)
        except Exception:
            break


def _ensure_worker():
    global _WORKER, _WORKER_LOCK
    import threading
    if _WORKER_LOCK is None:
        _WORKER_LOCK = threading.Lock()
    if _WORKER is None or not _WORKER["proc"].is_alive():
        ctx = _mp.get_context("fork")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_grader_worker_loop, args=(child,), daemon=True)
        proc.start()
        _WORKER = {"proc": proc, "conn": parent}
    return _WORKER


def _kill_worker():
    global _WORKER
    if _WORKER is not None:
        try:
            _WORKER["proc"].kill()
        except Exception:
            pass
        _WORKER = None


def _verify_hard(gold: str, pred: str, timeout: float = 4.0) -> bool:
    """Symbolic verify (both directions) bounded by a hard wall-clock kill."""
    if _WORKER_LOCK is None:
        _ensure_worker()
    with _WORKER_LOCK:
        try:
            w = _ensure_worker()
            w["conn"].send((gold, pred))
            if w["conn"].poll(timeout):
                return bool(w["conn"].recv())
            _kill_worker()  # child wedged in C recursion -> hard kill
            return False
        except Exception:
            _kill_worker()
            return False



@contextmanager
def _hard_timeout(seconds: int):
    """SIGALRM-based wall-clock guard around sympy parse/verify.

    math-verify's own timeout is disabled (asyncio-safe), but short-yet-pathological
    LaTeX (e.g. nested factorials from a degenerate peer output) can drive sympy into
    deep recursion that hangs the process. SIGALRM only works on the main thread, so
    this is a no-op elsewhere; the len()<=200 guard and fail-closed except still apply.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _on_alarm(signum, frame):
        raise _GraderTimeout()

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def verify_equiv(pred: str, gold: str) -> bool:
    """Return True if pred and gold are mathematically equivalent. Fail-closed."""
    if not MATH_VERIFY_AVAILABLE:
        return False
    # Cheap STRING-level pre-filter, run BEFORE parse()/verify() because parse()
    # itself can hang on these. math-verify turns \sum/\prod/\int/\lim/\infty into
    # sympy Sum/Product/Integral/Limit/oo, whose evalf drives euler_maclaurin into
    # un-interruptible C recursion (SIGALRM can't preempt it). Real final answers are
    # closed-form and never contain these, so skipping the symbolic path here never
    # rejects a valid match — the sig-fig fallback below still runs.
    if _HANG_PRONE_RE.search(str(pred)) or _HANG_PRONE_RE.search(str(gold)):
        return _sigfig_equal(pred, gold)
    # Bound the symbolic path with a HARD subprocess kill (SIGALRM can't preempt
    # sympy's C-level recursion — dmp_prem/logcombine/euler_maclaurin). Long strings
    # are usually garbage/truncated; cap to keep the worker cheap. The sig-fig
    # fallback still runs on timeout/miss.
    if len(str(pred)) <= 200 and len(str(gold)) <= 200:
        if _verify_hard(gold, pred, timeout=4.0):
            return True
    return _sigfig_equal(pred, gold)


# ---------------------------------------------------------------------------
# Significant-figures fallback for rounded measurement answers (e.g. minerva
# physics: gold 1.6 vs pred 1.57). math-verify treats these as unequal. We only
# apply this when at least one side is a decimal, and compare both rounded to the
# COARSER side's significant-figure count, so integer-answer benchmarks (aime,
# amc) are never affected.
# ---------------------------------------------------------------------------

def _to_number(text: str) -> float | None:
    t = str(text).strip().replace("$", "").replace(",", "")
    t = re.sub(r"\\times\s*10\^?\{?(-?\d+)\}?", r"e\1", t)
    t = re.sub(r"\\text\s*\{[^{}]*\}", "", t).strip()
    try:
        return float(Fraction(t))
    except Exception:
        try:
            return float(t)
        except ValueError:
            return None


def _is_decimal(s: str) -> bool:
    s = str(s).replace("$", "").strip()
    s = re.sub(r"\\text\s*\{[^{}]*\}", "", s).strip().replace(" ", "")
    return bool(re.fullmatch(r"-?\d*\.\d+(e-?\d+)?", s))


def _n_sig(s: str) -> int:
    s = str(s).replace("$", "").strip()
    s = re.sub(r"\\text\s*\{[^{}]*\}", "", s).strip().replace(" ", "")
    s = re.sub(r"e-?\d+$", "", s)
    digits = re.sub(r"[^\d]", "", s.lstrip("-").lstrip("0").replace(".", ""))
    return max(1, len(digits))


def _sig_round(x: float, sig: int) -> float:
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def _sigfig_equal(pred: str, gold: str) -> bool:
    if not (_is_decimal(pred) or _is_decimal(gold)):
        return False
    x, y = _to_number(pred), _to_number(gold)
    if x is None or y is None:
        return False
    sig = min(
        _n_sig(pred) if _is_decimal(pred) else 99,
        _n_sig(gold) if _is_decimal(gold) else 99,
    )
    try:
        return _sig_round(x, sig) == _sig_round(y, sig)
    except Exception:
        return False
