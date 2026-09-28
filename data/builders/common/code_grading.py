"""Sandboxed execution of model-generated Python for code-task correctness.

Used offline by ``data.builders.common.score_code_peers`` to compute pass@1 once, which
is then stored in each record's ``peer_correct`` map. The pipeline's
loop never executes code — it only reads that label. Keeping execution out of the
hot loop is what makes the code task safe and deterministic.

SECURITY: running model-generated code is inherently unsafe. We run each program
in a fresh subprocess with `python -I` (isolated mode), a wall-clock timeout, and
(on POSIX) RLIMIT address-space/CPU caps. This is NOT a real security sandbox —
run data prep on a disposable/containerised machine, never on a host with secrets
or network you care about. Set FEEDBACK_CODE_EXEC_ALLOW=1 to acknowledge.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT = 10.0
DEFAULT_MEM_MB = 1024


@dataclass
class ExecResult:
    passed: bool
    error: str = ""


_PREAMBLE = """\
import resource, sys
def _limit(mem_mb):
    try:
        soft = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    except Exception:
        pass
_limit({mem})
"""


def _build_runner(program: str, *, mem_mb: int, cpu_s: int) -> str:
    # On POSIX add resource limits; on other platforms skip the preamble.
    preamble = ""
    if os.name == "posix":
        preamble = _PREAMBLE.format(mem=mem_mb, cpu=cpu_s)
    return preamble + "\n" + program


def run_python(
    program: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    mem_mb: int = DEFAULT_MEM_MB,
    stdin: str | None = None,
    env: dict | None = None,
    tail: int = 500,
    private_tmp: bool = False,
) -> ExecResult:
    """Run a standalone Python program; pass == clean exit code 0.

    env: variables added to the inherited environment (e.g. single-threaded BLAS); tail: characters of stderr kept on failure;
    private_tmp: the program's tempfile.gettempdir() is its own directory (tests that write to gettempdir()/<fixed name>
    otherwise collide when programs run in parallel).
    """
    if os.environ.get("FEEDBACK_CODE_EXEC_ALLOW") != "1":
        raise RuntimeError(
            "Refusing to execute model-generated code. Set FEEDBACK_CODE_EXEC_ALLOW=1 "
            "(ideally inside a disposable container) to acknowledge the risk."
        )
    source = _build_runner(program, mem_mb=mem_mb, cpu_s=int(timeout) + 1)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "prog.py"
        script.write_text(source)
        extra = {str(k): str(v) for k, v in (env or {}).items()}
        if private_tmp:
            own = Path(tmp) / "tmp"
            own.mkdir()
            extra.update(TMPDIR=str(own), TEMP=str(own), TMP=str(own))
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                input=(stdin or "").encode() if stdin is not None else None,
                capture_output=True,
                timeout=timeout,
                cwd=tmp,
                env=dict(os.environ, **extra) if extra else None,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout")
        except Exception as exc:  # pragma: no cover - defensive
            return ExecResult(False, f"runner-error: {exc}")
    if proc.returncode == 0:
        return ExecResult(True, "")
    err = (proc.stderr or b"").decode(errors="replace")[-int(tail):]
    return ExecResult(False, err.strip() or f"exit {proc.returncode}")


# ---------------------------------------------------------------------------
# Per-format glue: assemble (peer_code + tests) into one runnable program.
# ---------------------------------------------------------------------------

def check_asserts(
    peer_code: str,
    test_code: str,
    *,
    entry_point: str | None = None,
    setup_code: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> ExecResult:
    """HumanEval / MBPP / BigCodeBench style: peer defines a function, tests assert.

    HumanEval ``test`` defines ``check(candidate)`` and needs an explicit call;
    MBPP/BigCodeBench tests are bare ``assert`` statements that call the function
    by name (already defined in ``peer_code``).
    """
    parts = [peer_code, "", setup_code, "", test_code, ""]
    if entry_point and "check(" in test_code and f"check({entry_point}" not in test_code:
        parts.append(f"check({entry_point})")
    program = "\n".join(parts)
    return run_python(program, timeout=timeout)


_UNITTEST_RUNNER = """
import unittest as _ut, sys as _sys
_suite = _ut.TestLoader().loadTestsFromModule(_sys.modules['__main__'])
_res = _ut.TextTestRunner(verbosity=0).run(_suite)
_sys.exit(0 if _res.wasSuccessful() else 1)
"""


def check_unittest(
    peer_code: str,
    test_code: str,
    *,
    setup_code: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> ExecResult:
    """BigCodeBench style: ``test_code`` defines a ``unittest.TestCase``."""
    program = "\n".join([peer_code, "", setup_code, "", test_code, _UNITTEST_RUNNER])
    return run_python(program, timeout=timeout)


def check_io(
    peer_code: str,
    cases: list[dict],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> ExecResult:
    """LiveCodeBench stdin/stdout style: run ``peer_code`` per case, diff stdout.

    Each case is ``{"input": str, "output": str}``. The peer program is expected
    to read stdin and print the answer. Passes iff every case's stripped stdout
    matches.
    """
    for case in cases:
        result = _run_io_case(peer_code, str(case.get("input", "")), str(case.get("output", "")), timeout)
        if not result.passed:
            return result
    return ExecResult(True, "")


def _run_io_case(peer_code: str, stdin: str, expected: str, timeout: float) -> ExecResult:
    if os.environ.get("FEEDBACK_CODE_EXEC_ALLOW") != "1":
        raise RuntimeError("Set FEEDBACK_CODE_EXEC_ALLOW=1 to execute model-generated code.")
    source = _build_runner(peer_code, mem_mb=DEFAULT_MEM_MB, cpu_s=int(timeout) + 1)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "prog.py"
        script.write_text(source)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                input=stdin.encode(),
                capture_output=True,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout")
    if proc.returncode != 0:
        return ExecResult(False, (proc.stderr or b"").decode(errors="replace")[-300:])
    got = (proc.stdout or b"").decode(errors="replace").strip()
    return ExecResult(got == expected.strip(), "" if got == expected.strip() else "output-mismatch")


def check_functional(
    peer_code: str,
    cases: list[dict],
    entry_point: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> ExecResult:
    """LiveCodeBench functional style: call ``entry_point(*input)`` and compare.

    Each case is ``{"input": <json-or-list>, "output": <json>}``. ``input`` is
    JSON-decoded and splatted as positional args.
    """
    harness = textwrap.dedent(
        f"""
        import json, sys
        _cases = json.loads({json.dumps(json.dumps(cases))})
        for _c in _cases:
            _inp = _c["input"]
            if isinstance(_inp, str):
                try:
                    _inp = json.loads(_inp)
                except Exception:
                    _inp = [_inp]
            if not isinstance(_inp, list):
                _inp = [_inp]
            _exp = _c["output"]
            if isinstance(_exp, str):
                try:
                    _exp = json.loads(_exp)
                except Exception:
                    pass
            _got = {entry_point}(*_inp)
            assert _got == _exp, f"expected {{_exp!r}} got {{_got!r}}"
        """
    )
    program = peer_code + "\n" + harness
    return run_python(program, timeout=timeout)


# ---------------------------------------------------------------------------
# Dispatch by code-record format (set by the dataset loader).
# ---------------------------------------------------------------------------

def score_code_record(record: dict, peer_code: str, *, timeout: float = DEFAULT_TIMEOUT) -> ExecResult:
    """Run one peer's code against a code record's tests. Returns ExecResult.

    ``record["code_format"]`` selects the harness:
      * "asserts"    -> check_asserts (HumanEval/MBPP/BigCodeBench)
      * "io"         -> check_io (LiveCodeBench stdin/stdout)
      * "functional" -> check_functional (LiveCodeBench call-based)
    """
    fmt = str(record.get("code_format", "asserts"))
    if not peer_code.strip():
        return ExecResult(False, "empty")
    if fmt == "io":
        return check_io(peer_code, list(record.get("test_cases", [])), timeout=timeout)
    if fmt == "functional":
        return check_functional(
            peer_code, list(record.get("test_cases", [])),
            str(record.get("entry_point", "")), timeout=timeout,
        )
    if fmt == "unittest":
        return check_unittest(
            peer_code, str(record.get("test", "")),
            setup_code=str(record.get("test_setup", "")), timeout=timeout,
        )
    return check_asserts(
        peer_code,
        str(record.get("test", "")),
        entry_point=record.get("entry_point") or None,
        setup_code=str(record.get("test_setup", "")),
        timeout=timeout,
    )
