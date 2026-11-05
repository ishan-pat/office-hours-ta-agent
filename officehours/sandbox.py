"""Run short Python snippets the model writes, to check arithmetic and formulas.

This is a *convenience* sandbox, not a security boundary: a separate
interpreter in isolated mode, an empty temp working directory, a stripped
environment (no API keys), a wall-clock timeout and capped output. It is fine
for a local study tool; anything multi-user would need a container.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

TIMEOUT_S = 5
MAX_OUTPUT = 4000


def run_python(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {TIMEOUT_S}s."
    out = (proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else ""))
    out = out.strip() or "(no output; use print())"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + "\n...[truncated]"
    return proc.returncode == 0, out
