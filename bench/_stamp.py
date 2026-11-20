"""Shared helper: stamp results/*.json with a provenance `_meta` block.

Run by `bench.run_all` as a post-run step (benches keep writing bare JSON —
zero edits to existing files). The stamp binds a result file to the exact
commit/tree state that produced it, which is what `run_all --check-stale`
audits.

Stamp schema (added as top-level key `_meta`):

    _meta = {
        "git_sha":         <full HEAD sha>,
        "git_dirty":       <bool: non-empty `git status --porcelain`>,
        "bench_module":    <module run, e.g. "bench.run_phase0">,
        "command":         <exact command line>,
        "timestamp_utc":   <ISO-8601 UTC, at stamp time>,
        "python_version":  <platform.python_version()>,
        "numpy_version":   <numpy.__version__ or None>,
        "scipy_version":   <scipy.__version__ or None>,
    }

Note on dirty trees: the stamp is written AFTER the bench finishes, so the
results files themselves show up in `git status`. A tree that only has
modified results/*.json (plus run_all.json) is the normal post-run state and
is reported as dirty=true — that is fine; staleness compares the recorded
sha against HEAD, and records the dirty flag at *run* time for provenance,
not as a pass/fail condition on its own.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git_state():
    """Return (sha, dirty) for the current tree; (None, None) outside a repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() != ""
        return (sha or None), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _versions():
    np = sp = None
    try:
        import numpy
        np = numpy.__version__
    except ImportError:
        pass
    try:
        import scipy
        sp = scipy.__version__
    except ImportError:
        pass
    return np, sp


def meta_for(bench_module: str, command: str) -> dict:
    sha, dirty = git_state()
    np, sp = _versions()
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "bench_module": bench_module,
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": np,
        "scipy_version": sp,
    }


def stamp_json(path: Path, bench_module: str, command: str) -> bool:
    """Add/refresh `_meta` in a results json. True if stamped."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    data["_meta"] = meta_for(bench_module, command)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True
