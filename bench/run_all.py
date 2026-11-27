"""Single-command bench runner: `python3 -m bench.run_all [--spice] [--fast] [--check-stale]`.

Runs the canonical Stage-A/B bench set (the AGENTS.md "Commands" list,
hardcoded in dependency-free order — glob discovery is fragile), one
sequential subprocess per bench with the environment pinned:

    OPENBLAS_NUM_THREADS=1  OMP_NUM_THREADS=1  MKL_NUM_THREADS=1

(64-thread OpenBLAS makes a 128x128 complex solve/SVD ~70x slower on this
box; every bench sets it internally via `os.environ.setdefault`, but the
runner pins it in the child env too so no bench can import numpy first with
it unset.)

After each bench finishes, its expected results/*.json is stamped with a
`_meta` provenance block (see bench/_stamp.py) — benches keep writing bare
JSON, stamping is a post-run step only. A missing expected json counts as a
failed bench. Exit code is nonzero if any bench fails or times out.

Options:
  --spice        also run the opt-in SPICE benches (twin.spice.extract /
                 twin.spice.extract_b2). They need ngspice; the child env has
                 LD_LIBRARY_PATH removed (the agent-injected libstdc++ breaks
                 ngspice: GLIBCXX_3.4.32 missing — see twin/spice notes).
                 Default: skipped.
  --fast         skip the long benches (pipeline, substrate, sde, multirate,
                 twin_experiments). For quick smoke runs only — not a green
                 run.
  --check-stale  do not run benches; audit results/*.json staleness against
                 the current HEAD and exit nonzero if any file is stale
                 (wrong sha, dirty-tree mismatch, or missing _meta).

Output: results/run_all.json with per-bench {exit, seconds, checks},
plus live per-bench tails in results/.run_all_logs/<bench>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
LOGS = RESULTS / ".run_all_logs"
PER_BENCH_TIMEOUT_S = 900          # default per-bench timeout
# Known-long benches get an extended cap: on this loaded box (load avg ~29,
# concurrent ngspice/yosys/openroad) run_pipeline measured >15 min wall and
# would be killed at the default 900 s. 45 min gives it air without ever
# masking a true hang forever.
LONG_TIMEOUT_S = 2700
# Total budget: the "uniform ~15 min" nominal from the spec is unrealizable
# on this loaded box — measured 2026-11-19: twin_experiments 347 s,
# phase0 116 s, transform 130 s, pipeline ~5 min ⇒ full non-SPICE suite
# ~20-24 min. The B2 joint closure (run_b2_joint: 3 anchors through
# compose.run_chain, ~5-10 min) was added on top; 60 min keeps the whole
# 25-suite set inside one run with margin. A bench that would start with
# <60 s remaining is skipped and recorded instead of half-run.
TOTAL_BUDGET_S = 60 * 60
TAIL_LINES = 12

# The canonical bench set, verbatim from AGENTS.md "Commands", in
# dependency-free order (nothing later imports from an earlier entry's
# fresh output; run_b2 only reads the pre-made results/b2_extract.json).
# json = the bench's own primary output file (existence-checked, then
# stamped); report = the human-readable report (existence-checked, parsed
# for "N/M checks pass").
BENCHES = [
    # module                        json                       report                        long
    ("run_twin_experiments",      "twin_experiments.json",   "report.txt",                  True),
    ("run_phase0",                "phase0.json",             "phase0_report.txt",           False),
    ("run_lyapunov",              "lyapunov.json",           "lyapunov_report.txt",         False),
    ("run_reduce",                "reduce.json",             "reduce_report.txt",           False),
    ("run_skew",                  "skew.json",               "skew_report.txt",             False),
    ("run_route",                 "route.json",              "route_report.txt",            False),
    ("run_transform",             "transform.json",          "transform_report.txt",        False),
    ("run_sde",                   "sde.json",                "sde_report.txt",              True),
    ("run_multirate",             "multirate.json",          "multirate_report.txt",        True),
    ("run_compiler",              "compiler.json",           "compiler_report.txt",         False),
    ("run_constraint",            "constraint.json",         "constraint_report.txt",       False),
    ("run_constraint_timing",     "constraint_timing.json",  "constraint_timing_report.txt", False),
    ("run_pipeline",              "pipeline.json",           "pipeline_report.txt",         "long"),
    ("run_substrate",             "substrate.json",          "substrate_report.txt",        True),
    ("run_stagea",                "stagea.json",             "stagea_report.txt",           False),
    ("run_closure_split",         "closure_split.json",      "closure_split_report.txt",    False),
    ("run_basis_interface",       "basis_interface.json",    "basis_interface_report.txt",  False),
    ("run_calib",                 "calib.json",              "calib_report.txt",            False),
    ("run_p7_netlist",            "p7_netlist.json",         "p7_netlist_report.txt",       False),
    ("run_route_contract",        "route_contract.json",     "route_contract_report.txt",   False),
    ("run_scaling",               "scaling.json",            None,                          False),
    ("run_error_control",         "error_control.json",      "error_control_report.txt",    False),
    ("run_compiler_e2e",          "compiler_e2e.json",       "compiler_e2e_report.txt",     False),
    ("run_composition",           "composition.json",        "composition_report.txt",      False),
    ("run_precision",             "precision.json",          "precision_report.txt",        False),
    ("run_b2",                    "b2.json",                 None,                          False),
    ("run_b2_behavioral",         "b2_behavioral.json",      None,                          False),
    ("run_b2_joint",              "b2_joint.json",           None,                          "long"),
]

# Opt-in only: real ngspice extraction (~70 s per PDK / ~10 min for b2).
SPICE_BENCHES = [
    ("twin.spice.extract",    "device_extract.json",  None, False),
    ("twin.spice.extract_b2", "b2_extract.json",      None, False),
]

# "N/M checks pass" style line in the *_report.txt files; some reports use
# Chinese variants, so fall back to any "a/b" followed by "check".
CHECK_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+checks?\s+pass")


def pinned_env(remove_ld_library_path: bool = False) -> dict:
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # PATH unchanged. Only strip the agent-injected libstdc++ for ngspice.
    if remove_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def parse_checks(report_path: Path):
    """(passed, total) from the report's 'N/M checks pass' line, or None."""
    try:
        text = report_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = CHECK_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def run_bench(module: str, timeout_s: int, remove_ld_library_path: bool):
    """Run `python3 -u -m <module>` from the repo root. Returns (exit, seconds)."""
    fq = module if module.startswith("twin.") else f"bench.{module}"
    log_path = LOGS / f"{module.replace('.', '_')}.log"
    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", fq], cwd=REPO,
            env=pinned_env(remove_ld_library_path),
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            exit_code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            exit_code = "timeout"
    seconds = round(time.monotonic() - t0, 1)
    tail(log_path)
    return exit_code, seconds


def tail(log_path: Path):
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if lines:
        print("    --- tail ---")
        for line in lines[-TAIL_LINES:]:
            print("    " + line)


def check_stale() -> int:
    """Audit every results/*.json that has (or should have) _meta vs HEAD."""
    sys.path.insert(0, str(REPO))
    from bench._stamp import git_state  # noqa: E402

    sha, dirty = git_state()
    if sha is None:
        print("check-stale: not a git repo, cannot audit")
        return 2
    print(f"check-stale: HEAD={sha[:12]} dirty={dirty}")
    tracked = {b[1] for b in BENCHES} | {b[1] for b in SPICE_BENCHES}
    default_tracked = {b[1] for b in BENCHES}
    stale = 0
    seen = 0
    untracked_skipped = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name == "run_all.json":
            continue
        if path.name not in tracked:
            # Auxiliary jsons no bench owns as its primary output (e.g.
            # ledger.json written alongside stagea.json) are not audited.
            untracked_skipped.append(path.name)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f).get("_meta")
        except (OSError, json.JSONDecodeError, AttributeError):
            meta = None
        reasons = []
        if not isinstance(meta, dict):
            # Missing _meta is stale only for the default (non-SPICE) set;
            # SPICE-primary outputs (b2_extract.json, device_extract.json)
            # are stamped only by opt-in runs, so a bare file there just
            # means SPICE hasn't been run under run_all yet.
            if path.name in default_tracked:
                reasons.append("missing _meta (never stamped)")
            else:
                untracked_skipped.append(path.name + " (spice, not stamped)")
                continue
        else:
            if meta.get("git_sha") != sha:
                reasons.append(f"json sha {str(meta.get('git_sha'))[:12]} != HEAD")
            if bool(meta.get("git_dirty")) != bool(dirty):
                reasons.append(f"json git_dirty={meta.get('git_dirty')} != current {dirty}")
        if reasons:
            stale += 1
            print(f"  STALE {path.name}: {'; '.join(reasons)}")
        else:
            seen += 1
    print(f"check-stale: {seen} fresh, {stale} stale "
          f"(tracked files without _meta count as stale — run the benches "
          f"to stamp them; {len(untracked_skipped)} auxiliary files skipped: "
          f"{', '.join(untracked_skipped) or 'none'})")
    return 1 if stale else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench.run_all")
    ap.add_argument("--spice", action="store_true",
                    help="also run the ngspice benches (twin.spice.extract*)")
    ap.add_argument("--fast", action="store_true",
                    help="skip long benches (pipeline/substrate/sde/multirate/twin_experiments)")
    ap.add_argument("--check-stale", action="store_true",
                    help="audit results/*.json _meta vs HEAD instead of running")
    args = ap.parse_args(argv)

    if args.check_stale:
        return check_stale()

    sys.path.insert(0, str(REPO))
    from bench._stamp import stamp_json  # noqa: E402

    skipped = []
    plan = list(BENCHES)
    if args.fast:
        plan = [b for b in plan if not b[3]]
        skipped += [b[0] for b in BENCHES if b[3]]
    if args.spice:
        plan += SPICE_BENCHES

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOGS.mkdir(exist_ok=True)
    from bench._stamp import git_state  # noqa: E402
    sha, dirty = git_state()

    print(f"run_all: {len(plan)} benches (spice={args.spice}, fast={args.fast}), "
          f"HEAD={str(sha)[:12] if sha else '?'}, dirty={dirty}")

    per_bench = {}
    failures = []
    t_all0 = time.monotonic()
    for module, json_name, report_name, is_long in plan:
        remaining = TOTAL_BUDGET_S - (time.monotonic() - t_all0)
        if remaining <= 60:
            print(f"[SKIP] {module}: total budget ({TOTAL_BUDGET_S}s) exhausted")
            per_bench[module] = {"exit": "skipped-budget", "seconds": 0.0,
                                 "checks": None}
            skipped.append(module)
            continue
        cap = LONG_TIMEOUT_S if is_long == "long" else PER_BENCH_TIMEOUT_S
        timeout = min(cap, int(max(remaining, 60)))
        print(f"[RUN ] {module} (timeout {timeout}s)")
        exit_code, seconds = run_bench(
            module, timeout, remove_ld_library_path=module.startswith("twin.spice"))

        entry = {"exit": exit_code, "seconds": seconds, "checks": None}
        json_path = RESULTS / json_name
        report_path = RESULTS / report_name if report_name else None

        if exit_code == 0:
            if not json_path.exists():
                print(f"    MISSING expected output {results_rel(json_path)}")
                entry["exit"] = "missing-output"
            elif stamp_json(json_path, f"bench.{module}", f"python3 -m bench.{module}"):
                entry["stamped"] = json_name
            if report_path and not report_path.exists():
                print(f"    MISSING expected report {results_rel(report_path)}")
                entry["exit"] = "missing-output"
            if report_path and entry["exit"] == 0:
                entry["checks"] = parse_checks(report_path)
        if exit_code != 0 or entry["exit"] != 0:
            failures.append(module)

        status = "ok" if entry["exit"] == 0 else f"FAIL({entry['exit']})"
        checks = entry["checks"]
        checks_s = f" {checks[0]}/{checks[1]} checks" if checks else ""
        print(f"[{status.upper():>5}] {module} in {seconds}s{checks_s}")
        per_bench[module] = entry

    total_s = round(time.monotonic() - t_all0, 1)
    summary = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "flags": {"spice": args.spice, "fast": args.fast},
        "total_seconds": total_s,
        "per_bench": per_bench,
        "skipped": skipped,
        "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "run_all.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nrun_all: {len(plan) - len(failures) - len([m for m in skipped if m in per_bench])}"
          f"/{len(plan)} ok in {total_s}s; failures: {failures or 'none'}; "
          f"skipped: {skipped or 'none'}")
    print(f"record: results/run_all.json (logs: results/.run_all_logs/)")
    return 1 if failures else 0


def results_rel(p: Path) -> str:
    return str(p.relative_to(REPO))


if __name__ == "__main__":
    sys.exit(main())
