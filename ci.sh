#!/bin/sh
# Regression harness entry point: full non-SPICE bench suite + staleness audit.
# Usage: ./ci.sh            (full, ~10-15 min on a loaded box)
#        ./ci.sh --fast     (skip long benches; not a green run)
set -eu

# REQUIRED: 64-thread OpenBLAS makes 128x128 complex solve/svd ~70x slower
# on this box; every bench sets it internally via os.environ.setdefault,
# but pin it here so nothing imports numpy with it unset.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NOTE (system-level): the SPICE benches (opt-in, `python3 -m bench.run_all
# --spice`) need ngspice and, inside the Trae agent shell, must run as
#   env -u LD_LIBRARY_PATH  <ngspice ...>
# because the agent injects a libstdc++ that ngspice cannot load
# (GLIBCXX_3.4.32 missing). run_all handles this for --spice children.
# A normal login shell needs neither workaround.

echo "== ci.sh: bench suite =="
python3 -m bench.run_all "$@"

echo "== ci.sh: staleness audit =="
python3 -m bench.run_all --check-stale

echo "== ci.sh: summary =="
python3 - <<'EOF'
import json
with open("results/run_all.json", encoding="utf-8") as f:
    r = json.load(f)
print(f"suite: {r['total_seconds']}s, failures: {r['failures'] or 'none'}, "
      f"skipped: {r['skipped'] or 'none'}")
for mod, e in r["per_bench"].items():
    checks = e.get("checks")
    cs = f" [{checks[0]}/{checks[1]} checks]" if checks else ""
    print(f"  {mod}: exit={e['exit']} {e['seconds']}s{cs}")
EOF
echo "ci.sh: OK"
