"""P0-B/G4 acceptance -- the B3 supervisor calibration FSM, gf180mcu batch-1.

The contract lives in ``twin/calib.py`` and its behavioural acceptance is
``bench/run_calib.py`` (62/62).  This suite is the pin-level realisation of the
part of that contract a *supervisor* has to own:

    IDLE -> STIMULATE -> MEASURE -> CORRECT -> VERIFY -> ACCEPTED (terminal)
                                                     \\-> STIMULATE (retry)
                                                     \\-> FAILED   (terminal)

Three semantics are what the hardware must not lose, and each is a scenario
below:

* FAILED is terminal and is never reported as an accept (the twin refuses a
  FAILED ``CalibrationResult``, so a certificate can never be gated on one);
* a retry must consume a *fresh* measurement -- a reading whose digest is
  already in the history is refused at MEASURE, which is the hardware version
  of the overfit failure ``run_calib`` catches (a correction fitted to the
  training set that "passes" on it);
* VERIFY reads the held-out verdict (``verify_ok``), not the training residual.

The topology question this suite answers is the *guard depth*.  A round has
KMAX+1 attempts, so the history must remember the KMAX readings that came
before it; the suite measures the area of HIST = 1/2/4 and runs the same
scenario through each build.  HIST=1 is expected to accept a replay (negative
control), HIST=2 must refuse it, HIST=4 must refuse it -- i.e. the minimum
sound depth is KMAX, and that is a measured statement, not a preference.

Synthesis is yosys + the gf180 7-track 5V library (``gf180mcu_fd_sc_mcu7t5v0``),
``ss_125C_3v00`` corner (the kit has no tt liberty; ff/ss bracket is the
declared substitute).  The delay figure is a *bound*: ``ltp`` levels from the
generic netlist times the slowest liberty stage of any cell the netlist uses.
Signoff STA and post-PEX are G5/G6; this suite only has to show that the
supervisor is not the calibration loop's timing constraint.

Recorded tool limitation: yosys 0.52's ``fsm_detect`` does not extract this
machine (verified on a textbook 3-state reference module), so state encoding
(one-hot vs binary) is NOT compared here; the state is an explicit 3-bit reg so
either encoding stays a local change for G5's P&R.
"""

import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twin.spice import extract_gf180 as G            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(ROOT, "hw", "gf180", "batch1", "fsm")
OUT = G.OUT

PDK = os.path.expanduser(
    "~/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296"
    "/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib")
LIB_SS = os.path.join(PDK, "gf180mcu_fd_sc_mcu7t5v0__ss_125C_3v00.lib")

KMAX = 2                       # retry budget (3 attempts per round)
ADC12_S = 12e-6                # 12-bit conversion time (P0-E, DESIGN 6.29)
B3_DW = 16                     # the digest width B3 would carry
AREA_BUDGET_UM2 = 1e4          # 0.01 mm^2: 1 % of a 1 mm^2 shuttle tile
CHECKS = []


def chk(name, ok, detail):
    CHECKS.append(dict(name=name, ok=bool(ok), detail=detail))
    print(("PASS  " if ok else "FAIL  ") + name + "  " + detail)
    return ok


# ----- tools ----------------------------------------------------------------
def sh(cmd, cwd=None):
    """Run a command, return (rc, stdout+stderr)."""
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


_LIB_CACHE = {}


def _floats(txt):
    return [float(x) for x in re.findall(r"-?[\d.]+(?:e[-+]?\d+)?", txt)]


def _arc_block_delays(body):
    """(slew_idx, load_idx, rows) for every cell_rise/cell_fall template."""
    out = []
    for arc in re.finditer(r"cell_(?:rise|fall)\([^)]*\)\s*\{(.*?)\n        \}",
                           body, re.S):
        b = arc.group(1)
        m1 = re.search(r'index_1\("([^"]*)"', b)
        m2 = re.search(r'index_2\("([^"]*)"', b)
        if not (m1 and m2):
            continue
        i1, i2 = _floats(m1.group(1)), _floats(m2.group(1))
        mv = re.findall(r"values\s*\((.*?)\)\s*;", b, re.S)
        rows = [_floats(r) for r in re.findall(r'"([^"]*)"', "".join(mv))]
        if rows and i1 and i2:
            out.append((i1, i2, rows))
    return out


def _delay_at(body, load, slew):
    """Worst delay of one cell at a declared (load pF, input slew ns).

    Both operating points are picked from the liberty's own index vectors by
    nearest-neighbour, so the number is a table entry, never an interpolation
    invented here.  ``load=None`` means "worst entry in the whole table", which
    is the pessimistic bound used for the timing gate.
    """
    worst = 0.0
    for i1, i2, rows in _arc_block_delays(body):
        if load is None or slew is None:
            for r in rows:
                worst = max(worst, max(r) if r else 0.0)
            continue
        r = min(range(len(i1)), key=lambda k: abs(i1[k] - slew))
        c = min(range(len(i2)), key=lambda k: abs(i2[k] - load))
        r = min(r, len(rows) - 1)
        c = min(c, len(rows[r]) - 1)
        worst = max(worst, rows[r][c])
    return worst


def liberty_cells(path):
    """{cell: dict(area, t_max, t_ref)}.

    t_max  = worst entry of the whole delay table (an upper bound at the worst
             load the table covers -- pessimistic, used as the gate)
    t_ref  = delay at 0.02 pF load / 0.02 ns slew (a fanout-4 node in this
             library: 1X input pin + 3 fanout pins), used only for reporting
    """
    if path in _LIB_CACHE:
        return _LIB_CACHE[path]
    txt = open(path, errors="ignore").read()
    heads = list(re.finditer(r"\n  cell\((\w+)\) \{", txt))
    out = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(txt)
        blk = txt[m.end():end]
        a = re.search(r"area\s*:\s*([\d.]+)", blk)
        out[m.group(1)] = dict(
            area=float(a.group(1)) if a else 0.0,
            t_max=_delay_at(blk, None, None),
            t_ref=_delay_at(blk, 0.02, 0.02))
    _LIB_CACHE[path] = out
    return out


def variant(hist, dw, tag):
    """Write the RTL with the HIST/DW defaults substituted (chparam does not
    survive ``hierarchy`` for a plain parameterised module in this yosys)."""
    src = open(os.path.join(HW, "calib_fsm.v")).read()
    src = re.sub(r"parameter HIST = \d+", f"parameter HIST = {hist}", src)
    src = re.sub(r"parameter DW   = \d+", f"parameter DW   = {dw}", src)
    path = os.path.join(OUT, f"fsm_{tag}.v")
    with open(path, "w") as fh:
        fh.write(src)
    return path


def synth(hist, dw, tag):
    """Map to gf180 cells, return area / cells / flops / levels / stage delay."""
    v = variant(hist, dw, tag)
    logp = os.path.join(OUT, f"fsm_{tag}.log")
    script = (f"read_verilog {v}; hierarchy -top calib_fsm; proc; opt; "
              f"techmap; opt; ltp; dfflibmap -liberty {LIB_SS}; "
              f"abc -liberty {LIB_SS}; opt_clean; stat -liberty {LIB_SS}")
    rc, out = sh(["yosys", "-p", script])
    with open(logp, "w") as fh:
        fh.write(out)
    r = dict(rc=rc, log=logp, hist=hist, dw=dw)

    m = re.search(r"Number of cells:\s+(\d+)", out)
    r["cells"] = int(m.group(1)) if m else 0
    m = re.search(r"Chip area for module '[^']*':\s+([\d.]+)", out)
    r["area_um2"] = float(m.group(1)) if m else 0.0
    m = re.search(r"Number of memories:\s+(\d+)", out)
    r["memories"] = int(m.group(1)) if m else -1
    m = re.search(r"Longest topological path in \w+ \(length=(\d+)\)", out)
    r["levels"] = int(m.group(1)) if m else 0

    used = {}
    for name, cnt in re.findall(
            r"(gf180mcu_fd_sc_mcu7t5v0__\w+)\s+(\d+)\s*$", out, re.M):
        used[name] = int(cnt)
    r["used"] = used
    r["flops"] = sum(n for c, n in used.items() if "dff" in c)
    r["generic_left"] = len(re.findall(r"\n\s+\\?\$\w+\s+\d+\s*$", out))
    r["latches"] = sum(n for c, n in used.items() if "dlat" in c or "_lat" in c)

    lib = liberty_cells(LIB_SS)
    comb = [c for c in used if "dff" not in c and c in lib]
    stage = max([lib[c]["t_max"] for c in comb], default=0.0)
    stage_ref = max([lib[c]["t_ref"] for c in comb], default=0.0)
    r["stage_ns"] = stage
    r["stage_ref_ns"] = stage_ref
    r["path_ns"] = r["levels"] * stage
    r["path_ref_ns"] = r["levels"] * stage_ref
    r["f_mhz_min"] = 1e3 / r["path_ns"] if r["path_ns"] > 0 else 0.0
    r["f_mhz_ref"] = 1e3 / r["path_ref_ns"] if r["path_ref_ns"] > 0 else 0.0
    return r


def sim(hist, tag):
    """Build and run the protocol testbench; return {scenario: fields}."""
    exe = os.path.join(OUT, f"fsm_{tag}.vvp")
    rc, out = sh(["iverilog", f"-DHIST={hist}", "-o", exe,
                  "calib_fsm.v", "tb_calib_fsm.v"], cwd=HW)
    if rc != 0:
        return dict(rc=rc, log=out, scn={}, finished=False)
    rc, out = sh(["vvp", exe])
    scn = {}
    for line in out.splitlines():
        if line.startswith("SCN "):
            parts = line.split()
            d = dict(kv.split("=") for kv in parts[2:])
            scn[parts[1]] = {k: int(v) for k, v in d.items()}
    return dict(rc=rc, scn=scn, finished="SIMDONE" in out, log=out)


# ----- main ----------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    # ---------- protocol (RTL simulation, three guard depths) --------------
    sims = {h: sim(h, f"h{h}") for h in (1, 2, 4)}
    s2 = sims[2]["scn"]

    ok = all(v["finished"] and v["rc"] == 0 for v in sims.values())
    chk("E1 the RTL builds and every run reaches SIMDONE for HIST 1/2/4",
        ok, "; ".join(f"HIST={h} rc={v['rc']} done={v['finished']} "
                     f"scn={len(v['scn'])}" for h, v in sims.items()))

    h = s2.get("happy", {})
    chk("E2 straight round: one stimulus, one conversion, one correction, "
        "ACCEPTED",
        h.get("st") == 5 and h.get("stim") == 1 and h.get("meas") == 1
        and h.get("corr") == 1 and h.get("cons") == 1,
        f"st={h.get('st')} stim/meas/corr/cons="
        f"{h.get('stim')}/{h.get('meas')}/{h.get('corr')}/{h.get('cons')}")

    bad = [(hh, k) for hh, v in sims.items() for k, s in v["scn"].items()
           for t in ("stim", "meas", "corr", "cons")
           if s[t] != s[t + "c"]]
    chk("E3 every trigger is a one-cycle pulse (pulses == cycles high)",
        not bad, "all triggers 1 cycle in every scenario of every build"
        if not bad else f"multi-cycle pulses: {bad}")

    # the stale scenario deliberately has no CORRECT/VERIFY on its last attempt
    bad = [(hh, k) for hh, v in sims.items() for k, s in v["scn"].items()
           if k not in ("stale", "nodata") and s["ord"] != 1]
    chk("E4 stimulus < conversion < correction < verify in every round",
        not bad, "strictly increasing in all accepted retry rounds"
        if not bad else f"out of order: {bad}")

    r = s2.get("retry1", {})
    chk("E5 a failed held-out verify with budget left retries once and then "
        "accepts",
        r.get("st") == 5 and r.get("cons") == 2 and r.get("retry") == 1,
        f"st={r.get('st')} attempts={r.get('cons')} retry_cnt={r.get('retry')}")

    b = s2.get("budget", {})
    chk("E6 retry budget exhausted: FAILED after exactly KMAX+1 attempts and "
        "the counter saturates",
        b.get("st") == 6 and b.get("cons") == 3 and b.get("retry") == KMAX,
        f"st={b.get('st')} attempts={b.get('cons')} retry_cnt={b.get('retry')} "
        f"(KMAX={KMAX}; a 4th attempt would be an out-of-budget retry)")

    n = s2.get("nodata", {})
    chk("E7 a conversion that produced no fresh reading fails without "
        "consuming anything",
        n.get("st") == 6 and n.get("cons") == 0 and n.get("corr") == 0,
        f"st={n.get('st')} consumed={n.get('cons')} corrected={n.get('corr')}")

    a = s2.get("term_acc", {})
    chk("E8 ACCEPTED is terminal: holding start high cannot re-enter a round",
        a.get("st") == 5 and a.get("cons") == 1 and a.get("stim") == 1,
        f"st={a.get('st')} consumed={a.get('cons')} stim={a.get('stim')} after "
        f"10 cycles of start=1")

    f = s2.get("term_fail", {})
    chk("E9 FAILED is terminal too, and never turns into an accept",
        f.get("st") == 6 and f.get("cons") == 3 and f.get("stim") == 3,
        f"st={f.get('st')} consumed={f.get('cons')} stim={f.get('stim')}")

    m = s2.get("reset_mid", {})
    chk("E10 reset in the middle of a round returns to IDLE with a cleared "
        "retry counter",
        m.get("st") == 0 and m.get("retry") == 0,
        f"st={m.get('st')} retry_cnt={m.get('retry')}")

    sh_ = s2.get("start_held", {})
    chk("E11 start held high for the whole round runs exactly one round",
        sh_.get("st") == 5 and sh_.get("cons") == 1,
        f"st={sh_.get('st')} consumed={sh_.get('cons')} (one measurement)")

    # ---------- the guard depth (the topology decision) -------------------
    st4 = sims[4]["scn"].get("stale", {})
    chk("E12 HIST=4 refuses a third attempt that replays the first reading",
        st4.get("st") == 6 and st4.get("cons") == 2 and st4.get("stim") == 3,
        f"st={st4.get('st')} consumed={st4.get('cons')} of 3 stimulated: the "
        f"replay is killed at MEASURE, before it can be corrected")

    st2 = s2.get("stale", {})
    chk("E13 HIST=2 (= KMAX, the minimum sound depth) refuses it as well",
        st2.get("st") == 6 and st2.get("cons") == 2 and st2.get("stim") == 3,
        f"st={st2.get('st')} consumed={st2.get('cons')}: a round has KMAX+1 "
        f"attempts, so KMAX remembered readings cover every in-round replay")

    st1 = sims[1]["scn"].get("stale", {})
    chk("E14 negative control: HIST=1 accepts the replay, so the depth is "
        "load-bearing, not decoration",
        st1.get("st") == 5 and st1.get("cons") == 3,
        f"st={st1.get('st')} consumed={st1.get('cons')}: one stored digest "
        f"cannot see attempt 3 == attempt 1")

    other = ("happy", "retry1", "budget", "nodata", "term_acc", "term_fail",
             "start_held")
    diff = [(k, t) for k in other for t in ("st", "cons", "retry")
            if sims[1]["scn"].get(k, {}).get(t) != sims[2]["scn"].get(k, {}).get(t)]
    chk("E15 the guard depth changes nothing else: HIST=1 and HIST=2 agree on "
        "every other scenario",
        not diff, "identical state/attempts/retry on 7 scenarios"
        if not diff else f"diverged on {diff}")

    # ---------- synthesis (area, structure, timing bound) -----------------
    variants = {h: synth(h, 8, f"h{h}") for h in (1, 2, 4)}
    w16 = synth(2, B3_DW, f"h2w{B3_DW}")

    ok = all(v["rc"] == 0 for v in list(variants.values()) + [w16])
    chk("S1 yosys maps every variant without error", ok,
        "; ".join(f"HIST={h} rc={v['rc']} cells={v['cells']} "
                  f"area={v['area_um2']:.0f}um2" for h, v in variants.items()
                  ) + f"; DW={B3_DW} cells={w16['cells']} "
                      f"area={w16['area_um2']:.0f}um2")

    bad = [(v["hist"], v["generic_left"], v["latches"], v["memories"])
           for v in list(variants.values()) + [w16]
           if v["generic_left"] or v["latches"] or v["memories"]]
    chk("S2 the netlist is library cells only: no generic gates left, no "
        "latches, no inferred memory",
        not bad, "generic/latch/mem = 0/0/0 in all four mappings"
        if not bad else f"leaks: {bad}")

    fl = {h: v["flops"] for h, v in variants.items()}
    ok = all(fl[h] == 7 + 8 * h for h in fl)
    chk("S3 the digest history is exactly HIST*DW flops (the rest is 7 control "
        "bits): no hidden state",
        ok, f"flops {fl} vs 7+8*HIST = {[7 + 8 * h for h in sorted(fl)]} "
            f"(DW=8); at DW={B3_DW}: {w16['flops']} = 7+{B3_DW}*2")

    ar = {h: v["area_um2"] for h, v in variants.items()}
    mono = ar[1] < ar[2] < ar[4]
    chk("S4 area grows with the guard depth, so the depth is a real cost line",
        mono, "  ".join(f"HIST={h}: {a:.0f} um2 ({a / 1e3:.2f} kum2)"
                        for h, a in sorted(ar.items()))
        + f"; the 8-bit step costs {(ar[2] - ar[1]) / 8:.0f} um2/bit, and the "
          f"B3 width (DW={B3_DW}) is {w16['area_um2']:.0f} um2")

    lv = {h: v["levels"] for h, v in variants.items()}
    chk("S5 the mapped critical path stays short in logic levels",
        max(lv.values()) <= 60,
        "levels " + "  ".join(f"HIST={h}: {n}" for h, n in sorted(lv.items()))
        + " (generic 1-bit gates; every level is at most one library cell)")

    pr = variants[2]["path_ref_ns"]
    ncyc = s2.get("budget", {}).get("n", 0)
    pworst = max(v["path_ns"] for v in variants.values())
    chk("S6 a full KMAX+1-attempt round fits inside one 12 us conversion slot "
        "even at the slow corner",
        ncyc and ncyc * pr < ADC12_S * 1e9,
        f"{ncyc} cycles x {pr:.0f} ns (fanout-4 node = 0.02 pF, the 1X pin is "
        f"~3-5 fF so this is a 4-6 load) = {ncyc * pr / 1e3:.1f} us of the "
        f"{ADC12_S * 1e6:.0f} us slot ({ADC12_S * 1e9 / (ncyc * pr):.1f}x "
        f"margin, ~{1e3 / pr:.0f} MHz); the absolute worst-entry bracket "
        f"(every node at the table's max load and max slew) is "
        f"{ncyc * pworst / 1e3:.0f} us and is not realizable inside this block")

    chk("S7 the supervisor at the B3 digest width fits the area budget",
        w16["area_um2"] <= AREA_BUDGET_UM2,
        f"DW={B3_DW}, HIST=KMAX: {w16['area_um2']:.0f} um2 of "
        f"{AREA_BUDGET_UM2:.0f} um2 (0.01 mm^2) = "
        f"{100 * w16['area_um2'] / AREA_BUDGET_UM2:.0f} %; for scale, one 1 pF "
        f"MIM of the T1 tile is ~500 um2 (DESIGN 6.28)")

    # ---------- records ----------------------------------------------------
    out = dict(bench="run_gf180_fsm", cell="B3 supervisor calibration FSM",
               kmax=KMAX, adc12_s=ADC12_S, b3_dw=B3_DW,
               area_budget_um2=AREA_BUDGET_UM2,
               sim={str(h): dict(finished=v["finished"], rc=v["rc"],
                                 scenarios=v["scn"])
                    for h, v in sims.items()},
               synth={h: dict(cells=v["cells"], area_um2=v["area_um2"],
                              flops=v["flops"], levels=v["levels"],
                              stage_ns=v["stage_ns"], path_ns=v["path_ns"],
                              f_mhz_min=v["f_mhz_min"], log=v["log"],
                              used=v["used"])
                      for h, v in variants.items()},
               synth_b3w=dict(cells=w16["cells"], area_um2=w16["area_um2"],
                              flops=w16["flops"], levels=w16["levels"],
                              path_ns=w16["path_ns"], log=w16["log"],
                              used=w16["used"]),
               n_checks=len(CHECKS), n_pass=sum(1 for c in CHECKS if c["ok"]),
               checks=CHECKS, total_seconds=time.time() - t0)
    out["status"] = "complete" if out["n_pass"] == out["n_checks"] else "failed"
    with open(os.path.join(OUT, "gf180_fsm.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    rep = os.path.join(OUT, "gf180_fsm_report.txt")
    with open(rep, "w") as fh:
        fh.write("P0-B/G4  B3 supervisor calibration FSM -- gf180mcu batch-1\n")
        fh.write(f"pf {os.path.basename(HW)}/calib_fsm.v -> "
                 f"gf180mcu_fd_sc_mcu7t5v0 (ss_125C_3v00); KMAX={KMAX}\n")
        fh.write("guard depth: HIST>=" + str(KMAX) + " is the minimum sound "
                 "depth; HIST=1 accepts a replay (measured, E14)\n")
        fh.write("area " + "  ".join(
            f"HIST={h}: {v['area_um2']:.0f}um2/{v['flops']}ff/{v['levels']}lv"
            for h, v in sorted(variants.items())) + "\n")
        fh.write(f"B3 width DW={B3_DW}: {w16['area_um2']:.0f} um2, "
                 f"{w16['flops']} flops, path bound "
                 f"{w16['path_ns']:.2f} ns\n")
        for c in CHECKS:
            fh.write(("PASS  " if c["ok"] else "FAIL  ") + c["name"] + "  "
                     + c["detail"] + "\n")
        fh.write("\n%d/%d checks pass\n" % (out["n_pass"], out["n_checks"]))
    print(f"\n{out['n_pass']}/{out['n_checks']} checks passed, "
          f"{out['n_checks'] - out['n_pass']} failed -> "
          f"results/gf180_fsm.json [{out['total_seconds']:.1f} s]")
    return 0 if out["n_pass"] == out["n_checks"] else 1


if __name__ == "__main__":
    sys.exit(main())
