"""P0-E: the realizable trim loop and the P table recomputed through it (§6.29).

§6.26's ``full`` column multiplies every measured axis by ``1 + MEAS_FLOOR``
(100 ppm): a *continuous* trim that removes the corner, the mismatch and the
temperature excursion at once. That is an upper bound, not a deployable
number. This suite replaces it with a closed loop that has the three things a
real chip has:

  * a coarse+fine two-stage DAC (finite codes, +-R clamp, overlap requirement);
  * a noisy readout (declared 100 ppm floor + ADC LSB), which is injected into
    the state on every pass -- so it is NOT dithered away;
  * an actuator gain error (from T9's measured LSB -4.4 % / INL -2.19 %), which
    makes one pass leave ``e*delta`` and each retry multiply that by ``e``.

The loop is the ``twin/calib.py`` FSM's semantics: measure -> correct -> a
fresh reading decides whether another pass is needed -> held-out VERIFY. A
chip that clamps a knob or fails VERIFY is *rejected* (P = 0 for that draw),
never silently shipped with a large residual.

Five blocks: A the codec (step, grid, overlap/gap), B the loop (quantization
bound, read-noise law, actuator ladder, VERIFY false-accept law), B2 the
readout (10 vs 12 bit realized residual), C the measured deviations (required
range, one-knob structural residual), D the QoI gain per axis plus the P
table over six modes (ideal_full / trim2 / trim1shot / trim1knob / trim_q /
trim_q12), per-device granularity, and the fail-closed accounting gates.

    python3 -m bench.run_gf180_trim     # ~4-8 min (the three anchors compile)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE) + "/results"
sys.path.insert(0, os.path.dirname(HERE))

import bench.run_gf180_joint as GJ                                # noqa: E402
from twin.trim import (TwoStageDAC, TrimSpec, close_loop,         # noqa: E402
                       required_range, residual_law, verify_accept_prob)

J = GJ.J                       # bench.run_b2_joint: qoi/perturb/anchors
MEAS_FLOOR = GJ.MEAS_FLOOR
TOLS = GJ.TOLS
CHECKS = []
_LIN = {}                      # per-anchor gain linearity, filled by E10


def chk(name, ok, detail):
    CHECKS.append(dict(name=name, ok=bool(ok), detail=detail))
    print(("PASS  " if ok else "FAIL  ") + name + "\n      " + detail)
    return ok


# ---------------------------------------------------------------------------
# the measured deviation model (exactly §6.26's uncal branch)
# ---------------------------------------------------------------------------
def _deltas(p, fam, k, mc_f, is_kcl):
    """True relative deviations of the three trim axes, chip #k.

    ``self`` scales the integrator (pole / base gm), ``gain`` the input
    transconductance (b), ``mirror`` the coupling edges. In §6.26's uncal
    branch the edge factor is ``gain*mirror``, so trimming ``gain`` alone does
    not zero the edges. One draw per CHIP (the trim is global), which matches
    the granularity §6.26's ``full`` mode assumes; the per-device granularity
    is measured separately in E9.
    """
    ks_mc, kg_mc, km_mc = mc_f[fam]
    if is_kcl:
        # KCL: the diagonal IS the sum of the edge conductances, so there is
        # exactly ONE knob (a gm scale) on A and b together. Its deviation is
        # the measured corner/gm factor; the steady-state QoI is invariant to a
        # *uniform* scale (x = G^-1 I), which is why §6.26's uncal poisson
        # column is already 1.00 at 2 %.
        return {"self": float(p["k_gm"] * kg_mc[k] * km_mc[k]) - 1.0}
    return {"self": float(p["k_pole"] * ks_mc[k]) - 1.0,
            "gain": float(p["k_gm"] * kg_mc[k]) - 1.0,
            "mirror": float(p["k_mirror"] * km_mc[k]) - 1.0}


def _realized(r, is_kcl):
    """Multiplicative factors after the trim, in perturb()'s three slots."""
    ks = 1.0 + r["residual"]["self"]
    if is_kcl:
        return ks, ks, ks
    kg = 1.0 + r["residual"]["gain"]
    km = 1.0 + r["residual"].get("mirror", 1.0)
    return ks, kg, kg * km


# ---------------------------------------------------------------------------
# A. the codec
# ---------------------------------------------------------------------------
def block_codec(out):
    print("\n--- A. two-stage DAC: step, grid, overlap ---")
    dac = TwoStageDAC(range_rel=0.35, coarse_bits=4, fine_bits=6, overlap=0.25)
    print("  R=+-35 %%, 4+6 bits, overlap 25 %%: step %.6f %% (%d codes), "
          "coarse LSB %.4f %%" % (100 * dac.step, dac.n_codes,
                                 100 * dac.lsb_coarse))

    ideal_step = 2.0 * 0.35 / 1024
    chk("E1 the step is 2R/2^(Nc+Nf) and the code count is 2^(Nc+Nf)",
        abs(dac.step / ideal_step - 1) < 1e-12 and dac.n_codes == 1024,
        "step %.9f vs %.9f (%.1e); codes %d"
        % (dac.step, ideal_step, abs(dac.step / ideal_step - 1),
           dac.n_codes))

    rng = np.random.default_rng(1)
    cmds = rng.uniform(-dac.range_rel, dac.range_rel, 4000)
    got = np.array([dac.quantize(c) for c in cmds])
    res = got - cmds
    grid = float(np.max(np.abs(got / dac.step - np.round(got / dac.step))))
    chk("E2 every command lands within step/2 and the code set is the uniform "
        "step grid",
        np.max(np.abs(res)) <= dac.step / 2 + 1e-15 and grid < 1e-9,
        "max |res| %.3e = %.2f step; max grid deviation %.1e step"
        % (np.max(np.abs(res)), np.max(np.abs(res)) / dac.step, grid))

    scan = np.linspace(-dac.range_rel, dac.range_rel, 400001)
    row = {}
    for fg in (0.0, 0.25, -0.25):
        got = np.array([dac.quantize(c, fg) for c in scan])
        d = np.diff(got)
        mx = float(np.max(np.abs(got - scan)))
        ana = dac.max_residual(fg)
        row[fg] = (mx, ana, float(d.min()), float(d.max()))
        print("  fine gain %+0.2f: monotone %s, max |res| %.4f %% = %.2f step "
              "(analytic %.4f %%, sampled/analytic %.2f), max step jump %.4f %%"
              % (fg, bool(d.min() >= -1e-15), 100 * mx, mx / dac.step,
                 100 * ana, mx / ana, 100 * d.max()))
    chk("E3 overlap: the analytic worst residual at the nominal fine gain IS "
        "step/2; a -25 %% fine-gain error opens a 0.25-LSB gap, grows the worst "
        "residual 16x, and a 25 %% overlap is exactly what absorbs it",
        abs(dac.max_residual(0.0) / (dac.step / 2) - 1) < 1e-15
        and row[0.0][0] / row[0.0][1] > 0.999
        and abs(row[-0.25][0] / row[-0.25][1] - 1) < 0.01
        and row[-0.25][0] / row[0.0][0] > 12
        and row[0.0][2] >= -1e-15,
        "nominal max %.4f %% = step/2 exactly (analytic); -25 %% gives "
        "%.4f %% = %.1fx the nominal (predicted gap 0.25 LSB = %.2f %%)"
        % (100 * row[0.0][0], 100 * row[-0.25][0],
           row[-0.25][0] / row[0.0][0], 100 * 0.25 * dac.lsb_coarse))
    out["dac"] = dict(range_rel=0.35, coarse_bits=4, fine_bits=6, overlap=0.25,
                      step=dac.step, n_codes=dac.n_codes,
                      lsb_coarse=dac.lsb_coarse,
                      max_res_nominal=dac.max_residual(0.0),
                      max_res_fine_m25=dac.max_residual(-0.25))
    return dac


# ---------------------------------------------------------------------------
# B. the loop laws
# ---------------------------------------------------------------------------
def block_loop(out, dac):
    print("\n--- B. loop laws: quantization, read noise, actuator, VERIFY ---")
    spec0 = TrimSpec(knobs=(("self",),), iters=1, read_sigma=0.0,
                     actuator_err=0.0)
    rng = np.random.default_rng(2)

    # ---- E4 quantization statistics (no noise, no actuator error) ----------
    deltas = rng.uniform(-0.6 * dac.range_rel, 0.6 * dac.range_rel, 4000)
    res = np.array([close_loop({"self": d}, {"self": dac}, spec0, rng)
                    ["residual"]["self"] for d in deltas])
    rms = float(np.sqrt(np.mean(res ** 2)))
    ideal = dac.step / math.sqrt(12.0)
    chk("E4 a quantization-limited loop: max <= step/2 and RMS = step/sqrt(12)",
        np.max(np.abs(res)) <= dac.step / 2 + 1e-15
        and abs(rms / ideal - 1) < 0.05,
        "max %.2f step, RMS %.4f %% vs step/sqrt(12) = %.4f %% (%.1f %%)"
        % (np.max(np.abs(res)) / dac.step, 100 * rms, 100 * ideal,
           100 * (rms / ideal - 1)))

    # ---- E5 read-noise law (one pass; the noise is INJECTED, not dithered) --
    sweep = []
    for mult in (0.003, 0.03, 0.3, 1.0, 3.0, 10.0, 30.0):
        sig = mult * dac.step
        sp = TrimSpec(knobs=(("self",),), iters=1, read_sigma=sig,
                      actuator_err=0.0, tol_verify=0.05)
        r = np.array([close_loop({"self": d}, {"self": dac}, sp, rng)
                      ["residual"]["self"] for d in deltas])
        rms = float(np.sqrt(np.mean(r ** 2)))
        pred = residual_law(sp, dac, passes=1)
        sweep.append((mult, rms / dac.step, pred / dac.step,
                      float(np.max(np.abs(r)) / dac.step)))
        print("  sigma_read %6.2f step: RMS %.3f step vs pred %.3f "
              "(%.1f %%), max %.2f step"
              % (mult, sweep[-1][1], sweep[-1][2],
                 100 * (sweep[-1][1] / sweep[-1][2] - 1), sweep[-1][3]))
    worst = max(abs(100 * (s[1] / s[2] - 1)) for s in sweep)
    big = [s for s in sweep if s[0] == 30.0][0]
    chk("E5 read noise adds in quadrature and is NOT dithered away: RMS = "
        "sqrt(step^2/12 + sigma^2) over four decades, and it keeps growing "
        "once sigma > step",
        worst < 8.0 and big[1] > 0.9 * 30.0,
        "worst deviation %.1f %% for sigma 0.003..30 step; at sigma=30 step "
        "the RMS is %.1f step, i.e. the measurement (step/2 = 0.5 is not a "
        "bound)" % (worst, big[1]))

    # ---- E6 actuator ladder on a 24-bit DAC (quantization removed) ---------
    fine = TwoStageDAC(range_rel=0.35, coarse_bits=4, fine_bits=32,
                       overlap=0.25)
    ladder = []
    for e in (0.003, 0.03, 0.10):
        for k in (1, 2, 3):
            sp = TrimSpec(knobs=(("self",),), iters=k, read_sigma=0.0,
                          actuator_err=e, tol_verify=-1.0)
            r = abs(close_loop({"self": 0.20}, {"self": fine}, sp,
                               rng)["residual"]["self"])
            pred = e ** k * 0.20
            ladder.append((e, k, r, pred))
            print("  e=%.0e k=%d: |res| %.3e vs e^k*delta %.3e (%.1f %%)"
                  % (e, k, r, pred, 100 * (r / pred - 1)))
    worst = max(abs(100 * (x[2] / x[3] - 1)) for x in ladder)
    row3 = [x for x in ladder if x[0] == 0.03]
    chk("E6 the actuator-gain ladder: one pass leaves e*delta and each retry "
        "multiplies by e (with quantization removed the law is exact)",
        worst < 2.0,
        "worst deviation %.1f %% over e=0.3/3/10 %% and k=1..3 (32-bit DAC); "
        "e=3 %%: %.3e (k=1) -> %.3e (k=2) -> %.3e (k=3)"
        % (worst, row3[0][2], row3[1][2], row3[2][2]))

    # ---- E6b a retry saturates at the code floor ---------------------------
    sat = []
    for k in (1, 2, 3, 4):
        sp = TrimSpec(knobs=(("self",),), iters=k, read_sigma=0.0,
                      actuator_err=0.03, tol_verify=-1.0)
        sat.append(abs(close_loop({"self": 0.20}, {"self": dac}, sp, rng)
                       ["residual"]["self"]))
    print("  10-bit DAC, e=3 %%, delta=20 %%: k=1 %.3e, k=2 %.3e, k=3 %.3e, "
          "k=4 %.3e (code floor step/2 = %.3e)"
          % (sat[0], sat[1], sat[2], sat[3], dac.step / 2))
    chk("E6b a retry saturates: k=3 changes nothing measurable and k=2 already "
        "sits at/below the code floor, so k_max=2 is the right loop budget",
        abs(sat[2] - sat[1]) <= 1e-3 * max(sat[1], 1e-30)
        and abs(sat[3] - sat[2]) <= 1e-3 * max(sat[2], 1e-30)
        and sat[1] <= dac.step / 2,
        "k=1 %.3e -> k=2 %.3e (code floor step/2 = %.3e) -> k=3 %.3e -> "
        "k=4 %.3e: two passes reach the floor, further passes are bit-exact "
        "no-ops" % (sat[0], sat[1], dac.step / 2, sat[2], sat[3]))

    # ---- E7 held-out VERIFY false-accept law -------------------------------
    sp = TrimSpec(knobs=(("self",),), iters=1, read_sigma=100e-6,
                  actuator_err=0.0, tol_verify=0.01)
    rows = []
    n_mc = 4000
    for frac in (0.2, 0.5, 0.9, 0.98, 1.0, 1.02, 1.5):
        r0 = frac * sp.tol_verify
        acc = float(np.mean([abs(r0 + rng.normal(0.0, sp.read_sigma))
                             <= sp.tol_verify for _ in range(n_mc)]))
        ana = verify_accept_prob(r0, sp)
        se = math.sqrt(max(ana * (1 - ana), 1e-9) / n_mc)
        rows.append((frac, acc, ana, 2 * se))
        print("  r/tol=%.2f: MC accept %.4f vs analytic %.4f (2 s.e. %.4f)"
              % (frac, acc, ana, 2 * se))
    mid = [x for x in rows if x[0] == 1.0][0]
    worst = max(abs(x[1] - x[2]) / x[3] for x in rows if x[3] > 0)
    chk("E7 the held-out VERIFY is a probabilistic gate: MC matches "
        "Phi((tol-r)/sigma) - Phi((-tol-r)/sigma) at every r; a chip at "
        "r = 1.0 tol is accepted only ~50 %% of the time",
        worst < 1.0 and abs(mid[1] - 0.5) < 0.05,
        "worst |MC - analytic| = %.2f s.e. (MC n=%d); r=1.0 tol accepts %.3f"
        % (worst, n_mc, mid[1]))
    out["loop"] = dict(quant_rms_step=rms / dac.step,
                       noise_sweep=[list(map(float, s)) for s in sweep],
                       actuator_ladder=[list(map(float, s)) for s in ladder],
                       retry_saturation=[float(v) for v in sat],
                       code_floor=dac.step / 2,
                       verify_rows=[list(map(float, s)) for s in rows])
    return sp


# ---------------------------------------------------------------------------
# C. the measured deviations: required range and the one-knob residual
# ---------------------------------------------------------------------------
def block_axes(out, pts, mc_f, n_mc):
    print("\n--- C. measured deviations: range spec and knob count ---")
    free = {f: (np.ones(n_mc), np.ones(n_mc), np.ones(n_mc)) for f in GJ.FAMS}
    req, sep = {}, {}
    for fam in GJ.FAMS:
        pool = {k: [] for k in ("self", "gain", "mirror")}
        sep_rows = []
        for p in pts:
            if p["fam"] != fam:
                continue
            for k in range(n_mc):
                for a, v in _deltas(p, fam, k, mc_f, False).items():
                    pool[a].append(v)
            d0 = _deltas(p, fam, 0, free, False)
            sep_rows.append((abs(d0["self"] - d0["gain"]) / 2.0, p["tag"],
                             d0["self"], d0["gain"]))
        req[fam] = {a: required_range(v) for a, v in pool.items() if v}
        sep[fam] = max(sep_rows, key=lambda t: t[0])
        print("  %s required range: %s" % (fam, ", ".join(
            "%s +-%.1f %%" % (a, 100 * r) for a, r in req[fam].items())))
        print("  %s worst one-knob residual |ds-dg|/2 = %.2f %% at %s "
              "(ds %+.1f %%, dg %+.1f %%)"
              % (fam, 100 * sep[fam][0], sep[fam][1],
                 100 * sep[fam][2], 100 * sep[fam][3]))

    n_bad = 0
    for p in pts:
        if p["fam"] != "3v3":
            continue
        d0 = _deltas(p, "3v3", 0, free, False)
        if abs(d0["self"] - d0["gain"]) / 2.0 > TOLS[0]:
            n_bad += 1
    chk("S1 one joint knob cannot trim 3v3 to 1 %%: the measured self/gain "
        "separation exceeds the 1 %% workload at %d of the 16 3v3 points "
        "(worst %.2f %% at %s)"
        % (n_bad, 100 * sep["3v3"][0], sep["3v3"][1]),
        n_bad >= 1 and sep["3v3"][0] > TOLS[0],
        "worst %.2f %% at %s vs the 1 %% workload; 6v0 worst %.2f %% at %s "
        "-- the knob count, not the DAC bit count, is binding"
        % (100 * sep["3v3"][0], sep["3v3"][1],
           100 * sep["6v0"][0], sep["6v0"][1]))

    need = max(max(req[f].values()) for f in req)
    chk("S2 the required trim range is the measured worst deviation, not a "
        "round number: +-%.1f %% (a +-25 %% DAC clamps real chips)"
        % (100 * need),
        need > 0.25,
        "%s; with +-25 %% those chips are REJECTED by VERIFY, not silently "
        "trimmed" % "; ".join("%s %.1f %%" % (f, 100 * max(req[f].values()))
                              for f in req))
    out["required_range"] = {f: dict(v) for f, v in req.items()}
    out["one_knob_worst"] = {f: [float(sep[f][0]), sep[f][1]] for f in sep}
    out["n_3v3_points_above_1pct"] = int(n_bad)
    return need


# ---------------------------------------------------------------------------
# D. the P table through the loop
# ---------------------------------------------------------------------------
MODES = {
    "ideal_full": (None, 0),        # §6.26's continuous-trim upper bound
    "trim2": ("split", 3),          # one DAC per axis, retry allowed
    "trim1shot": ("split", 1),      # same topology, no retry
    "trim1knob": ("joint", 3),      # self+gain share one DAC
    "trim_q": ("split_q", 3),       # QoI-weighted verify, 10-bit DAC/ADC
    "trim_q12": ("split_q", 3),     # QoI-weighted verify, 12-bit DAC/ADC
}
# hardware per mode: (DAC, readout sigma)
HW = {}        # filled in main()


def _mode_spec(mode, gains=None):
    topo, kmax = MODES[mode]
    if topo is None:
        return None
    dac, read_sigma = HW["12"] if mode.endswith("12") else HW["10"]
    knobs = (("self", "gain"), ("mirror",)) if topo.startswith("joint") else \
            (("self",), ("gain",), ("mirror",))
    spec = TrimSpec(knobs=knobs, iters=kmax, read_sigma=read_sigma,
                    actuator_err=0.03, tol_verify=min(TOLS))
    if topo.endswith("_q") and gains:
        # A single coefficient axis moves the QoI by up to g: the sensitive
        # direction is a *difference* of two axis residuals (a uniform scale
        # cancels), so bounding each axis by tol/(2 g) bounds the difference
        # by tol/g.
        gmax = max(max(gains.values()), 1.0)
        spec.tol_axis = {a: min(TOLS) / (2.0 * gmax) for a in gains}
    return spec


def block_table(out, dac, read_sigma, pts, mc_f, n_mc):
    print("\n--- D. P(workload passes) through the loop ---")
    compiled = {}
    for name in J.ANCHORS:
        t0 = time.time()
        bundle, res, A, b, T, shift = J.compile_anchor(name)
        x_ref = J.qoi_step(A, b, T, shift)
        compiled[name] = dict(A=A, b=b, T=T, shift=shift, x_ref=x_ref)
        print("  %-13s n=%d compiled [%.1f s]" % (name, A.shape[0],
                                                  time.time() - t0), flush=True)

    # ---- E10 the QoI gain per coefficient axis -----------------------------
    print("\n  E10 QoI gain per axis (one-axis probe at r=1e-3, linearity "
          "1e-4..1e-2):")
    gains_all = {}
    uni = {}
    for name, c in compiled.items():
        A, b, T, shift, x_ref = c["A"], c["b"], c["T"], c["shift"], c["x_ref"]
        is_kcl = name == "poisson-2d"
        axes = ("self",) if is_kcl else ("self", "gain", "mirror")
        g, lin = {}, {}
        _LIN[name] = lin
        for a in axes:
            vals = []
            for r0 in (1e-4, 1e-3, 1e-2):
                ks = 1.0 + (r0 if a == "self" else 0.0)
                kg = 1.0 + (r0 if a == "gain" else 0.0)
                km = 1.0 + (r0 if a == "mirror" else 0.0)
                if is_kcl:
                    ks = kg = km = 1.0 + r0
                ke = kg if is_kcl else kg * km
                e = J.qoi_err(J.qoi_step(*J.perturb(A, b, np.full(len(b), ks),
                                                    np.full(len(b), kg),
                                                    np.full(len(b), ke)),
                                         T, shift), x_ref)
                vals.append(e / r0)
            g[a] = vals[1]
            lin[a] = (max(vals) / max(min(vals), 1e-30)
                      if abs(vals[1]) >= 0.05 else 1.0)
        gains_all[name] = g
        # the *uniform* probe is §6.21's experiment (all axes scaled together);
        # it is the control that says the amplifier is the NON-uniformity
        e = J.qoi_err(J.qoi_step(*J.perturb(A, b,
                                            np.full(len(b), 1.0 + 1e-3),
                                            np.full(len(b), 1.0 + 1e-3),
                                            np.full(len(b), 1.0 + 1e-3)),
                                    T, shift), x_ref)
        uni[name] = e / 1e-3
        print("    %-13s gmax %.2f  uniform %.2f  %s"
              % (name, max(g.values()), uni[name], "  ".join(
            "%s %.2f (lin %.2fx)" % (a, g[a], lin[a]) for a in axes)))
    worst_lin = max(l for v in _LIN.values() for l in v.values())
    chk("E10 the QoI gain is linear in the residual over two decades, and it "
        "is NOT 1: the edge/mirror axis is amplified",
        worst_lin <= 1.15
        and max(gains_all["burgers-1d"]["mirror"],
                gains_all["poisson-2d"]["self"]) >= 5.0,
        "; ".join("%s gmax %.1f" % (k, max(v.values()))
                  for k, v in gains_all.items())
        + " -- a 1 %% coefficient residual on the edge axis is a >5 %% QoI "
          "residual on at least one anchor, so a coefficient-space verify is "
          "too loose")
    out["qoi_gains"] = {k: dict(v, uniform=uni.get(k, 0.0))
                       for k, v in gains_all.items()}

    table = {}
    for name, c in compiled.items():
        A, b, T, shift, x_ref = c["A"], c["b"], c["T"], c["shift"], c["x_ref"]
        n = A.shape[0]
        is_kcl = name == "poisson-2d"
        rows = []
        t0 = time.time()
        for p in pts:
            fam = p["fam"]
            acc = {m: [] for m in MODES}
            its = {m: [] for m in MODES}
            n_reject = {m: 0 for m in MODES}
            n_clamp = {m: 0 for m in MODES}
            for k in range(n_mc):
                d = _deltas(p, fam, k, mc_f, is_kcl)
                kf = np.full(n, 1.0 + MEAS_FLOOR)
                acc["ideal_full"].append(J.qoi_err(
                    J.qoi_step(*J.perturb(A, b, kf, kf, kf), T, shift), x_ref))
                for mode in ("trim2", "trim1shot", "trim1knob", "trim_q",
                             "trim_q12"):
                    sp = _mode_spec(mode, gains_all[name])
                    if is_kcl:
                        sp.knobs = (("self",),)
                    dacs = {a: (HW["12"][0] if mode.endswith("12")
                                else HW["10"][0]) for a in d}
                    r = close_loop(d, dacs, sp,
                                   np.random.default_rng(1000 + k))
                    its[mode].append(r["iterations"])
                    n_clamp[mode] += int(r["clamped"])
                    if not r["verified"]:
                        n_reject[mode] += 1
                        acc[mode].append(float("inf"))
                        continue
                    ks, kg, ke = _realized(r, is_kcl)
                    acc[mode].append(J.qoi_err(
                        J.qoi_step(*J.perturb(A, b, np.full(n, ks),
                                              np.full(n, kg), np.full(n, ke)),
                                   T, shift), x_ref))

            def prob(m, tol):
                if not p["bias_ok"]:
                    return 0.0
                return sum(1 for e in acc[m] if e <= tol) / len(acc[m])

            def med(m):
                fin = [e for e in acc[m] if np.isfinite(e)]
                return float(np.median(fin)) if fin else float("inf")

            rows.append(dict(
                tag=p["tag"], fam=fam, source=p["source"], temp_c=p["temp_c"],
                bias_ok=p["bias_ok"],
                **{("p_" + m): {("tol_%d" % int(t * 100)): prob(m, t)
                                for t in TOLS} for m in MODES},
                **{("n_reject_" + m): n_reject[m] for m in MODES},
                **{("n_clamp_" + m): n_clamp[m] for m in MODES},
                iters_mean=float(np.mean(its["trim2"])) if its["trim2"] else 0.0,
                iters_mean_q12=(float(np.mean(its["trim_q12"]))
                                if its.get("trim_q12") else 0.0),
                iters_max=int(max(its["trim2"])) if its["trim2"] else 0,
                **{("err_med_" + m): med(m) for m in MODES}))
        table[name] = rows
        print("  %-13s P table done [%.1f s]" % (name, time.time() - t0),
              flush=True)

    # ---- E9 per-device granularity: what a GLOBAL trim cannot reach --------
    print("\n  E9 per-device mismatch (one draw per NODE) under the same "
          "global trim, at 3v3/27 C:")
    per_node = {}
    tgt = [p for p in pts if p["fam"] == "3v3" and p["source"] == "corner"
           and abs(p["temp_c"] - 27.0) < 1e-9][0]
    for name, c in compiled.items():
        A, b, T, shift, x_ref = c["A"], c["b"], c["T"], c["shift"], c["x_ref"]
        n = A.shape[0]
        is_kcl = name == "poisson-2d"
        ks_mc, kg_mc, km_mc = mc_f["3v3"]
        errs, glob = [], []
        for k in range(n_mc):
            idx = (np.arange(n) + k) % n_mc
            if is_kcl:
                f_g = tgt["k_gm"] * kg_mc[idx] * km_mc[idx]
                f_g = f_g / float(np.mean(f_g))
                Ap, bp = J.perturb(A, b, 1.0, f_g, f_g)
            else:
                f_s = tgt["k_pole"] * ks_mc[idx]
                f_g = tgt["k_gm"] * kg_mc[idx]
                f_m = tgt["k_mirror"] * km_mc[idx]
                f_s = f_s / float(np.mean(f_s))
                f_g = f_g / float(np.mean(f_g))
                f_m = f_m / float(np.mean(f_m))
                Ap, bp = J.perturb(A, b, f_s, f_g, f_g * f_m)
            errs.append(J.qoi_err(J.qoi_step(Ap, bp, T, shift), x_ref))
            d = _deltas(tgt, "3v3", k, mc_f, is_kcl)
            sp = _mode_spec("trim2")
            if is_kcl:
                sp.knobs = (("self",),)
            r = close_loop(d, {a: dac for a in d}, sp,
                           np.random.default_rng(1000 + k))
            ks, kg, ke = _realized(r, is_kcl)
            glob.append(J.qoi_err(J.qoi_step(
                *J.perturb(A, b, np.full(n, ks), np.full(n, kg),
                           np.full(n, ke)), T, shift), x_ref))
        med = float(np.median([e for e in errs if np.isfinite(e)]))
        gmed = float(np.median([e for e in glob if np.isfinite(e)]))
        per_node[name] = dict(median=med, global_median=gmed,
                              ratio=med / gmed)
        print("    %-13s per-node median %.2e vs global-trim median %.2e "
              "(%.0fx)" % (name, med, gmed, med / gmed))
    chk("E9 a global trim cannot touch per-DEVICE mismatch: with one mismatch "
        "draw per node the untrimmable floor is >=3x the global-trim residual "
        "at every anchor",
        min(v["ratio"] for v in per_node.values()) >= 3.0,
        "; ".join("%s %.1fx" % (k, v["ratio"]) for k, v in per_node.items())
        + " -- §6.26's `full` assumes a global trim; that granularity, not the "
          "DAC, is the next budget item")
    out["per_node_granularity"] = per_node

    # ---- gates -------------------------------------------------------------
    flat = [(name, r) for name, rows in table.items() for r in rows]
    worst_full = min(r["p_ideal_full"]["tol_1"] for _, r in flat)
    worst_2 = min(r["p_trim2"]["tol_1"] for _, r in flat)
    worst_q = min(r["p_trim_q"]["tol_1"] for _, r in flat)
    worst_q12 = min(r["p_trim_q12"]["tol_1"] for _, r in flat)
    gap = [(n, r) for n, r in flat if r["p_trim2"]["tol_1"] < 1.0 - 1e-9]
    chk("T1 a coefficient-space verify at 1 %% does NOT reproduce the ideal "
        "full column: trim2 loses the 1 %% column at %d of 96 cells (worst "
        "P1 %.2f) -- the QoI gain is the missing factor" % (len(gap), worst_2),
        len(gap) >= 1 and worst_2 < worst_full - 1e-9,
        "min P1: ideal %.3f vs trim2 %.3f; the losses sit on the anchor whose "
        "edge axis is amplified %.1fx"
        % (worst_full, worst_2, gains_all["burgers-1d"]["mirror"]))
    chk("T1q a QoI-weighted verify (each axis <= tol/(2*gmax)) plus a 12-bit "
        "DAC/readout lifts the 1 %% column from %.2f to >=0.75 at every cell"
        % worst_2,
        worst_q12 >= 0.75 and worst_q12 >= worst_2 + 0.4,
        "min P1: trim2 %.2f (coefficient verify) -> trim_q12 %.3f (QoI "
        "verify, 12 bit); the residue is the false-reject rate of the tight "
        "gate, not a residual error" % (worst_2, worst_q12))

    one = [(n, r) for n, r in flat if r["p_trim1shot"]["tol_1"] < 1.0 - 1e-9]
    worst_2s = min(r["p_trim1shot"]["tol_2"] for _, r in flat)
    worst_5s = min(r["p_trim1shot"]["tol_5"] for _, r in flat)
    who = "; ".join("%s %s P1 %.2f (%.1f passes avg)"
                    % (n.split(":")[0], r["tag"], r["p_trim1shot"]["tol_1"],
                       r["iters_mean"]) for n, r in one[:3]) or "none"
    chk("T2 dropping the retry costs the 1 %% column at %d cells (the 3 %% "
        "actuator error, not the DAC, is the cost)" % len(one),
        len(one) >= 1,
        "%s -- worst cell at 2 %%/5 %% over all 96: %.2f/%.2f"
        % (who, worst_2s, worst_5s))

    knob = [(n, r) for n, r in flat
            if not n.startswith("poisson") and r["p_trim1knob"]["tol_1"] < 1.0 - 1e-9]
    chk("T3 one joint knob costs the 1 %% column at %d cells (the measured "
        "self/gain separation, not the DAC, is binding)" % len(knob),
        len(knob) >= 1,
        ("; ".join("%s %s P1 %.2f P2 %.2f"
                   % (n.split(":")[0], r["tag"], r["p_trim1knob"]["tol_1"],
                      r["p_trim1knob"]["tol_2"]) for n, r in knob[:4]) or "none"))

    rej = {m: sum(r["n_reject_" + m] for _, r in flat) for m in MODES}
    cla = {m: sum(r["n_clamp_" + m] for _, r in flat) for m in MODES}
    over = []
    for n, r in flat:
        for m in MODES:
            for t in TOLS:
                key = "tol_%d" % int(t * 100)
                cap = (n_mc - r["n_reject_" + m]) / n_mc
                if r["p_" + m][key] > cap + 1e-12:
                    over.append((n, r["tag"], m, key))
    chk("T4 fail-closed accounting: no cell scores more passes than it has "
        "verified chips",
        not over and (rej["trim1shot"] + rej["trim1knob"]) > 0,
        "rejections of 4608 draws per mode: trim2 %d, trim1shot %d, "
        "trim1knob %d, trim_q %d; clamp events %d/%d/%d/%d; %d cells exceed "
        "their verified fraction"
        % (rej["trim2"], rej["trim1shot"], rej["trim1knob"], rej["trim_q"],
           cla["trim2"], cla["trim1shot"], cla["trim1knob"], cla["trim_q"],
           len(over)))

    print("\n  P table (median QoI error in %% / P1 / P2 / P5):")
    print("  %-13s %-22s %8s %8s %6s %6s %6s %6s %8s"
          % ("anchor", "point", "trim2", "trim_q12", "P1", "P2", "P5",
             "Q1", "Q12"))
    for name, rows in table.items():
        for r in rows:
            print("  %-13s %-22s %8.4f %8.5f %6.3f %6.3f %6.3f %6.3f %6.3f%s"
                  % (name.split("-")[0], r["tag"],
                     100 * r["err_med_trim2"], 100 * r["err_med_trim_q12"],
                     r["p_trim2"]["tol_1"], r["p_trim2"]["tol_2"],
                     r["p_trim2"]["tol_5"], r["p_trim_q"]["tol_1"],
                     r["p_trim_q12"]["tol_1"],
                     "" if r["bias_ok"] else "  [bias-invalid]"))

    tol_q = min(TOLS) / (2.0 * max(max(gains_all["burgers-1d"].values()), 1.0))
    sig12 = HW["12"][1]
    pred = []
    for n, r in flat:
        k = max(r["iters_mean_q12"], 1.0)
        z = tol_q / (sig12 * math.sqrt(k + 1.0))
        pred.append(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
    pred_rate = float(np.mean(pred))
    meas_rate = rej["trim_q12"] / float(len(flat) * n_mc)
    chk("T5 the QoI gate's yield loss is analytic: the trim residual is noise "
        "accumulated over k passes and VERIFY adds one more draw, so P(reject) "
        "= 2(1-Phi(tol/(sigma sqrt(k+1))))",
        0.5 <= meas_rate / max(pred_rate, 1e-9) <= 2.0,
        "tol %.0f ppm, sigma12 %.0f ppm, mean k %.1f -> predicted %.1f %%, "
        "measured %.1f %% (ratio %.2f); the 10-bit sigma %.0f ppm would put "
        "this gate at %.1f sigma and reject most good chips"
        % (tol_q * 1e6, sig12 * 1e6,
           float(np.mean([r["iters_mean_q12"] for _, r in flat])),
           100 * pred_rate, 100 * meas_rate, meas_rate / max(pred_rate, 1e-9),
           read_sigma * 1e6, tol_q / read_sigma))

    out["p_table"] = table
    out["modes"] = {m: dict(topology=MODES[m][0], max_passes=MODES[m][1])
                    for m in MODES}
    return table


def main(argv=None):
    t0 = time.time()
    ext = GJ.load_gf180_extract()
    pts = GJ._points(ext)
    mc_f = {f: GJ._mc_factors(ext, f) for f in GJ.FAMS}
    n_mc = len(ext["mc"]["3v3"]["gm_fd"]["values"])
    out = dict(design="gf180mcu batch-1 realizable trim loop (P0-E)",
               note="§6.26 `full` is a continuous-trim upper bound; this suite "
                    "closes the loop with finite trim codes, an injected "
                    "readout noise and an actuator gain error, then recomputes "
                    "P on the same 3 anchors x 32 points x 48 draws")

    dac = block_codec(out)
    block_loop(out, dac)
    need = block_axes(out, pts, mc_f, n_mc)

    adc_lsb = 2 * dac.range_rel / 1024
    read_sigma = math.sqrt(MEAS_FLOOR ** 2 + (adc_lsb / math.sqrt(12.0)) ** 2)
    realized = residual_law(TrimSpec(read_sigma=read_sigma), dac)
    dac12 = TwoStageDAC(range_rel=0.35, coarse_bits=4, fine_bits=8,
                        overlap=0.25)
    read12 = math.sqrt(MEAS_FLOOR ** 2
                       + ((2 * dac12.range_rel / 4096) / math.sqrt(12.0)) ** 2)
    realized12 = residual_law(TrimSpec(read_sigma=read12), dac12)
    HW["10"] = (dac, read_sigma)
    HW["12"] = (dac12, read12)
    print("  12-bit option: DAC step %.0f ppm, ADC sigma %.0f ppm -> "
          "realized %.0f ppm (%.1fx floor)"
          % (dac12.step * 1e6, read12 * 1e6, realized12 * 1e6,
             realized12 / MEAS_FLOOR))
    out["readout"] = dict(floor_ppm=MEAS_FLOOR * 1e6,
                          bits10=dict(adc_lsb=adc_lsb, sigma=read_sigma,
                                      realized_ppm=1e6 * realized),
                          bits12=dict(dac_step=dac12.step, sigma=read12,
                                      realized_ppm=1e6 * realized12))
    chk("E8b the 12-bit option meets the plan's <2.5x spec: realized residual "
        "%.0f ppm = %.2fx the 100 ppm floor" % (1e6 * realized12,
                                               realized12 / MEAS_FLOOR),
        realized12 / MEAS_FLOOR <= 2.5,
        "10-bit gives %.0f ppm (%.1fx); 12-bit gives %.0f ppm (%.2fx) -- the "
        "readout floor, not the DAC, now dominates"
        % (1e6 * realized, realized / MEAS_FLOOR, 1e6 * realized12,
           realized12 / MEAS_FLOOR))
    print("\n  readout: %.0f ppm floor + 10-bit ADC LSB %.0f ppm -> sigma "
          "%.0f ppm (%.1fx the floor)" % (MEAS_FLOOR * 1e6, adc_lsb * 1e6,
                                           read_sigma * 1e6,
                                           read_sigma / MEAS_FLOOR))
    print("  realized residual RMS %.0f ppm vs the 100 ppm continuous-trim "
          "bound -> %.1fx" % (1e6 * realized, realized / MEAS_FLOOR))
    chk("E8 the readout is not the bottleneck: sigma_read <= the DAC step/2",
        read_sigma <= dac.step / 2,
        "sigma %.0f ppm vs step/2 %.0f ppm -- quantization-limited, so extra "
        "ADC bits buy nothing until the DAC gains a bit; the realized RMS "
        "residual is %.0f ppm (%.1fx the ideal 100 ppm)"
        % (read_sigma * 1e6, dac.step / 2 * 1e6, 1e6 * realized,
           realized / MEAS_FLOOR))

    block_table(out, dac, read_sigma, pts, mc_f, n_mc)

    out["required_range_rel"] = need
    out["n_checks"] = len(CHECKS)
    out["n_pass"] = sum(1 for c in CHECKS if c["ok"])
    out["checks"] = CHECKS
    out["total_seconds"] = time.time() - t0
    out["status"] = "complete" if out["n_pass"] == len(CHECKS) else "failed"
    with open(os.path.join(OUT, "gf180_trim.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d/%d checks passed, %d failed -> results/gf180_trim.json [%.1f s]"
          % (out["n_pass"], out["n_checks"], out["n_checks"] - out["n_pass"],
             out["total_seconds"]))
    return 0 if out["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
