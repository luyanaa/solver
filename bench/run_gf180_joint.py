"""G2: gf180mcu joint closure — corner x 48-draw mismatch x calibration-in-
the-loop on the three anchors, P(workload passes) at 1/2/5 %.

    python3 -m bench.run_gf180_joint     # ~4-6 min (burgers compile dominates)

The chain is B2's (`bench.run_b2_joint`) so the two processes are compared on
the SAME compiled descriptors and the SAME QoI definition (zero-state step
response x(T) on the compiled physical-time descriptor):

    build_problem / build_poisson_2d
      → lowering.compose.run_chain (boundary→closure→split→scaling→basis→
        cert→interface→contracts→p7)
      → A_v = alpha^-1 S A S^-1 (the compiler's physical-time coefficients)
      → perturb by the G1-MEASURED nonideality factors
      → QoI error vs the nominal compile; P = fraction of 48 draws within tol

What gf180 changes vs the IHP lane (all from results/gf180_extract.json):

  * THREE separate axes, never one global scale: k_pole (self coefficient,
    1/(r_o C) = lambda*Id/C — the ideal-current-source pole cell measures
    exactly this), k_gm (input gain, from the resistive-load DC gate sweep,
    now present at every PVT/joint point), k_mirror (edge axis, from the
    independent-mirror ratio at that point).
  * INTER-PROCESS FINDING: in 3v3 the self and gain axes are nearly
    collinear (corr 0.96, max separation 14 %) — one trim knob covers both;
    in 6v0 they separate (corr 0.39, max 36 %) — two knobs are needed.
  * MISMATCH: 48 measured draws. k_self_mc = Id_pole/Id_TT (the pole is
    lambda*Id/C with lambda per-device constant), k_gm_mc = gm/gm_TT,
    k_mirror_mc = ratio/1.0 (the MC mirror is the clamped 1:1 copy).
  * NOISE PROVENANCE: gf180 MOS noise is .noise-MEASURED (10.97 nV/rtHz,
    white factor 0.978) — unlike IHP where PSP103 OSDI noise returns zero
    spectra and the label had to be "derived-not-measured".
  * BIAS GATE: the gf180 pole cell is driven by an ideal current source, so
    no measured point leaves saturation — the IHP lane's 4 hot-corner
    bias-invalid points have no counterpart here (recorded, not hidden).

Calibration modes (identical semantics to B2, floor = the same declared
100 ppm readout bound so the two processes stay comparable):

  uncal     corner + mismatch enter directly.
  no_track  foreground trim at test (27 C) removes the process corner and
            static mismatch; only the measured temperature excursion remains.
  bg_track  + replica background tracking, declared 1 C refresh granularity.
  full      continuous trim at the operating point → 100 ppm floor (bound).

Honesty: the 48 mismatch draws were measured at TT/27 C only and are applied
as per-chip RELATIVE factors at every corner (declared composition, not a
MC re-run at each corner); `full` is an upper bound (the real trim code range
is a B3 characterization item); the QoI is the operator-error proxy.
"""
from __future__ import annotations

import json
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

import bench.run_b2_joint as J                                   # noqa: E402
from twin.macromodel import (gf180_noise_model, gf180_pvt_factors,  # noqa: E402
                             load_gf180_extract)

MEAS_FLOOR = J.MEAS_FLOOR      # 100 ppm declared readout floor (same as B2)
BG_UPDATE_DT_C = J.BG_UPDATE_DT_C
TOLS = J.TOLS
FAMS = ("3v3", "6v0")
# cross-process reference from the IHP lane (§6.19): pole 2.82-10.0 MHz
IHP_POLE_SPREAD = 10.0 / 2.82


def _one_pole_rms_wide(path):
    """1-pole log-magnitude fit with a pole search down to 1 kHz.

    B2's `_one_pole_rms` searches 1e5..3e8 Hz because the IHP poles are
    2.8-10 MHz; gf180's 3.3 V pole cell sits at 56-210 kHz, BELOW that
    bound, so the b2 fit returns a bound artifact (rms 0.37 on a clean
    record). Same fit, wider search:
    """
    d = np.atleast_2d(np.loadtxt(path))
    f = d[:, 0]
    mag = np.abs(d[:, 1] + 1j * d[:, 2])
    a0 = float(mag[0])
    m = (f >= 1e4) & (f <= 1e8)
    y = np.log(mag[m])
    best = (float("inf"), float("nan"))
    for fp in np.logspace(3.0, 9.0, 2400):
        pred = np.log(a0) - 0.5 * np.log(1.0 + (f[m] / fp) ** 2)
        r = float(np.sqrt(np.mean((pred - y) ** 2)))
        if r < best[0]:
            best = (r, float(fp))
    return best[0], best[1]


def _mc_factors(ext, fam):
    """Per-draw relative factors from that family's 48-sample record."""
    tt = ext["corners"]["%s_tt" % fam]
    mc = ext["mc"][fam]
    ids = np.asarray(mc["id_pole"]["values"], dtype=float)
    gms = np.asarray(mc["gm_fd"]["values"], dtype=float)
    mir = np.asarray(mc["ratio_mis"]["values"], dtype=float)
    k_self = ids / float(tt["id_at_gmprobe_a"])
    k_gm = gms / float(tt["gm_fd"])
    k_mir = mir / 1.0                  # clamped 1:1 mirror: nominal ratio 1
    return k_self, k_gm, k_mir


def _points(ext):
    """Every measured point, plus the SAME-PROCESS 27 C reference it must be
    compared to: the foreground trim at test measures THIS chip, so the
    process corner is absorbed there and only the temperature excursion is
    left for no_track/bg_track (B2's frac_a/frac_b semantics)."""
    pts = []
    for fam in FAMS:
        fac = gf180_pvt_factors(ext, fam)
        ref = {v["proc"]: v for v in fac.values()
               if v["source"] == "corner" and abs(v["temp_c"] - 27.0) < 1e-9}
        for tag, f in fac.items():
            r = ref[f["proc"]]
            pts.append(dict(
                tag="%s:%s" % (fam, tag), fam=fam, **f,
                frac_pole=f["pole_hz"] / r["pole_hz"] - 1.0,
                frac_gm=f["gm"] / r["gm"] - 1.0,
                frac_mirror=f["mirror_ratio"] / r["mirror_ratio"] - 1.0))
    pts.sort(key=lambda p: (p["fam"], p["source"] != "corner", p["temp_c"],
                            p["tag"]))
    return pts


def main(argv=None):
    ext = load_gf180_extract()
    checks = []
    fails = []

    def check(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))
        if not ok:
            fails.append(name)
        print("%s  %-62s %s" % ("PASS" if ok else "FAIL", name, detail))

    pts = _points(ext)
    n_mc = len(ext["mc"]["3v3"]["gm_fd"]["values"])
    fits = {f: gf180_pvt_factors(ext, f) for f in FAMS}
    mc_f = {f: _mc_factors(ext, f) for f in FAMS}

    # ---- K0 inputs ---------------------------------------------------------
    check("K0: extraction complete; 6 corners + 18 PVT + 8 joint points, "
          "48 draws per family",
          ext.get("status") == "complete"
          and len(fits["3v3"]) == 16 and len(fits["6v0"]) == 16
          and n_mc == 48,
          "status=%s, points/family=%d, draws=%d"
          % (ext.get("status"), len(fits["3v3"]), n_mc))

    # ---- K1 1-pole validity of the corner AC records -----------------------
    rms = {}
    for fam in FAMS:
        for tag in ("tt", "ss", "ff"):
            r, fp = _one_pole_rms_wide(OUT + "/gf_ac_%s_%s.dat" % (fam, tag))
            rms["%s_%s" % (fam, tag)] = (r, fp)
    worst = max(rms, key=lambda t: rms[t][0])
    check("K1: every corner AC is 1-pole (log-mag fit rms < 2e-3, search "
          "1 kHz..1 GHz)",
          max(r[0] for r in rms.values()) < 2e-3,
          "worst %s rms %.2e (fit pole %.1f kHz)"
          % (worst, rms[worst][0], rms[worst][1] / 1e3))

    # ---- K2/K3 the two axes, measured separately ---------------------------
    stat = {}
    for fam in FAMS:
        p = np.array([v["k_pole"] for v in fits[fam].values()])
        g = np.array([v["k_gm"] for v in fits[fam].values()])
        stat[fam] = dict(pole=(float(p.min()), float(p.max())),
                         gm=(float(g.min()), float(g.max())),
                         corr=float(np.corrcoef(p, g)[0, 1]),
                         sep=float(np.max(np.abs(p / g - 1.0))))
    check("K2: both axes have a real measured spread (>1.4x)",
          all(max(v["pole"][1] / v["pole"][0], v["gm"][1] / v["gm"][0]) > 1.4
              for v in stat.values()),
          "; ".join("%s pole %.2fx (%.3f..%.3f), gm %.2fx (%.3f..%.3f)"
                    % (f, v["pole"][1] / v["pole"][0], *v["pole"],
                       v["gm"][1] / v["gm"][0], *v["gm"])
                    for f, v in stat.items()))
    check("K3: 3v3 axes are collinear (one trim knob) and 6v0 axes separate "
          "(two knobs)",
          stat["3v3"]["corr"] > 0.90 and stat["6v0"]["corr"] < 0.60,
          "3v3 corr=%.2f max separation %.0f %%; 6v0 corr=%.2f separation "
          "%.0f %% (both move with Id; at 6 V the vds/geometry mix breaks "
          "the collinearity)"
          % (stat["3v3"]["corr"], 100 * stat["3v3"]["sep"],
             stat["6v0"]["corr"], 100 * stat["6v0"]["sep"]))
    check("K3b: gf180's pole corner spread is quieter than IHP's (%.2fx)"
          % IHP_POLE_SPREAD,
          stat["3v3"]["pole"][1] / stat["3v3"]["pole"][0] < IHP_POLE_SPREAD,
          "gf180 3v3 pole %.2fx vs IHP §6.19 %.2fx across ss/tt/ff"
          % (stat["3v3"]["pole"][1] / stat["3v3"]["pole"][0],
             IHP_POLE_SPREAD))
    check("K3c: gf180 noise is .noise-MEASURED (unlike IHP's derived label)",
          gf180_noise_model(ext).source == "measured-mos-gf180",
          "source=%s, %.2f nV/rtHz, white factor %.3f vs gamma=1"
          % (gf180_noise_model(ext).source,
             gf180_noise_model(ext).density_v_rthz * 1e9,
             gf180_noise_model(ext).factor))

    # ---- K4 compile --------------------------------------------------------
    print("\n--- A. compile the three anchors (same descriptors as B2) ---")
    compiled = {}
    for name in J.ANCHORS:
        t0 = time.time()
        bundle, res, A, b, T, shift = J.compile_anchor(name)
        x_ref = J.qoi_step(A, b, T, shift)
        compiled[name] = dict(A=A, b=b, T=T, shift=shift, x_ref=x_ref, res=res)
        print("  %-13s n=%d alpha=%.3e T=%.3e s cert=%s [%.1f s]"
              % (name, A.shape[0], res.scaling.alpha, T, res.cert.status,
                 time.time() - t0), flush=True)
    check("K4: all three anchors compile and the config edge set == the "
          "contract edge set (F5)",
          all(c["res"].edge_sets_equal and c["x_ref"] is not None
              for c in compiled.values()),
          "anchors=%s" % ", ".join("%s(n=%d)" % (k, v["A"].shape[0])
                                   for k, v in compiled.items()))

    # ---- K5 the P table ----------------------------------------------------
    print("\n--- B. corner x mismatch x calibration: P(workload passes) ---")
    table = {}
    for name, c in compiled.items():
        A, b, T, shift, x_ref = c["A"], c["b"], c["T"], c["shift"], c["x_ref"]
        n = A.shape[0]
        is_kcl = name == "poisson-2d"
        rows = []
        t0 = time.time()
        for p in pts:
            fam = p["fam"]
            k_self_mc, k_gm_mc, k_mir_mc = mc_f[fam]
            base = p["k_gm"] if is_kcl else p["k_pole"]
            e_un = []
            for k in range(n_mc):
                idx = (np.arange(n) + k) % n_mc
                if is_kcl:
                    dk = np.full(n, base)
                    ek = base
                    gk = p["k_gm"] * k_gm_mc[idx] * k_mir_mc[idx]
                else:
                    dk = base * k_self_mc[idx]
                    gk = p["k_gm"] * k_gm_mc[idx]
                    ek = p["k_gm"] * k_gm_mc[idx] * p["k_mirror"] * k_mir_mc[idx]
                Ap, bp = J.perturb(A, b, dk, gk, ek)
                e_un.append(J.qoi_err(J.qoi_step(Ap, bp, T, shift), x_ref))

            dt_op = abs(p["temp_c"] - 27.0)
            track = min(1.0, BG_UPDATE_DT_C / dt_op) if dt_op > 1e-9 else 0.0
            frac_d_proc = p["frac_gm"] if is_kcl else p["frac_pole"]
            frac_b_proc = p["frac_gm"]
            k_nt_a = np.full(n, 1.0 + frac_d_proc + MEAS_FLOOR)
            k_nt_b = np.full(n, 1.0 + frac_b_proc + MEAS_FLOOR)
            k_bg_a = np.full(n, 1.0 + frac_d_proc * track + MEAS_FLOOR)
            k_bg_b = np.full(n, 1.0 + frac_b_proc * track + MEAS_FLOOR)
            k_full = np.full(n, 1.0 + MEAS_FLOOR)
            e_nt = J.qoi_err(J.qoi_step(*J.perturb(A, b, k_nt_a, k_nt_b, k_nt_b),
                                        T, shift), x_ref)
            e_bg = J.qoi_err(J.qoi_step(*J.perturb(A, b, k_bg_a, k_bg_b, k_bg_b),
                                        T, shift), x_ref)
            e_full = J.qoi_err(J.qoi_step(*J.perturb(A, b, k_full, k_full,
                                                     k_full), T, shift), x_ref)

            def prob(errs, tol):
                if not p["bias_ok"]:
                    return 0.0
                return sum(1 for e in errs if e <= tol) / len(errs)

            rows.append(dict(
                tag=p["tag"], fam=fam, source=p["source"], proc=p["proc"],
                temp_c=p["temp_c"], k_pole=p["k_pole"], k_gm=p["k_gm"],
                k_mirror=p["k_mirror"], bias_ok=p["bias_ok"],
                p_uncal={("tol_%d" % int(t * 100)): prob(e_un, t)
                         for t in TOLS},
                p_no_track={("tol_%d" % int(t * 100)): prob([e_nt], t)
                            for t in TOLS},
                p_bg_track={("tol_%d" % int(t * 100)): prob([e_bg], t)
                            for t in TOLS},
                p_full={("tol_%d" % int(t * 100)): prob([e_full], t)
                        for t in TOLS},
                err_full=e_full,
                err_uncal_med=float(np.median(e_un))))
        table[name] = rows
        print("  %-13s P table done [%.1f s]" % (name, time.time() - t0),
              flush=True)

    hdr = ("%-13s %-22s %6s %6s %7s %7s %7s %7s"
           % ("anchor", "point", "k_pole", "k_gm", "P_un", "P_notrk",
              "P_bg", "P_full"))
    print("\n  " + hdr)
    print("  " + "-" * len(hdr))
    for name, rows in table.items():
        for r in rows:
            print("  %-13s %-22s %6.3f %6.3f %7.2f %7.2f %7.2f %7.2f%s"
                  % (name, r["tag"], r["k_pole"], r["k_gm"],
                     r["p_uncal"]["tol_1"], r["p_no_track"]["tol_1"],
                     r["p_bg_track"]["tol_1"], r["p_full"]["tol_1"],
                     "" if r["bias_ok"] else "  [bias invalid]"))

    # ---- K6..K13 checks on the table ---------------------------------------
    valid = [(n, r) for n in table for r in table[n] if r["bias_ok"]]
    mono = all(r[m]["tol_1"] <= r[m]["tol_2"] <= r[m]["tol_5"]
               for n in table for r in table[n]
               for m in ("p_uncal", "p_no_track", "p_bg_track", "p_full"))
    check("K5: P(tol) monotone in the 1/2/5 % windows (all anchors, modes)",
          mono, "%d bias-valid rows, %d draws each" % (len(valid), n_mc))
    check("K6: full calibration (100 ppm floor) passes every point at 5 %",
          all(r["p_full"]["tol_5"] == 1.0 for _, r in valid),
          "%d points" % len(valid))
    sens = [(n, r["tag"], r["p_uncal"]["tol_1"]) for n, r in valid
            if r["p_uncal"]["tol_1"] < 1.0]
    check("K7: at least two anchors show a real uncalibrated failure at 1 %",
          len({n for n, _, _ in sens}) >= 2,
          "failing rows: %d, e.g. %s"
          % (len(sens), ", ".join("%s/%s=%.2f" % s for s in sens[:6])))
    recov = all(r["p_bg_track"]["tol_1"] >= r["p_uncal"]["tol_1"]
                for _, r in valid)
    check("K8: background tracking never lowers P vs uncalibrated", recov,
          "bg >= uncal at every valid point")
    tt_rows = [(n, r) for n, r in valid if r["tag"].endswith("3v3_tt")
               or r["tag"].endswith("6v0_tt")]
    check("K9: TT/27 mismatch is live and full cal reduces it >=10x",
          all(r["err_uncal_med"] > 0.0 for _, r in tt_rows)
          and all(r["err_full"] < r["err_uncal_med"] / 10.0
                  for _, r in tt_rows),
          ", ".join("%s: med %.4g -> floor %.4g"
                    % (n, r["err_uncal_med"], r["err_full"])
                    for n, r in tt_rows))
    check("K10: no bias-invalid point (ideal-current-source pole cell cannot "
          "leave saturation — the IHP lane's hot-corner gate has no "
          "counterpart here)",
          all(r["bias_ok"] for n in table for r in table[n]),
          "bias_ok on all %d rows; the IHP lane flagged TT/125 and FF/125 "
          "with Vd=82/26 mV" % sum(len(v) for v in table.values()))
    check("K11: 6V family has the wider corner spread (lambda 2.7x)",
          stat["6v0"]["pole"][1] / stat["6v0"]["pole"][0]
          >= stat["3v3"]["pole"][1] / stat["3v3"]["pole"][0],
          "6v0 %.2fx vs 3v3 %.2fx"
          % (stat["6v0"]["pole"][1] / stat["6v0"]["pole"][0],
             stat["3v3"]["pole"][1] / stat["3v3"]["pole"][0]))

    # ---- summary ------------------------------------------------------------
    n_pass = sum(c["passed"] for c in checks)
    summary = dict(
        n_checks=len(checks), n_pass=n_pass, n_fail=len(fails), n_mc=n_mc,
        process="gf180mcu (3v3 + 6v0)", anchors={
            n: dict(n_states=int(c["A"].shape[0]),
                    alpha=float(c["res"].scaling.alpha),
                    horizon_s=float(c["T"]),
                    cert_status=c["res"].cert.status)
            for n, c in compiled.items()},
        calibration=dict(floor_ppm=MEAS_FLOOR * 1e6,
                         bg_update_dt_c=BG_UPDATE_DT_C,
                         modes=["uncal", "no_track", "bg_track", "full"]),
        axes={f: dict(k_pole=v["pole"], k_gm=v["gm"], corr=v["corr"],
                      separation=v["sep"]) for f, v in stat.items()},
        p_table=table,
        validity_limits=[
            "the 48 mismatch draws were measured at TT/27 C; they are applied "
            "as per-chip RELATIVE factors at every corner (declared "
            "composition, not a MC re-run per corner)",
            "full calibration is the continuous-trim upper bound (100 ppm "
            "floor, same declared readout bound as the IHP lane so the two "
            "processes are comparable); the real trim code range is B3",
            "6v0 rows apply the 6V family's own measured relative factors to "
            "the same compiled descriptors: they answer 'if this architecture "
            "were built on 6 V devices, how would the corner sensitivity "
            "look', not 'the chip uses 6 V devices here'",
            "poisson-2d is the 8x8 (64-state) mesh with the declared 1 ns "
            "model-time deadline mapped to alpha*1 ns (steady state); the "
            "16x16 anchor is run_pipeline's workload",
            "the step-response QoI is the operator-error proxy, not a "
            "nonlinear PDE re-run",
            "MOS noise is measured here (unlike IHP's PSP103 OSDI gap), but "
            "it enters this lane only through the map, not through the QoI "
            "(the QoI is deterministic operator error)",
        ],
    )
    with open(OUT + "/gf180_joint.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\n%d/%d checks passed, %d failed -> results/gf180_joint.json"
          % (n_pass, len(checks), len(fails)))
    if fails:
        print("FAILURES: " + "; ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
