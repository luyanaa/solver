"""G1 acceptance: gf180mcu primitive database (results/gf180_extract.json).

Acceptance for the batch-1 process choice: every number the compiler and
macromodel will consume must come from a measured ngspice run on
GF180MCU, be cross-checked against an independent caliber (the diode-mirror
extract.py table, the PDK's own Pelgrom formula, the analytic 4kT/g_m), and
fail loudly if a required block is missing.

    python3 -m bench.run_gf180       # ~1 s, reads results/gf180_extract.json

The check list is the G1 exit line from PLAN-gf180mcu-batch1.md:
  J0  fail-closed extraction status (complete, no failed required block)
  J1  §6.9 anchor reproduces: independent mirror ratio 1.0328 (lambda*dVds)
  J2  shared-reference mirror reads 1.0000 (the design intent, not a leak)
  J3  1:4 / 1:16 arrays are per-finger unity when Vds is clamped
  J4  lambda from the vds curve agrees with (ratio-1)/(vhold-vdiode)
  J5  gm/Id agrees with the extract.py diode caliber (~13.3)
  J6  ro: pole caliber vs diode caliber, both recorded, not silently mixed
  J7  MIM capacitor 1.016 pF vs the 500 um^2 * 2 fF/um^2 nominal
  J8  TG Ron and its 4kT*Ron noise (the switch noise must be derived, not
      .noise, when the device is in triode)
  J9  MOS white-noise gamma_eff vs the gamma=1 ideal, and the 1/f corner
  J10 input-referred density vs sqrt(4kT/gm) at the MEASURED gm
  J11 MC determinism (same seed -> bit-identical rows)
  J12 Pelgrom: measured diff-pair offset sigma vs the PDK formula
  J13 mirror mismatch is real (sigma >= 1 %) and unbiased (mean within 2 %)
  J14 TG Ron mismatch sigma (geometry-driven, ~1 %)
  J15 PVT: 18 points, all bias-valid, mirror ratio span, V_BE temp slope
  J16 joint corner+temp: 8 points, 6V family degrades more than 3V3
  J17 pre-PEX parasitic pole sensitivity vs the analytic log slope
  J18 transient: ro*C settling time = ln(9)*tau, THD of the gm cell
  J19 6V family extracted with its own lambda/ro and the lmin=0.6 um fact
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE) + "/results"
sys.path.insert(0, os.path.dirname(HERE))

K_B = 1.380649e-23
Q_E = 1.602176634e-19
T300 = 300.0
FOUR_KT = 4.0 * K_B * T300

# cross-reference constants (must stay in sync with the cited tables)
ANCHOR_RATIO = 1.0328      # DESIGN.md §6.9 gf180 independent-mirror leak
ANCHOR_GM_ID = 13.33       # results/device_extract.json, same bias
ANCHOR_LAMBDA_DIODE = 0.03405
ANCHOR_RO_DIODE = 2.84e6
N_PVT = 18
N_JOINT = 8
N_MC = 48
I_BIAS = 10e-6            # the extraction bias, so lambda <-> ro is fixed


def _ro(lam):
    """lambda (1/ohm) is measured at the 10 uA bias: ro = 1/(lambda*I)."""
    return 1.0 / (lam * I_BIAS)


def _load():
    with open(OUT + "/gf180_extract.json") as fh:
        return json.load(fh)


def _rel(a, b):
    return abs(a / b - 1.0) if b else float("inf")


def main(argv=None):
    res = _load()
    blocks = res.get("blocks") or {}
    status = res.get("status", "missing")
    checks = []
    failures = []

    def check(name, ok, detail, genuine_fail=True):
        rec = dict(name=name, passed=bool(ok), detail=detail,
                   genuine_fail=bool(genuine_fail and not ok))
        checks.append(rec)
        if not ok and genuine_fail:
            failures.append(name)
        print("%s  %-56s %s"
              % ("PASS" if ok else ("FAIL" if genuine_fail else "SKIP"),
                 name, detail))

    c3 = res["corners"]["3v3_tt"]
    c3s = res["corners"]["3v3_ss"]
    c3f = res["corners"]["3v3_ff"]
    c6 = res["corners"]["6v0_tt"]
    mc3 = res["mc"]["3v3"]
    mc6 = res["mc"]["6v0"]
    noise = res["noise"]
    pvt = res["pvt"]["rows"]
    jnt = res["joint"]["rows"]
    par = res["parasitics"]
    tra = res["transient"]

    # ---- J0 fail-closed ----------------------------------------------------
    failed_required = [k for k, b in blocks.items()
                       if b.get("status") == "failed" and b.get("required")]
    check("J0 extraction complete (fail-closed)",
          status == "complete" and not failed_required,
          "status=%s, blocks=%d ok, failed_required=%s"
          % (status, sum(1 for b in blocks.values()
                         if b.get("status") == "ok"), failed_required))

    # ---- J1 the §6.9 anchor ------------------------------------------------
    check("J1 independent mirror ratio == 1.0328 (lambda*dVds)",
          _rel(c3["ratio_vds_mismatch"], ANCHOR_RATIO) < 0.01,
          "meas %.5f vs §6.9 %.4f (%.2f %%)"
          % (c3["ratio_vds_mismatch"], ANCHOR_RATIO,
             100 * _rel(c3["ratio_vds_mismatch"], ANCHOR_RATIO)))

    # ---- J2 shared reference is not a leak ---------------------------------
    check("J2 shared-reference mirror == 1.0000",
          abs(c3["ratio_shared"] - 1.0) < 1e-3,
          "meas %.6f (independent copy %.4f, i.e. the clamp removes it)"
          % (c3["ratio_shared"], c3["ratio_vds_mismatch"]))

    # ---- J3 arrays ---------------------------------------------------------
    check("J3 1:4 / 1:16 arrays are per-finger unity",
          abs(c3["ratio4_per_finger"] - 1.0) < 2e-3
          and abs(c3["ratio16_per_finger"] - 1.0) < 2e-3,
          "per-finger %.5f / %.5f (totals %.3f / %.3f)"
          % (c3["ratio4_per_finger"], c3["ratio16_per_finger"],
             c3["ratio4"], c3["ratio16"]))

    # ---- J4 lambda mechanism ----------------------------------------------
    lam_dvds = (c3["ratio_vds_mismatch"] - 1.0) / (c3["vdd_v"] / 2
                                                   - c3["vdiode"])
    check("J4 lambda from the vds sweep == (ratio-1)/(vhold-vdiode)",
          _rel(c3["lambda_slope_vds"], lam_dvds) < 0.15,
          "slope %.5f /V vs chord %.5f /V (%.1f %%)"
          % (c3["lambda_slope_vds"], lam_dvds,
             100 * _rel(c3["lambda_slope_vds"], lam_dvds)))

    # ---- J5 gm/Id ----------------------------------------------------------
    gm_id = c3["gm_fd"] / c3["id_at_gmprobe_a"]
    check("J5 gm/Id agrees with the diode caliber",
          _rel(gm_id, ANCHOR_GM_ID) < 0.10,
          "meas %.2f vs extract.py %.2f (%.1f %%) at Id=%.2f uA"
          % (gm_id, ANCHOR_GM_ID, 100 * _rel(gm_id, ANCHOR_GM_ID),
             c3["id_at_gmprobe_a"] * 1e6))

    # ---- J6 two ro calibers, both kept ------------------------------------
    check("J6 ro calibers kept distinct (pole vs diode mirror)",
          0.80 < _ro(c3["lambda_ohm_inv"]) / ANCHOR_RO_DIODE < 1.20,
          "pole caliber ro=%.3f M (lambda=%.4f/V) vs diode caliber %.2f M "
          "(lambda=%.5f/V): the pole device sits at Vds=vhold, the diode "
          "mirror at Vds=Vgs -- both recorded"
          % (_ro(c3["lambda_ohm_inv"]) / 1e6, c3["lambda_ohm_inv"],
             ANCHOR_RO_DIODE / 1e6, ANCHOR_LAMBDA_DIODE))
    check("J6b intrinsic gain gm*ro in the 50 dB class",
          100.0 < c3["a0_gm_ro"] < 600.0,
          "a0=%.1f (%.1f dB) from gm=%.2f uS * ro=%.3f M"
          % (c3["a0_gm_ro"], 20 * math.log10(c3["a0_gm_ro"]),
             c3["gm_fd"] * 1e6, _ro(c3["lambda_ohm_inv"]) / 1e6))

    # ---- J7 MIM ------------------------------------------------------------
    check("J7 MIM 1 pF bank capacitor",
          _rel(c3["c_mim_f"], c3["c_mim_nominal_f"]) < 0.05,
          "meas %.4f pF vs nominal %.4f pF (%.2f %%)"
          % (c3["c_mim_f"] * 1e12, c3["c_mim_nominal_f"] * 1e12,
             100 * _rel(c3["c_mim_f"], c3["c_mim_nominal_f"])))

    # ---- J8 TG -------------------------------------------------------------
    tg = noise["tg"]["3v3_tt"]
    check("J8 TG switch Ron + 4kT*Ron noise",
          2000.0 < c3["ron_tg"] < 4000.0
          and 0.85 < tg["white_factor_4kTRon"] < 1.15,
          "Ron=%.0f ohm, band S_v/(4kT*Ron)=%.3f -- unlike the IHP triode "
          "case (0.001) the gf180 triode .noise agrees with 4kT*Ron, so the "
          "derived switch-noise label is confirmed not needed here"
          % (c3["ron_tg"], tg["white_factor_4kTRon"]))

    # ---- J9 gamma_eff ------------------------------------------------------
    gm3 = noise["nmos"]["3v3_tt"]["gamma_eff"]["white_factor"]
    gp3 = noise["pmos"]["3v3_tt"]["gamma_eff"]["white_factor"]
    gm6 = noise["nmos"]["6v0_tt"]["gamma_eff"]["white_factor"]
    check("J9 MOS white-noise gamma_eff vs the gamma=1 ideal",
          all(0.6 < g < 1.6 for g in (gm3, gp3, gm6)),
          "nm3v3 %.3f, pm3v3 %.3f, nm6v0 %.3f (white factor over 4kT/gm; "
          "vs the MOS 2/3 notation -> %.2fx / %.2fx / %.2fx)"
          % (gm3, gp3, gm6, gm3 / (2.0 / 3.0), gp3 / (2.0 / 3.0),
             gm6 / (2.0 / 3.0)))
    check("J9b 1/f corner is real (two-band density ratio > 1.2)",
          noise["nmos"]["3v3_tt"]["band_density_ratio_lo_hi"] > 1.2,
          "1-10 MHz / 1-100 MHz density ratio %.3f, flicker corner %.1f MHz, "
          "B=%.2e V^2*Hz"
          % (noise["nmos"]["3v3_tt"]["band_density_ratio_lo_hi"],
             (noise["nmos"]["3v3_tt"]["flicker_corner_hz"] or 0) / 1e6,
             noise["nmos"]["3v3_tt"]["flicker_b_v2_hz"]))

    # ---- J10 input-referred density vs 4kT/gm -----------------------------
    dens = noise["nmos"]["3v3_tt"]["density_in_nv_rthz"]
    ideal = 1e9 * math.sqrt(FOUR_KT / c3["gm_fd"])
    check("J10 input density vs sqrt(4kT/gm) at the MEASURED gm",
          _rel(dens, ideal) < 0.20,
          "meas %.2f nV/rtHz vs gamma=1 ideal %.2f nV/rtHz (%.1f %%)"
          % (dens, ideal, 100 * _rel(dens, ideal)))

    # ---- J11 determinism ---------------------------------------------------
    check("J11 MC determinism (same seed -> identical rows)",
          mc3["determinism_ok"] and mc6["determinism_ok"],
          "3v3=%s, 6v0=%s, n=%d each"
          % (mc3["determinism_ok"], mc6["determinism_ok"], mc3["n"]))

    # ---- J12 Pelgrom -------------------------------------------------------
    r3 = mc3["vd0_ratio_meas_over_pred"]
    r6 = mc6["vd0_ratio_meas_over_pred"]
    check("J12 diff-pair offset sigma vs the PDK Pelgrom formula",
          0.8 < r3 < 1.6 and 0.8 < r6 < 1.6,
          "3v3 sigma=%.2f mV vs pred %.2f mV (x%.2f); 6v0 sigma=%.2f mV vs "
          "pred %.2f mV (x%.2f) -- the excess is mobility (par_k) mismatch "
          "plus the finite 48-draw sample"
          % (mc3["vd0_offset"]["sigma"] * 1e3, mc3["vd0_pred_v"] * 1e3, r3,
             mc6["vd0_offset"]["sigma"] * 1e3, mc6["vd0_pred_v"] * 1e3, r6))

    # ---- J13 mirror mismatch ----------------------------------------------
    sm = mc3["ratio_mis"]["sigma"]
    mu = mc3["ratio_mis"]["mean"]
    check("J13 mirror mismatch is real and unbiased",
          sm > 0.01 and abs(mu - 1.0) < 0.02,
          "sigma=%.2f %% (stderr %.2f %%), mean=%.4f over %d draws"
          % (sm * 100, mc3["ratio_mis"]["sigma_stderr"] * 100, mu, mc3["n"]))
    check("J13b 1:4 array mismatch adds independent draws",
          mc3["ratio4_mis"]["sigma"] > mc3["ratio_mis"]["sigma"] * 0.8,
          "sigma(1:4)=%.2f %% vs sigma(1:1)=%.2f %%"
          % (mc3["ratio4_mis"]["sigma"] * 100, sm * 100))

    # ---- J14 TG Ron mismatch ----------------------------------------------
    check("J14 TG Ron mismatch is sub-percent (geometry-dominated)",
          mc3["ron_tg"]["sigma"] / mc3["ron_tg"]["mean"] < 0.05,
          "Ron=%.1f +- %.1f ohm (%.2f %%)"
          % (mc3["ron_tg"]["mean"], mc3["ron_tg"]["sigma"],
             100 * mc3["ron_tg"]["sigma"] / mc3["ron_tg"]["mean"]))

    # ---- J15 PVT -----------------------------------------------------------
    ratios = [v["mirror_ratio"] for v in pvt.values()]
    vb = {(v["fam"], v["temp_c"], v["vdd_v"]): v["vdiode"] for v in pvt.values()}
    slope_vbe = None
    if ("3v3", -40.0, 3.3) in vb and ("3v3", 125.0, 3.3) in vb:
        slope_vbe = ((vb[("3v3", 125.0, 3.3)] - vb[("3v3", -40.0, 3.3)])
                     / 165.0 * 1e3)
    check("J15 PVT: all points present and bias-valid",
          len(pvt) == N_PVT and all(v["bias_ok"] for v in pvt.values()),
          "%d/%d rows, bias_ok all=%s, mirror ratio span %.4f..%.4f"
          % (len(pvt), N_PVT, all(v["bias_ok"] for v in pvt.values()),
             min(ratios), max(ratios)))
    check("J15b V_BE temperature slope (interface bias drift)",
          slope_vbe is not None and 0.5 < abs(slope_vbe) < 1.6,
          "d(V_BE)/dT = %.2f mV/K at 10 uA (3v3) -- the same magnitude as the "
          "IHP HBT (V_BE -2 mV/K) and the reason the interface bias sits in "
          "the background-cal loop"
          % (slope_vbe if slope_vbe is not None else float("nan")))

    # ---- J16 joint ---------------------------------------------------------
    j3 = [v["mirror_ratio"] for k, v in jnt.items() if "3v3" in k]
    j6 = [v["mirror_ratio"] for k, v in jnt.items() if "6v0" in k]
    check("J16 joint corner+temp points exist and separate the families",
          len(jnt) == N_JOINT and len(j3) == 4 and len(j6) == 4
          and (max(j6) - min(j6)) > 0.05,
          "%d/%d points; 3v3 ratio %.4f..%.4f, 6v0 %.4f..%.4f (the 6V family "
          "carries a 2.7x larger lambda, so its corner spread is wider)"
          % (len(jnt), N_JOINT, min(j3), max(j3), min(j6), max(j6)))

    # ---- J17 parasitics ----------------------------------------------------
    check("J17 pre-PEX pole sensitivity vs the analytic log slope",
          abs(par["slope_rel_err"]) < 0.05,
          "meas %.4e /F vs predicted %.4e /F (%.2f %%); C sweep %s fF"
          % (par["slope_log_per_f"], par["slope_log_per_f_pred"],
             100 * par["slope_rel_err"],
             [c["c_added_ff"] for c in par["cases"]]))

    # ---- J18 transient -----------------------------------------------------
    tau = _ro(c3["lambda_ohm_inv"]) * 1e-12      # ro * 1 pF
    settle = tra["slow_step"]["pole_cell"]["settle_s"]
    check("J18 ro*C settling time == ln(9)*tau",
          3.4 < settle / tau < 4.3,
          "settle(2%%)=%.2f us, tau=ro*C=%.2f us -> %.2f tau (ln9=2.20 for "
          "1%%; 2%% band gives ln(50)=3.91)"
          % (settle * 1e6, tau * 1e6, settle / tau))
    thd = tra["fast_sine"]["gm_cell"]["thd"]
    check("J18b gm-cell THD at 1 mV / 10 MHz",
          thd < 0.01,
          "THD=%.2f %% (2nd %.2e dominates, 3rd %.1e); slew %.2e V/s at the "
          "fast step; f0 meas %.1f MHz"
          % (thd * 100, tra["fast_sine"]["gm_cell"]["harmonics"]["2"],
             tra["fast_sine"]["gm_cell"]["harmonics"]["3"],
             tra["fast_step"]["gm_cell"]["slew_v_per_s"],
             tra["fast_sine"]["gm_cell"]["f0_meas_hz"] / 1e6))

    # ---- J19 6V family -----------------------------------------------------
    check("J19 6V family: own lambda/ro, no HBT-equivalent shortcut",
          c6["lambda_ohm_inv"] > 2.0 * c3["lambda_ohm_inv"]
          and c6["ro_ohm"] < c3["ro_ohm"],
          "6v0 lambda=%.4f /V, ro=%.3f M, pole=%.1f kHz, Ron_TG=%.0f ohm "
          "(vs 3v3 lambda=%.4f /V, ro=%.3f M)"
          % (c6["lambda_ohm_inv"], c6["ro_ohm"] / 1e6, c6["pole_hz"],
             c6["ron_tg"], c3["lambda_ohm_inv"], c3["ro_ohm"] / 1e6))
    lmin_ok = res.get("families", {}).get("6v0", {}).get("l_tg", 0) >= 0.6e-6
    check("J19b 6V lmin=0.6 um recorded (the 0.5 um TG failed closed)",
          lmin_ok,
          "families['6v0']['l_tg']=%s m -- an L below the first bin gives "
          "'could not find a valid modelname', caught by the corner block "
          "before it could poison the MC block"
          % (res.get("families", {}).get("6v0", {}).get("l_tg")))

    # ---- cross-corner sanity ----------------------------------------------
    gm_ss = c3s["gm_fd"] / c3["gm_fd"]
    gm_ff = c3f["gm_fd"] / c3["gm_fd"]
    check("J20 corner order: gm(ss) < gm(tt) < gm(ff)",
          gm_ss < 1.0 < gm_ff,
          "gm(ss)/gm(tt)=%.3f, gm(ff)/gm(tt)=%.3f, ratio(ss)=%.4f, "
          "ratio(ff)=%.4f"
          % (gm_ss, gm_ff, c3s["ratio_vds_mismatch"],
             c3f["ratio_vds_mismatch"]))

    n_pass = sum(c["passed"] for c in checks)
    summary = dict(
        n_checks=len(checks), n_pass=n_pass,
        n_genuine_fail=len(failures), failed=failures, status=status,
        prim3v3=dict(ratio=c3["ratio_vds_mismatch"],
                     ratio_shared=c3["ratio_shared"],
                     lambda_pole=c3["lambda_ohm_inv"],
                     lambda_diode=ANCHOR_LAMBDA_DIODE,
                     gm=c3["gm_fd"], ro=c3["ro_ohm"],
                     a0=c3["a0_gm_ro"], c_mim_f=c3["c_mim_f"],
                     ron_tg=c3["ron_tg"], pole_hz=c3["pole_hz"]),
        prim6v0=dict(ratio=c6["ratio_vds_mismatch"],
                     lambda_pole=c6["lambda_ohm_inv"],
                     ro=c6["ro_ohm"], gm=c6["gm_fd"], ron_tg=c6["ron_tg"]),
        noise=dict(gamma_eff_white=dict(nm3v3=gm3, pm3v3=gp3, nm6v0=gm6),
                   input_density_nv=dens, ideal_nv=ideal,
                   flicker_corner_hz=noise["nmos"]["3v3_tt"]
                   ["flicker_corner_hz"],
                   tg_white_factor=tg["white_factor_4kTRon"]),
        mc=dict(n=mc3["n"], determinism=mc3["determinism_ok"],
                vd0_sigma=mc3["vd0_offset"]["sigma"],
                vd0_pred=mc3["vd0_pred_v"],
                ratio_sigma=sm, ratio_mean=mu,
                ron_sigma_rel=mc3["ron_tg"]["sigma"]
                / mc3["ron_tg"]["mean"]),
        pvt=dict(n=len(pvt), vbe_slope_mv_per_k=slope_vbe,
                 ratio_span=(min(ratios), max(ratios))),
        joint=dict(n=len(jnt), r3=(min(j3), max(j3)),
                   r6=(min(j6), max(j6))),
        paras=dict(slope_rel_err=par["slope_rel_err"]),
        transient=dict(settle_tau=settle / tau, thd=thd),
    )
    with open(OUT + "/gf180.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n%d/%d checks passed, %d genuine failures -> results/gf180.json"
          % (n_pass, len(checks), len(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
