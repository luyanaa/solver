"""B2 macromodel acceptance: behavioral primitive models fed from the
MEASURED b2 record, and a re-evaluation of P(workload passes) with more
fidelity than the linear-param-spread version in run_b2.py.

    python3 -m bench.run_b2_behavioral      # ~2 s, reads results/b2_extract.json + b2.json

Models under test (twin/macromodel.py, all numpy-only closed forms):

  IntegratorStage — finite DC gain A0 = gm/g_out_total (device output
      conductance adds to the load), pole omega_p = g_out_total/C
      (GBW/A0 identity), output swing saturation, diff-pair sech^2
      compression; NOT a solver.
  CurrentMirror / TGSwitch — measured ratio/Ron + Pelgrom sigma samplers.
  NoiseModel — HBT white density .noise-MEASURED (6.38 nV/sqrt(Hz),
      white factor 1.897); MOS path DERIVED 4kT*gamma/gm, honestly
      labeled "derived-not-measured".

Headline measured fact (reproduced here from the b2 AC record): the MOS
integrator self-coefficient is (gm + g_out)/C, not gm/C.  The derived
device output conductance at the TT bias is 15.93 uS (ro=62.8k,
lambda_fit=1.147/V at Id=13.9uA), adding ~+10.6% to the pole the linear
gm/C convention assigns.  The b2.json P(workload) was computed on the
gm axis alone (sigma 1.5%) and never saw this leak nor the mirror
sigma (5.5%) — the deltas below quantify both.

Honesty: every "measured" number comes from results/b2_extract.json or
the b2_*.dat records; model-only quantities (v_swing headroom, TG
c_par/q_inj_fc, charge injection) are declared as such, never presented
as data.  A macromodel that cannot reproduce its measured point fails
loudly (genuine_fail).
"""
from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE) + "/results"
sys.path.insert(0, os.path.dirname(HERE))

from lowering.netlist import (BehavioralNetlist, BehavioralIntegrator,  # noqa: E402
                              BehavioralEdge, BehavioralPort, BehavioralTap,
                              to_behavioral, attach_nonidealities)
from twin.macromodel import (IntegratorStage, CurrentMirror, TGSwitch,  # noqa: E402
                             NoiseModel, load_b2_extract, hbt_ro,
                             mos_corner_fits, b2_nonideal_map,
                             VT, FOURKT, RL_AC_OHM, C_BANK_F, LAMBDA_HBT)
from twin.calib import calibrate_foreground  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))  # noqa: E402 (already above)

C_PF = 1.0
MEAS_NOISE_PPM = 100.0      # same foreground-cal floor as run_b2.py
GAIN_PAIR_OFFSET = 12       # stage2 draw index (fixed pairing of MC samples)


def _load_dat(name):
    return np.atleast_2d(np.loadtxt(OUT + "/" + name))


def _ac_A0(tag):
    """Measured low-frequency gain of the b2 gm-cell-B AC stage."""
    d = _load_dat("b2_ac_%s.dat" % tag)
    mag = np.abs(d[:, 1] + 1j * d[:, 2])
    return float(mag[0])


def _diff_slope_tt():
    """Measured corner diff-pair slope dV(dc1)/dVd over the b2 sweep."""
    d = _load_dat("b2_diff_tt.dat")
    f, v1 = d[:, 0], d[:, 1]
    return float((v1[-1] - v1[0]) / (f[-1] - f[0]))


# ---------------------------------------------------------------------------
# 2-integrator coupled system (the workload vehicle)
# ---------------------------------------------------------------------------
class TwoIntegrator:
    """Cascade x1 -> x2 built ONLY from measured b2 quantities.

    Both stages are the same MOS gm/C primitive (B0 T1):
        stage k:  dx_k = (gm_k * in_k - G_k * x_k)/C
    G_k = gm_k + g_dev (device output conductance, measured AC fit) is
    the NONIDEAL self-coefficient; the linear b2 convention uses
    self = gm_k (g_dev = 0).  The coupling edge carries the measured
    mirror ratio (T5 single-flux-source mirror semantics: gain
    gm_c*ratio/C).  Stage1 gm / Id draws: MC sample k; stage2: sample
    (k+12) mod n_mc (fixed pairing; n_mc is read from the measured record).
    """

    def __init__(self, ext):
        tt = ext["corners"]["tt"]
        mc = ext["mc"]
        self.C = C_BANK_F
        self.gm_tt = tt["gm_mos_fd"]
        self.gm_c = tt["gm_mos_fd"]          # coupling edge gm (TT)
        self.r_tt = tt["mirror_ratio"]
        self.id_tt = tt["id_mos_pole"]
        fits = mos_corner_fits(ext)
        f_tt = fits["tt"]
        self.g_dev_tt = f_tt["g_out_total"] - 1.0 / RL_AC_OHM   # S, device share
        self.lam_tt = self.g_dev_tt / self.id_tt                # V^-1
        self.gm_vals = list(mc["gm_mos_fd"]["values"])
        self.id_vals = list(mc["id_mos_pole"]["values"])
        self.r_vals = list(mc["mirror_ratio"]["values"])
        # Truth comes from the measured sample count (b2_extract.json mc_n /
        # the recorded value lists), never a stale module constant: the
        # 24-sample constant vs 48 recorded values was exactly the bug that
        # produced probabilities > 1 in the linear-only row.
        self.n_mc = len(self.gm_vals)
        if not (len(self.id_vals) == len(self.r_vals) == self.n_mc):
            raise ValueError(
                "MC value lists disagree: gm=%d id=%d ratio=%d"
                % (len(self.gm_vals), len(self.id_vals), len(self.r_vals)))
        if ext.get("mc_n") not in (None, self.n_mc):
            raise ValueError("mc_n=%s but %d recorded MC samples"
                             % (ext.get("mc_n"), self.n_mc))

    def instance(self, k, nonideal=True):
        gm1 = self.gm_vals[k]
        gm2 = self.gm_vals[(k + GAIN_PAIR_OFFSET) % self.n_mc]
        if nonideal:
            id1 = self.id_vals[k]
            id2 = self.id_vals[(k + GAIN_PAIR_OFFSET) % self.n_mc]
            G1 = gm1 + self.lam_tt * id1
            G2 = gm2 + self.lam_tt * id2
        else:
            G1, G2 = gm1, gm2               # linear b2 convention: leak = 0
        return dict(gm1=gm1, gm2=gm2, G1=G1, G2=G2,
                    r=self.r_vals[k],
                    a1=G1 / self.C, a2=G2 / self.C,
                    b1=gm1 / self.C, k12=self.gm_c * self.r_vals[k] / self.C)

    def dc_gain(self, inst):
        """u -> x2 DC gain of the assembled cascade."""
        x1ss = inst["b1"] / inst["a1"]
        return inst["k12"] / inst["a2"] * x1ss


def _rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
def main(argv=None):
    ext = load_b2_extract()
    with open(OUT + "/b2.json") as fh:
        b2 = json.load(fh)
    checks = []
    fails = []

    def check(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))
        if not ok:
            fails.append(name)
        print("%s  %-58s %s" % ("PASS" if ok else "FAIL", name, detail))

    tt = ext["corners"]["tt"]
    mc = ext["mc"]
    n_mc = len(mc["mirror_ratio"]["values"])

    # =====================================================================
    # A. IntegratorStage reproduces the measured AC point (corner sweep)
    # =====================================================================
    print("\n--- A. integrator stage vs measured b2 AC record ---")
    for tag in ("tt", "ss", "ff"):
        ext["corners"][tag]["_a0_ac"] = _ac_A0(tag)
    fits = mos_corner_fits(ext)

    # A1: measured AC magnitude is 1-pole (justifies the model class)
    def _fit_1pole(f, mag, a0):
        m = (f >= 1e4) & (f <= 1e8)
        y = np.log(mag[m])
        best = (float("inf"), float("nan"))
        for fp in np.logspace(5.3, 7.5, 4000):
            pred = np.log(a0) - 0.5 * np.log(1.0 + (f[m] / fp) ** 2)
            r = float(np.sqrt(np.mean((pred - y) ** 2)))
            if r < best[0]:
                best = (r, float(fp))
        return best

    worst_fit = 0.0
    fp_fits = {}
    for tag in ("tt", "ss", "ff"):
        d = _load_dat("b2_ac_%s.dat" % tag)
        f, mag = d[:, 0], np.abs(d[:, 1] + 1j * d[:, 2])
        r, fp = _fit_1pole(f, mag, fits[tag]["A0"])
        worst_fit = max(worst_fit, r)
        fp_fits[tag] = fp
    check("A1: measured AC magnitude is 1-pole (log-mag rms < 1e-3)",
          worst_fit < 1e-3,
          "worst rms %.2e over tt/ss/ff; fitted fp %.2f/%.2f/%.2f MHz"
          % (worst_fit, fp_fits["tt"] / 1e6, fp_fits["ss"] / 1e6,
             fp_fits["ff"] / 1e6))

    # A2: macromodel pole (GBW/A0 identity) reproduces pole_hz per corner
    deltas = {t: fits[t]["pole_pred_hz"] / fits[t]["pole_meas_hz"] - 1.0
              for t in ("tt", "ss", "ff")}
    worst = max(abs(v) for v in deltas.values())
    check("A2: predicted pole = gm/(A0*C) matches measured pole_hz (<6%)",
          worst < 0.06,
          "tt %+.2f%%  ss %+.2f%%  ff %+.2f%% (worst %.2f%%)"
          % (deltas["tt"] * 100, deltas["ss"] * 100, deltas["ff"] * 100,
             worst * 100))

    # A3: derived ro / lambda_fit — corner-dependent, honest vs ledger
    lams = {t: fits[t]["lambda_fit"] for t in ("tt", "ss", "ff")}
    ok_lam = all(0.3 < l < 3.0 for l in lams.values())
    check("A3: derived device ro > 0 and lambda_fit in [0.3, 3.0] /V "
          "(corner-dependent; ledger 0.970/V is a point value)",
          ok_lam and all(math.isfinite(fits[t]["ro_ohm"])
                         for t in ("tt", "ss", "ff")),
          "ro tt=%.1fk ss=%.0fk ff=%.1fk; lambda_fit %.2f/%.2f/%.2f /V"
          % (fits["tt"]["ro_ohm"] / 1e3, fits["ss"]["ro_ohm"] / 1e3,
             fits["ff"]["ro_ohm"] / 1e3,
             lams["tt"], lams["ss"], lams["ff"]))

    # A4: HBT interface stage is a near-ideal integrator
    ro_hbt = hbt_ro(ext)
    st_hbt = IntegratorStage(gm=tt["gm_hbt"], C=C_BANK_F, ro=ro_hbt,
                             v_swing=0.5, drive="diffpair", name="hbt-if")
    leak_frac = st_hbt.leak_rad / (tt["gm_hbt"] / C_BANK_F)
    check("A4: HBT interface leak/gmC < 0.1%% (ro = lambda*Ic, "
          "lambda=%.4f/V)" % LAMBDA_HBT,
          0 < leak_frac < 1e-3,
          "ro=%.2f MOhm, A0=%.0f, leak/gmC=%.4f%%"
          % (ro_hbt / 1e6, st_hbt.A0, leak_frac * 100))

    # A5: diff-pair tanh law reproduces the measured sweep slope (VBIC
    # residual ~4% reported as validity limit)
    slope_meas = _diff_slope_tt()
    RC_LOAD, I_TAIL = 10e3, 40e-6
    vsat = 2.0 * VT
    grid = np.linspace(-0.01, 0.01, 201)
    i_over = I_TAIL / 2.0 * (1.0 + np.tanh(grid / vsat))   # Q3 side: base at +Vd
    slope_model = float((i_over[-1] - i_over[0]) / (grid[-1] - grid[0])
                        * (-RC_LOAD))
    d5 = slope_model / slope_meas - 1.0
    check("A5: macromodel diff-pair tanh slope reproduces measured "
          "b2_diff_tt slope (<6%; VBIC residual reported)",
          abs(d5) < 0.06 and slope_meas < 0 and slope_model < 0,
          "model %.3f vs measured %.3f V/V (%+.1f%%)" %
          (slope_model, slope_meas, d5 * 100))

    # A6: saturation consistency (linearization + clamp)
    st_mos = IntegratorStage(gm=tt["gm_mos_fd"], C=C_BANK_F,
                             ro=fits["tt"]["ro_ohm"], v_swing=0.5,
                             name="t1-mos")
    lin_err = abs((st_mos.respond(1e-6) / 1e-6) / st_mos.A0 - 1.0)
    check("A6: respond() linearizes to A0 (1e-9) and clamps at v_swing",
          lin_err < 1e-9 and abs(st_mos.respond(10.0)) <= st_mos.v_swing,
          "drespond/dvin(0) rel err %.1e; respond(10V)=%.3g V <= %.2f"
          % (lin_err, float(st_mos.respond(10.0)), st_mos.v_swing))

    # =====================================================================
    # B. mirror / TG / noise reproduction
    # =====================================================================
    print("\n--- B. mirror / TG / noise samplers vs measured MC ---")
    mir = CurrentMirror(ratio=tt["mirror_ratio"],
                        sigma_ratio=mc["mirror_ratio"]["sigma"],
                        rng=np.random.default_rng(2026))
    draws = mir.sample(240000)
    sig_hat = float(np.std(draws))
    mean_mc = mc["mirror_ratio"]["mean"]
    sig_mean_mc = mc["mirror_ratio"]["sigma"] / math.sqrt(n_mc)
    check("B1: mirror sampler sigma == measured Pelgrom sigma (<1%)",
          abs(sig_hat / mc["mirror_ratio"]["sigma"] - 1.0) < 0.01,
          "sigma_hat=%.5f vs measured %.5f" % (
              sig_hat, mc["mirror_ratio"]["sigma"]))
    check("B2: TT mirror ratio is a consistent MC center "
          "(|ratio_tt - MC mean| < 2 sigma_mean)",
          abs(tt["mirror_ratio"] - mean_mc) < 2 * sig_mean_mc,
          "ratio_tt=%.6f, MC mean=%.6f, sigma_mean=%.6f (mismatch spread "
          "is centered on the corner within sampling error)"
          % (tt["mirror_ratio"], mean_mc, sig_mean_mc))

    tg = TGSwitch(ron=tt["ron_tg"], sigma_ron=mc["ron_tg"]["sigma"],
                  c_par=30e-15, q_inj_fc=0.5,       # declared, not measured
                  rng=np.random.default_rng(2026))
    ron_hat = tg.sample(240000)
    check("B3: TG Ron sampler: tt point + sigma + corner spread",
          abs(float(np.std(ron_hat)) / mc["ron_tg"]["sigma"] - 1.0) < 0.01
          and abs(ext["corners"]["ss"]["ron_tg"] - 599.3) < 1.0
          and abs(ext["corners"]["ff"]["ron_tg"] - 331.3) < 1.0,
          "ron_tt=%.1f Ohm, sigma=%.2f, corners %.0f/%.0f (ss/ff measured)"
          % (tg.ron, mc["ron_tg"]["sigma"],
             ext["corners"]["ss"]["ron_tg"], ext["corners"]["ff"]["ron_tg"]))
    tg_pole = tg.series_rc_pole_hz(C_BANK_F)
    stage_pole_hz = fits["tt"]["pole_pred_hz"]
    check("B4: TG series-RC pole sits far above the stage pole "
          "(model prediction, c_par declared not measured)",
          tg_pole > 20 * stage_pole_hz,
          "Ron=452 Ohm into 1 pF: %.0f MHz vs stage pole %.2f MHz "
          "(%.0fx)" % (tg_pole / 1e6, stage_pole_hz / 1e6,
                       tg_pole / stage_pole_hz))

    n_hbt = NoiseModel.from_b2_hbt(ext)
    dens_recheck = math.sqrt(ext["noise"]["hbt"]["tt"]["var_inoise_white"]
                             / ext["noise"]["hbt"]["gamma_eff"]["bw"])
    check("B5: HBT white density reproduced from the measured record "
          "(var_white/bw -> %.3f nV/rootHz)" % (n_hbt.density_v_rthz * 1e9),
          abs(n_hbt.density_v_rthz - dens_recheck) < 1e-18
          and n_hbt.source == "measured-hbt",
          "density=%.4f nV/sqrt(Hz), source=%s" % (
              n_hbt.density_v_rthz * 1e9, n_hbt.source))
    var_band = n_hbt.variance(1e3, 1e9)
    rel = var_band / ext["noise"]["hbt"]["tt"]["var_inoise_white"] - 1.0
    check("B6: NoiseModel band integration reproduces var_inoise_white "
          "(<1e-6 rel)", abs(rel) < 1e-6,
          "var(1k-1G)=%.6e vs measured %.6e (%+.1e rel)"
          % (var_band, ext["noise"]["hbt"]["tt"]["var_inoise_white"], rel))

    gm_hbt = tt["gm_hbt"]
    e_n_gamma1 = math.sqrt(FOURKT * 1.0 / gm_hbt)
    fac_calc = (n_hbt.density_v_rthz / e_n_gamma1) ** 2
    fac_rec = ext["noise"]["hbt"]["gamma_eff"]["white_factor"]
    check("B7: white factor cross-check (e_n_meas/e_n[gamma=1])^2 "
          "== recorded 1.897 (<1%)",
          abs(fac_calc / fac_rec - 1.0) < 0.01,
          "computed %.4f vs recorded %.4f (SNR cost %.2f dB)"
          % (fac_calc, fac_rec, 10 * math.log10(fac_rec)))

    n_mos = NoiseModel.derived_mos(tt["gm_mos_fd"])
    dens_expect = math.sqrt(FOURKT * 0.5 / tt["gm_mos_fd"])
    check("B8: MOS density is the 4kT*gamma/gm derivation, labeled "
          "'derived-not-measured' (no fabricated number)",
          abs(n_mos.density_v_rthz - dens_expect) < 1e-18
          and n_mos.source == "derived-not-measured",
          "%.3f nV/sqrt(Hz) derived (gamma=0.5); measured gamma_eff "
          "UNAVAILABLE (PSP103 OSDI zero spectra)" % (
              n_mos.density_v_rthz * 1e9))

    # =====================================================================
    # C. 2-integrator coupled system: ideal vs nonideal
    # =====================================================================
    print("\n--- C. 2-integrator coupled system (measured quantities only) ---")
    sys2 = TwoIntegrator(ext)
    assert sys2.n_mc == n_mc
    inst_tt = dict(gm1=tt["gm_mos_fd"], gm2=tt["gm_mos_fd"],
                   G1=tt["gm_mos_fd"] + sys2.g_dev_tt,
                   G2=tt["gm_mos_fd"] + sys2.g_dev_tt,
                   r=tt["mirror_ratio"])
    inst_tt.update(a1=inst_tt["G1"] / C_BANK_F, a2=inst_tt["G2"] / C_BANK_F,
                   b1=inst_tt["gm1"] / C_BANK_F,
                   k12=sys2.gm_c * inst_tt["r"] / C_BANK_F)
    pole_tt = inst_tt["a1"]
    G_tt = sys2.dc_gain(inst_tt)
    leak_pct = sys2.g_dev_tt / tt["gm_mos_fd"]
    check("C1: measured leak shifts the integrator pole by g_out/gm "
          "= %+.1f%% vs the linear gm/C convention (headline)"
          % (leak_pct * 100),
          0.08 < leak_pct < 0.13,
          "g_dev=%.2f uS (ro=%.1f kOhm), (gm+g_out)/gm = %.4f"
          % (sys2.g_dev_tt * 1e6, 1.0 / sys2.g_dev_tt / 1e3,
             1.0 + leak_pct))

    A_ideal = np.array([[-tt["gm_mos_fd"], 0.0],
                        [inst_tt["k12"], -tt["gm_mos_fd"]]])
    A_non = np.array([[-inst_tt["a1"], 0.0],
                      [inst_tt["k12"], -inst_tt["a2"]]])
    eig_pred = np.array([-inst_tt["a1"], -inst_tt["a2"]])
    eig_direct = np.sort(np.linalg.eigvals(A_non).real)
    check("C2: nonideal eigenvalues == closed-form leak prediction "
          "(1e-9 rel)",
          np.allclose(eig_direct, np.sort(eig_pred), rtol=1e-9)
          and np.allclose(np.sort(np.linalg.eigvals(A_ideal).real),
                          [-tt["gm_mos_fd"]] * 2, rtol=1e-12),
          "nonideal %.4g/%.4g rad/s (ideal %.4g); leak is systematic, "
          "not a spread" % (eig_direct[0], eig_direct[1],
                            tt["gm_mos_fd"] / C_BANK_F))

    # C3: DC gain — macromodel closed form vs assembled linear algebra
    B = np.array([[inst_tt["b1"]], [0.0]])
    Cm = np.array([[0.0, 1.0]])
    G_alg = float((Cm @ np.linalg.inv(-A_non) @ B).item())
    check("C3: DC gain closed form == (-A)^-1 assembly (1e-9 rel)",
          abs(G_alg / G_tt - 1.0) < 1e-9,
          "G_tt = %.4f V/V (linear-model convention would say %.4f — the "
          "measured leak reduces the cascade DC gain by %.1f%%)"
          % (G_tt, inst_tt["r"], (1.0 - G_tt / inst_tt["r"]) * 100))

    # C4: RK4 time-domain step vs expm of the assembled system
    from scipy.linalg import expm
    M = np.zeros((3, 3))
    M[:2, :2] = A_non
    M[:2, 2] = B[:, 0]
    dt, T = 0.05e-9, 0.5e-6
    x = np.zeros(2)
    xs = []
    bvec = B[:, 0]
    for _ in range(int(T / dt)):
        x = _rk4_step(lambda v: A_non @ v + bvec, x, dt)
        xs.append(x.copy())
    E = expm(M * T)
    x_exact = E[:2, 2]
    err_c4 = float(np.max(np.abs(xs[-1] - x_exact)) / max(abs(x_exact[1]), 1e-30))
    check("C4: RK4 on the macromodel ODE == expm step response (1e-6)",
          err_c4 < 1e-6,
          "max rel err %.2e at t=0.5us (dt=50ps, poles %.2e rad/s)"
          % (err_c4, pole_tt))

    # C5: static-nonlinearity contract: respond() is exactly the readout
    # tanh composed on the open-loop stage steady state (x_ss = A0*u),
    # the small-signal slope is A0, and the 1%-compression drive inverts
    # tanh consistently (model integrity, like the run_transform
    # isometry checks)
    u_grid = np.array([1e-3, 0.01, 0.05, 0.1, 0.3])
    x_open = st_mos.A0 * u_grid                       # open-loop stage x_ss
    readout = st_mos.v_swing * np.tanh(x_open / st_mos.v_swing)
    comp_err = float(np.max(np.abs(readout - st_mos.respond(u_grid))))
    z99 = float(np.arctanh(0.99))                     # 2.6466
    u_1pct = z99 * st_mos.v_swing / st_mos.A0
    resp_1pct = float(st_mos.respond(u_1pct))
    check("C5: respond() == readout(open-loop x_ss) on a grid (1e-12), "
          "and the 1%%-compression drive inverts tanh consistently",
          comp_err < 1e-12
          and abs(resp_1pct / (0.99 * st_mos.v_swing) - 1.0) < 1e-9,
          "max grid err %.1e; u_1pct=%.4f V (A0=%.2f, v_swing=%.2f) -> "
          "respond=%.4f V = 0.99*swing" % (
              comp_err, u_1pct, st_mos.A0, st_mos.v_swing, resp_1pct))

    # C6: interface offset eats diff-pair gm AT the operating point
    # (local sech^2) — why foreground vos cal is mandatory, not cosmetic
    vos_max = float(np.max(np.abs(mc["vd0_offset_raw"]["values"])))
    gm_fac = st_hbt.drive_gm_factor(vos_max)
    slope_fac = float(st_hbt.drive_gain_factor(vos_max))
    # cross-check: local factor == derivative of the drive law
    eps = 1e-6
    deriv = float((st_hbt.v_sat * np.tanh((vos_max + eps) / st_hbt.v_sat)
                   - st_hbt.v_sat * np.tanh((vos_max - eps) / st_hbt.v_sat))
                  / (2 * eps))
    check("C6: worst-chip interface offset (%.1f mV) costs local diff-pair "
          "gm x%.2f (%.0f%% loss uncalibrated); offset cal (C3 "
          "foreground) nulls it" % (vos_max * 1e3, gm_fac,
                                    (1.0 - gm_fac) * 100),
          0.7 < gm_fac < 0.85
          and abs(deriv - gm_fac) < 1e-9
          and abs(float(np.cosh(vos_max / st_hbt.v_sat) ** -2) - gm_fac)
          < 1e-12,
          "local sech^2=%.3f, averaged sweep factor=%.3f (both < 1); "
          "v_sat=2V_T=%.1f mV" % (gm_fac, slope_fac, st_hbt.v_sat * 1e3))

    # =====================================================================
    # D. P(workload passes): linear-only vs nonideal vs calibrated
    # =====================================================================
    print("\n--- D. P(workload passes) over the %d measured MC chips ---" % n_mc)
    tols = (0.10, 0.05, 0.02)

    # D1: reproduce run_b2.py's linear-param-only numbers exactly
    p_lin = {}
    for tol in tols:
        ok = [abs(g / tt["gm_mos_fd"] - 1.0) <= tol for g in sys2.gm_vals]
        p_lin[tol] = sum(ok) / n_mc
    p_b2 = b2["p_workload"]
    check("D1: linear-param-only P reproduces run_b2.py (gm axis only)",
          abs(p_lin[0.10] - p_b2["tol_10"]) < 1e-9
          and abs(p_lin[0.05] - p_b2["tol_5"]) < 1e-9
          and abs(p_lin[0.02] - p_b2["tol_2"]) < 1e-9,
          "P10=%.3f P5=%.3f P2=%.3f (b2.json: %.2f/%.2f/%.2f)"
          % (p_lin[0.10], p_lin[0.05], p_lin[0.02],
             p_b2["tol_10"], p_b2["tol_5"], p_b2["tol_2"]))

    # D2: nonideal joint criterion (pole AND gain), TT nonideal reference
    p_non = {t: 0.0 for t in tols}
    p_cal = {t: 0.0 for t in tols}
    p_target = {t: 0.0 for t in tols}   # vs the LINEAR design target gm/C
    for k in range(n_mc):
        inst = sys2.instance(k, nonideal=True)
        pole_k = inst["a1"]
        G_k = sys2.dc_gain(inst)
        for tol in tols:
            ok_pole = abs(pole_k / pole_tt - 1.0) <= tol
            ok_gain = abs(G_k / G_tt - 1.0) <= tol
            p_non[tol] += ok_pole and ok_gain
            # foreground cal: DC gain corrected to the 100 ppm floor
            cal = calibrate_foreground([G_k], [G_tt],
                                       meas_noise_ppm=MEAS_NOISE_PPM)
            gain_res = cal.residual_ppm / 1e6
            p_cal[tol] += ok_pole and (gain_res <= tol)
            # vs the linear design target gm/C (pole axis only)
            p_target[tol] += abs(pole_k / (tt["gm_mos_fd"] / C_BANK_F)
                                 - 1.0) <= tol
    p_non = {t: p_non[t] / n_mc for t in tols}
    p_cal = {t: p_cal[t] / n_mc for t in tols}
    p_target = {t: p_target[t] / n_mc for t in tols}

    check("D2: nonideal joint P(10/5/2%%) = %.2f/%.2f/%.2f — strictly "
          "below the linear numbers (mirror axis + leak spread enter)"
          % (p_non[0.10], p_non[0.05], p_non[0.02]),
          p_non[0.10] <= p_lin[0.10] and p_non[0.05] < p_lin[0.05]
          and p_non[0.02] < p_lin[0.02],
          "linear %.2f/%.2f/%.2f -> nonideal %.2f/%.2f/%.2f"
          % (p_lin[0.10], p_lin[0.05], p_lin[0.02],
             p_non[0.10], p_non[0.05], p_non[0.02]))

    p_pole = {t: sum(1 for k in range(n_mc)
                     if abs(sys2.instance(k, True)["a1"] / pole_tt
                            - 1.0) <= t) / n_mc for t in tols}
    check("D3: foreground gain cal recovers the gain axis (residual = "
          "100 ppm floor) but NOT the pole: P_cal == pole-only P",
          all(abs(p_cal[t] - p_pole[t]) < 1e-9 for t in tols)
          and p_cal[0.10] == 1.0 and p_cal[0.05] == 1.0,
          "P_cal %.2f/%.2f/%.2f (gain floor %.0f ppm; pole spread "
          "lambda*sigma_Id/gm = %.2f%%)" % (
              p_cal[0.10], p_cal[0.05], p_cal[0.02], MEAS_NOISE_PPM,
              sys2.lam_tt * mc["id_mos_pole"]["sigma"] / tt["gm_mos_fd"] * 100))

    check("D4: judged against the LINEAR design target gm/C, the leak "
          "empties the 5%% and 2%% windows (P5=P2=0) and halves P10",
          p_target[0.05] == 0.0 and p_target[0.02] == 0.0
          and 0.3 <= p_target[0.10] <= 0.7,
          "P vs gm/C target: %.2f/%.2f/%.2f (systematic %+.1f%%) — "
          "target_gm_over_c must be defined on (gm+g_out)/C or the cal "
          "loop must trim gm code" % (
              p_target[0.10], p_target[0.05], p_target[0.02],
              leak_pct * 100))

    # D5: axis attribution recorded for the report
    attrs = dict(
        systematic_leak_pct=leak_pct * 100.0,
        sigma_mirror_pct=mc["mirror_ratio"]["sigma"] * 100.0,
        sigma_gm_pct=mc["gm_mos_fd"]["sigma"] / tt["gm_mos_fd"] * 100.0,
        sigma_Id_pct=mc["id_mos_pole"]["sigma"] / tt["id_mos_pole"] * 100.0,
        sigma_pole_nonideal_pct=float(
            np.std([sys2.instance(k, True)["a1"] for k in range(n_mc)])
            / pole_tt * 100.0),
        vos_sigma_mV=mc["vd0_offset_raw"]["sigma"] * 1e3,
    )
    check("D5: axis attribution recorded (mirror %.1f%% > Id %.1f%% > "
          "gm %.1f%%; systematic leak %+.1f%% dominates the mean)"
          % (attrs["sigma_mirror_pct"], attrs["sigma_Id_pct"],
             attrs["sigma_gm_pct"], attrs["systematic_leak_pct"]),
          attrs["sigma_mirror_pct"] > attrs["sigma_Id_pct"]
          > attrs["sigma_gm_pct"] > 0.0,
          json.dumps({k2: round(v, 3) for k2, v in attrs.items()}))

    # =====================================================================
    # E. noise through the system
    # =====================================================================
    print("\n--- E. HBT noise through the 2-integrator system ---")
    # stage1 input-referred HBT noise propagates: H(s) = [gm1/(sC+G1)] *
    # [k12/(sC+G2)] from e_n to x2.  Both densities flat -> the white
    # factor must land as a pure variance scale.
    f_grid = np.linspace(0.0, 50 * inst_tt["a2"] / (2 * math.pi), 400001)
    w = 2 * math.pi * f_grid
    H = (inst_tt["b1"] / (1j * w + inst_tt["a1"])) * \
        (inst_tt["k12"] / (1j * w + inst_tt["a2"]))
    var_num = float(np.trapezoid(np.abs(H) ** 2, f_grid))   # one-sided f>=0
    sig_out_hbt = n_hbt.density_v_rthz * math.sqrt(var_num)
    sig_out_ideal = n_hbt.density_v_rthz / math.sqrt(n_hbt.factor) \
        * math.sqrt(var_num)
    ratio_out = (sig_out_hbt / sig_out_ideal) ** 2
    snr_cost = 10 * math.log10(n_hbt.factor)
    check("E1: white factor lands in the output as a pure variance "
          "scale for flat in-band spectra",
          abs(ratio_out / n_hbt.factor - 1.0) < 1e-9,
          "var ratio %.6f vs factor %.4f (sigma_out=%.3f mV, swing 0.5 V)"
          % (ratio_out, n_hbt.factor, sig_out_hbt * 1e3))
    check("E2: SNR optimism cost %.2f dB matches the recorded 2.8 dB "
          "(risk-3 verdict intact)" % snr_cost,
          abs(snr_cost - 2.8) < 0.1,
          "10*log10(1.897) = %.2f dB" % snr_cost)
    check("E3: MOS path noise stays derived-labeled through the "
          "attachment (no measured MOS/TG gamma exists)",
          n_mos.source == "derived-not-measured",
          "MOS density %.3f nV/sqrt(Hz) is 4kT*gamma/gm only"
          % (n_mos.density_v_rthz * 1e9))

    # =====================================================================
    # F. netlist attachment (backward compatibility + JSON round-trip)
    # =====================================================================
    print("\n--- F. BehavioralNetlist nonideal attachment ---")
    # toy netlist built directly from dataclasses
    toy = BehavioralNetlist(
        name="toy2", alpha=1.0,
        integrators=[BehavioralIntegrator(
            name="x1", pole=-inst_tt["a1"],
            target_gm_over_c=abs(inst_tt["a1"]), C_bank=1.0,
            C_code=0, gm_code=0, s_x=1.0, alpha=1.0, noise_sigma_scale=1.0,
            cal_method="gm_c"),
            BehavioralIntegrator(
            name="x2", pole=-inst_tt["a2"],
            target_gm_over_c=abs(inst_tt["a2"]), C_bank=1.0,
            C_code=0, gm_code=0, s_x=1.0, alpha=1.0, noise_sigma_scale=1.0,
            cal_method="gm_c")],
        edges=[BehavioralEdge(src="x1", dst="x2", gain=inst_tt["k12"],
                              tile="mvm-tile", route_contract=None)],
        ports=[BehavioralPort(input=0, dst="x1", gain=inst_tt["b1"])],
        taps=[BehavioralTap(tap=0, src="x2", gain=1.0)])
    A_before = toy.effective_A()
    nmap = b2_nonideal_map(toy, ext, device="mos")
    attach_nonidealities(toy, nmap)
    check("F1: attach_nonidealities fills every integrator, keys valid, "
          "effective_A unchanged",
          set(nmap) == {"x1", "x2"}
          and np.array_equal(A_before, toy.effective_A()),
          "nonideal keys %s; ro=%.1f kOhm, leak=%.2e rad/s attached"
          % (sorted(toy.nonideal), nmap["x1"]["ro_ohm"] / 1e3,
             nmap["x1"]["leak_rad_s"]))
    js = json.dumps(toy.as_dict())
    back = json.loads(js)
    check("F2: netlist as_dict JSON round-trip carries the nonideal "
          "block", "nonideal" in back
          and back["nonideal"]["x1"]["device"] == "mos"
          and back["nonideal"]["x2"]["noise_source"] == "derived-not-measured",
          "json bytes=%d, x1.device=%s, x2.noise_source=%s"
          % (len(js), back["nonideal"]["x1"]["device"],
             back["nonideal"]["x2"]["noise_source"]))
    try:
        attach_nonidealities(toy, {"x1": {}})
        dup_ok = False
    except ValueError:
        dup_ok = True
    try:
        attach_nonidealities(BehavioralNetlist(
            name="t", alpha=1.0, integrators=[], edges=[], ports=[],
            taps=[]), {"nope": {}})
        unk_ok = False
    except ValueError:
        unk_ok = True
    check("F3: double-attach and unknown-element attach raise "
          "(no silent overwrite/drop)", dup_ok and unk_ok, "both raise")

    # real generated config through the same path (diag(-1,-100))
    from analysis.descriptor import Descriptor
    from lowering.budget import BudgetInput, compile_budget
    from lowering.discretize import PDESpec, discretize
    from lowering.p7 import generate_config
    from lowering.scaling import gramian_scaling
    from analysis.slack import HardwareSpec
    desc = Descriptor(np.eye(2), np.diag([-1.0, -100.0]),
                      np.array([[1.0], [1.0]]), np.array([[1.0, 0.0]]))
    sc = gramian_scaling(desc, v_swing=1.0, B_hw=1e6)
    cert = compile_budget(BudgetInput(
        sys=desc, spec=HardwareSpec(B_hw=1e6, sigma_req=30.0), scaling=sc))
    cfg = generate_config(desc, cert, sc)
    net_real = to_behavioral(cfg, name="diag2-b2")
    nmap_real = b2_nonideal_map(net_real, ext)
    attach_nonidealities(net_real, nmap_real)
    ok_real = (set(nmap_real) == {i.name for i in net_real.integrators}
               and all(v["ro_ohm"] > 0 for v in nmap_real.values()))
    check("F4: real generated config (diag(-1,-100)) accepts the b2 "
          "attachment; ideal default preserved when not attached",
          ok_real and BehavioralNetlist(
              name="ideal", alpha=1.0, integrators=[], edges=[], ports=[],
              taps=[]).nonideal == {},
          "attached %d integrators; mos ro=%.1f kOhm at the TT reference "
          "bias" % (len(nmap_real), fits["tt"]["ro_ohm"] / 1e3))

    # =====================================================================
    n_pass = sum(c["passed"] for c in checks)
    summary = dict(
        n_checks=len(checks), n_pass=n_pass, n_fail=len(fails),
        corner_fits={t: {k2: (float(v) if np.isscalar(v) and np.isfinite(v)
                              or isinstance(v, (int, float, str))
                              else v)
                         for k2, v in fits[t].items()}
                     for t in ("tt", "ss", "ff")},
        measured=dict(
            gm_mos2_tt=tt["gm_mos2_param"], gm_fd_tt=tt["gm_mos_fd"],
            g_dev_tt_S=sys2.g_dev_tt, ro_mos_tt_ohm=1.0 / sys2.g_dev_tt,
            lambda_fit_tt=sys2.lam_tt,
            ro_hbt_ohm=ro_hbt, A0_hbt=st_hbt.A0,
            A0_ac={t: ext["corners"][t]["_a0_ac"] for t in ("tt", "ss", "ff")},
            pole_meas_hz={t: fits[t]["pole_meas_hz"] for t in ("tt", "ss", "ff")},
            pole_pred_hz={t: fits[t]["pole_pred_hz"] for t in ("tt", "ss", "ff")},
            pole_fit_delta_pct={t: deltas[t] * 100 for t in ("tt", "ss", "ff")},
            diff_slope_model=slope_model, diff_slope_meas=slope_meas,
            noise_hbt=n_hbt.as_dict(), noise_mos=n_mos.as_dict()),
        system=dict(pole_tt_rad_s=pole_tt, dc_gain_tt=G_tt,
                    leak_pct=leak_pct * 100,
                    ideal_model_dc_gain=inst_tt["r"]),
        p_workload=dict(
            linear_only={("tol_%d" % int(t * 100)): p_lin[t] for t in tols},
            nonideal_joint={("tol_%d" % int(t * 100)): p_non[t] for t in tols},
            nonideal_cal={("tol_%d" % int(t * 100)): p_cal[t] for t in tols},
            nonideal_vs_linear_target={
                ("tol_%d" % int(t * 100)): p_target[t] for t in tols},
            b2_json_reference=b2["p_workload"], n_mc=n_mc),
        axis_attribution=attrs,
        validity_limits=[
            "1-pole small-signal model: valid where the measured AC "
            "magnitude is 1-pole (log-mag rms < 1e-3, checked)",
            "ro/lambda_fit are corner- and bias-dependent (lambda_fit "
            "0.66-2.32 /V across ss/tt/ff); the ledger 0.970/V is a "
            "point value; ro is NOT rescaled with per-instance gm",
            "diff-pair tanh law ignores VBIC Early/high-injection "
            "(measured-vs-model slope residual ~%.1f%%)" % (abs(d5) * 100),
            "v_swing (0.5 V), TG c_par (30 fF) and q_inj_fc (0.5) are "
            "declared model parameters, NOT measurements",
            "MOS/TG noise: derived 4kT*gamma/gm only — PSP103 OSDI "
            "device noise unimplemented in ngspice-44.2",
        ],
        notes=list(ext.get("notes", [])),
    )
    with open(OUT + "/b2_behavioral.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\n%d/%d checks passed, %d failed -> results/b2_behavioral.json"
          % (n_pass, len(checks), len(fails)))
    if fails:
        print("FAILURES: " + "; ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
