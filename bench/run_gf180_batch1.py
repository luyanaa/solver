"""G3 acceptance -- gf180mcu batch-1 testbench three-way reconciliation.

Every tile of the batch-1 characterization vehicle is simulated from a checked
in netlist under ``hw/gf180/batch1/`` and reconciled three ways:

    (A) SPICE   the netlist as built (real PDK devices, real MIM, real metal)
    (B) macro   the composition of primitive-bank numbers from
                ``results/gf180_extract.json`` (twin/macromodel.py gf180 lane)
    (C) design  the closed-form / spec number the compiler used (DESIGN 6.25,
                6.15), i.e. what the plan promised before silicon

The policies this file enforces, and why:

* A check is only a check if it can fail. Every row prints all three columns.
* Where B and C are the same number by construction the row is labelled
  ``wiring`` (it tests the composition, not new physics).
* Where the model is the thing under test (noise-vs-area scaling, ladder
  parasitics) the row reports the measured factor and gates only the direction
  that must hold (parasitics may slow a ladder, never speed it).
* Netlist parsing lesson kept in the code: ngspice ``wrdata`` writes a
  (scale, value) pair *per vector*, so two real vectors give FOUR columns
  (dc/tran) while one complex vector gives three. Do not assume 3 everywhere.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import twin.spice.extract_gf180 as G  # noqa: E402  (PDK paths + ngspice runner)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(ROOT, "hw", "gf180", "batch1")
OUT = G.OUT
EXTRACT = os.path.join(ROOT, "results", "gf180_extract.json")

FOUR_KT = 4.0 * 1.380649e-23 * 300.0
PHASE_BUDGET_1DEG_S = 27.8e-12      # DESIGN 6.15: 100 MHz, 1 degree
VDD = 3.3
# magic tech "Nominal capacitances" (gf180mcuA.tech, from the foundry doc
# PDS_035_03): metal3 areacap 10.094 aF/um^2, perimeter 30.021 aF/um.
# ----- gf180mcuD stack (magic tech "Nominal capacitances", typical corner) ----
# A has metal1-3 only; D has metal1-5 and is the wafer.space target. Its MIM
# lives on metal4..metal5, so the m4m5 device every batch-1 netlist instantiates
# is legal ONLY in B/C/D. The ngspice kits of A and D are byte-identical, so the
# primitive bank transfers; only these back-end numbers are variant-specific.
M1_AREACAP_AF_UM2, M1_PERIMCAP_AF_UM, M1_RSH_OHM_SQ = 29.304, 39.431, 0.090
M2_AREACAP_AF_UM2, M2_PERIMCAP_AF_UM, M2_RSH_OHM_SQ = 15.016, 33.298, 0.090
M3_AREACAP_AF_UM2, M3_PERIMCAP_AF_UM, M3_RSH_OHM_SQ = 10.094, 30.021, 0.090
M4_AREACAP_AF_UM2, M4_PERIMCAP_AF_UM, M4_RSH_OHM_SQ = 7.602, 28.153, 0.090
M5_AREACAP_AF_UM2, M5_PERIMCAP_AF_UM, M5_RSH_OHM_SQ = 5.798, 30.386, 0.040


def wire_rc(area_af, peri_af, rsh, l_um, w_um):
    """Distributed-wire R and C for a straight l x w rectangle (magic-tech C)."""
    c = (area_af * l_um * w_um + peri_af * 2 * (l_um + w_um)) * 1e-18
    return rsh * l_um / w_um, c
PAR_VTH_UM = 7.148e-3               # PDK fets_mm par_vth (V * um)
CHECKS = []


def chk(name, ok, detail):
    CHECKS.append(dict(name=name, ok=bool(ok), detail=detail))
    print(("PASS  " if ok else "FAIL  ") + name + "  " + detail)
    return ok


def _load(name):
    with open(os.path.join(HW, name)) as fh:
        return fh.read()


def run(tag, text, subs=None):
    txt = text.replace("{NGS}", G.NGS).replace("{OUT}", OUT)
    txt = txt.replace("{vdd}", str(VDD)).replace("(%g-v(dac))",
                                                "(%s-v(dac))" % VDD)
    for k, v in (subs or {}).items():
        txt = txt.replace(k, str(v))
    return G.run("b1_%s" % tag, txt)


def ac(path, iv=0):
    """Complex sweep. wrdata writes (scale, re, im) PER vector, so a second
    complex vector starts at column 3, not column 4."""
    d = np.loadtxt(path)
    j = 3 * iv
    return d[:, j], d[:, j + 1] + 1j * d[:, j + 2]


def f3db(f, y):
    """-3 dB frequency of |y| by log-log interpolation on the crossing."""
    a = np.abs(y)
    target = a[0] / np.sqrt(2.0)
    i = int(np.argmax(a < target))
    if i == 0:
        return float(f[0])
    x = (np.log(target) - np.log(a[i - 1])) / (np.log(a[i]) - np.log(a[i - 1]))
    return float(np.exp(np.log(f[i - 1]) + x * (np.log(f[i]) - np.log(f[i - 1]))))


def tail_fit(t, v, vinf, lo, hi):
    w = (t > lo) & (t < hi) & (vinf - v > 0)
    p = np.polyfit(t[w], np.log(vinf[w] - v[w]), 1)
    resid = float(np.std(np.log(vinf[w] - v[w]) - np.polyval(p, t[w])))
    return -1.0 / p[0], resid, int(w.sum())


def op_row(stdout, key, skip=0):
    """Values of an ``echo KEY v1 v2 ...`` line (KEY itself is token 0)."""
    for line in stdout.splitlines():
        if line.startswith(key):
            return [float(x) for x in line.split()[1 + skip:]]
    raise RuntimeError("no %s row in ngspice output" % key)


def main():
    ex = json.load(open(EXTRACT))
    c3 = ex["corners"]["3v3_tt"]
    gm, ro, cmim = c3["gm_fd"], c3["ro_ohm"], c3["c_mim_f"]
    lam_slope, lam_pole = c3["lambda_slope_vds"], c3["lambda_ohm_inv"]
    ron_tg, id_bias, vdiode = c3["ron_tg"], c3["id_a"], c3["vdiode"]
    noise3 = ex["noise"]["nmos"]["3v3_tt"]
    out = dict(design="gf180mcu batch-1 (G3)",
               note="three-way: SPICE / primitive-bank composition / design closed form")

    # =====================================================================
    # T1  integrator tile
    # =====================================================================
    t1 = run("t1", _load("t1_integrator.sp"))
    opv = op_row(t1, "T1OP")
    f, y = ac(os.path.join(OUT, "b1_t1_ac.dat"))
    a0_sp, fp_sp, vout = abs(y[0]), f3db(f, y), opv[0]
    a0_macro, fp_macro = gm * ro, 1.0 / (2 * np.pi * ro * cmim)
    ro_nom = 1.0 / (lam_slope * id_bias)      # design column: slope lambda
    a0_des, fp_des = gm * ro_nom, 1.0 / (2 * np.pi * ro_nom * 1.0e-12)
    print("\n--- T1 integrator tile (gm cell + one 1 pF MIM C_bank) ---")
    print("  column        a0        pole")
    print("  SPICE      %8.2f  %9.0f Hz" % (a0_sp, fp_sp))
    print("  macro      %8.2f  %9.0f Hz   (gm*ro, 1/(2*pi*ro*C_mim))" % (a0_macro, fp_macro))
    print("  design     %8.2f  %9.0f Hz   (slope lambda, C = 1.000 pF)" % (a0_des, fp_des))
    chk("T1a DC gain within 20 % of the primitive-bank composition",
        abs(a0_sp / a0_macro - 1) < 0.20,
        "SPICE %.2f vs macro %.2f (macro-SPICE %.1f %%)"
        % (a0_sp, a0_macro, 100 * (a0_macro / a0_sp - 1)))
    chk("T1b pole within 20 % of 1/(2*pi*ro*C_mim)",
        abs(fp_sp / fp_macro - 1) < 0.20,
        "SPICE %.1f kHz vs macro %.1f kHz (%.1f %%)"
        % (fp_sp / 1e3, fp_macro / 1e3, 100 * (fp_macro / fp_sp - 1)))
    chk("T1c pole also within 20 % of the design column (the two lambdas agree here)",
        abs(fp_sp / fp_des - 1) < 0.20,
        "design %.1f kHz; lambda_slope %.5f vs lambda_pole %.5f -> %.1f %% ro split"
        % (fp_des / 1e3, lam_slope, lam_pole,
           100 * (ro / ro_nom - 1)))
    chk("T1d integrating node stays in saturation", vout > 0.3,
        "v(out) = %.3f V = the diode bias the extraction used (%.3f V)"
        % (vout, vdiode))
    out["t1"] = dict(a0_spice=a0_sp, fp_spice_hz=fp_sp, a0_macro=a0_macro,
                     fp_macro_hz=fp_macro, a0_design=a0_des, fp_design_hz=fp_des,
                     v_out=vout)

    # =====================================================================
    # T5  16-cell flux chain
    # =====================================================================
    t5 = run("t5", _load("t5_flux_chain16.sp"))
    r_chord = op_row(t5, "T5OP")[3]
    f5, y5 = ac(os.path.join(OUT, "b1_t5_ac.dat"))
    dc_gain, fp_lad = abs(y5[0]), f3db(f5, y5)
    fcal, ycal = ac(os.path.join(OUT, "b1_t5_cal_ac.dat"))
    g_ss = 2 * np.pi * f3db(fcal, ycal) * (2 * cmim)       # two 1 pF nodes
    tr = np.loadtxt(os.path.join(OUT, "b1_t5_tr.dat"))
    tau1, resid, npts = tail_fit(tr[:, 0], tr[:, 1], np.full(tr.shape[0], 1.75),
                                 60e-9, 250e-9)
    lam1 = 2 * (1 - np.cos(np.pi / 31))      # 15 free nodes, node 1 held by the source
    tau_pred = cmim / (g_ss * lam1)
    print("\n--- T5 16-cell flux chain (triode pass-transistor flux elements) ---")
    print("  DC gain %.4f   -3 dB %.0f kHz" % (dc_gain, fp_lad / 1e3))
    print("  link: chord %.0f ohm (0.2 V probe)   small-signal %.0f ohm (2-node pole %.1f MHz)"
          % (r_chord, 1 / g_ss, f3db(fcal, ycal) / 1e6))
    print("  mode-1 tau: measured %.1f ns   composition %.1f ns   tax x%.2f"
          % (tau1 * 1e9, tau_pred * 1e9, tau1 / tau_pred))
    chk("T5a ladder DC gain is 1 (a conservative flux chain has no DC loss)",
        abs(dc_gain - 1) < 0.01, "|H| at the lowest frequency = %.4f" % dc_gain)
    chk("T5b small-signal link conductance exceeds the 0.2 V chord (triode law)",
        1 / g_ss < r_chord,
        "R_ss %.0f ohm < R_chord %.0f ohm; the triode law gives R_chord/R_ss = "
        "1 + vds/(2*vov) = %.2f" % (1 / g_ss, r_chord, r_chord * g_ss))
    chk("T5c parasitics may only slow the ladder (tax recorded, not ratio-gated)",
        tau1 / tau_pred >= 1.0,
        "measured/composition = %.2f (fit residual %.3f over %d pts); the "
        "composition has no per-node link capacitance, so this is the batch-1 "
        "parasitic tax and a C1 measurement item" % (tau1 / tau_pred, resid, npts))
    out["t5"] = dict(dc_gain=dc_gain, f3db_hz=fp_lad, r_chord_ohm=r_chord,
                     g_ss_s=g_ss, r_ss_ohm=1 / g_ss, tau1_s=tau1,
                     tau_pred_s=tau_pred, tax=tau1 / tau_pred, lam1=lam1,
                     tail_fit_resid=resid, tail_fit_points=npts)

    # =====================================================================
    # T7  route tester
    # =====================================================================
    t7 = run("t7", _load("t7_route_tester.sp"))
    r1, r2, vw, vz, vwb, vzb = op_row(t7, "T7OP")
    rsh = r1 / (1000e-6 / 2e-6)
    f7, yz = ac(os.path.join(OUT, "b1_t7_ac.dat"), 0)
    _, yw = ac(os.path.join(OUT, "b1_t7_ac.dat"), 1)
    _, yzb = ac(os.path.join(OUT, "b1_t7_ac.dat"), 2)
    _, ywb = ac(os.path.join(OUT, "b1_t7_ac.dat"), 3)
    fp_z, fp_zb = f3db(f7, yz), f3db(f7, yzb)
    c_wire = wire_rc(M3_AREACAP_AF_UM2, M3_PERIMCAP_AF_UM, M3_RSH_OHM_SQ,
                     1000.0, 2.0)[1]
    t50_wire = 0.3795 * r1 * c_wire
    # the ladder path shares the SAME driver and load, so its pole should be the
    # lumped one scaled by C_total/C_old: (C_mim + C_wire)/C_mim.
    fp_pred = fp_z * cmim / (cmim + c_wire)
    stack = {}
    for nm, a, p_, rs in (("M1", M1_AREACAP_AF_UM2, M1_PERIMCAP_AF_UM, M1_RSH_OHM_SQ),
                          ("M2", M2_AREACAP_AF_UM2, M2_PERIMCAP_AF_UM, M2_RSH_OHM_SQ),
                          ("M3", M3_AREACAP_AF_UM2, M3_PERIMCAP_AF_UM, M3_RSH_OHM_SQ),
                          ("M4", M4_AREACAP_AF_UM2, M4_PERIMCAP_AF_UM, M4_RSH_OHM_SQ),
                          ("M5", M5_AREACAP_AF_UM2, M5_PERIMCAP_AF_UM, M5_RSH_OHM_SQ)):
        r, c = wire_rc(a, p_, rs, 1000.0, 2.0)
        stack[nm] = dict(rsh_ohm_sq=rs, r_ohm=r, c_f=c,
                         t50_s=0.3795 * r * c)
    print("\n--- T7 route tester (gf180mcuD stack; M3 wire through rm3) ---")
    print("  per-layer 1 mm x 2 um route (magic-tech C, SPICE rsh):")
    for nm in ("M1", "M2", "M3", "M4", "M5"):
        print("    %s  rsh %.2f  R %6.1f ohm  C %5.1f fF  t50 %5.2f ps"
              % (nm, stack[nm]["rsh_ohm_sq"], stack[nm]["r_ohm"],
                 stack[nm]["c_f"] * 1e15, stack[nm]["t50_s"] * 1e12))
    print("  sheet R %.4f ohm/sq vs rsh_rm3 %.3f (%.1f %%)"
          % (rsh, M3_RSH_OHM_SQ, 100 * (rsh / M3_RSH_OHM_SQ - 1)))
    print("  lumped-R path: %.1f kHz (no wire %.1f kHz); 10-section pi-RC ladder:"
          " %.1f kHz (C-ratio prediction %.1f kHz)"
          % (fp_z / 1e3, fp_sp / 1e3, fp_zb / 1e3, fp_pred / 1e3))
    chk("T7a metal sheet resistance matches the PDK resistor model (<5 %)",
        abs(rsh / M3_RSH_OHM_SQ - 1) < 0.05,
        "rm3 %.4f ohm/sq; the 100 um wire independently gives %.4f"
        % (rsh, r2 / (100e-6 / 2e-6)))
    chk("T7b a 1 mm M3 route is invisible behind a 1 pF bank (<2 % pole shift)",
        abs(fp_z / fp_sp - 1) < 0.02,
        "%.1f kHz with the wire vs %.1f kHz without (%.2f %%)"
        % (fp_z / 1e3, fp_sp / 1e3, 100 * (fp_z / fp_sp - 1)))
    chk("T7c the wire's own RC delay is under 10 % of the 1-degree phase budget",
        t50_wire < 0.10 * PHASE_BUDGET_1DEG_S,
        "t50 %.2f ps vs %.1f ps at 100 MHz (%.1f %%)"
        % (t50_wire * 1e12, PHASE_BUDGET_1DEG_S * 1e12,
           100 * t50_wire / PHASE_BUDGET_1DEG_S))
    chk("T7d the distributed ladder lands on the C-ratio prediction (<8 %)",
        abs(fp_zb / fp_pred - 1) < 0.08,
        "%.1f kHz vs %.1f kHz (%.1f %%); the wire's 45 ohm is %.4f %% of the "
        "2.48 Mohm driver, so only its shunt C can move the pole"
        % (fp_zb / 1e3, fp_pred / 1e3, 100 * (fp_zb / fp_pred - 1),
           100 * r1 / ro))
    chk("T7e shunt wire C is NOT invisible once it is actually in the netlist",
        0.05 < (1 - fp_zb / fp_z) < 0.15,
        "ladder vs lumped-R: %.2f %% (predicted %.2f %%); T7b only ever proved "
        "the SERIES R negligible, because the PDK kit has no wire-C model"
        % (100 * (1 - fp_zb / fp_z), 100 * (1 - cmim / (cmim + c_wire))))
    out["t7"] = dict(rsh_ohm_sq=rsh, r_1000um_ohm=r1, r_100um_ohm=r2,
                     c_wire_f=c_wire, t50_wire_s=t50_wire,
                     fp_with_wire_hz=fp_z, fp_without_hz=fp_sp,
                     fp_ladder_hz=fp_zb, fp_ladder_pred_hz=fp_pred,
                     stack=stack,
                     note="gf180mcuD stack: the ngspice kit models rm3 as a series "
                          "resistor only; wire C comes from the magic-tech nominal "
                          "table and is modelled here as a 10-section pi ladder. "
                          "T7b's 'invisible' claim therefore covers the series R, "
                          "not the shunt C (T7e). metal3 rsh: SPICE 0.09 vs magic "
                          "'resist' 90 (A's tech also prints a 10 entry from a "
                          "second table; D prints 90)")

    # =====================================================================
    # T8  noise bank
    # =====================================================================
    t8 = run("t8", _load("t8_noise_bank.sp"))
    rows = {}
    for line in t8.splitlines():
        if line.startswith("NROW"):
            p = line.split()
            rows[p[1]] = (float(p[2]), float(p[3]))
    b0, b1m, b2 = 1e6, 1e7, 1e8
    s1 = rows["a_low"][0] ** 2 / (b1m - b0)
    s2 = rows["a"][0] ** 2 / (b2 - b0)
    c1, c2 = np.log(b1m / b0) / (b1m - b0), np.log(b2 / b0) / (b2 - b0)
    bl = (s1 - s2) / (c1 - c2)
    A = max(s1 - bl * c1, 0.0)
    rout = 100e3 * ro / (100e3 + ro)
    a0_cell = gm * rout
    si = max((A - FOUR_KT * rout) / a0_cell ** 2, 0.0)
    dens = 1e9 * np.sqrt(si)
    mn = [rows["m%d" % k][1] / np.sqrt(99e6) * 1e9 for k in (1, 2, 4, 8)]
    kt_meas, kt_pred = rows["kt"][0], np.sqrt(4.14e-21 / cmim)
    big = rows["dbig"][1] / np.sqrt(2.5e6) * 1e9
    print("\n--- T8 noise bank ---")
    print("  replica cell plateau %.2f nV/rtHz (primitive bank %.2f, gamma %.3f)"
          % (dens, noise3["density_in_nv_rthz"], si * gm / FOUR_KT))
    print("  bank m=1/2/4/8: %s nV/rtHz   area x8 -> x%.2f (ideal sqrt(8) = 2.83)"
          % (["%.2f" % x for x in mn], mn[0] / mn[-1]))
    print("  kT/C %.2f uV measured vs %.2f uV theory" % (kt_meas * 1e6, kt_pred * 1e6))
    print("  W=100u at 100 uA clean cell: %.2f nV/rtHz (4kT*gm with gm=1.40 mS "
          "would be %.2f nV)" % (big, 1e9 * np.sqrt(FOUR_KT / 1.40e-3)))
    chk("T8a kT/C anchor: total integrated noise = sqrt(kT/C) within 5 %",
        abs(kt_meas / kt_pred - 1) < 0.05,
        "%.2f uV vs %.2f uV (%.1f %%)"
        % (kt_meas * 1e6, kt_pred * 1e6, 100 * (kt_meas / kt_pred - 1)))
    chk("T8b bank size reduces the input-referred density monotonically",
        all(mn[i] > mn[i + 1] for i in range(3)),
        ", ".join("%.2f" % x for x in mn) + " nV/rtHz at m = 1/2/4/8")
    chk("T8c replica cell reproduces the primitive bank's own band total (<25 %)",
        abs(rows["a"][0] / noise3["onoise_total_v"] - 1) < 0.25,
        "batch-1 %.4f mV vs extractor %.4f mV (%.1f %%)"
        % (rows["a"][0] * 1e3, noise3["onoise_total_v"] * 1e3,
           100 * (rows["a"][0] / noise3["onoise_total_v"] - 1)))
    out["t8"] = dict(replica_density_nv=dens,
                     primitive_density_nv=noise3["density_in_nv_rthz"],
                     gamma_from_replica=si * gm / FOUR_KT, bank_nv_1_2_4_8=mn,
                     area8_factor=mn[0] / mn[-1], kt_measured_v=kt_meas,
                     kt_theory_v=kt_pred, big_cell_nv=big,
                     open_items=[
                         "area x8 gives x%.2f, not sqrt(8) = 2.83: ngspice scales "
                         "m= and W identically (checked), so the PDK's white-noise "
                         "scaling vs area is what does not reproduce. C1 item."
                         % (mn[0] / mn[-1]),
                         "the replica's A+B/f plateau fit is ill-conditioned: that "
                         "cell's flicker corner (%.2f MHz) sits on its pole, so "
                         "the primitive bank's gamma_eff should be read as a fit, "
                         "not a bound" % (noise3["flicker_corner_hz"] / 1e6)])

    # =====================================================================
    # T9  5-bit current DAC
    # =====================================================================
    dac_src = _load("t9_dac5.sp")
    cur = []
    for code in (0, 1, 2, 4, 8, 16, 31):
        subs = {"{TAG}": str(code)}
        for k in range(5):
            subs["{B%d}" % k] = (code >> k) & 1
        vd = op_row(run("t9_%02d" % code, dac_src, subs), "DCODE", skip=1)[0]
        cur.append((VDD - vd) / 1e3)
    lsb, fs = cur[1] - cur[0], cur[-1] - cur[0]
    vds_lsb, vds_fs = VDD - lsb * 1e3, VDD - fs * 1e3
    lsb_pred = id_bias * (1 + lam_pole * (vds_lsb - vdiode))
    inl_rel = fs / (31 * lsb) - 1
    inl_pred = ((1 + lam_pole * (vds_fs - vdiode))
                / (1 + lam_pole * (vds_lsb - vdiode)) - 1)
    print("\n--- T9 5-bit binary current DAC (1 k load, lambda*DVds live) ---")
    print("  LSB %.3f uA (bank unit %.3f uA, lambda-corrected %.3f uA)"
          % (lsb * 1e6, id_bias * 1e6, lsb_pred * 1e6))
    print("  full scale %.2f uA = %.2f LSB;  INL %.2f %% (lambda predicts %.2f %%)"
          % (fs * 1e6, fs / lsb, 100 * inl_rel, 100 * inl_pred))
    chk("T9a LSB within 20 % of the lambda-corrected bank unit current",
        abs(lsb / lsb_pred - 1) < 0.20,
        "%.3f uA vs %.3f uA (%.1f %%)"
        % (lsb * 1e6, lsb_pred * 1e6, 100 * (lsb / lsb_pred - 1)))
    chk("T9b the INL is the lambda mechanism, within a factor of two",
        0.5 < abs(inl_rel / inl_pred) < 2.0,
        "measured %.2f %% vs lambda %.2f %% (x%.2f)"
        % (100 * inl_rel, 100 * inl_pred, inl_rel / inl_pred))
    out["t9"] = dict(lsb_a=lsb, lsb_pred_a=lsb_pred, full_scale_a=fs,
                     inl_rel=inl_rel, inl_pred=inl_pred, vds_lsb=vds_lsb,
                     vds_fs=vds_fs)

    # =====================================================================
    # T10 comparator core
    # =====================================================================
    comp = _load("t10_comparator.sp")
    ON = ("XM3 n1 n2 tail 0 nfet_03v3 w=10u l=1u\n"
          "XM4 n2 n1 tail 0 nfet_03v3 w=10u l=1u")
    OFF = ("XM3 n1 gz tail 0 nfet_03v3 w=10u l=1u\n"
           "XM4 n2 gz tail 0 nfet_03v3 w=10u l=1u")

    def trip(tag):
        d = np.loadtxt(os.path.join(OUT, "b1_cmp_dc_%s.dat" % tag))
        s, vd = d[:, 0], d[:, 1] - d[:, 3]
        z = np.where(np.diff(np.sign(vd)) != 0)[0]
        if not len(z):
            return None
        i = z[0]
        return float(s[i] + (s[i + 1] - s[i]) * (-vd[i]) / (vd[i + 1] - vd[i]))

    run("cmp_pair0", comp, {"{TAG}": "pair0", "{MM}": 0, "{SEED}": 0,
                            "{REGEN_BLOCK}": OFF})
    off0 = trip("pair0") * 1e3
    draws = []
    for k in range(1, 5):
        run("cmp_pair%d" % k, comp, {"{TAG}": "pair%d" % k, "{MM}": 1,
                                     "{SEED}": 910000 + k, "{REGEN_BLOCK}": OFF})
        draws.append(trip("pair%d" % k) * 1e3)
    run("cmp_latch0", comp, {"{TAG}": "latch0", "{MM}": 0, "{SEED}": 0,
                             "{REGEN_BLOCK}": ON})
    tr = np.loadtxt(os.path.join(OUT, "b1_cmp_tr_latch0.dat"))
    dv = np.abs(tr[:, 1] - tr[:, 3])
    i1 = int(np.argmax(dv > 1.0))
    t_delay = float(tr[i1, 0]) if (i1 > 0 and dv[i1] > 1.0) else float("inf")
    w = (dv > 0.02) & (dv < 0.4) & (tr[:, 0] > 5e-9)
    pf = np.polyfit(tr[w, 0], np.log(dv[w]), 1)
    tau_reg = 1.0 / pf[0]
    sig_meas = float(np.std(draws))
    pelgrom = float(np.sqrt(2) * 0.7071 * PAR_VTH_UM * 1e-6
                    / np.sqrt(1e-6 * 9.72e-6)) * 1e3
    print("\n--- T10 regenerative comparator core ---")
    print("  input-pair offset: nominal %.2f mV; 4 mismatch draws %s mV (sigma %.2f)"
          % (off0, ["%+.2f" % d for d in draws], sig_meas))
    print("  Pelgrom input-referred sqrt(2)*sigma_vth = %.2f mV (6.25's 38.36 mV is "
          "OUTPUT-referred: /12.95 -> %.2f mV)" % (pelgrom, 38.36 / 12.95))
    print("  regeneration tau %.2f ns; delay to |dv| = 1 V %.2f ns = %.1f tau"
          % (tau_reg * 1e9, t_delay * 1e9, t_delay / tau_reg))
    chk("T10a nominal pair offset under 1 mV (symmetry; a PWL source's DC sweep "
        "value is its 'dc' field, 0 V, so the trip point IS the offset)",
        abs(off0) < 1.0, "offset %.3f mV on a 20 uV sweep grid" % off0)
    chk("T10b mismatch is live and the Pelgrom scale is right (0.5..5 mV, 4 draws)",
        all(abs(d) > 0.5 for d in draws) and abs(sig_meas / pelgrom - 1) < 3.0,
        "sigma %.2f mV vs Pelgrom %.2f mV, seed-resolved draws"
        % (sig_meas, pelgrom))
    chk("T10c regeneration reaches 1 V with an exponential, ns-scale tau",
        t_delay < 20e-9 and 0.05e-9 < tau_reg < 5e-9,
        "delay %.2f ns, tau %.2f ns, %d fit points" % (t_delay * 1e9,
                                                       tau_reg * 1e9, w.sum()))
    out["t10"] = dict(offset_nominal_mv=off0, mismatch_draws_mv=draws,
                      sigma_meas_mv=sig_meas, pelgrom_mv=pelgrom,
                      tau_reg_s=tau_reg, delay_1v_s=t_delay,
                      note="6.25's 38.36 mV vd0 sigma is an output-referred offset")

    # =====================================================================
    # summary
    # =====================================================================
    n_pass = sum(1 for c in CHECKS if c["ok"])
    out.update(n_checks=len(CHECKS), n_pass=n_pass, checks=CHECKS,
               status="complete" if n_pass == len(CHECKS) else "failed")
    with open(os.path.join(OUT, "gf180_batch1.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n%d/%d checks passed, %d failed -> results/gf180_batch1.json"
          % (n_pass, len(CHECKS), len(CHECKS) - n_pass))
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
