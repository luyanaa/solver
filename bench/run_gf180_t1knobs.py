"""P0-B acceptance -- T1 four-knob integrator cell, gf180mcu batch-1 vehicle.

The P7 coefficient graph asks one integrator cell for four things at once, and
``hw/gf180/batch1/t1_four_knob.sp`` is the topology that delivers them:

    a_ii      a binary-weighted NMOS gm bank, 4 legs at m=1/2/4/8 off a replica
              gate rail -> 15 codes, a_ii(k) = k * gm_unit / C
    leakage   a triode NMOS whose gate is a continuous voltage (a *codable*
              resistor in the a_ii sense, not a code)
    b_ports   two input transconductors, fixed relative weights (m=1 and m=2),
              each with an enable
    STATE_WR  the write/hold switch, in three variants so the charge injection
              can be compared: an ideal S element (zero-injection reference), a
              CMOS transmission gate, and an NMOS-only pass device

Measurement method (2-port admittance at a *pinned* node). The cell's node is
held by an ideal source behind a 0.01 ohm resistor while an AC drive is applied
to one terminal; the current through that source is then exactly the quantity:

    gm probe    AC on the gate rail   -> |i| = gm          (ro does not enter)
    C probe     AC on the node        -> |i| = |g_ds+jwC|, C from the top decade
    leak probe  DC sweep +-2 mV       -> I and g_ds in one run (no AC needed)

That is why this suite does not lean on the AC "a0 * w_p" estimator: at a pinned
node gm is a single number and C is a single number, and the ratio is a_ii with
no ro in it.

Three netlist traps are load-bearing and are recorded in the deck's comments
because each one cost a debugging pass:

* ngspice's SW element is fine in ``op`` and ``tran`` but its AC linearisation
  sits at Roff, so every gate driven through a switch floats in an AC run. The
  code therefore also travels on bypass resistors ({RB0}..{RB3}, {RP1},
  {RP2}); the switch stays for the nominal switching transient.
* MOS subckt port order is d g s b. A transposition puts the gate on the source
  and leaves a 37 uS short from a 3.3 V rail into the node (measured).
* A PMOS with its bulk tied to a switchable source forward-biases its drain-bulk
  diode at a 0.1 V drain: another 37 uS floor. The bulk stays on vdd.

Every number below is measured here; nothing is carried in from the spec except
the isolated MIM (1.0163 pF, P0-A/G3) and the 12-bit trim floor (P0-E).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twin.spice import extract_gf180 as G            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW = os.path.join(ROOT, "hw", "gf180", "batch1")
OUT = G.OUT
NET = os.path.join(HW, "t1_four_knob.sp")

ON, OFF = 0.01, 1e12                     # bypass resistor on / off
VDD = 3.3
VSTATE = 1.2                             # the cell's quiescent state voltage
C_MIM = 1.0163e-12                       # isolated MIM, P0-A/G3 (DESIGN 6.25)
STEP12 = 1.7090e-4                       # 12-bit DAC step, P0-E (DESIGN 6.29)
ADC12_S = 12e-6                          # 12-bit conversion time, P0-E
CHECKS = []


def chk(name, ok, detail):
    CHECKS.append(dict(name=name, ok=bool(ok), detail=detail))
    print(("PASS  " if ok else "FAIL  ") + name + "  " + detail)
    return ok


def _load():
    with open(NET) as fh:
        return fh.read()


def subs_base(**kw):
    """All tokens of the deck, with the AC/dc/tran defaults (everything off)."""
    d = dict(B0=0, B1=0, B2=0, B3=0, VGL=0, VSB=0, MLOAD=1, VPL=0,
             EN1="dc 0", EN2="dc 0", AC1=0, AC2=0, VW=VSTATE,
             EW1="dc 0", EWP="dc 3.3", EWN="dc 0", EW3="dc 0",
             VDS=0, EDS="dc 0", AC_G=0, AC_I=0, AC_V=0, RDS=1e12, RCM="1e12",
             RB0=OFF, RB1=OFF, RB2=OFF, RB3=OFF, RP1=OFF, RP2=OFF)
    d.update(kw)
    return {"{%s}" % k: str(v) for k, v in d.items()}


def code_subs(code, **kw):
    bits = [(code >> i) & 1 for i in range(4)]
    d = dict((("B%d" % i), b) for i, b in enumerate(bits))
    d.update((("RB%d" % i), ON if b else OFF) for i, b in enumerate(bits))
    return subs_base(**dict(d, **kw))


def _m(d, **kw):
    """Merge bare keyword overrides into a token dict whose keys are braced."""
    out = dict(d)
    out.update({("{%s}" % k): str(v) for k, v in kw.items()})
    return out


def run(tag, kw, ctrl):
    txt = _load().replace("{CTRL}", ctrl)
    txt = txt.replace("{NGS}", G.NGS).replace("{OUT}", OUT)
    for k, v in kw.items():
        txt = txt.replace(k, v)
    return G.run("t1k_%s" % tag, txt)


AC_DECK = "op\nac dec 20 1 1e8\nwrdata {OUT}/t1k_ac.dat i(vds)"
TR_DECK = "op\ntran %s\nwrdata {OUT}/t1k_tr.dat v(out)"


def ac_cur(path=None):
    d = np.loadtxt(path or os.path.join(OUT, "t1k_ac.dat"))
    return d[:, 0], d[:, 1] + 1j * d[:, 2]


def tr_v(path=None):
    d = np.loadtxt(path or os.path.join(OUT, "t1k_tr.dat"))
    return d[:, 0], d[:, 1]


def gm_of(tag, kw):
    """Transconductance at a pinned node: AC on the gate rail, read i(vds)."""
    run(tag, _m(kw, AC_G=1, VDS=VSTATE, RDS=ON, EDS="dc 0"), AC_DECK)
    f, y = ac_cur()
    return abs(y[0])


def c_of(tag, kw):
    """Node capacitance at a pinned node: AC on the node, |i|/(w) at the top."""
    run(tag, _m(kw, AC_V=1, VDS=VSTATE, RDS=ON, EDS="dc 0"), AC_DECK)
    f, y = ac_cur()
    return abs(y[-1]) / (2 * np.pi * f[-1])


def dc_probe(tag, kw, v0):
    """I and g at a node voltage from one +-2 mV DC sweep (i(vds) is 2 columns)."""
    ctrl = ("dc Vds %.6f %.6f 0.002\nwrdata {OUT}/t1k_dc.dat i(vds)"
            % (v0 - 0.002, v0 + 0.002))
    run(tag, _m(kw, RDS=ON, EDS="dc 0", AC_V=0), ctrl)
    d = np.loadtxt(os.path.join(OUT, "t1k_dc.dat"))
    # i(vds) < 0 means current leaves the node through the pin; report the
    # node current as a positive sink and the conductance as +dI/dV.
    i_mid = -float(d[len(d) // 2, 1])
    g = -float((d[-1, 1] - d[0, 1]) / (d[-1, 0] - d[0, 0]))
    return i_mid, g


def tr_open(tag, kw, tstop="8u", tstep="4n"):
    """Write/hold: closed 0..1 us, opens at 1 us; returns (t, v)."""
    run(tag, kw, TR_DECK % (tstep + " " + tstop))
    return tr_v()


def slope_at(t, v, t0, t1):
    m = (t > t0) & (t < t1)
    return float(np.polyfit(t[m], v[m], 1)[0])


def main():
    t0 = time.time()
    out = dict(bench="run_gf180_t1knobs", cell="T1 four-knob integrator")
    print("T1 four-knob integrator cell -- 2-port admittance at a pinned node\n")

    # =================================================== A. a_ii ladder
    print("--- A. a_ii: binary gm bank (4 legs m=1/2/4/8), state at %.1f V ---"
          % VSTATE)
    codes = [1, 2, 4, 8, 15]
    lad = {}
    for k in codes:                        # drains fixed (load m=15, all attached)
        kw = code_subs(k, MLOAD=15, VPL=VDD)
        lad[k] = (gm_of("fa%d" % k, kw), c_of("fb%d" % k, kw))
    gmu = float(np.mean([lad[k][0] / k for k in codes]))
    ratios = {k: lad[k][0] / k / gmu for k in codes}
    print("  code  gm(mS)   gm/(k*gm_u)  C(pF)   a_ii(rad/s)")
    for k in codes:
        print("  %4d  %7.4f  %+9.5f    %7.4f  %9.3e"
              % (k, lad[k][0] * 1e3, ratios[k] - 1, lad[k][1] * 1e12,
                 lad[k][0] / lad[k][1]))
    chk("A1 the binary weights are exact in gm: gm(k)/k is one number over the "
        "whole ladder",
        max(abs(r - 1) for r in ratios.values()) < 1e-3,
        "gm_u = %.4f mS, worst |gm/(k*gm_u)-1| = %.2e over codes %s"
        % (gmu * 1e3, max(abs(r - 1) for r in ratios.values()), codes))

    a1 = lad[1][0] / lad[1][1]
    a15 = lad[15][0] / lad[15][1]
    span_fixed = a15 / a1
    print("  a_ii(1) %.3e -> a_ii(15) %.3e rad/s: span %.3fx of the 15x code"
          % (a1, a15, span_fixed))
    chk("A2 with every drain attached (load m=15 fixed, bank gate-switched) the "
        "coefficient ladder is linear in k",
        abs(span_fixed / 15.0 - 1) < 0.01,
        "span %.3fx vs 15.000 nominal (%+.2f %%) -- the residual is the legs' "
        "Cgd, not the weights" % (span_fixed, 100 * (span_fixed / 15.0 - 1)))

    # the counter-example: a load whose m tracks the code
    lad_trk = {}
    for k in (1, 15):
        kw = code_subs(k, MLOAD=k, VPL=VDD)
        lad_trk[k] = (gm_of("ft%d" % k, kw), c_of("fu%d" % k, kw))
    span_trk = (lad_trk[15][0] / lad_trk[15][1]) / (lad_trk[1][0] / lad_trk[1][1])
    cper = (lad[k][1] - lad[1][1]) / 14.0
    cper_trk = (lad_trk[15][1] - lad_trk[1][1]) / 14.0
    print("  load tracks the code: C(1) %.4f pF, C(15) %.4f pF -> %.1f fF/code "
          "(fixed-drain load: %.2f fF/code)" % (lad_trk[1][1] * 1e12,
                                                lad_trk[15][1] * 1e12,
                                                cper_trk * 1e15, cper * 1e15))
    chk("A3 a code-tracking load capacitance breaks the ladder: the same 15 "
        "codes come out %.1f %% short at the top" % (100 * (1 - span_trk / 15)),
        span_trk / 15 < 0.97, "span %.3fx vs 15.000 (%+.2f %%) because C grows "
        "%.1f fF/code; the tile must switch the LOAD's gates too (drains stay "
        "attached) -- the bank already does"
        % (span_trk, 100 * (span_trk / 15 - 1), cper_trk * 1e15))

    chk("A4 the ladder spans more than a decade and the top code is the tile's "
        "fastest coefficient",
        10 < span_fixed < 20 and 3.0e9 < a15 < 3.7e9,
        "a_ii %.3e .. %.3e rad/s (%.1fx); the top code is %.0f MHz in rad/s, "
        "the bottom one %.1f MHz" % (a1, a15, span_fixed,
                                      a15 / 2 / np.pi / 1e6, a1 / 2 / np.pi / 1e6))

    cpar = lad[1][1] / C_MIM - 1
    chk("A5 the node's parasitic capacitance is declared, not assumed away",
        0 < cpar < 0.30,
        "C(1) %.4f pF over the isolated MIM %.4f pF = +%.1f %% (bank legs + "
        "load + leak + switch drain junctions); a_ii is computed with C_eff "
        "from here on, never with the MIM alone"
        % (lad[1][1] * 1e12, C_MIM * 1e12, 100 * cpar))

    # the node capacitance the hold test actually sees (load off, bank off)
    c_hold = c_of("ch", subs_base(MLOAD=1, VPL=0))

    print("  hold node capacitance (load and bank off): %.4f pF" % (c_hold * 1e12))

    # =================================================== B. leakage knob
    print("\n--- B. leakage: triode NMOS, continuous gate {VGL} ---")
    tri = []
    for vg in (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.3):
        i, g = dc_probe("lk%03d" % int(vg * 100),
                        subs_base(VGL=vg, VSB=0, MLOAD=1, VPL=0), 0.1)
        tri.append((vg, g, i))
    tri = np.array(tri)
    print("  VGL(V)  g(uS)  R(kOhm)   I(uA) @Vds=0.1")
    for vg, g, i in tri:
        print("  %5.2f  %7.2f  %8.1f  %8.2f"
              % (vg, g * 1e6, (1 / g / 1e3 if g > 0 else np.inf), i * 1e6))
    m = (tri[:, 0] >= 0.8) & (tri[:, 0] <= 1.5)
    p = np.polyfit(tri[m, 0], tri[m, 1], 1)
    vt_leak = -p[1] / p[0]
    res = np.max(np.abs(np.polyval(p, tri[m, 0]) / tri[m, 1] - 1))
    chk("B1 the leak is a triode conductance at Vds=0.1 V: g = k*(VGL-Vt) over "
         "the knee region",
         0.60 < vt_leak < 0.85 and 110e-6 < p[0] < 160e-6 and res < 0.10,
         "knee fit (VGL 0.8..1.5): Vt %.3f V, k %.0f uS/V, worst residual "
         "%.1f %% -- a continuous resistor the compiler can code, not a code"
         % (vt_leak, p[0] * 1e6, 100 * res))

    g_lo = tri[2, 1]                        # VGL=0.6, the first resolvable point
    g_hi = tri[-1, 1]
    span_dec = np.log10(g_hi / max(g_lo, 1e-9))
    chk("B2 the knob spans four decades of conductance",
         span_dec > 3.5,
         "%.3f uS (VGL 0.6) .. %.1f uS (3.3) = %.2f decades; below 0.6 V g is "
         "under the 0.01 uS probe floor, so the true span is larger (R %.0f "
         "kOhm .. %.1f kOhm)" % (g_lo * 1e6, g_hi * 1e6, span_dec,
                                 1 / g_lo / 1e3, 1 / g_hi / 1e3))

    sl_lo = (tri[7, 1] - tri[6, 1]) / (tri[7, 0] - tri[6, 0])   # 1.5..2.0
    sl_hi = (tri[-1, 1] - tri[-2, 1]) / (tri[-1, 0] - tri[-2, 0])
    print("  local slope %.0f uS/V at 1.75 V -> %.0f uS/V at 3.15 V (mobility "
          "degradation: monotone but sub-linear)" % (sl_lo * 1e6, sl_hi * 1e6))
    chk("B3 the triode law is not extrapolated past its data: the slope falls "
         "with VGL", sl_hi < sl_lo, "%.0f -> %.0f uS/V (%.2fx)"
         % (sl_lo * 1e6, sl_hi * 1e6, sl_hi / sl_lo))

    # the same device asked for a leak at the *state* voltage
    st0 = []
    for vg in (0.8, 1.2, 1.5, 1.8, 2.2, 3.3):
        i, g = dc_probe("s0%03d" % int(vg * 100),
                        subs_base(VGL=vg, VSB=0, MLOAD=1, VPL=0), VSTATE)
        st0.append((vg, g, i))
    st0 = np.array(st0)
    r_15 = st0[2, 2] / max(st0[2, 1], 1e-12)    # I/g in volts
    print("  at the %.1f V state, ground source: VGL 1.5 -> g %.2f uS, I %.0f "
          "uA, I/g %.0f V" % (VSTATE, st0[2, 1] * 1e6, st0[2, 2] * 1e6, r_15))
    chk("B4 a ground-source leak degenerates at the state voltage: I/g is tens "
         "of volts, i.e. a current sink, not a conductor",
         st0[2, 1] < 1e-6 and st0[2, 2] > 1e-5 and r_15 > 10,
         "VGL 1.5: g %.2f uS but I %.0f uA (I/g %.0f V); a proportional leak "
         "needs g, and at V=%.1f V the device is in saturation (Vov=%.2f < "
         "Vds)" % (st0[2, 1] * 1e6, st0[2, 2] * 1e6, r_15, VSTATE,
                     1.5 - vt_leak))

    st1 = []
    for vg in (0.0, 1.6, 2.0, 2.4, 2.8, 3.3):
        i, g = dc_probe("s1%03d" % int(vg * 100),
                        subs_base(VGL=vg, VSB=1.0, MLOAD=1, VPL=0), VSTATE)
        st1.append((vg, g, i))
    st1 = np.array(st1)
    r_33 = st1[-1, 2] / max(st1[-1, 1], 1e-12)
    print("  level-shifted source at 1.0 V: VGL 3.3 -> g %.0f uS, I %.0f uA, "
          "I/g %.3f V" % (st1[-1, 1] * 1e6, st1[-1, 2] * 1e6, r_33))
    chk("B5 the level-shifted source (VSB=1.0 V) is a conductor AT the %.1f V "
         "state: I/g equals its own Vds, not tens of volts" % VSTATE,
         st1[-1, 1] > 100e-6 and 0.15 < r_33 < 0.35 and st1[1, 2] < 1e-9,
         "g 0 -> %.0f uS over VGL 2.0..3.3 with I/g %.3f V (= Vds 0.2 V), and "
         "I(VGL<=1.6) under 1 nA; the body effect moves Vt to ~1.0 V so the "
         "useful range starts at 2.0 V" % (st1[-1, 1] * 1e6, r_33))

    g_18 = tri[np.argmin(np.abs(tri[:, 0] - 1.5)), 1]
    chk("B6 the knob's range depends on the state voltage (triode needs Vds < "
         "Vov), so one table cannot document the cell",
         g_18 / max(st0[2, 1], 1e-12) > 20,
         "same VGL=1.5: %.1f uS at a 0.1 V state vs %.2f uS at a %.1f V state "
         "(%.0fx); a compiled a_ii must be taken at the tile's declared state"
         % (g_18 * 1e6, st0[2, 1] * 1e6, VSTATE, g_18 / st0[2, 1]))

    # =================================================== C. b_ports
    print("\n--- C. b_ports: two input transconductors, weights 1 and 2 ---")
    pq = dict(MLOAD=15, VPL=VDD, VGL=0, VSB=0)
    gm1 = gm_of("pp1", _m(subs_base(**pq), RP1=ON, AC1=1, AC2=0))
    gm2 = gm_of("pp2", _m(subs_base(**pq), RP2=ON, AC2=1, AC1=0))
    gm0 = gm_of("pp0", _m(subs_base(**pq), RP1=OFF, AC1=1, AC2=0))
    gmb = gm_of("ppb", _m(subs_base(**pq), RP1=ON, RP2=ON, AC1=1, AC2=1))
    print("  gm(port1) %.4f mS, gm(port2) %.4f mS, disabled %.5f mS, both "
          "%.4f mS" % (gm1 * 1e3, gm2 * 1e3, gm0 * 1e3, gmb * 1e3))
    chk("C1 the port weights are exact: m=2 gives exactly twice m=1",
        abs(gm2 / gm1 - 2) < 0.005,
        "%.5f vs %.5f mS (%+.3f %%)" % (gm2 * 1e3, gm1 * 1e3,
                                        100 * (gm2 / gm1 / 2 - 1)))
    chk("C2 a port unit and a bank unit are the same device at the same bias "
        "(one replica rail)", abs(gm1 / gmu - 1) < 0.01,
        "gm_port1 %.5f mS vs gm_unit %.5f mS (%+.2f %%)"
        % (gm1 * 1e3, gmu * 1e3, 100 * (gm1 / gmu - 1)))
    chk("C3 the port enable really disables (the gate goes to ground, it is not "
        "just a floating node)", gm0 / gm1 < 0.01,
        "disabled %.6f mS = %.2f %% of enabled" % (gm0 * 1e3, 100 * gm0 / gm1))
    chk("C4 with both ports driven the node current is the sum (no cross-talk "
        "beyond the port ratio)", abs(gmb / (gm1 + gm2) - 1) < 0.01,
        "%.4f vs %.4f mS (%+.2f %%)" % (gmb * 1e3, (gm1 + gm2) * 1e3,
                                        100 * (gmb / (gm1 + gm2) - 1)))

    # =================================================== D. STATE_WR
    print("\n--- D. STATE_WR: charge injection, three variants ---")
    drop = "pulse(3.3 0 1u 10n 10n 100u 200u)"
    rise = "pulse(0 3.3 1u 10n 10n 100u 200u)"
    cen = lad[1][1]
    inj = {}
    for name, kw in (("ideal", dict(EW1=drop)), ("tg", dict(EWN=drop, EWP=rise)),
                     ("nmos", dict(EW3=drop))):
        t, v = tr_open("wr%s" % name, subs_base(MLOAD=1, VPL=0, VGL=0, **kw),
                       tstop="6u", tstep="2n")
        pre = float(np.mean(v[(t > 0.9e-6) & (t < 0.99e-6)]))
        post = float(v[np.argmax(t >= 1.06e-6)])
        inj[name] = (pre, post, post - pre)
        print("  %-5s v_pre %.6f V, v_post %.6f V -> injection %+.4f mV "
              "(Q %+.3f fC)" % (name, pre, post, (post - pre) * 1e3,
                                (post - pre) * c_hold * 1e15))
    q_lsb = STEP12 * VSTATE * c_hold
    chk("D1 the ideal switch is a zero-injection reference and the TG beats the "
        "NMOS-only pass device by more than 5x",
        abs(inj["ideal"][2]) < 1e-5 and inj["nmos"][2] < -1e-3
        and abs(inj["nmos"][2] / inj["tg"][2]) > 5,
        "ideal %+.4f mV, TG %+.3f mV, NMOS %+.3f mV -> NMOS/TG %.1fx"
        % (inj["ideal"][2] * 1e3, inj["tg"][2] * 1e3, inj["nmos"][2] * 1e3,
           abs(inj["nmos"][2] / inj["tg"][2])))
    chk("D2 the injection is quoted in trim LSBs: the TG's is below one 12-bit "
        "LSB, the NMOS-only's is several",
        abs(inj["tg"][2]) * c_hold / q_lsb < 1.0
        and abs(inj["nmos"][2]) * c_hold / q_lsb > 3.0,
        "12-bit LSB %.3f fC: TG %.2f LSB, NMOS %.2f LSB = %.3f %% of the 1.2 V "
        "state range per write"
        % (q_lsb * 1e15, abs(inj["tg"][2]) * c_hold / q_lsb,
           abs(inj["nmos"][2]) * c_hold / q_lsb,
           100 * abs(inj["nmos"][2]) * c_hold / q_lsb * STEP12))

    # hold: the floor, its attribution, the leak knob (two ends)
    t, v = tr_open("hdf", subs_base(EWN=drop, EWP=rise, VGL=0))
    rate_f = slope_at(t, v, 1.2e-6, 7.8e-6)
    i_f, _ = dc_probe("hdfi", subs_base(VGL=0, MLOAD=1, VPL=0), VSTATE)
    pred_f = i_f / c_hold
    print("  hold floor: dv/dt %.2f V/s, pinned-node DC current %.1f pA -> "
          "I/C %.2f V/s" % (rate_f, i_f * 1e12, pred_f))
    chk("D3 the hold floor is the off-state leakage of the write switches, and "
        "the same current is seen two ways",
        abs(-rate_f / pred_f - 1) < 0.15 and 1e-11 < i_f < 1e-10,
        "transient %.2f V/s vs I/C %.2f V/s (I %.1f pA, C_hold %.4f pF) -> "
        "%.1f %%; the DC probe is at exactly 1.2 V, the transient window "
        "averages 1.2 -> 1.05 V" % (rate_f, pred_f, i_f * 1e12, c_hold * 1e12,
                                    100 * abs(-rate_f / pred_f - 1)))

    body = (("XWn2 out enwn w2 0 nfet_03v3 w=2u l=0.28u", "* removed"),
            ("XWp2 out enwp w2 vdd pfet_03v3 w=4u l=0.28u", "* removed"),
            ("XWn3 out ewn3 w3 0 nfet_03v3 w=2u l=0.28u", "* removed"),
            ("XMlk out vgl vsb 0 nfet_03v3 w=2u l=2u", "* removed"))
    txt = _load().replace("{CTRL}", TR_DECK % ("4n 8u"))
    txt = txt.replace("{NGS}", G.NGS).replace("{OUT}", OUT)
    for k, v_ in subs_base(EWN=drop, EWP=rise, VGL=0).items():
        txt = txt.replace(k, v_)
    for a, b in body:
        txt = txt.replace(a, b)
    G.run("t1k_hdfloor", txt)
    t, v = tr_v()
    rate_0 = slope_at(t, v, 1.2e-6, 7.8e-6)
    chk("D4 attribution: removing the three write devices and the leak device "
        "takes the floor to zero, so the %.0f pA is theirs, not the MIM or the "
        "fixture" % (i_f * 1e12),
        abs(rate_0) < 0.5,
        "dv/dt %.3f V/s with the write devices and the leak device removed "
        "(vs %.2f V/s with them)" % (rate_0, rate_f))

    # relative drift per us: (rate/V) has units 1/s, so ppm per us is
    # (rate/V) * 1e6 * 1e-6 = rate/V
    ppm_us = -rate_f / VSTATE
    hold12 = 122.0 / ppm_us
    print("  drift %.1f ppm/us of the state -> a 122 ppm (12-bit) hold lasts "
          "%.1f us, vs the %.0f us ADC conversion"
          % (ppm_us, hold12, ADC12_S * 1e6))
    chk("D5 consequence for the calibration contract: at 12 bits the cell "
        "cannot hold its value across one ADC conversion",
        hold12 * 1e-6 < ADC12_S,
        "hold budget %.1f us < ADC %.0f us (%.1fx short): the background loop "
        "must re-write (foreground, or a bigger C, or a lower-leakage switch), "
        "not just track -- a measured design gap for the B3 cell, not a bench "
        "failure" % (hold12, ADC12_S * 1e6, ADC12_S / (hold12 * 1e-6)))

    kw_l = subs_base(EWN=drop, EWP=rise, VGL=2.0, VSB=1.0)
    t, v = tr_open("hdlk", kw_l, tstop="4u", tstep="2n")
    rate_l = slope_at(t, v, 1.02e-6, 1.10e-6)
    i_l, g_l = dc_probe("hdli", subs_base(VGL=2.0, VSB=1.0, MLOAD=1, VPL=0),
                        VSTATE)
    pred_i = -i_l / c_hold                     # current-driven droop
    pred_g = -g_l * (VSTATE - 1.0) / c_hold    # what a conductor would give
    print("  level-shifted leak at VGL=2.0: dv/dt %.1f kV/s; I/C predicts "
          "%.1f kV/s, g*(V-1 V)/C would predict %.1f kV/s"
          % (rate_l / 1e3, pred_i / 1e3, pred_g / 1e3))
    chk("D6 the hold droop is the DC leak current, not the small-signal "
        "conductance: dv/dt follows I/C",
        abs(rate_l / pred_i - 1) < 0.25 and abs(rate_l) > 5 * abs(pred_g),
        "%.1f kV/s vs I/C %.1f kV/s (%+.0f %%) and vs g*(V-1 V)/C %.1f kV/s "
        "(%.0fx); I %.0f nA, g %.0f nS -- in subthreshold I/g is 13 V, so the "
        "current is the right variable (window 1.02..1.10 us averages "
        "1.2 -> 1.14 V, which is why the measured slope sits under the 1.2 V "
        "prediction)" % (rate_l / 1e3, pred_i / 1e3,
                        100 * abs(rate_l / pred_i - 1), pred_g / 1e3,
                        abs(rate_l / pred_g), i_l * 1e9, g_l * 1e9))

    t, v = tr_open("hdfast", subs_base(EWN=drop, EWP=rise, VGL=3.3, VSB=0),
                   tstop="4u")
    i5 = np.argmax(v <= 0.05 * VSTATE)
    t_coll = (t[i5] - 1.0e-6) * 1e9
    i_33 = st0[-1, 2]
    print("  ground-source leak at VGL=3.3: the state is gone in %.0f ns "
          "(%.0f uA sink)" % (t_coll, i_33 * 1e6))
    chk("D7 the top of the ground-source knob is not a hold setting: the state "
        "collapses in tens of nanoseconds",
        t_coll < 50 and 1 < t_coll * 1e-9 / (c_hold * VSTATE / i_33) < 5,
        "%.0f ns to 5 %% of the state vs C*V/I = %.0f ns (%.0f uA sink) -- the "
        "knob's usable hold range is its low end, which is why D6 is measured "
        "at the level-shifted low end" % (t_coll,
                                          c_hold * VSTATE / i_33 * 1e9,
                                          i_33 * 1e6))

    # =================================================== E. cross-checks
    print("\n--- E. cross-checks ---")
    chk("E1 one formula predicts the ladder: a_ii(k) = k*gm_u/C(k) reproduces "
        "the measured top and bottom",
        abs(gmu / lad[1][1] / a1 - 1) < 0.01
        and abs(15 * gmu / lad[15][1] / a15 - 1) < 0.01,
        "a_ii(1) %.3e (formula %.3e), a_ii(15) %.3e (formula %.3e)"
        % (a1, gmu / lad[1][1], a15, 15 * gmu / lad[15][1]))
    frac = st1[-1, 1] / c_hold / a1
    chk("E2 the leak is a fine knob in a_ii units (about half a code), not a "
        "second coarse ladder",
        0.2 < frac < 1.0,
        "g_leak(max)/C_hold = %.3e rad/s = %.2f of a_ii(code 1) (%.3e)"
        % (st1[-1, 1] / c_hold, frac, a1))

    # =================================================== summary
    out.update(gm_unit_s=gmu, ladder={str(k): dict(gm_s=lad[k][0],
                                                   c_f=lad[k][1],
                                                   a_ii=lad[k][0] / lad[k][1])
                                      for k in codes},
               span_fixed=span_fixed, span_track=span_trk,
               c_per_code_f=cper, c_per_code_track_f=cper_trk,
               c_eff_1_f=lad[1][1], c_hold_f=c_hold,
               c_mim_f=C_MIM, c_par_over_mim=cpar,
               leak=dict(vt=vt_leak, k_s_per_v=p[0],
                         table=[[float(a), float(b), float(c)] for a, b, c in tri],
                         state_ground=[[float(a), float(b), float(c)]
                                       for a, b, c in st0],
                         state_shift=[[float(a), float(b), float(c)]
                                      for a, b, c in st1],
                         span_decades=span_dec, g_max_s=g_hi),
               ports=dict(gm1_s=gm1, gm2_s=gm2, gm_disabled_s=gm0,
                          gm_both_s=gmb),
               state_wr=dict(injection=dict((k, dict(v_pre=v[0], v_post=v[1],
                                                     dv=v[2],
                                                     q_fc=v[2] * c_hold * 1e15))
                                            for k, v in inj.items()),
                             q_lsb_fc=q_lsb * 1e15,
                             hold_floor_v_s=rate_f, hold_floor_i_a=i_f,
                             hold_removed_v_s=rate_0, hold_leak_v_s=rate_l,
                             hold_leak_i_a=i_l, collapse_ns=t_coll,
                             drift_ppm_us=ppm_us, hold12_us=hold12,
                             adc12_us=ADC12_S * 1e6),
               n_checks=len(CHECKS), n_pass=sum(1 for c in CHECKS if c["ok"]),
               checks=CHECKS, total_seconds=time.time() - t0)
    out["status"] = "complete" if out["n_pass"] == out["n_checks"] else "failed"
    with open(os.path.join(OUT, "gf180_t1knobs.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    rep = os.path.join(OUT, "gf180_t1knobs_report.txt")
    with open(rep, "w") as fh:
        fh.write("P0-B  T1 four-knob integrator cell -- gf180mcu batch-1\n")
        fh.write("gm_unit %.4f mS, a_ii %.3e..%.3e rad/s (%.2fx), C_eff(1) "
                 "%.4f pF (+%.1f %% over the MIM), C_hold %.4f pF\n"
                 % (gmu * 1e3, a1, a15, span_fixed, lad[1][1] * 1e12,
                    100 * cpar, c_hold * 1e12))
        fh.write("leak Vt %.3f V k %.0f uS/V at Vds=0.1, %.2f decades; at the "
                 "%.1f V state the ground source is a sink (I/g %.0f V) and the "
                 "level-shifted one is a conductor (I/g %.2f V, %.0f uS)\n"
                 % (vt_leak, p[0] * 1e6, span_dec, VSTATE, r_15, r_33,
                    st1[-1, 1] * 1e6))
        fh.write("ports %.4f/%.4f mS (1:2), disabled %.5f mS; injection TG "
                 "%.3f mV (%.2f LSB12) vs NMOS %.3f mV (%.2f LSB12); hold "
                 "floor %.1f pA = %.1f ppm/us, collapse %.0f ns\n"
                 % (gm1 * 1e3, gm2 * 1e3, gm0 * 1e3, inj["tg"][2] * 1e3,
                    abs(inj["tg"][2]) * c_hold / q_lsb, inj["nmos"][2] * 1e3,
                    abs(inj["nmos"][2]) * c_hold / q_lsb, i_f * 1e12, ppm_us,
                    t_coll))
        for c in CHECKS:
            fh.write(("PASS  " if c["ok"] else "FAIL  ") + c["name"] + "  "
                     + c["detail"] + "\n")
        fh.write("\n%d/%d checks pass\n" % (out["n_pass"], out["n_checks"]))
    print("\n%d/%d checks passed, %d failed -> results/gf180_t1knobs.json "
          "[%.1f s]" % (out["n_pass"], out["n_checks"],
                        out["n_checks"] - out["n_pass"], out["total_seconds"]))
    return 0 if out["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
