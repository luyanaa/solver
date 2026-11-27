"""gf180mcu pre-silicon primitive extraction (G1 of PLAN-gf180mcu-batch1).

Mirrors twin/spice/extract_b2.py's fail-closed framework on the GF180MCU
open PDK (MOS-only vehicle #1; IHP SG13G2 stays vehicle #2 with the HBT
interface). Everything here is measured through ngspice against the PDK
the repo already extracts (twin/spice/extract.py PDKS["gf180mcu"],
results/device_extract.json: lambda=0.034/V @10 uA on nfet_03v3 10u/1u).

Blocks (all fail-closed, results/gf180_extract.json carries status):

  corners        tt/ss/ff x {3v3, 6v0} full probe: mirror ratio curve
                 (Vh sweep -> the lambda*DVds mechanism of §6.9 measured
                 directly), 1:4/1:16 shared-reference ratios, gm-cell pole
                 (ideal load + 1 pF C_bank -> ro = 1/(2*pi*fpole*C)),
                 .noise gamma_eff on nFET/pFET cells (BSIM4 noise is native
                 in ngspice — the MOS gamma_eff the IHP lane could NOT
                 measure: PSP103 OSDI noise is unimplemented there),
                 transmission-gate Ron + 4kT*Ron sanity, MIM cap C meas.
  mismatch_mc    48 seeds x {3v3, 6v0} of the PDK's OWN Pelgrom machinery
                 (fets_mm subckt wrappers + sw_stat_mismatch=1, per-parse
                 agauss redraw via `.option seed`), separate processes per
                 seed (a plain reset does NOT redraw agauss — same lesson
                 as B2). Diff-pair offset, shared mirror ratio, gm, Id,
                 TG Ron. Same-seed reproducibility is verified by re-running
                 seed 1 and comparing rows byte-for-byte.
  pvt            temps(-40/27/125) x vdd(+-10%) per family, trimmed probe.
  joint          ss/ff x -40/125 C with corner lib AND .temp in the SAME
                 run (no multiplicative composition), per family.
  parasitics     added C at the integrator node 0/10/100 fF -> pole slope
                 per FARAD vs the exact two-point log-slope prediction
                 ("pre-PEX sensitivity estimate", same label as B2).
  transient      fast (gm cell, step+sine THD) and slow (pole cell, the
                 ro*C integrator settling) stimulus, 3v3 family.

Device families: the PDK (version 792ef670) ships 3.3 V and 6.0 V analog
devices (nfet_03v3 / nfet_06v0); PLAN-gf180mcu-batch1 said "5 V" — the PDK
offers 6 V, recorded as the honest deviation. Mismatch constants (par_vth
7.148 mV*um nfet3v3 / 6.66 pfet3v3 / 11.55 nfet6v0) are parsed from the
PDK's fets_mm section at runtime; sigma_vth = 0.7071*par_vth*1e-6/
sqrt(L_eff*W_eff) is the PDK's own formula.

Reuse: analysis helpers are imported from extract_b2 (same wrdata/spectrum
conventions, same noise band 1 kHz..1 GHz so gamma_eff numbers are directly
comparable to the IHP table).
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from twin.spice.extract_b2 import (NL, K_B, Q_E, OUT, N_MC_LOCAL, IHP,
                                   _load_wrdata, load_ac, ac_pole_hz,
                                   integrate_psd, gamma_eff, _mc_summary,
                                   _settle, _thd)

CIEL = os.path.expanduser("~/.ciel/ciel")
GF = sorted(glob.glob(CIEL + "/gf180mcu/versions/*/gf180mcuA"))[0]
NGS = GF + "/libs.tech/ngspice"

FOUR_KT = 4.0 * K_B * 300.0
MC_SEED_BASE = 910000
N_MC_SEEDS = N_MC_LOCAL            # 48 seeds, one ngspice process each
W_INT, L_INT = 10e-6, 1e-6         # integrator/interface device geometry
W_TG_N, L_TG = 2e-6, 0.5e-6
W_TG_P = 4e-6
I_BIAS = 10e-6

FAMILIES = {
    "3v3": dict(dev_n="nfet_03v3", dev_p="pfet_03v3", vdd=3.3, l_tg=0.5e-6,
                vdds_pvt=(3.0, 3.3, 3.6), vhold=1.65, rl_a=100e3,
                vcm=1.65, vh_sw=(0.2, 3.0, 0.01), vg_sw=(-0.08, 0.08, 0.002),
                vg_mc=(-0.03, 0.03, 0.006)),
    "6v0": dict(dev_n="nfet_06v0", dev_p="pfet_06v0", vdd=6.0, l_tg=0.7e-6,
                vdds_pvt=(5.4, 6.0, 6.6), vhold=3.0, rl_a=200e3,
                vcm=3.0, vh_sw=(0.2, 5.5, 0.02), vg_sw=(-0.15, 0.15, 0.005),
                vg_mc=(-0.05, 0.05, 0.01)),
}
FAM_SECTIONS = (("tt", "typical"), ("ss", "ss"), ("ff", "ff"))
PVT_TEMPS = (-40.0, 27.0, 125.0)
JOINT = (("ss", -40.0), ("ss", 125.0), ("ff", -40.0), ("ff", 125.0))
PARAS_C_FF = (0.0, 10.0, 100.0)
TRAN = dict(step_v=1e-3, sine_v=1e-3, f0=10e6, step_t0=20e-9,
            step_tstop=400e-9, sine_tstop=1e-6, dt=0.2e-9, maxstep=1e-9,
            settle_tol=0.01, thd_harmonics=5,
            slow_step_v=1e-3, slow_t0=10e-6, slow_tstop=100e-6,
            slow_dt=0.2e-6, slow_rise=1e-6)
CAP_NAME = "cap_mim_2f0_m4m5_noshield"
CAP_L = CAP_W = 22.36e-6           # 500 um^2 * 2 fF/um^2 = 1 pF nominal
CAP_F_AC = 1e6

# the PDK's own Pelgrom constants, parsed from the fets_mm section
_MM_PAR = {}


def _parse_mm_constants():
    """par_vth / par_l / par_w per device from the PDK fets_mm section."""
    if _MM_PAR:
        return _MM_PAR
    txt = open(NGS + "/sm141064.spice").read()
    for m in re.finditer(
            r"\.subckt\s+(nfet_03v3|pfet_03v3|nfet_06v0|pfet_06v0)\s+.*?"
            r"\.ends\s+\1", txt, re.S | re.I):
        name = m.group(1).lower()
        blk = m.group(0)
        pv = re.search(r"\.param\s+par_vth=([0-9.eE+-]+)", blk)
        pl = re.search(r"\.param\s+par_l=([0-9.eE+-]+)", blk)
        pw = re.search(r"\.param\s+par_w=([0-9.eE+-]+)", blk)
        if pv:
            _MM_PAR[name] = dict(par_vth=float(pv.group(1)),
                                 par_l=float(pl.group(1)) if pl else 0.0,
                                 par_w=float(pw.group(1)) if pw else 0.0)
    return _MM_PAR


def sigma_vth(dev, w, l):
    """The PDK's own sigma formula: 0.7071*par_vth*1e-6/sqrt(L_eff*W_eff)."""
    p = _parse_mm_constants()[dev.lower()]
    leff = l - p["par_l"]
    weff = (w - p["par_w"])
    if leff <= 0 or weff <= 0:
        return float("nan")
    return 0.7071 * p["par_vth"] * 1e-6 / np.sqrt(leff * weff)


# ---------------------------------------------------------------------------
# netlists
# ---------------------------------------------------------------------------
def netlist_corner(fam, sec, tag):
    """One probe file per (family, corner): DC operating points, mirror
    ratio vs Vds (the lambda mechanism), gm probe with a DC gate sweep,
    pole cell (ideal load + 1 pF -> ro), three noise cells, TG Ron, MIM C.

    Biasing lesson recorded in the code, not the log: a PMOS with its bulk
    at ground and its source at mid-rail forward-biases the source-body
    diode (measured v(b)=0.64 V on the first pass); the p-bulk goes to vdd
    here, and the TG probe current is 2 uA so both legs stay in triode
    (10 uA pushed a 2u/0.5u TG into saturation and read as 100 kOhm).
    """
    f = FAMILIES[fam]
    iu = I_BIAS * 1e6
    p = [
        "* gf180 corner probe %s (%s, %s)" % (tag, fam, sec),
        '.lib "%s/sm141064.ngspice" %s' % (NGS, sec),
        ".param fnoicor=0",
        ".param sw_stat_global=0",
        ".param sw_stat_mismatch=0",
        "* --- NMOS diode reference (replica bias) + 1:1 mirror + 1:4 / 1:16",
        "Vdd vdd 0 %g" % f["vdd"],
        "Iref vdd cdr dc %gu" % iu,
        "XM0 cdr cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "XM1 cmout cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "Vh cmout 0 dc %g" % f["vhold"],
        "* a second 1:1 copy with Vds forced to V(cdr): the shared-bias intent",
        "XM1s cmouts cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "Es cmouts 0 cdr 0 1",
        "* arrays: a VCVS holds Vds = V(cdr) -- the shared-reference intent",
        "XM4 cm4 cdr 0 0 %s w=10u l=1u m=4" % f["dev_n"],
        "Eh4 cm4 0 cdr 0 1",
        "XM16 cm16 cdr 0 0 %s w=10u l=1u m=16" % f["dev_n"],
        "Eh16 cm16 0 cdr 0 1",
        "* --- gm probe: resistive load + DC gate sweep (gm = dId/dVg)",
        "Vgg ggm cdr dc 0",
        "XMgg dgg ggm 0 0 %s w=10u l=1u" % f["dev_n"],
        "Rlg vdd dgg %gk" % (f["rl_a"] / 1e3),
        "* --- diode-node buffers (VCVS, zero output Z) so the series AC",
        "*     source really applies 1 V across Vgs instead of dividing",
        "*     against the diode's own 1/gm (measured 13 vs 340 on pass 1)",
        "Eb gb 0 cdr 0 1",
        "Ebp gbp 0 cdp 0 1",
        "* --- noise cell (nMOS): resistive load + series AC noise reference",
        "Vgac gb gn1 0 dc 0 ac 1",
        "XMn da gn1 0 0 %s w=10u l=1u" % f["dev_n"],
        "Rla vdd da %gk" % (f["rl_a"] / 1e3),
        "* --- pole cell: ideal current-source load + 1 pF C_bank -> ro",
        "Vgb gb gb2 0 dc 0 ac 1",
        "XMn2 d2b gb2 0 0 %s w=10u l=1u" % f["dev_n"],
        "Ila2 vdd d2b dc %gu" % iu,
        "C1 d2b 0 1p",
        "* --- pFET diode + noise cell (p-bulk on vdd = the n-well)",
        "Irefp cdp 0 dc %gu" % iu,
        "XP0 cdp cdp vdd vdd %s w=10u l=1u" % f["dev_p"],
        "Vgacp gbp gp1 0 dc 0 ac 1",
        "XPna dap gp1 vdd vdd %s w=10u l=1u" % f["dev_p"],
        "Rlap 0 dap %gk" % (f["rl_a"] / 1e3),
        "* --- transmission gate 2u n // 4u p @ 0.5u, 2 uA so both legs are",
        "*     in triode (n-bulk=0, p-bulk=vdd is how the TG is really built)",
        "Vtg a 0 dc %g ac 1" % f["vcm"],
        "XMtn b gtn a 0 %s w=2u l=%gu" % (f["dev_n"], f["l_tg"] * 1e6),
        "XMtp b gtp a vdd %s w=4u l=%gu" % (f["dev_p"], f["l_tg"] * 1e6),
        "Vgn gtn 0 %g" % f["vdd"],
        "Vgp gtp 0 0",
        "Itg b 0 dc 2u",
        "* --- MIM cap probe: %.2fu x %.2fu = 500 um2 @ 2 fF/um2 = 1 pF nominal"
        % (CAP_L * 1e6, CAP_W * 1e6),
        '.lib "%s/sm141064.ngspice" mimcap_typical' % NGS,
        "Vcb cbn 0 dc 1.5 ac 1",
        "Xcb cbn 0 %s c_length=%.2fu c_width=%.2fu"
        % (CAP_NAME, CAP_L * 1e6, CAP_W * 1e6),
        ".control",
        "op",
        "* 1:1 mirror with the output Vds held at vhold (lambda*DVds error)",
        "let ratio_mis = -i(vh)/%gu" % iu,
        "let ratio_shared = -i(es)/%gu" % iu,
        "let ratio4 = -i(eh4)/%gu" % iu,
        "let ratio16 = -i(eh16)/%gu" % iu,
        "let vdiode = v(cdr)",
        "let vdiodep = v(cdp)",
        "let ida = (%g-v(da))/%gk" % (f["vdd"], f["rl_a"] / 1e3),
        "let idap = v(dap)/%gk" % (f["rl_a"] / 1e3),
        "let vd2b = v(d2b)",
        "let vb = v(b)",
        "echo OPROW $&ratio_mis $&ratio4 $&ratio16 $&ratio_shared $&vdiode "
        "$&vdiodep $&ida $&idap $&vd2b $&vb",
        "dc Vh %g %g %g" % f["vh_sw"],
        "wrdata %s/gf_mirror_%s.dat i(vh)" % (OUT, tag),
        "dc Vgg %g %g %g" % (-0.03, 0.03, 0.002),
        "wrdata %s/gf_gm_%s.dat v(dgg)" % (OUT, tag),
        "ac dec 40 0.1 1e8",
        "wrdata %s/gf_ac_%s.dat v(d2b)" % (OUT, tag),
        "ac lin 2 %g %g" % (CAP_F_AC, CAP_F_AC),
        "wrdata %s/gf_cap_%s.dat i(vcb)" % (OUT, tag),
        "* ngspice-44 leaves the CURRENT plot on the INTEGRATED noise plot",
        "* after each analysis (spectral densities live in the sibling plot",
        "* and their plot number is not stable across analysis mixes), so",
        "* the band-integrated onoise_total/inoise_total are echoed directly",
        "* -- S_v = onoise_total^2/(f1-f0) over the flat band.",
        "noise v(da) Vgac dec 20 1e6 1e8",
        "let n_o = onoise_total",
        "let n_i = inoise_total",
        "echo NROW nmos $&n_o $&n_i",
        "* decade-narrow band: residual 1/f shows up as a higher density",
        "noise v(da) Vgac dec 20 1e6 1e7",
        "let n1_o = onoise_total",
        "let n1_i = inoise_total",
        "echo NROW nmos_low $&n1_o $&n1_i",
        "noise v(b) Vtg dec 20 1e6 1e9",
        "let t_o = onoise_total",
        "let t_i = inoise_total",
        "echo NROW tg $&t_o $&t_i",
        "noise v(dap) Vgacp dec 20 1e6 1e8",
        "let p_o = onoise_total",
        "let p_i = inoise_total",
        "echo NROW pmos $&p_o $&p_i",
        "noise v(dap) Vgacp dec 20 1e6 1e7",
        "let p1_o = onoise_total",
        "let p1_i = inoise_total",
        "echo NROW pmos_low $&p1_o $&p1_i",
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


def netlist_mc(fam, seed, tag):
    """Mismatch MC sample: the PDK's fets_mm wrappers redraw per parse, so
    each seed is its own ngspice process (`.option seed` reseeded)."""
    f = FAMILIES[fam]
    p = [
        "* gf180 mismatch MC %s (seed %d)" % (tag, seed),
        '.lib "%s/sm141064.ngspice" typical' % NGS,
        ".param fnoicor=0",
        ".param sw_stat_global=0",
        ".param sw_stat_mismatch=1",
        ".option seed=%d" % seed,
        "Vdd vdd 0 %g" % f["vdd"],
        "* --- diff pair (offset)",
        "Itail tail 0 dc 20u",
        "XQ1 dc1 db tail 0 %s w=10u l=1u" % f["dev_n"],
        "XQ2 dc2 dcm tail 0 %s w=10u l=1u" % f["dev_n"],
        "Rc1 vdd dc1 %gk" % (f["rl_a"] / 1e3),
        "Rc2 vdd dc2 %gk" % (f["rl_a"] / 1e3),
        "Vcm dcm 0 %g" % f["vcm"],
        "Vd db dcm dc 0 ac 1",
        "* --- shared-reference mirror 1:1 + 1:4 (ratio mismatch)",
        "Iref vdd cdr dc %gu" % (I_BIAS * 1e6),
        "XM0 cdr cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "XM1 cmout cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "Eh cmout 0 cdr 0 1",
        "XM4 cm4 cdr 0 0 %s w=10u l=1u m=4" % f["dev_n"],
        "Eh4 cm4 0 cdr 0 1",
        "* --- gm cell (resistive load, Vgb sweep for gm_fd)",
        "Vgb gb cdr dc 0",
        "XMn2 d2b gb 0 0 %s w=10u l=1u" % f["dev_n"],
        "Rl2 vdd d2b %gk" % (f["rl_a"] / 1e3),
        "* --- TG",
        "Vtg a 0 dc %g ac 1" % f["vcm"],
        "XMtn b gtn a 0 %s w=2u l=%gu" % (f["dev_n"], f["l_tg"] * 1e6),
        "XMtp b gtp a vdd %s w=4u l=%gu" % (f["dev_p"], f["l_tg"] * 1e6),
        "Vgn gtn 0 %g" % f["vdd"],
        "Vgp gtp 0 0",
        "Itg b 0 dc 2u",
        ".control",
        "op",
        "let vd0 = v(dc1)-v(dc2)",
        "let ratio = -i(eh)/%gu" % (I_BIAS * 1e6),
        "let ratio4 = -i(eh4)/%gu" % (4 * I_BIAS * 1e6),
        "let vdiode = v(cdr)",
        "let vb = v(b)",
        "echo MCOP $&vd0 $&ratio $&ratio4 $&vdiode $&vb",
        "dc Vd -0.005 0.005 0.001",
        "let gain = (v(dc1)[10]-v(dc1)[0])/0.01",
        "echo MCGN $&gain",
        "dc Vgb %g %g %g" % f["vg_mc"],
        "let i0 = (%g-v(d2b)[0])/%gk" % (f["vdd"], f["rl_a"] / 1e3),
        "let i1 = (%g-v(d2b)[5])/%gk" % (f["vdd"], f["rl_a"] / 1e3),
        "let i2 = (%g-v(d2b)[10])/%gk" % (f["vdd"], f["rl_a"] / 1e3),
        "let gmc = abs((i2-i0)/%g)" % (f["vg_mc"][1] - f["vg_mc"][0]),
        "let idop = i1",
        "echo MCGM $&gmc $&idop",
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


def netlist_pvt(fam, sec, temp_c, vdd, tag):
    f = FAMILIES[fam]
    p = [
        "* gf180 PVT %s (%s, %s, %g C, %g V)" % (tag, fam, sec, temp_c, vdd),
        '.lib "%s/sm141064.ngspice" %s' % (NGS, sec),
        ".param fnoicor=0",
        ".param sw_stat_global=0",
        ".param sw_stat_mismatch=0",
        ".temp %g" % temp_c,
        "Vdd vdd 0 %g" % vdd,
        "Iref vdd cdr dc %gu" % (I_BIAS * 1e6),
        "XM0 cdr cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "XM1 cmout cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "* INDEPENDENT mirror: drain held at vhold (lambda*dVds error shows)",
        "Vh cmout 0 dc %g" % f["vhold"],
        "* gb follows the diode: without this the AC cell is unbiassed",
        "Eb gb 0 cdr 0 1",
        "Vgb gb gb2 0 dc 0 ac 1",
        "XMn2 d2b gb2 0 0 %s w=10u l=1u" % f["dev_n"],
        "Ila2 vdd d2b dc %gu" % (I_BIAS * 1e6),
        "C1 d2b 0 1p",
        "Vgac cdr gn1 0 dc 0 ac 1",
        "XMn da gn1 0 0 %s w=10u l=1u" % f["dev_n"],
        "Rla vdd da %gk" % (f["rl_a"] / 1e3),
        "* gm probe: resistive load + DC gate sweep, the corner netlist's cell",
        "Vgg ggm cdr dc 0",
        "XMgg dgg ggm 0 0 %s w=10u l=1u" % f["dev_n"],
        "Rlg vdd dgg %gk" % (f["rl_a"] / 1e3),
        ".control",
        "op",
        "let ratio = -i(vh)/%gu" % (I_BIAS * 1e6),
        "let vdiode = v(cdr)",
        "let ida = (%g-v(da))/%gk" % (f["vdd"], f["rl_a"] / 1e3),
        "let vd2b = v(d2b)",
        "echo OPROW $&ratio $&vdiode $&ida $&vd2b",
        "dc Vgb %g %g %g" % f["vg_sw"],
        "wrdata %s/gf_gmsweep_%s.dat v(d2b)" % (OUT, tag),
        "dc Vgg -0.03 0.03 0.002",
        "wrdata %s/gf_pvtgm_%s.dat v(dgg)" % (OUT, tag),
        "ac dec 40 0.1 1e8",
        "wrdata %s/gf_ac_%s.dat v(d2b)" % (OUT, tag),
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


def netlist_paras(c_ff, tag):
    f = FAMILIES["3v3"]
    extra = "" if c_ff <= 0 else "Cpar d2b 0 %gf\n" % c_ff
    p = [
        "* gf180 pre-PEX parasitic probe: added C = %g fF at d2b" % c_ff,
        '.lib "%s/sm141064.ngspice" typical' % NGS,
        ".param fnoicor=0",
        ".param sw_stat_global=0",
        ".param sw_stat_mismatch=0",
        "Vdd vdd 0 %g" % f["vdd"],
        "Iref vdd cdr dc %gu" % (I_BIAS * 1e6),
        "XM0 cdr cdr 0 0 %s w=10u l=1u" % f["dev_n"],
        "* bias follow: gb tracks the diode, gb2 = gb - Vgb",
        "Eb gb 0 cdr 0 1",
        "Vgb gb gb2 0 dc 0 ac 1",
        "XMn2 d2b gb2 0 0 %s w=10u l=1u" % f["dev_n"],
        "Ila2 vdd d2b dc %gu" % (I_BIAS * 1e6),
        "C1 d2b 0 1p",
        extra.rstrip("\n"),
        ".control",
        "ac dec 40 0.1 1e8",
        "wrdata %s/gf_ac_%s.dat v(d2b)" % (OUT, tag),
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


def netlist_tran():
    f = FAMILIES["3v3"]
    vg_op = 0.765       # 3v3 diode Vgs @10uA (device_extract.json); the tran
    # stimulus is recorded verbatim in the JSON so the post-run can reproduce
    T = TRAN
    p = [
        "* gf180 transient: fast gm-cell (step+sine THD) + slow pole cell",
        '.lib "%s/sm141064.ngspice" typical' % NGS,
        ".param fnoicor=0",
        ".param sw_stat_global=0",
        ".param sw_stat_mismatch=0",
        "* --- fast step copy: gm cell A, %g mV step @ t0=%gns, Rl=%gk"
        % (T["step_v"] * 1e3, T["step_t0"] * 1e9, f["rl_a"] / 1e3),
        "Vdd_s vdd_s 0 %g" % f["vdd"],
        "Vg_s g_s 0 PULSE(%g %g %g 0.5n 0.5n %g %g)"
        % (vg_op, vg_op + T["step_v"], T["step_t0"],
           T["step_tstop"] - T["step_t0"], 2 * T["step_tstop"]),
        "XMn_s da_s g_s 0 0 %s w=10u l=1u" % f["dev_n"],
        "Rla_s vdd_s da_s %gk" % (f["rl_a"] / 1e3),
        "* --- fast sine copy (THD): %g mV @ %g MHz"
        % (T["sine_v"] * 1e3, T["f0"] * 1e-6),
        "Vdd_n vdd_n 0 %g" % f["vdd"],
        "Vg_n g_n 0 SIN(%g %g %g)" % (vg_op, T["sine_v"], T["f0"]),
        "XMn_n da_n g_n 0 0 %s w=10u l=1u" % f["dev_n"],
        "Rla_n vdd_n da_n %gk" % (f["rl_a"] / 1e3),
        "* --- slow copy: pole cell (ideal load + 1 pF), %g mV gate step"
        % (T["slow_step_v"] * 1e3),
        "Vdd_q vdd_q 0 %g" % f["vdd"],
        "Vg_q g_q 0 PULSE(%g %g %g %gu %gu %g %g)"
        % (vg_op, vg_op + T["slow_step_v"], T["slow_t0"],
           T["slow_rise"] * 1e6, T["slow_rise"] * 1e6,
           T["slow_tstop"] - T["slow_t0"], 2 * T["slow_tstop"]),
        "XMn_q d2b_q g_q 0 0 %s w=10u l=1u" % f["dev_n"],
        "Ila_q vdd_q d2b_q dc %gu" % (I_BIAS * 1e6),
        "C1_q d2b_q 0 1p",
        ".control",
        "tran %gn %gn 0 %gn" % (T["dt"] * 1e9, T["step_tstop"] * 1e9,
                                T["maxstep"] * 1e9),
        "linearize",
        "wrdata %s/gf_tran_fast_step.dat v(da_s)" % OUT,
        "tran %gn %gn 0 %gn" % (T["dt"] * 1e9, T["sine_tstop"] * 1e9,
                                T["maxstep"] * 1e9),
        "linearize",
        "wrdata %s/gf_tran_fast_sine.dat v(da_n)" % OUT,
        "tran %gu %gu 0 %gu" % (T["slow_dt"] * 1e6, T["slow_tstop"] * 1e6,
                                T["slow_dt"] * 1e6),
        "linearize",
        "wrdata %s/gf_tran_slow_step.dat v(d2b_q)" % OUT,
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


# ---------------------------------------------------------------------------
# runner + parsers
# ---------------------------------------------------------------------------
def run(tag, netlist_text, timeout=900):
    os.makedirs(OUT, exist_ok=True)
    path = OUT + "/gf180_%s.sp" % tag
    with open(path, "w") as fh:
        fh.write(netlist_text)
    cmd = ["env", "-u", "LD_LIBRARY_PATH", "ngspice", "-b", path]
    env = os.environ.copy()
    # the machine-wide ~/.spiceinit is the IHP one and preloads OSDI libs
    # from $PDK_ROOT/$PDK; those models are irrelevant for gf180 but the
    # variables must resolve or ngspice exits nonzero
    env["PDK_ROOT"] = os.path.dirname(IHP)
    env["PDK"] = os.path.basename(IHP)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=NGS, env=env)
    with open(OUT + "/gf180_%s.log" % tag, "w") as fh:
        fh.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError("ngspice %s failed, see results/gf180_%s.log"
                           % (tag, tag))
    return r.stdout


def pole_interp(path):
    """-3 dB crossing by log-log interpolation on the wrdata grid.

    `ac_pole_hz` picks the nearest grid point, which quantizes ro by one
    sweep step (+-13 % at dec 10); this one interpolates between the two
    bracketing points, so a dec-40 sweep is accurate to ~1 %.
    """
    f, mag = load_ac(path)
    gdc = mag[0]
    tgt = gdc / np.sqrt(2.0)
    below = np.nonzero(mag <= tgt)[0]
    if len(below) == 0:
        return float(f[-1])
    k = int(below[0])
    if k == 0:
        return float(f[0])
    lf0, lf1 = np.log(f[k - 1]), np.log(f[k])
    lm0, lm1 = np.log(mag[k - 1]), np.log(mag[k])
    lt = np.log(tgt)
    w = (lt - lm0) / (lm1 - lm0) if lm1 != lm0 else 0.0
    return float(np.exp(lf0 + w * (lf1 - lf0)))

def _row(out, key, n):
    for line in out.splitlines():
        if line.startswith(key):
            vals = [float(x) for x in line.split()[1:]]
            if len(vals) != n:
                raise RuntimeError("%s has %d values, want %d"
                                   % (key, len(vals), n))
            return vals
    raise RuntimeError("no %s in output" % key)


def integrate_band(freq, amp, f0, f1):
    """Integrate S(f)=amp^2 over [f0,f1] (amp is an amplitude density)."""
    f = np.asarray(freq, float)
    sq = np.asarray(amp, float) ** 2
    m = (f >= f0) & (f <= f1)
    return float(np.trapezoid(sq[m], f[m]))


def parse_corner(out, fam, tag):
    f = FAMILIES[fam]
    v = _row(out, "OPROW", 10)
    d = dict(ratio_vds_mismatch=v[0], ratio4=v[1], ratio16=v[2],
             ratio_shared=v[3], vdiode=v[4], vdiodep=v[5], id_a=v[6],
             id_p=v[7], v_d2b=v[8], ron_tg=(f["vcm"] - v[9]) / 2e-6)
    d["ratio4_per_finger"] = d["ratio4"] / 4.0
    d["ratio16_per_finger"] = d["ratio16"] / 16.0
    # mirror ratio vs output Vds -> the lambda slope at the 6.9 point
    vh, io = _load_wrdata(OUT + "/gf_mirror_%s.dat" % tag)[0]
    ratio = -io / I_BIAS
    k = int(np.argmin(np.abs(vh - f["vhold"])))
    lo, hi = max(k - 8, 0), min(k + 8, len(vh) - 1)
    d["lambda_slope_vds"] = float(np.polyfit(vh[lo:hi + 1],
                                             ratio[lo:hi + 1], 1)[0])
    d["ratio_from_curve_at_vhold"] = float(ratio[k])
    # gm from the resistive-load gate sweep (Id = (vdd - v(dgg))/Rl)
    vg, vdgg = _load_wrdata(OUT + "/gf_gm_%s.dat" % tag)[0]
    idd = (f["vdd"] - vdgg) / f["rl_a"]
    d["gm_fd"] = float(np.polyfit(vg, idd, 1)[0])
    d["id_at_gmprobe_a"] = float(idd[np.argmin(np.abs(vg))])
    # ro from the ideal-load pole cell (1 pF bank)
    d["pole_hz"] = pole_interp(OUT + "/gf_ac_%s.dat" % tag)
    d["ro_ohm"] = float(1.0 / (2 * np.pi * d["pole_hz"] * 1e-12))
    d["lambda_ohm_inv"] = float(1.0 / (d["ro_ohm"] * I_BIAS))
    d["a0_gm_ro"] = float(d["gm_fd"] * d["ro_ohm"])
    # MIM: |i| = omega*C at 1 MHz with a 1 V AC drive
    fc, ic = load_ac(OUT + "/gf_cap_%s.dat" % tag)
    d["c_mim_f"] = float(np.mean(ic) / (2 * np.pi * CAP_F_AC))
    d["c_mim_nominal_f"] = (CAP_L * 1e6) * (CAP_W * 1e6) * 2e-15
    d["temp_c"] = 27.0
    d["vdd_v"] = f["vdd"]
    return d


def parse_mc(out, fam):
    v = _row(out, "MCOP", 5)
    g = _row(out, "MCGN", 1)[0]
    gm = _row(out, "MCGM", 2)
    return dict(vd0_offset=v[0], ratio_mis=v[1], ratio4_mis=v[2],
                vdiode=v[3], vb_tg=v[4],
                ron_tg=(FAMILIES[fam]["vcm"] - v[4]) / 2e-6,
                gain_diff=g, gm_fd=gm[0], id_pole=gm[1])


# ---------------------------------------------------------------------------
def extract_all():
    t0 = time.time()
    res = dict(kind="gf180_extract", meta=dict(
        pdk="gf180mcu", version=os.path.basename(os.path.dirname(GF)),
        i_bias_a=I_BIAS, w_um=W_INT * 1e6, l_um=L_INT * 1e6,
        noise_band=(1e3, 1e9), n_mc=N_MC_SEEDS, mc_seed_base=MC_SEED_BASE),
        corners={}, mc={}, noise={}, pvt={}, joint={}, parasitics=None,
        transient=None, mm_constants=_parse_mm_constants(),
        notes=[], blocks={})

    def block(name, ok, detail="", required=True):
        res["blocks"][name] = dict(status=("ok" if ok else "failed"),
                                   required=required, detail=detail)
        return ok

    def fam_tag(fam, sec, tag):
        return "%s_%s" % (fam, tag)

    # ---- corners + noise (6 runs) ------------------------------------------
    for fam in FAMILIES:
        for tag, sec in FAM_SECTIONS:
            ft = fam_tag(fam, sec, tag)
            print("== gf180 corner %s (%s/%s)" % (ft, fam, sec))
            try:
                out = run(ft, netlist_corner(fam, sec, ft))
                d = parse_corner(out, fam, ft)
                res["corners"][ft] = d
                # Band-integrated noise rows (see the netlist comment):
                # S_v = onoise_total^2/(f1-f0) over the FLAT band. The
                # resistive cells use 1 MHz..100 MHz (above the 1/f corner,
                # below the ~160 MHz Miller pole); the TG node is resistive
                # to >1 GHz so it uses 1 MHz..1 GHz.
                NB = {"nmos": (1e6, 1e9), "pmos": (1e6, 1e9),
                      "tg": (1e6, 1e9)}
                rl = FAMILIES[fam]["rl_a"]
                rout = rl * d["ro_ohm"] / (rl + d["ro_ohm"])
                a0 = d["gm_fd"] * rout
                rows = {}
                for line in out.splitlines():
                    if line.startswith("NROW"):
                        v = line.split()
                        rows[v[1]] = (float(v[2]), float(v[3]))
                def plateaus(wide, low):
                    """A + B/f least-squares from two band means: a residual
                    1/f shows up as a higher density in the narrow band."""
                    (v_w, _), (v_l, _) = wide, low
                    b0, b1m, b2 = 1e6, 1e7, 1e8
                    s1 = v_l ** 2 / (b1m - b0)          # mean over 1..10 MHz
                    s2 = v_w ** 2 / (b2 - b0)           # mean over 1..100 MHz
                    c1 = np.log(b1m / b0) / (b1m - b0)
                    c2 = np.log(b2 / b0) / (b2 - b0)
                    bcoef = (s1 - s2) / (c1 - c2)
                    a = s1 - bcoef * c1
                    return max(a, 0.0), max(bcoef, 0.0), s1, s2

                for prim, (b0, b1) in NB.items():
                    vo, vi = rows[prim]
                    sv = vo ** 2 / (b1 - b0)
                    ent = dict(band_hz=(b0, b1), onoise_total_v=vo,
                               inoise_total_v=vi, sv_band_mean_v2_per_hz=sv)
                    if prim in ("nmos", "pmos"):
                        a_fl, b_fl, s1, s2 = plateaus((vo, vi),
                                                      rows[prim + "_low"])
                        ent["sv_white_v2_per_hz"] = a_fl
                        ent["flicker_b_v2_hz"] = b_fl
                        ent["flicker_corner_hz"] = (b_fl / a_fl
                                                    if a_fl > 0 else None)
                        ent["band_density_ratio_lo_hi"] = s1 / s2
                    if prim == "tg":
                        ent["ron_ohm"] = d["ron_tg"]
                        ent["white_factor_4kTRon"] = sv / (FOUR_KT *
                                                           d["ron_tg"])
                    else:
                        gm = (d["gm_fd"] if prim == "nmos"
                              else d["gm_fd"] * d["id_p"] / max(d["id_a"],
                                                                1e-18))
                        # subtract the load in the OUTPUT domain, but use the
                        # 1/f-free plateau as the thermal floor
                        si = max((ent["sv_white_v2_per_hz"]
                                  - FOUR_KT * rout) / a0 ** 2, 0.0)
                        ent["a0_gm_rout"] = a0
                        ent["rout_ohm"] = rout
                        ent["si_white_a2_per_hz"] = si
                        ent["sv_4ktRout"] = FOUR_KT * rout
                        ent["density_in_nv_rthz"] = 1e9 * np.sqrt(si)
                        ent["gm_fd"] = gm
                        white = float(si * gm / FOUR_KT)
                        ent["gamma_eff_whiteband"] = white
                        ent["gamma_eff"] = dict(
                            white_factor=white,
                            gamma_eff_vs_assumed=white / (2.0 / 3.0),
                            gamma_assumed=2.0 / 3.0, si=si, gm=gm,
                            note=("white_factor is vs the gamma=1 ideal "
                                  "(4kT/gm); gamma_eff_vs_assumed vs the "
                                  "MOS 2/3 derived notation" +
                                  ("; pmos gm is a scaled proxy "
                                   "(gm_n*Id_p/Id_n), not a direct sweep"
                                   if prim == "pmos" else "")))
                    res["noise"].setdefault(prim, {})[ft] = ent
            except Exception as e:
                res["notes"].append("corner %s FAILED: %r" % (ft, e))
                print("   FAILED: %r" % (e,))

    # ---- mismatch MC (48 seeds x 2 families, deterministic per seed) -------
    def mc_jobs():
        jobs = []
        for fam in FAMILIES:
            for k in range(N_MC_SEEDS):
                jobs.append((fam, MC_SEED_BASE + k, "%s_s%02d" % (fam, k)))
            jobs.append((fam, MC_SEED_BASE, "%s_s00_rep" % fam))  # determinism
        def one(job):
            fam, seed, tag = job
            try:
                out = run(tag, netlist_mc(fam, seed, tag))
                return fam, tag, parse_mc(out, fam)
            except Exception as e:                      # fail-closed, per draw
                res["notes"].append("mc %s FAILED: %r" % (tag, e))
                return fam, tag, None
        with ThreadPoolExecutor(max_workers=6) as ex:
            done = list(ex.map(one, jobs))
        for fam in FAMILIES:
            rows = [(t, d) for f2, t, d in done if f2 == fam
                    and not t.endswith("_rep") and d is not None]
            rep_rows = [d for f2, t, d in done if f2 == fam
                        and t.endswith("_rep") and d is not None]
            if len(rows) < max(8, int(0.9 * N_MC_SEEDS)) or not rep_rows:
                res["notes"].append("mc %s: only %d/%d draws parsed"
                                    % (fam, len(rows), N_MC_SEEDS))
                continue
            rows.sort()
            arrs = {k: np.array([d[k] for _, d in rows]) for k in
                    ("vd0_offset", "ratio_mis", "ratio4_mis", "ron_tg",
                     "gain_diff", "gm_fd", "id_pole", "vb_tg")}
            mc = _mc_summary(arrs, len(rows))
            # determinism: seed-1 rerun must reproduce seed-1 rows exactly
            rep = rep_rows[0]
            first = dict(rows[0][1])
            rep_ok = all(np.isclose(rep[k2], first[k2], rtol=1e-12, atol=0)
                         for k2 in first)
            mc["determinism_ok"] = bool(rep_ok)
            mc["n"] = len(rows)
            # Pelgrom prediction for the diff-pair offset
            sv = sigma_vth(FAMILIES[fam]["dev_n"], W_INT, L_INT)
            gain = float(np.mean(arrs["gain_diff"]))
            mc["sigma_vth_pred_v"] = float(sv)
            # vd0 = v(dc1)-v(dc2) is the differential OUTPUT offset, while
            # gain_diff is differential-in -> single-ended-out, so the
            # output-referred sigma is 2*|A_se|*sigma(Vos_in) and the pair's
            # input offset is sqrt(2)*sigma_vth (two independent draws)
            mc["vd0_pred_v"] = float(2.0 * abs(gain) * np.sqrt(2.0) * sv)
            mc["vd0_ratio_meas_over_pred"] = float(
                abs(mc["vd0_offset"]["sigma"]) / mc["vd0_pred_v"])
            res["mc"][fam] = mc
            print("== gf180 MC %s: %d samples, det=%s, vd0 sigma=%.2fmV "
                  "(pred %.2fmV, x%.2f)"
                  % (fam, len(rows), rep_ok,
                     mc["vd0_offset"]["sigma"] * 1e3,
                     mc["vd0_pred_v"] * 1e3,
                     mc["vd0_ratio_meas_over_pred"]))
        return True

    # ---- PVT + joint + parasitics + tran (trimmed, parallel) ---------------
    def pvt_jobs():
        def one(job):
          try:
            fam, temp_c, vdd = job
            tag = "pvt_%s_t%d_v%d" % (fam, int(temp_c), int(vdd * 10))
            sec = "typical"
            out = run(tag, netlist_pvt(fam, sec, temp_c, vdd, tag))
            v = _row(out, "OPROW", 4)
            f = FAMILIES[fam]
            _vg, vdgg = _load_wrdata(OUT + "/gf_pvtgm_%s.dat" % tag)[0]
            gm_p = float(np.polyfit(_vg, (f["vdd"] - vdgg) / f["rl_a"], 1)[0])
            vb, vd2 = _load_wrdata(OUT + "/gf_gmsweep_%s.dat" % tag)[0]
            # resistive? PVT cell B has IDEAL load: the Vgb sweep moves Vds
            # -> intrinsic gain A = dVds/dVgs; gm needs Id... Id IS pinned
            # by the ideal source; gm(T) instead from cell A? cell A gate is
            # Vgac (dc 0) not swept. Use the AC pole for the leak and record
            # the intrinsic gain; gm(T) via the joint MC lane is TARGET.
            a_fd = float(np.polyfit(vb, vd2, 1)[0])
            pole = pole_interp(OUT + "/gf_ac_%s.dat" % tag)
            ro = 1.0 / (2 * np.pi * pole * 1e-12)
            return tag, dict(fam=fam, temp_c=temp_c, vdd_v=vdd,
                             mirror_ratio=v[0], vdiode=v[1], id_a=v[2],
                             v_d2b=v[3], gm_fd=gm_p,
                             a_intrinsic_fd=a_fd, pole_hz=pole,
                             ro_ohm_from_pole=ro,
                             lambda_from_pole=1.0 / (ro * I_BIAS),
                             bias_ok=bool(v[2] * f["rl_a"] <=
                                          f["vdd"] - 0.3))
          except Exception as e:
            res["notes"].append("pvt %r FAILED: %r" % (job, e))
            return None
        jobs = [(fam, t, v) for fam in FAMILIES for t in PVT_TEMPS
                for v in FAMILIES[fam]["vdds_pvt"]]
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = [r for r in ex.map(one, jobs) if r is not None]
        res["pvt"] = dict(temps_c=list(PVT_TEMPS),
                          rows={t: d for t, d in rows},
                          note=("temperature via ngspice .temp; nominal "
                                "process; gm(T) comes from the resistive-"
                                "load DC gate sweep (the corner netlist's "
                                "gm cell), not from a fixed-Id Vds slope"))
        return len(rows)

    def joint_jobs():
        def one(job):
          try:
            fam, (sec, temp_c) = job
            tag = "jt_%s_%s_%s" % (fam, sec, ("m40" if temp_c < 0 else "p125"))
            f = FAMILIES[fam]
            out = run(tag, netlist_pvt(fam, sec, temp_c, f["vdd"], tag))
            v = _row(out, "OPROW", 4)
            f = FAMILIES[fam]
            _vg, vdgg = _load_wrdata(OUT + "/gf_pvtgm_%s.dat" % tag)[0]
            gm_p = float(np.polyfit(_vg, (f["vdd"] - vdgg) / f["rl_a"], 1)[0])
            vb, vd2 = _load_wrdata(OUT + "/gf_gmsweep_%s.dat" % tag)[0]
            a_fd = float(np.polyfit(vb, vd2, 1)[0])
            pole = pole_interp(OUT + "/gf_ac_%s.dat" % tag)
            ro = 1.0 / (2 * np.pi * pole * 1e-12)
            return tag, dict(fam=fam, section=sec, temp_c=temp_c,
                             vdd_v=f["vdd"], mirror_ratio=v[0],
                             vdiode=v[1], id_a=v[2], v_d2b=v[3],
                             gm_fd=gm_p, a_intrinsic_fd=a_fd, pole_hz=pole,
                             ro_ohm_from_pole=ro,
                             lambda_from_pole=1.0 / (ro * I_BIAS))
          except Exception as e:
            res["notes"].append("joint %r FAILED: %r" % (job, e))
            return None
        jobs = [(fam, j) for fam in FAMILIES for j in JOINT]
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows = [r for r in ex.map(one, jobs) if r is not None]
        res["joint"] = dict(rows={t: d for t, d in rows},
                            note=("corner lib AND .temp in the SAME run; "
                                  "P(joint) consumption applies the TT "
                                  "relative mismatch draws on top "
                                  "(declared composition)"))
        return len(rows)

    def paras_jobs():
        def one(c_ff):
            tag = "paras_c%d" % int(c_ff)
            try:
                run(tag, netlist_paras(c_ff, tag))
                return c_ff, pole_interp(OUT + "/gf_ac_%s.dat" % tag)
            except Exception as e:
                res["notes"].append("paras %s FAILED: %r" % (tag, e))
                return None
        with ThreadPoolExecutor(max_workers=3) as ex:
            got = [r for r in ex.map(one, list(PARAS_C_FF)) if r is not None]
        got.sort()
        f0 = got[0][1]
        entries = []
        for c_ff, pole in got:
            rel = (pole - f0) / f0
            entries.append(dict(c_added_ff=c_ff, pole_hz=pole,
                                pole_rel_shift=rel,
                                pole_rel_shift_per_ff=rel / c_ff
                                if c_ff > 0 else 0.0))
        dc_f = PARAS_C_FF[-1] - PARAS_C_FF[0]
        c0 = 1e-12 + PARAS_C_FF[0] * 1e-15
        c1 = 1e-12 + PARAS_C_FF[-1] * 1e-15
        slope_meas = np.log(got[-1][1] / got[0][1]) / (dc_f * 1e-15)
        slope_pred = -np.log(c1 / c0) / (dc_f * 1e-15)
        res["parasitics"] = dict(
            node="d2b", c_bank_ff=1000.0, cases=entries,
            slope_log_per_f=float(slope_meas),
            slope_log_per_f_pred=float(slope_pred),
            slope_rel_err=float(slope_meas / slope_pred - 1.0),
            label="pre-PEX sensitivity estimate, not a PEX result",
            note=("pole = 1/(ro*C_tot) on the ideal-load cell; the exact "
                  "two-point log slope is the prediction"))
        return len(got)

    def tran_job():
        out = run("tran", netlist_tran())
        del out
        d = np.atleast_2d(np.loadtxt(OUT + "/gf_tran_fast_step.dat"))
        t = d[:, 0]
        fast = dict(gm_cell=dict(t0_s=TRAN["step_t0"], in_v=TRAN["step_v"],
                                 **_settle(t, d[:, 1], TRAN["step_t0"],
                                           TRAN["settle_tol"])),
                    dt_s=TRAN["dt"], tstop_s=TRAN["step_tstop"])
        v = d[:, 1]
        m = t >= TRAN["step_t0"]
        fast["gm_cell"]["slew_v_per_s"] = float(np.max(np.abs(
            np.gradient(v[m], t[m]))))
        d2 = np.atleast_2d(np.loadtxt(OUT + "/gf_tran_fast_sine.dat"))
        thd = dict(gm_cell=dict(in_v=TRAN["sine_v"],
                                **_thd(d2[:, 0], d2[:, 1], TRAN["f0"],
                                       TRAN["thd_harmonics"])),
                   dt_s=TRAN["dt"], tstop_s=TRAN["sine_tstop"],
                   fft="python Hann-windowed rFFT on ngspice wrdata output")
        d3 = np.atleast_2d(np.loadtxt(OUT + "/gf_tran_slow_step.dat"))
        slow = dict(pole_cell=dict(
            t0_s=TRAN["slow_t0"], in_v=TRAN["slow_step_v"],
            **_settle(d3[:, 0], d3[:, 1], TRAN["slow_t0"], 0.02)),
            dt_s=TRAN["slow_dt"], tstop_s=TRAN["slow_tstop"],
            note="settling of the ro*C integrator node (tau = ro*C)")
        res["transient"] = dict(fast_step=fast, fast_sine=thd, slow_step=slow,
                                stimulus=dict(TRAN, vg_op_fixed_v=0.765,
                                              note="vg_op from "
                                                   "device_extract.json "
                                                   "diode point @10uA"),
                                fam="3v3")
        return True

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_mc = ex.submit(mc_jobs)
        f_pvt = ex.submit(pvt_jobs)
        f_jnt = ex.submit(joint_jobs)
        n_par = paras_jobs()
        ok_tran = tran_job()
        try:
            ok_mc = f_mc.result()
        except Exception as e:
            res["notes"].append("mc block FAILED: %r" % e)
            ok_mc = False
        try:
            n_pvt = f_pvt.result()
        except Exception as e:
            res["notes"].append("pvt block FAILED: %r" % e)
            res["pvt"] = dict(rows={}, temps_c=list(PVT_TEMPS))
            n_pvt = 0
        try:
            n_jnt = f_jnt.result()
        except Exception as e:
            res["notes"].append("joint block FAILED: %r" % e)
            res["joint"] = dict(rows={})
            n_jnt = 0

    # ---- fail-closed blocks -------------------------------------------------
    need_c = {"%s_%s" % (f, t) for f in FAMILIES for t, _ in FAM_SECTIONS}
    ok_corners = need_c <= set(res["corners"])
    block("corners", ok_corners, "%s" % sorted(res["corners"]))
    for prim in ("nmos", "pmos"):
        ok_n = all("%s_tt" % f in res["noise"].get(prim, {})
                   and "gamma_eff" in res["noise"][prim]["%s_tt" % f]
                   and res["noise"][prim]["%s_tt" % f].get(
                       "gamma_eff", {}).get("white_factor", 0) > 0
                   for f in FAMILIES)
        block("%s_noise" % prim, ok_n,
              "gamma_eff tt: " + "; ".join(
                  "%.3f white" % res["noise"][prim]["%s_tt" % f]
                  ["gamma_eff"]["white_factor"] for f in FAMILIES
                  if "%s_tt" % f in res["noise"].get(prim, {})
                  and "gamma_eff" in res["noise"][prim]["%s_tt" % f]))
    tg_ok = all("white_factor_4kTRon" in res["noise"].get("tg", {})
                .get("%s_tt" % f, {}) and
                0.5 < res["noise"]["tg"]["%s_tt" % f]["white_factor_4kTRon"]
                < 2.0 for f in FAMILIES)
    block("tg_noise_4ktron", tg_ok,
          "; ".join("%.3f" % res["noise"]["tg"]["%s_tt" % f]
                    ["white_factor_4kTRon"] for f in FAMILIES
                    if "white_factor_4kTRon" in
                    res["noise"]["tg"].get("%s_tt" % f, {})))
    det_ok = all(res["mc"].get(f, {}).get("determinism_ok") for f in FAMILIES)
    n_mc_ok = all(res["mc"].get(f, {}).get("n") == N_MC_SEEDS for f in FAMILIES)
    pel_ok = all(0.5 < res["mc"].get(f, {}).get("vd0_ratio_meas_over_pred", 0)
                 < 2.0 for f in FAMILIES)
    block("mismatch_mc", bool(ok_mc and det_ok and n_mc_ok),
          "48x2 samples, determinism=%s" % det_ok)
    block("pelgrom_offset", pel_ok,
          "; ".join("x%.2f" % res["mc"][f]["vd0_ratio_meas_over_pred"]
                    for f in FAMILIES if f in res["mc"]))
    block("pvt", n_pvt == 18, "%d/18 PVT combos" % n_pvt)
    block("joint_corners", n_jnt == 8, "%d/8 joint points" % n_jnt)
    block("parasitics", res["parasitics"] is not None
          and len(res["parasitics"]["cases"]) == 3,
          "slope rel err %.3f" % (res["parasitics"]["slope_rel_err"]
                                  if res["parasitics"] else float("nan")))
    block("transient", bool(ok_tran), "fast settle+THD + slow ro*C settling")

    failed = [k for k, b in res["blocks"].items()
              if b["status"] == "failed" and b["required"]]
    res["status"] = "failed" if len(failed) > 2 else (
        "partial" if failed else "complete")
    res["failed_blocks"] = failed
    res["families"] = {k: dict(v) for k, v in FAMILIES.items()}
    res["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT + "/gf180_extract.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("-> results/gf180_extract.json  status=%s (%.1fs)"
          % (res["status"], res["runtime_s"]))
    if failed:
        print("   FAILED BLOCKS: %s" % failed)
    return res


def main(argv):
    extract_all()


if __name__ == "__main__":
    main(sys.argv)
