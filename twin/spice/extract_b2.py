"""B2 pre-silicon primitive extraction: .noise, corners vs Monte Carlo,
PVT, transient, pre-PEX parasitic sensitivity, fail-closed status.

Closes the B2 obligations on IHP SG13G2 (DESIGN.md §7 Phase 7.5 / §9 risk 3):

(1) Real ngspice `.noise` on behavioral primitives (NOT the diode mirror of
    extract.py) -> gamma_eff = measured_PSD / theoretical. The §6.9 noise
    column was DERIVED from 4kTγ/g_m; this module measures it.
(2) Corner (global process: mos_tt/ss/ff + hbt_typ/wcs/bcs) run SEPARATELY
    from Monte Carlo (local Pelgrom mismatch: mos_tt_mismatch + mm_ok=1,
    48 setseed/mc_source samples in one ngspice process), then compared.
(3) PVT: temperature (-40/27/125 °C via ngspice `.temp`, global `temper`)
    x supply ±10% (1.35/1.5/1.65 V) on the MOS gm-cell and HBT pair.
(4) Transient: .tran step + sine on the HBT diff pair and MOS gm-cell;
    settling + slew measured in time domain, THD by python FFT on the
    wrdata output (amplitudes/frequencies recorded in the JSON).
(5) Pre-PEX parasitic sensitivity: no layout exists for the primitives, so
    PEX is impossible pre-layout; instead an estimated wire/load C
    (0/10/100 fF) is added at the integrator output node and the pole is
    re-extracted -> pole shift per added fF. Label: "pre-PEX sensitivity
    estimate, not a PEX result".

Fail-closed contract: the JSON carries top-level "status"
("complete"|"partial"|"failed") plus per-block "blocks{}" statuses. A
required block that fails to run or parse forces status != "complete";
bench/run_b2.py reports that as a genuine failure. A documented
KNOWN_LIMITATION (PSP103 OSDI device noise) is a *skip*, distinct from a
failure.

Primitives (B0 tile spec, §7 Phase 7.5):
  * HBT diff pair (npn13G2, 20 µA/side, 10k loads) — the interface gm-stage
    + input-referred offset/noise/gain.  [T2/T5 interface]
  * HBT current mirror (npn13G2, 20 µA, Vce sweep) — mirror ratio + ro.
  * LV MOS gm-cell (sg13_lv_nmos 10u/0.13u @ 10 µA, current-source load)
    — the state integrator transconductor: gm + .noise gamma_eff.  [T1]
  * LV MOS gm-cell copy with Rl≈ro load + 1 pF — measured AC pole (gm/C).
  * Transmission gate (2u nmos ∥ 4u pmos @ 10 µA) — T7/T6 parasitic Ron
    + Ron mismatch + 4kT·Ron noise sanity check.

Honesty contract: if a model's .noise fails, the failure is recorded, no
number is fabricated. All numbers land in results/b2_extract.json.

Reuse of extract.py conventions: run() cwd = PDK model dir (relative
.includes), env -u LD_LIBRARY_PATH is applied by the caller shell wrapper;
IHP MOS needs PDK_ROOT/PDK (PSP103 OSDI). W/L in SI-with-suffix.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(os.path.dirname(HERE)) + "/results"
CIEL = os.path.expanduser("~/.ciel/ciel")
NL = chr(10)
IHP = sorted(__import__("glob").glob(CIEL + "/ihp-sg13/versions/*/ihp-sg13g2"))[0]
MODELS = IHP + "/libs.tech/ngspice/models"

K_B = 1.380649e-23
T_K = 300.0
Q_E = 1.602176634e-19
VT = K_B * T_K / Q_E          # 25.85 mV
FOUR_KT = 4.0 * K_B * T_K     # 1.66e-20 J

# MC depth: local Pelgrom mismatch bumped 24 -> 48 samples so the sigma
# estimates tighten; sigma_stderr = sigma/sqrt(2(N-1)) is reported next to
# each sigma. The global-process MC stays at 24 (it exists only to show the
# separation from mismatch, not to be a production statistic).
N_MC_LOCAL = 48
N_MC_GLOBAL = 24
MC_SEED = 1234567

# PVT grid: ngspice `.temp` sets the global `temper`. Both model stacks
# reference it (VBIC expressions use `(temper+273)/300`; PSP103 uses the
# simulator temperature against its tnom=27 parameter reference), so the
# variation is real model temperature, NOT a re-extraction of the nominal
# parameter set (tnom stays pinned at 27 °C in the model cards — that is
# how the PDK ships it, recorded honestly).
PVT_TEMPS = (-40.0, 27.0, 125.0)
PVT_VDDS = (1.35, 1.50, 1.65)     # ±10% around 1.5 V
PVT_MOS_SEC, PVT_HBT_SEC = "mos_tt", "hbt_typ"

# Transient stimulus (recorded verbatim in the JSON so the behavioral-model
# lane can reproduce the comparison point). THD frequency is 10 MHz for both
# primitives: with maxstep=1 ns the tran completes in seconds on this loaded
# box (a 100 MHz PSP tran crawls to minutes — measured), and the 5th
# harmonic (50 MHz) still gets ~20 print points/period on the uniform
# linearize grid.
TRAN = dict(
    hbt_step_v=10e-3, hbt_sine_v=10e-3, hbt_f0=10e6,
    mos_step_v=1e-3, mos_sine_v=1e-3, mos_f0=10e6,
    step_t0=20e-9, step_tstop=400e-9, sine_tstop=1e-6,   # 10 periods @10 MHz
    dt=0.2e-9, maxstep=1e-9,
    settle_tol=0.01, thd_harmonics=5,
)

# Pre-PEX parasitic probe: added C at the integrator output node (d2b,
# already loaded with the 1 pF C_bank). Pole(fF) is re-extracted per value.
PARAS_C_FF = (0.0, 10.0, 100.0)

# corner = (mos lib section, hbt lib section, tag, mismatch_instances)
# Global process corners: plain corner libs, mm_ok=0 (devices track together).
CORNERS = [
    ("mos_tt", "hbt_typ", "tt", False),
    ("mos_ss", "hbt_typ", "ss", False),
    ("mos_ff", "hbt_typ", "ff", False),
    ("mos_tt", "hbt_wcs", "hbtwcs", False),
    ("mos_tt", "hbt_bcs", "hbtbcs", False),
]
# Two DIFFERENT Monte Carlo mechanisms (kept separate on purpose):
#  * "mc"       = LOCAL Pelgrom mismatch: sg13g2_moslv_mod_mismatch.lib's
#                 per-instance agauss delvto/factuo (sigma ~ 1/sqrt(WL)) and
#                 npn13G2's mm_ok agauss (is/ibei/re/...) -> mm_ok=1 on every
#                 instance. Per-device draws: Q3 and Q4 differ.
#  * "mcproc"   = GLOBAL process MC: sg13g2_moslv_stat.lib / hbt_stat.lib
#                 gauss() params are GLOBAL (.param) — every device gets the
#                 same draw. NOT mismatch. Kept to show the separation.
MC_RUN = ("mos_tt_mismatch", "hbt_typ", "mc", True)
MC_PROC_RUN = ("mos_tt_stat", "hbt_typ_stat", "mcproc", False)


def netlist(mos_sec, hbt_sec, tag, mc=False, mismatch=False,
            n_mc=N_MC_LOCAL):
    """One file, all primitives. mc=True adds the reset-loop MC sampler."""
    mm = " mm_ok=1" if mismatch else ""
    p = [
        "* B2 primitives: HBT diff/mirror, LV MOS gm-cell x2, transmission gate",
        '.lib "%s/cornerMOSlv.lib" %s' % (MODELS, mos_sec),
        '.lib "%s/cornerHBT.lib" %s' % (MODELS, hbt_sec),
        "* --- HBT diff pair: 20uA/side, 10k loads (the interface gm-stage)",
        "Vdd vdd 0 1.5",
        "Itail tail 0 dc 40u",
        "XQ3 dc1 db tail 0 npn13G2%s" % mm,
        "XQ4 dc2 dcm tail 0 npn13G2%s" % mm,
        "Rc1 vdd dc1 10k",
        "Rc2 vdd dc2 10k",
        "Vcm dcm 0 0.8",
        "Vd db dcm dc 0 ac 1",
        "* --- HBT current mirror: 20uA, out at 0.9V",
        "Iref vdd cmin dc 20u",
        "XQ1 cmin cmin 0 0 npn13G2%s" % mm,
        "XQ2 cmout cmin 0 0 npn13G2%s" % mm,
        "Vh cmout 0 dc 0.9",
        "* --- LV MOS gm-cell A (noise): 10u/0.13u, 10uA current-source load",
        "Vg g 0 dc 0.39 ac 1",
        "XMn da g 0 0 sg13_lv_nmos w=10u l=0.13u%s" % mm,
        "Ila vdd da 0 dc 10u",
        "* --- LV MOS gm-cell B (pole): Rl=65k~ro load + 1pF, Vgs=0.3837 = the",
        "*     10uA diode point (saturation; verified vs isolated diode sweep)",
        "Vg2 g2 0 dc 0.3837 ac 1",
        "XMn2 d2b g2 0 0 sg13_lv_nmos w=10u l=0.13u%s" % mm,
        "Rl2 vdd d2b 65k",
        "C1 d2b 0 1p",
        "* --- transmission gate: 2u nmos // 4u pmos, 10uA probe",
        "Vtg a 0 dc 0.75 ac 1",
        "XMtn b gtn a 0 sg13_lv_nmos w=2u l=0.13u%s" % mm,
        "XMtp b gtp a 0 sg13_lv_pmos w=4u l=0.13u%s" % mm,
        "Vgn gtn 0 1.5",
        "Vgp gtp 0 0",
        "Itg b 0 dc 10u",
        ".control",
        "set seed = %d" % MC_SEED,
        "op",
        "let ratio = -i(vh)/20u",
        "let gmos = @n.xmn.nsg13_lv_nmos[gm]",
        "let gmos2 = @n.xmn2.nsg13_lv_nmos[gm]",
        "let vd0 = v(dc1)-v(dc2)",
        "let vdc1 = v(dc1)",
        "let vb = v(b)",
        "let idb = (1.5-v(d2b))/65k",
        "echo OPROW $&ratio $&gmos $&gmos2 $&vd0 $&vdc1 $&vb $&idb",
    ]
    if not mc:
        p += [
            "dc Vd -0.01 0.01 0.0002",
            "wrdata %s/b2_diff_%s.dat v(dc1) v(dc2)" % (OUT, tag),
            "dc Vh 0.3 1.2 0.01",
            "wrdata %s/b2_mirror_%s.dat i(vh) v(cmin)" % (OUT, tag),
            "dc Vg2 0.3 0.7 0.005",
            "wrdata %s/b2_gmsweep_%s.dat v(d2b)" % (OUT, tag),
            "ac dec 20 1e3 1e9",
            "wrdata %s/b2_ac_%s.dat v(d2b)" % (OUT, tag),
            "noise v(da) Vg dec 20 1e3 1e9",
            "setplot noise3",
            "wrdata %s/b2_noise_mos_%s.dat onoise_spectrum inoise_spectrum" % (OUT, tag),
            "noise v(dc1) Vd dec 20 1e3 1e9",
            "setplot noise5",
            "wrdata %s/b2_noise_hbt_%s.dat onoise_spectrum inoise_spectrum" % (OUT, tag),
            "noise v(b) Vtg dec 20 1e3 1e9",
            "setplot noise7",
            "wrdata %s/b2_noise_tg_%s.dat onoise_spectrum inoise_spectrum" % (OUT, tag),
        ]
    else:
        # Per-instance Pelgrom mismatch re-draw, following the PDK's own
        # gnucap/tests/ngspice recipe (verified to reproduce their reference
        # .out exactly on this ngspice-44.2): setseed + mc_source + op.
        # NOTE: a plain `reset` loop does NOT re-draw the per-instance
        # agauss() params (measured: matched-pair dVgs/dVbe identically 0).
        p += [
            "let i = 0",
            "dowhile i < %d" % n_mc,
            "  let seed = i + 1",
            "  setseed $&seed",
            "  mc_source",
            "  op",
            "  let ratio = -i(vh)/20u",
            "  let vd0 = v(dc1)-v(dc2)",
            "  let vdc1 = v(dc1)",
            "  let vb = v(b)",
            "  echo MCOP $&ratio $&vd0 $&vdc1 $&vb",
            "  dc Vd -0.005 0.005 0.001",
            "  let gain = (v(dc1)[10]-v(dc1)[0])/0.01",
            "  echo MCGN $&gain",
            "  dc Vg2 0.3737 0.3937 0.01",
            "  let i0 = (1.5-v(d2b)[0])/65k",
            "  let i1 = (1.5-v(d2b)[1])/65k",
            "  let i2 = (1.5-v(d2b)[2])/65k",
            "  let gmc = (i2-i0)/0.02",
            "  echo MCGM $&gmc $&i1",
            "  let i = i + 1",
            "end",
        ]
    p += [".endc", ".end", ""]
    return NL.join(p)


def netlist_pvt(temp_c, vdd, tag):
    """Trimmed PVT netlist: op row + gm sweep + AC pole. No noise sweeps
    (PSP noise is a documented no-op; HBT PVT noise is not a B2 target).
    tag makes the .dat names unique (runs execute concurrently)."""
    p = [
        "* B2 PVT probe: temp=%g C, vdd=%g V (mos_tt/hbt_typ)" % (temp_c, vdd),
        '.lib "%s/cornerMOSlv.lib" mos_tt' % MODELS,
        '.lib "%s/cornerHBT.lib" hbt_typ' % MODELS,
        ".temp %g" % temp_c,
        "Vdd vdd 0 %g" % vdd,
        "Itail tail 0 dc 40u",
        "XQ3 dc1 db tail 0 npn13G2",
        "XQ4 dc2 dcm tail 0 npn13G2",
        "Rc1 vdd dc1 10k",
        "Rc2 vdd dc2 10k",
        "Vcm dcm 0 0.8",
        "Vd db dcm dc 0 ac 1",
        "Iref vdd cmin dc 20u",
        "XQ1 cmin cmin 0 0 npn13G2",
        "XQ2 cmout cmin 0 0 npn13G2",
        "Vh cmout 0 dc 0.9",
        "Vg g 0 dc 0.39 ac 1",
        "XMn da g 0 0 sg13_lv_nmos w=10u l=0.13u",
        "Ila vdd da 0 dc 10u",
        "Vg2 g2 0 dc 0.3837 ac 1",
        "XMn2 d2b g2 0 0 sg13_lv_nmos w=10u l=0.13u",
        "Rl2 vdd d2b 65k",
        "C1 d2b 0 1p",
        "Vtg a 0 dc 0.75 ac 1",
        "XMtn b gtn a 0 sg13_lv_nmos w=2u l=0.13u",
        "XMtp b gtp a 0 sg13_lv_pmos w=4u l=0.13u",
        "Vgn gtn 0 %g" % vdd,
        "Vgp gtp 0 0",
        "Itg b 0 dc 10u",
        ".control",
        "op",
        "let ratio = -i(vh)/20u",
        "let gmos = @n.xmn.nsg13_lv_nmos[gm]",
        "let gmos2 = @n.xmn2.nsg13_lv_nmos[gm]",
        "let vd0 = v(dc1)-v(dc2)",
        "let vdc1 = v(dc1)",
        "let vb = v(b)",
        "let idb = (1.5-v(d2b))/65k",
        "echo OPROW $&ratio $&gmos $&gmos2 $&vd0 $&vdc1 $&vb $&idb",
        "dc Vg2 0.3 0.7 0.005",
        "wrdata %s/b2_gmsweep_%s.dat v(d2b)" % (OUT, tag),
        "ac dec 20 1e3 1e9",
        "wrdata %s/b2_ac_%s.dat v(d2b)" % (OUT, tag),
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


def netlist_paras(c_f, tag):
    """Parasitic-sensitivity probe: nominal TT with an ADDED C at the
    integrator output node d2b (in addition to the 1 pF C_bank). Unique
    .dat name per case (runs execute concurrently)."""
    extra = "" if c_f <= 0 else "Cpar d2b 0 %gf\n" % c_f
    p = [
        "* B2 pre-PEX parasitic probe: added C = %g fF at d2b" % c_f,
        '.lib "%s/cornerMOSlv.lib" mos_tt' % MODELS,
        '.lib "%s/cornerHBT.lib" hbt_typ' % MODELS,
        "Vdd vdd 0 1.5",
        "Itail tail 0 dc 40u",
        "XQ3 dc1 db tail 0 npn13G2",
        "XQ4 dc2 dcm tail 0 npn13G2",
        "Rc1 vdd dc1 10k",
        "Rc2 vdd dc2 10k",
        "Vcm dcm 0 0.8",
        "Vd db dcm dc 0 ac 1",
        "Iref vdd cmin dc 20u",
        "XQ1 cmin cmin 0 0 npn13G2",
        "XQ2 cmout cmin 0 0 npn13G2",
        "Vh cmout 0 dc 0.9",
        "Vg g 0 dc 0.39 ac 1",
        "XMn da g 0 0 sg13_lv_nmos w=10u l=0.13u",
        "Ila vdd da 0 dc 10u",
        "Vg2 g2 0 dc 0.3837 ac 1",
        "XMn2 d2b g2 0 0 sg13_lv_nmos w=10u l=0.13u",
        "Rl2 vdd d2b 65k",
        "C1 d2b 0 1p",
        extra.rstrip("\n"),
        "Vtg a 0 dc 0.75 ac 1",
        "XMtn b gtn a 0 sg13_lv_nmos w=2u l=0.13u",
        "XMtp b gtp a 0 sg13_lv_pmos w=4u l=0.13u",
        "Vgn gtn 0 1.5",
        "Vgp gtp 0 0",
        "Itg b 0 dc 10u",
        ".control",
        "op",
        "ac lin 6001 1e5 2e7",
        "wrdata %s/b2_ac_%s.dat v(d2b)" % (OUT, tag),
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


def netlist_tran():
    """Transient stimulus. TWO independent circuit copies so one ngspice
    run does both analyses (wall-time): a step copy (settling + slew) and
    a sine copy (THD via python FFT on the wrdata output). Amplitudes and
    frequencies are the TRAN constants above and are written to the JSON."""
    dt_ns = TRAN["dt"] * 1e9
    p = [
        "* B2 transient: step + sine on HBT diff pair and MOS gm-cell",
        '.lib "%s/cornerMOSlv.lib" mos_tt' % MODELS,
        '.lib "%s/cornerHBT.lib" hbt_typ' % MODELS,
        "* --- step copy (suffix _s): HBT diff pair, 10 mV input step",
        "Vdd_s vdd_s 0 1.5",
        "Itail_s tail_s 0 dc 40u",
        "XQ3s dc1_s db_s tail_s 0 npn13G2",
        "XQ4s dc2_s dcm_s tail_s 0 npn13G2",
        "Rc1_s vdd_s dc1_s 10k",
        "Rc2_s vdd_s dc2_s 10k",
        "Vcm_s dcm_s 0 0.8",
        "Vd_s db_s dcm_s PULSE(0 %g %g 0.5n 0.5n %g %g)"
        % (TRAN["hbt_step_v"], TRAN["step_t0"],
           TRAN["step_tstop"] - TRAN["step_t0"], 2 * TRAN["step_tstop"]),
        "* --- step copy: MOS gm-cell B, 1 mV input step, 1 pF load",
        "Vg2_s g2_s 0 PULSE(0.3837 %g %g 0.5n 0.5n %g %g)"
        % (0.3837 + TRAN["mos_step_v"], TRAN["step_t0"],
           TRAN["step_tstop"] - TRAN["step_t0"], 2 * TRAN["step_tstop"]),
        "XMn2_s d2b_s g2_s 0 0 sg13_lv_nmos w=10u l=0.13u",
        "Rl2_s vdd_s d2b_s 65k",
        "C1_s d2b_s 0 1p",
        "* --- sine copy (suffix _n): HBT diff pair, %g mV @ %g MHz"
        % (TRAN["hbt_sine_v"] * 1e3, TRAN["hbt_f0"] * 1e-6),
        "Vdd_n vdd_n 0 1.5",
        "Itail_n tail_n 0 dc 40u",
        "XQ3n dc1_n db_n tail_n 0 npn13G2",
        "XQ4n dc2_n dcm_n tail_n 0 npn13G2",
        "Rc1_n vdd_n dc1_n 10k",
        "Rc2_n vdd_n dc2_n 10k",
        "Vcm_n dcm_n 0 0.8",
        "Vd_n db_n dcm_n SIN(0 %g %g)" % (TRAN["hbt_sine_v"], TRAN["hbt_f0"]),        "* --- sine copy: MOS gm-cell B, %g mV @ %g MHz"
        % (TRAN["mos_sine_v"] * 1e3, TRAN["mos_f0"] * 1e-6),
        "Vg2_n g2_n 0 SIN(0.3837 %g %g)" % (TRAN["mos_sine_v"], TRAN["mos_f0"]),
        "XMn2_n d2b_n g2_n 0 0 sg13_lv_nmos w=10u l=0.13u",
        "Rl2_n vdd_n d2b_n 65k",
        "C1_n d2b_n 0 1p",
        ".control",
        "tran %gn %gn 0 %gn" % (TRAN["dt"] * 1e9, TRAN["step_tstop"] * 1e9,
                                TRAN["maxstep"] * 1e9),
        "linearize",
        "wrdata %s/b2_tran_step.dat v(dc1_s) v(dc2_s) v(d2b_s)" % OUT,
        "tran %gn %gn 0 %gn" % (TRAN["dt"] * 1e9, TRAN["sine_tstop"] * 1e9,
                                TRAN["maxstep"] * 1e9),
        "linearize",
        "wrdata %s/b2_tran_sine.dat v(dc1_n) v(dc2_n) v(d2b_n)" % OUT,
        ".endc",
        ".end",
        "",
    ]
    return NL.join(p)


def run(tag, mos_sec, hbt_sec, mc=False, mismatch=False, timeout=1800,
        n_mc=N_MC_LOCAL, netlist_text=None):
    os.makedirs(OUT, exist_ok=True)
    path = OUT + "/b2_%s.sp" % tag
    if netlist_text is None:
        netlist_text = netlist(mos_sec, hbt_sec, tag, mc, mismatch, n_mc)
    with open(path, "w") as fh:
        fh.write(netlist_text)
    env = os.environ.copy()
    env["PDK_ROOT"] = os.path.dirname(IHP)
    env["PDK"] = os.path.basename(IHP)
    cmd = ["env", "-u", "LD_LIBRARY_PATH", "ngspice", "-b", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=MODELS, env=env)
    with open(OUT + "/b2_%s.log" % tag, "w") as fh:
        fh.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError("ngspice %s failed, see results/b2_%s.log" % (tag, tag))
    return r.stdout


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def _load_wrdata(path):
    d = np.atleast_2d(np.loadtxt(path))
    return [(d[:, k], d[:, k + 1]) for k in range(0, d.shape[1], 2)]


def load_ac(path):
    """wrdata writes complex AC vectors as (freq, re, im) columns."""
    d = np.atleast_2d(np.loadtxt(path))
    if d.shape[1] < 3:
        return d[:, 0], d[:, 1]
    return d[:, 0], np.abs(d[:, 1] + 1j * d[:, 2])


def ac_pole_hz(path):
    f, mag = load_ac(path)
    gdc = mag[0]
    k = int(np.argmin(np.abs(mag - gdc / np.sqrt(2))))
    return float(f[k])


def integrate_psd(freq, amp):
    """ngspice spectrum is amplitude density (V/sqrt(Hz)); integrate S^2 df."""
    f = np.asarray(freq, float)
    s = np.asarray(amp, float) ** 2
    m = (f >= 1e3) & (f <= 1e9)
    f, s = f[m], s[m]
    var = np.trapezoid(s, f)                   # full band 1k..1G
    w = f >= 1e8                               # white band 100M..1G
    var_white = np.trapezoid(s[w], f[w]) / (f[w][-1] - f[w][0]) * (1e9 - 1e3)
    return var, var_white


def parse_oprow(stdout):
    for line in stdout.splitlines():
        if line.startswith("OPROW"):
            vals = [float(x) for x in line.split()[1:]]
            keys = ["mirror_ratio", "gm_mos_param", "gm_mos2_param",
                    "vd0_offset_raw", "vdc1", "vb_tg", "id_pole"]
            d = dict(zip(keys, vals))
            # HBT gm derived from measured Ic (gm/IC = 1/VT, exact physics)
            d["ic_hbt"] = (1.5 - d["vdc1"]) / 10e3
            d["gm_hbt"] = d["ic_hbt"] / VT
            d["ron_tg"] = (0.75 - d["vb_tg"]) / 10e-6
            return d
    raise RuntimeError("no OPROW in output")


def parse_mcrows(stdout, n):
    """MC loop emits three rows per sample (each right after its analysis,
    so $&name resolves in the plot that owns the vectors):
      MCOP ratio vd0 vdc1 vb      (after op)
      MCGN gain                   (after dc Vd)
      MCGM gmc i1                 (after dc Vg2)
    """
    ops, gns, gms = [], [], []
    for line in stdout.splitlines():
        if line.startswith("MCOP"):
            ops.append([float(x) for x in line.split()[1:]])
        elif line.startswith("MCGN"):
            gns.append([float(x) for x in line.split()[1:]])
        elif line.startswith("MCGM"):
            gms.append([float(x) for x in line.split()[1:]])
    if not (len(ops) == len(gns) == len(gms) == n):
        raise RuntimeError("MC rows: op=%d gain=%d gm=%d, want %d"
                           % (len(ops), len(gns), len(gms), n))
    keys_op = ["mirror_ratio", "vd0_offset_raw", "vdc1", "vb_tg"]
    a = np.array(ops)
    out = {k: a[:, j] for j, k in enumerate(keys_op)}
    out["gain_diff"] = np.array([g[0] for g in gns])
    gma = np.array(gms)
    out["gm_mos_fd"] = gma[:, 0]
    out["id_mos_pole"] = gma[:, 1]
    out["ic_hbt"] = (1.5 - out["vdc1"]) / 10e3
    out["gm_hbt"] = out["ic_hbt"] / VT
    out["ron_tg"] = (0.75 - out["vb_tg"]) / 10e-6
    return out


BW_NOISE = 1e9 - 1e3            # integration band 1 kHz .. 1 GHz


def gamma_eff(gm, var_in, kind):
    """measured input-referred PSD var vs theoretical white-noise budget.

    var_in is the input-referred variance integrated over BW_NOISE;
    white factor = (var/BW) / (4kT/gm), i.e. vs the gamma=1 ideal;
    gamma_eff vs the DERIVED assumption (MOS 2/3, BJT 1/2).
    """
    white = var_in * gm / (FOUR_KT * BW_NOISE)
    g_assumed = 2.0 / 3.0 if kind == "mos" else 0.5
    return dict(white_factor=white, gamma_eff_vs_assumed=white / g_assumed,
                gamma_assumed=g_assumed, var_in=var_in, bw=1e9 - 1e3)


def _mc_summary(mc, n):
    out = {}
    for k, v in mc.items():
        sig = float(np.std(v, ddof=1))
        out[k] = dict(values=list(map(float, v)),
                      mean=float(np.mean(v)),
                      sigma=sig,
                      # sampling error bar on the sigma estimate itself
                      sigma_stderr=sig / np.sqrt(2.0 * (n - 1)))
    return out


def _settle(t, v, t0, tol):
    """Step-response settling: baseline = pre-step (t < 0.8*t0, clear of the
    0.5 ns rise edge), final = last 10% of the record; settle time =
    (last time |v-final| > tol*|delta|) - t0 — the whole record counts, so
    the rise edge itself is inside the metric (min possible = rise time)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    pre = t < 0.8 * t0
    init = float(np.mean(v[pre]))
    fin = float(np.mean(v[-max(1, len(v) // 10):]))
    delta = fin - init
    bad = np.abs(v - fin) > tol * abs(delta)
    ts = (float(t[np.max(np.nonzero(bad))]) - float(t0)) if bad.any() else 0.0
    # overshoot BEYOND the final value in the direction of the step (the
    # naive max(v)-fin catches the pre-step baseline when delta < 0)
    ext = float(np.max(v)) if delta > 0 else float(np.min(v))
    ov = float((ext - fin) / delta) if abs(delta) > 0 else 0.0
    return dict(final_v=fin, delta_v=delta, settle_s=ts, overshoot_frac=ov,
                settle_tol=tol)


def _thd(t, v, f0_expect, n_harm):
    """Hann-windowed FFT THD: sqrt(sum h2..hN^2)/h1 on the wrdata output.
    The linearize grid closes exactly n periods at f0_expect -> the trailing
    sample is dropped so the fundamental/harmonics land on exact FFT bins
    (no leakage; the 501-sample version splits the fundamental across bins
    1 and 2 and a naive argmax then reads the leaked bin as the fundamental)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    span = t[-1] - t[0]
    nper = int(round(span * f0_expect))
    if nper >= 1 and abs(span / nper - 1.0 / f0_expect) < 1e-9:
        t, v = t[:-1], v[:-1]      # integer-period record
    v = v - np.mean(v)
    w = np.hanning(len(v))
    sp = np.abs(np.fft.rfft(v * w))
    dt = float(t[1] - t[0])
    f = np.fft.rfftfreq(len(v), d=dt)
    k0 = max(int(round(f0_expect * len(v) * dt)), 1)
    h1 = float(sp[k0])
    harm = {}
    for n in range(2, n_harm + 1):
        kn = n * k0
        if kn >= len(sp) or h1 <= 0:
            harm[str(n)] = 0.0
            continue
        # Hann main lobe is +/-1 bin; exact-bin alignment keeps harmonics
        # away from the fundamental (kn-k0 >= k0-1 > 1 for k0 >= 3)
        harm[str(n)] = float(np.max(sp[kn - 1:kn + 2])) / h1
    thd = float(np.sqrt(sum((hv * h1) ** 2 for hv in harm.values())) / h1) \
        if h1 > 0 else float("nan")
    amp = 2.0 * h1 / float(np.sum(w))        # coherent-gain corrected peak
    return dict(f0_expect_hz=f0_expect, f0_meas_hz=float(f[k0]),
                fund_peak_v=amp, thd=thd, harmonics=harm)


def _pvt_combo(tag, temp_c, vdd):
    out = run(tag, PVT_MOS_SEC, PVT_HBT_SEC,
              netlist_text=netlist_pvt(temp_c, vdd, tag))
    d = parse_oprow(out)
    # PVT runs bias the diff-pair supply at vdd, so ic_hbt / ron_tg must be
    # re-derived with the ACTUAL supply, not the hardcoded 1.5.
    d["ic_hbt"] = (vdd - d["vdc1"]) / 10e3
    d["gm_hbt"] = d["ic_hbt"] / VT
    f, v = _load_wrdata(OUT + "/b2_gmsweep_%s.dat" % tag)[0]
    i_d = (vdd - v) / 65e3
    gm_fd = np.gradient(i_d, f)
    k = int(np.argmin(np.abs(f - 0.3837)))
    d["gm_mos_fd"] = float(gm_fd[k])
    d["id_mos_pole"] = float(i_d[k])
    d["pole_hz"] = ac_pole_hz(OUT + "/b2_ac_%s.dat" % tag)
    d["temp_c"] = temp_c
    d["vdd_v"] = vdd
    return d


def extract_all():
    """Run corners + both MC mechanisms + PVT + transient + parasitic probe,
    post-process, write the JSON with fail-closed status/blocks."""
    os.makedirs(OUT, exist_ok=True)
    t0 = __import__("time").time()
    res = {"corners": {}, "mc": None, "mc_global": None, "noise": {},
           "notes": [], "blocks": {}, "pvt": None, "transient": None,
           "parasitics": None, "mc_n": N_MC_LOCAL}
    blocks = res["blocks"]

    def block(name, ok, detail="", kind="required"):
        blocks[name] = dict(status=("ok" if ok else "failed"),
                            required=(kind == "required"), detail=detail)
        return ok

    # ---- corners + MC (serial: these carry the noise sweeps) ---------------
    for mos_sec, hbt_sec, tag, mm in CORNERS + [MC_RUN, MC_PROC_RUN]:
        mc = tag.startswith("mc")
        print("== b2 run %s (%s/%s)%s" % (tag, mos_sec, hbt_sec,
              (" MC x%d%s" % ((N_MC_LOCAL if tag == "mc" else N_MC_GLOBAL),
                              ", mm_ok=1" if mm else "")) if mc else ""))
        try:
            out = run(tag, mos_sec, hbt_sec, mc=mc, mismatch=mm,
                      n_mc=(N_MC_LOCAL if tag == "mc" else N_MC_GLOBAL))
        except Exception as e:
            res["notes"].append("RUN %s FAILED: %r" % (tag, e))
            print("   FAILED: %r" % (e,))
            continue

        if not mc:
            try:
                op = parse_oprow(out)
                res["corners"][tag] = op
            except Exception as e:
                res["notes"].append("OPROW %s FAILED: %r" % (tag, e))
                continue
            # noise
            for prim, fname in (("mos", "b2_noise_mos_%s.dat"),
                                ("hbt", "b2_noise_hbt_%s.dat"),
                                ("tg", "b2_noise_tg_%s.dat")):
                path = OUT + "/" + (fname % tag)
                try:
                    (f, on), (_, inn) = _load_wrdata(path)
                    var_o, var_o_w = integrate_psd(f, on)
                    var_i, var_i_w = integrate_psd(f, inn)
                    res["noise"].setdefault(prim, {})[tag] = dict(
                        var_onoise=var_o, var_onoise_white=var_o_w,
                        var_inoise=var_i, var_inoise_white=var_i_w)
                except Exception as e:
                    res["notes"].append("noise %s/%s FAILED: %r" % (prim, tag, e))
            # ac pole (wrdata writes complex AC as freq/re/im columns)
            try:
                res["corners"][tag]["pole_hz"] = ac_pole_hz(
                    OUT + "/b2_ac_%s.dat" % tag)
            except Exception as e:
                res["notes"].append("ac %s FAILED: %r" % (tag, e))
            # MOS gm ground truth: numeric derivative of the Id(Vg) sweep
            try:
                f, v = _load_wrdata(OUT + "/b2_gmsweep_%s.dat" % tag)[0]
                i_d = (1.5 - v) / 65e3
                gm_fd = np.gradient(i_d, f)
                k = int(np.argmin(np.abs(f - 0.3837)))
                res["corners"][tag]["gm_mos_fd"] = float(gm_fd[k])
                res["corners"][tag]["id_mos_pole"] = float(i_d[k])
            except Exception as e:
                res["notes"].append("gmsweep %s FAILED: %r" % (tag, e))
            print("   ratio=%.4f gm_mos(param/fd)=%.0f/%.0fuS gm_hbt=%.0fuS "
                  "vd0=%.3fmV ron=%.0fOhm pole=%.2fMHz"
                  % (op["mirror_ratio"], op["gm_mos_param"] * 1e6,
                     res["corners"][tag].get("gm_mos_fd", 0) * 1e6,
                     op["gm_hbt"] * 1e6, op["vd0_offset_raw"] * 1e3,
                     op["ron_tg"], res["corners"][tag].get("pole_hz", 0) / 1e6))
        else:
            key = "mc" if tag == "mc" else "mc_global"
            n = N_MC_LOCAL if tag == "mc" else N_MC_GLOBAL
            try:
                parsed = parse_mcrows(out, n)
                res[key] = _mc_summary(parsed, n)
                print("   MC done: %d samples (%s)" % (n, key))
            except Exception as e:
                res["notes"].append("%s parse FAILED: %r" % (key, e))
                print("   %s parse FAILED: %r" % (key, e))

    # ---- PVT + parasitics + transient (parallel, trimmed netlists) ---------
    def _try(name, fn):
        try:
            return fn()
        except Exception as e:
            res["notes"].append("%s FAILED: %r" % (name, e))
            print("   %s FAILED: %r" % (name, e))
            return None

    def pvt_jobs():
        jobs = []
        for temp_c in PVT_TEMPS:
            for vdd in PVT_VDDS:
                tag = "pvt_t%d_v%d" % (int(temp_c), int(vdd * 100))
                jobs.append((tag, temp_c, vdd))

        def one(job):
            tag, temp_c, vdd = job
            return tag, _pvt_combo(tag, temp_c, vdd)

        with ThreadPoolExecutor(max_workers=3) as ex:
            rows = [r for r in ex.map(one, jobs) if r[1] is not None]
        if rows:
            rows.sort(key=lambda r: (r[1]["temp_c"], r[1]["vdd_v"]))
            nom = [d for _, d in rows
                   if d["temp_c"] == 27.0 and d["vdd_v"] == 1.5][0]
            shift = {}
            for tag, d in rows:
                shift[tag] = {k: d[k] - nom[k] for k in
                              ("gm_mos_fd", "gm_hbt", "mirror_ratio",
                               "vd0_offset_raw", "ron_tg", "pole_hz",
                               "id_mos_pole")}
            res["pvt"] = dict(
                temps_c=list(PVT_TEMPS), vdds_v=list(PVT_VDDS),
                nominal=dict(temp_c=27.0, vdd_v=1.5), rows={t: d for t, d in rows},
                shifts_vs_nominal=shift,
                note=("temperature via ngspice .temp (global temper); tnom=27 C "
                      "stays pinned in the model cards (PDK ships it that way), "
                      "so this is real simulation temperature, not a "
                      "re-extracted parameter set. Bias sources are ideal "
                      "fixed currents (as in the nominal primitive bank), so "
                      "supply variation probes headroom/Early effects only; "
                      "bias-network supply sensitivity is a B3 measurement."),
            )
        return len(rows)

    def paras_jobs():
        def one(c_f):
            tag = "paras_c%d" % int(c_f)
            out = run(tag, PVT_MOS_SEC, PVT_HBT_SEC,
                      netlist_text=netlist_paras(c_f, tag))
            return c_f, ac_pole_hz(OUT + "/b2_ac_%s.dat" % tag)
        with ThreadPoolExecutor(max_workers=3) as ex:
            got = [r for r in ex.map(one, list(PARAS_C_FF)) if r[1]]
        if len(got) == len(PARAS_C_FF):
            got.sort()
            f0 = got[0][1]
            entries = []
            for c_f, pole in got:
                rel = (pole - f0) / f0
                per_ff = rel / c_f if c_f > 0 else 0.0
                entries.append(dict(c_added_ff=c_f, pole_hz=pole,
                                    pole_rel_shift=rel,
                                    pole_rel_shift_per_ff=per_ff))
            slope_meas = (np.log(got[-1][1] / got[0][1])
                          / (PARAS_C_FF[-1] - PARAS_C_FF[0]))
            slope_pred = -1.0 / (1e-12 + PARAS_C_FF[-1] * 1e-15)
            res["parasitics"] = dict(
                node="d2b", c_bank_ff=1000.0, cases=entries,
                slope_log_per_f=slope_meas, slope_log_per_f_pred=slope_pred,
                label="pre-PEX sensitivity estimate, not a PEX result",
                note=("no layout exists for the primitives -> PEX is "
                      "impossible pre-layout. This probe adds an estimated "
                      "wire/load C at the integrator output and re-extracts "
                      "the pole; predicted slope -1/(C_tot) in log space is "
                      "shown for cross-check."))
        return len(got)

    def tran_job():
        out = run("tran", PVT_MOS_SEC, PVT_HBT_SEC,
                  netlist_text=netlist_tran())
        del out  # tran results come from the wrdata files
        d = np.atleast_2d(np.loadtxt(OUT + "/b2_tran_step.dat"))
        t = d[:, 0]
        step = dict(
            hbt_diff=dict(t0_s=TRAN["step_t0"], in_v=TRAN["hbt_step_v"],
                          **_settle(t, d[:, 1] - d[:, 3], TRAN["step_t0"],
                                    TRAN["settle_tol"])),
            mos_gcell=dict(t0_s=TRAN["step_t0"], in_v=TRAN["mos_step_v"],
                           **_settle(t, d[:, 5], TRAN["step_t0"],
                                     TRAN["settle_tol"])),
            dt_s=TRAN["dt"], tstop_s=TRAN["step_tstop"],
        )
        # slew rate on the gm-cell output (the integrator node)
        v = d[:, 5]
        m = t >= TRAN["step_t0"]
        dvdt = np.gradient(v[m], t[m])
        step["mos_gcell"]["slew_v_per_s"] = float(np.max(np.abs(dvdt)))
        step["mos_gcell"]["slew_implied_a"] = float(
            np.max(np.abs(dvdt)) * 1e-12)   # I = C * dV/dt, C = 1 pF
        d2 = np.atleast_2d(np.loadtxt(OUT + "/b2_tran_sine.dat"))
        t2 = d2[:, 0]
        sine = dict(
            hbt_diff=dict(in_v=TRAN["hbt_sine_v"],
                          **_thd(t2, d2[:, 1], TRAN["hbt_f0"],
                                 TRAN["thd_harmonics"])),
            mos_gcell=dict(in_v=TRAN["mos_sine_v"],
                           **_thd(t2, d2[:, 5], TRAN["mos_f0"],
                                  TRAN["thd_harmonics"])),
            dt_s=TRAN["dt"], tstop_s=TRAN["sine_tstop"],
            fft="python Hann-windowed rFFT on ngspice wrdata output",
        )
        res["transient"] = dict(step=step, sine=sine, stimulus=TRAN)
        return True

    # serial-ish parallel batch: PVT (9), parasitics (3), transient (1);
    # ThreadPool keeps ngspice process count modest on this loaded box.
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_pvt = ex.submit(pvt_jobs)
        f_par = ex.submit(paras_jobs)
        f_trn = ex.submit(tran_job)
        n_pvt, n_par = f_pvt.result(), f_par.result()
        ok_tran = f_trn.result()

    # ---- gamma_eff from nominal TT noise ------------------------------------
    tt = res["corners"].get("tt")
    hbt_ok = False
    for prim, kind, gm_key in (("hbt", "bjt", "gm_hbt"),):
        nd = res["noise"].get(prim, {}).get("tt")
        if nd and tt:
            var_in = nd["var_inoise"]
            if var_in <= 0.0:
                res["notes"].append("noise %s: all-zero spectra, not measurable" % prim)
                continue
            res["noise"][prim]["gamma_eff"] = gamma_eff(tt[gm_key], var_in, kind)
            res["noise"][prim]["gamma_eff_whiteband"] = gamma_eff(
                tt[gm_key], nd["var_inoise_white"], kind)
            hbt_ok = "gamma_eff" in res["noise"][prim]
    for prim in ("mos", "tg"):
        nd = res["noise"].get(prim, {}).get("tt")
        if nd and nd.get("var_inoise", 1.0) <= 0.0:
            res["notes"].append(
                "noise %s: PSP103 OSDI device noise returns identically-zero "
                "onoise/inoise spectra in ngspice-44.2 (verified in isolation) "
                "-> gamma_eff NOT measurable with this toolchain; only HBT "
                "(native VBIC) noise is real. No number fabricated." % prim)
    if res["noise"].get("tg", {}).get("tt") and tt:
        ron = tt["ron_tg"]
        var_i = res["noise"]["tg"]["tt"]["var_inoise"]
        res["noise"]["tg"]["ron_ohm"] = ron
        if var_i > 0.0:
            res["noise"]["tg"]["white_factor_4kT_Ron"] = var_i / (
                FOUR_KT * ron * BW_NOISE)

    # gamma_eff denominator cross-check: numeric-derivative MOS gm
    fd = res["corners"].get("tt", {}).get("gm_mos_fd")
    ge = res["noise"].get("mos", {}).get("gamma_eff")
    if ge is None and fd:
        var_in = res["noise"].get("mos", {}).get("tt", {}).get("var_inoise")
        if var_in and var_in > 0.0:
            res["noise"]["mos"]["gamma_eff_fd_gm"] = gamma_eff(fd, var_in, "mos")

    # ---- fail-closed block statuses -----------------------------------------
    need = {"tt", "ss", "ff", "hbtwcs", "hbtbcs"}
    ok_corners = need <= set(res["corners"]) and all(
        "pole_hz" in res["corners"][t] and "gm_mos_fd" in res["corners"][t]
        for t in need)
    block("corners", ok_corners,
          "%s" % sorted(res["corners"]))
    n_mc_parsed = len((res["mc"] or {}).get("mirror_ratio", {}).get("values", []))
    block("mismatch_mc", n_mc_parsed == N_MC_LOCAL,
          "%d/%d samples" % (n_mc_parsed, N_MC_LOCAL))
    n_mcg_parsed = len((res["mc_global"] or {}).get("mirror_ratio", {})
                       .get("values", []))
    block("global_mc", n_mcg_parsed == N_MC_GLOBAL,
          "%d/%d samples" % (n_mcg_parsed, N_MC_GLOBAL))
    block("hbt_noise", hbt_ok,
          "gamma_eff white_factor=%.3f" % res["noise"]["hbt"]["gamma_eff"]["white_factor"]
          if hbt_ok else "HBT .noise not measurable")
    block("ac", tt is not None and "pole_hz" in res["corners"].get("tt", {}),
          "tt pole=%.3f MHz" % (res["corners"].get("tt", {}).get("pole_hz", 0) / 1e6))
    # KNOWN_LIMITATION: PSP103 OSDI device noise — a documented skip, NOT a
    # failure, and NOT required for status=complete.
    for prim in ("mos", "tg"):
        blocks["%s_noise" % prim] = dict(
            status="known_limitation", required=False,
            detail="PSP103 OSDI device noise not implemented in ngspice-44.2 "
                   "(all-zero spectra, verified in isolation)")
    block("transient", ok_tran and res["transient"] is not None,
          "settling+THD on HBT diff pair and MOS gm-cell")
    block("pvt", res["pvt"] is not None and len(res["pvt"]["rows"]) == 9,
          "%d/9 PVT combos" % (len(res["pvt"]["rows"]) if res["pvt"] else 0))
    block("parasitics", res["parasitics"] is not None
          and len(res["parasitics"]["cases"]) == 3,
          "%d/3 parasitic C cases" % (len(res["parasitics"]["cases"])
                                      if res["parasitics"] else 0))

    failed = [k for k, b in blocks.items()
              if b["status"] == "failed" and b["required"]]
    if failed:
        res["status"] = "failed" if len(failed) > 2 else "partial"
    else:
        res["status"] = "complete"
    res["failed_blocks"] = failed
    res["runtime_s"] = round(__import__("time").time() - t0, 1)

    with open(OUT + "/b2_extract.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("-> results/b2_extract.json  status=%s (%.1fs)"
          % (res["status"], res["runtime_s"]))
    if failed:
        print("   FAILED BLOCKS: %s" % failed)
    return res


def main(argv):
    extract_all()


if __name__ == "__main__":
    main(sys.argv)
