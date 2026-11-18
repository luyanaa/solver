"""B2 pre-silicon primitive extraction: .noise, corners vs Monte Carlo.

Closes two B2 obligations on IHP SG13G2 (DESIGN.md §7 Phase 7.5 / §9 risk 3):

(1) Real ngspice `.noise` on behavioral primitives (NOT the diode mirror of
    extract.py) -> gamma_eff = measured_PSD / theoretical. The §6.9 noise
    column was DERIVED from 4kTγ/g_m; this module measures it.
(2) Corner (global process: mos_tt/ss/ff + hbt_typ/wcs/bcs) run SEPARATELY
    from Monte Carlo (local Pelgrom mismatch: mos_tt_stat + hbt_typ_stat,
    24 reset-loop samples in one ngspice process), then compared.

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
N_MC = 24
MC_SEED = 1234567

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


def netlist(mos_sec, hbt_sec, tag, mc=False, mismatch=False):
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
            "dowhile i < %d" % N_MC,
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


def run(tag, mos_sec, hbt_sec, mc=False, mismatch=False, timeout=1800):
    os.makedirs(OUT, exist_ok=True)
    path = OUT + "/b2_%s.sp" % tag
    with open(path, "w") as fh:
        fh.write(netlist(mos_sec, hbt_sec, tag, mc, mismatch))
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


def _mc_summary(mc):
    return {k: dict(values=list(map(float, v)),
                    mean=float(np.mean(v)),
                    sigma=float(np.std(v, ddof=1)))
            for k, v in mc.items()}


def extract_all():
    """Run corners + both MC mechanisms, post-process, write the JSON."""
    os.makedirs(OUT, exist_ok=True)
    res = {"corners": {}, "mc": None, "mc_global": None, "noise": {},
           "notes": []}

    for mos_sec, hbt_sec, tag, mm in CORNERS + [MC_RUN, MC_PROC_RUN]:
        mc = tag.startswith("mc")
        print("== b2 run %s (%s/%s)%s" % (tag, mos_sec, hbt_sec,
              (" MC x%d%s" % (N_MC, ", mm_ok=1" if mm else "")) if mc else ""))
        try:
            out = run(tag, mos_sec, hbt_sec, mc=mc, mismatch=mm)
        except Exception as e:
            res["notes"].append("RUN %s FAILED: %r" % (tag, e))
            print("   FAILED: %r" % (e,))
            continue

        if not mc:
            op = parse_oprow(out)
            res["corners"][tag] = op
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
            try:
                parsed = parse_mcrows(out, N_MC)
                res[key] = _mc_summary(parsed)
                print("   MC done: %d samples (%s)" % (N_MC, key))
            except Exception as e:
                res["notes"].append("%s parse FAILED: %r" % (key, e))
                print("   %s parse FAILED: %r" % (key, e))

    # gamma_eff from nominal TT noise.
    # Honest coverage note: PSP103 OSDI device noise is NOT implemented in
    # ngspice-44.2 (verified with an isolated one-device run: onoise/inoise
    # spectra are identically 0 and the op [gm] parameter is a ~0 constant).
    # HBT uses the native VBIC (level 9) model -> real .noise works.
    tt = res["corners"].get("tt")
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

    with open(OUT + "/b2_extract.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("-> results/b2_extract.json")
    return res


def main(argv):
    extract_all()


if __name__ == "__main__":
    main(sys.argv)
