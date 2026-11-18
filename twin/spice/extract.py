"""Extract g_m-level interface parameters from real PDKs via ngspice."""
from __future__ import annotations

import glob
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


def _first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


SKY = _first(CIEL + "/sky130/versions/*/sky130A")
GF = _first(CIEL + "/gf180mcu/versions/*/gf180mcuA")
IHP = _first(CIEL + "/ihp-sg13/versions/*/ihp-sg13g2")

PDKS = {
    "sky130": dict(
        inc=['.lib "%s/libs.tech/ngspice/sky130.lib.spice" tt' % SKY],
        dev="sky130_fd_pr__nfet_01v8", kind="X", wl="w=10 l=1",
        vdd=1.8, ibias="10u", vsweep=(0.15, 1.2), iend="40u",
    ),
    "gf180mcu": dict(
        # design.ngspice 里 fnoicor=0；直接 include 模型文件时必须显式补上，
        # 否则 noia/noib 表达式里的 fnoicor 未定义，ngspice 报 "Cannot compute substitute"。
        inc=['.lib "%s/libs.tech/ngspice/sm141064.ngspice" typical' % GF,
             '.param fnoicor=0'],
        dev="nfet_03v3", kind="M", wl="w=10u l=1u",
        vdd=3.3, ibias="10u", vsweep=(0.2, 2.4), iend="40u",
    ),
    "ihp-sg13g2": dict(
        # cornerMOSlv.lib 定义 sg13_lv_nmos 子电路并 include parm 文件；
        # 只 include parm 文件会得到 "could not find a valid modelname"。
        inc=['.lib "%s/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt' % IHP],
        dev="sg13_lv_nmos", kind="X", wl="w=10u l=0.13u",
        vdd=1.5, ibias="10u", vsweep=(0.15, 1.0), iend="40u",
    ),
    "ihp-hbt": dict(
        inc=['.lib "%s/libs.tech/ngspice/models/cornerHBT.lib" hbt_typ' % IHP],
        dev="npn13G2", kind="X", wl="",
        vdd=1.8, ibias="20u", vsweep=(0.3, 1.6), iend="60u",
    ),
}


def p_inc(pdk):
    return PDKS[pdk]["inc"][0]


def netlist(pdk):
    p = PDKS[pdk]
    pre = "X" if p["kind"] == "X" else "M"
    lines = ["* g_m interface extraction: " + pdk]
    lines += p["inc"]
    lines += [
        "Iin 0 d1 dc " + p["ibias"],
        pre + "M1 d1 d1 0 0 " + p["dev"] + " " + p["wl"],
        pre + "M2 d2 d1 0 0 " + p["dev"] + " " + p["wl"],
        "Vout d2 0 dc %.3f" % (0.5 * p["vdd"]),
        ".control",
        "op",
        "let io = -i(vout)",
        "let vgs = v(d1)",
        "print io vgs",
        "dc Iin 1u %s 1u" % p["iend"],
        "wrdata " + OUT + "/isweep_" + pdk + ".dat i(vout) v(d1)",
        "alter Iin dc = " + p["ibias"],
        "dc Vout %.2f %.2f 0.01" % p["vsweep"],
        "wrdata " + OUT + "/vsweep_" + pdk + ".dat i(vout)",
        ".endc",
        ".end",
        "",
    ]
    return NL.join(lines)


def _cols(path):
    d = np.atleast_2d(np.loadtxt(path))
    return [(d[:, k], d[:, k + 1]) for k in range(0, d.shape[1], 2)]


def analyse(pdk):
    ios = _cols(OUT + "/isweep_" + pdk + ".dat")
    vs = _cols(OUT + "/vsweep_" + pdk + ".dat")
    i_in, i_out = ios[0]
    _, vgs = ios[1]
    i_out = -i_out
    gm = np.gradient(i_out, vgs)
    j = int(np.argmin(np.abs(i_in - 10e-6)))
    v_out, i2 = vs[0]
    i2 = -i2
    gds = np.gradient(i2, v_out)
    k = int(np.argmin(np.abs(v_out - 0.5 * PDKS[pdk]["vdd"])))
    return dict(
        pdk=pdk, i_in=float(i_in[j]), i_out=float(i_out[j]),
        ratio=float(i_out[j] / i_in[j]), vgs=float(vgs[j]),
        gm=float(gm[j]), gm_over_id=float(gm[j] / i_out[j]),
        gds=float(gds[k]), gds_over_id=float(gds[k] / i_out[j]),
        ro=float(1.0 / gds[k]), lambda_eff=float(gds[k] / i_out[j]),
    )


def run(pdk):
    os.makedirs(OUT, exist_ok=True)
    path = OUT + "/ext_" + pdk + ".sp"
    with open(path, "w") as fh:
        fh.write(netlist(pdk))
    inc_path = re.search(r'"([^"]+)"', PDKS[pdk]["inc"][0]).group(1)
    cwd = os.path.dirname(inc_path)
    env = os.environ.copy()
    if pdk == "ihp-sg13g2":
        # PSP103 以 OSDI 加载；spinit 用 $PDK_ROOT/$PDK 解析 .osdi 路径。
        env["PDK_ROOT"] = os.path.dirname(IHP)
        env["PDK"] = os.path.basename(IHP)
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True,
                       timeout=900, cwd=cwd, env=env)
    with open(OUT + "/ext_" + pdk + ".log", "w") as fh:
        fh.write(r.stdout + r.stderr)
    return analyse(pdk)


def main(argv):
    pdks = argv[1:] or list(PDKS)
    out = []
    for p in pdks:
        print("== " + p)
        try:
            res = run(p)
            print("   ratio=%.5f  gm/ID=%.1f/V  gds/ID=%.4f/V  ro=%.1f kOhm  Vgs=%.3f"
                  % (res["ratio"], res["gm_over_id"], res["gds_over_id"],
                     res["ro"] / 1e3, res["vgs"]))
            out.append(res)
        except Exception as e:
            print("   FAILED: %r" % (e,))
    with open(OUT + "/device_extract.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(NL + "-> results/device_extract.json")


if __name__ == "__main__":
    main(sys.argv)
