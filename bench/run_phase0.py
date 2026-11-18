"""Phase 0 验收：可行性判定器（DESIGN.md §7 Phase 0）。

    python3 -m bench.run_phase0

五个算例覆盖四条失败轴：diffusion/biharmonic → κ，Burgers → 表示，Robertson → κ。
每个算例都同时给出解析对照，验收线是 ±0.5 decade（κ）与 ±3 dB（H∞/Γ/σ_y）。
"""

from __future__ import annotations

import json
import os

# 本机 OpenBLAS 默认开 64 线程，128×128 的复矩阵 solve/SVD 会慢到 1 s 一次。
# 这些算例的 n≤1024，单线程反而快 40×；允许用户覆盖。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.integrate import solve_ivp

from analysis import (Descriptor, HardwareSpec, evaluate, hinf_norm,
                      resolvent_peak, shock_sensor, solve_generalized_lyapunov)
from bench.run_twin_experiments import burgers, face_flux

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HW = HardwareSpec(kappa_hw=1e6, B_hw=10e6, etail_req=1e-4)
_LINES: list[str] = []


def emit(s: str = ""):
    print(s)
    _LINES.append(s)


def hr(title: str):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


def table(header, rows, fmts):
    cells = [[(f % v) if (isinstance(f, str) and f) else str(v)
              for f, v in zip(fmts, r)] for r in rows]
    widths = [max(len(header[j]), max((len(c[j]) for c in cells), default=0))
              for j in range(len(header))]
    emit("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for c in cells:
        emit("  " + " ".join(v.rjust(w) for v, w in zip(c, widths)))


def _fmt(v, spec="%.3g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "inf" if v == float("inf") else "n/a"
    return spec % v


# ==========================================================================
# 算例
# ==========================================================================
def build_diffusion(n: int, nu: float = 1e-3):
    k = np.arange(1, n + 1, dtype=float)
    sys = Descriptor(np.eye(n), np.diag(-nu * k * k), np.eye(n)[:, :1], np.eye(n))
    return sys, dict(kind="diffusion", nu=nu, kappa_analytic=float(n * n),
                     hinf_analytic=float(1.0 / nu), sigma_y_analytic=float(np.sqrt(1.0 / (2.0 * nu))))


def build_biharmonic(n: int, nu: float = 1e-8):
    k = np.arange(1, n + 1, dtype=float)
    sys = Descriptor(np.eye(n), np.diag(-nu * k ** 4), np.eye(n)[:, :1], np.eye(n))
    return sys, dict(kind="biharmonic", nu=nu, kappa_analytic=float(n ** 4),
                     hinf_analytic=float(1.0 / nu), sigma_y_analytic=float(np.sqrt(1.0 / (2.0 * nu))))


def inviscid_burgers_exact(x, T: float = 0.3):
    """u0 = 0.5+0.5 sin(2πx) 的特征线解；T < 1/π 时单调，可直接反插值。"""
    x0 = np.linspace(-0.5, 1.5, 20001)
    u0 = 0.5 + 0.5 * np.sin(2.0 * np.pi * x0)
    X = x0 + u0 * T
    order = np.argsort(X)
    Xs, us = X[order], u0[order]
    if np.any(np.diff(Xs) <= 0):
        raise ValueError("T 已过激波形成时刻，特征线多值")
    return np.interp(x, Xs, us)


def build_burgers(n: int = 128, T: float = 0.3):
    x = (np.arange(n) + 0.5) / n
    u = inviscid_burgers_exact(x, T)
    h = 1.0 / n
    alpha = 1.6

    def rhs(v):
        F = face_flux(v, "first", alpha, h)      # 一阶 Roe（可微、稳定）
        return -(F - np.roll(F, 1)) / h

    eps = 1e-6
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        J[:, j] = (rhs(u + e) - rhs(u - e)) / (2.0 * eps)

    B = np.sin(2.0 * np.pi * x)
    B = B / np.linalg.norm(B)                    # 零均值：不驱动守恒总质量
    sys = Descriptor(np.eye(n), J, B, np.eye(n))
    reg = shock_sensor(u)
    u0 = 0.5 + 0.5 * np.sin(2.0 * np.pi * x)
    u_w = burgers(u0, T, "weno5", cfl=0.35)
    reg_w = shock_sensor(u_w)
    meta = dict(kind="burgers", T=T, kappa_analytic=None, hinf_analytic=None,
                sigma_y_analytic=None, p=float(reg["p"]), E_tail=float(reg["E_tail"]),
                WENO_p=float(reg_w["p"]), WENO_E_tail=float(reg_w["E_tail"]))
    return sys, meta, reg


def build_robertson(T_obs: float = 1e4):
    def f(t, y):
        y1, y2, y3 = y
        return [-0.04 * y1 + 1e4 * y2 * y3,
                0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2,
                3e7 * y2 * y2]

    def J(y):
        y1, y2, y3 = y
        return np.array([[-0.04, 1e4 * y3, 1e4 * y2],
                         [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
                         [0.0, 6e7 * y2, 0.0]])

    sol = solve_ivp(f, [0.0, T_obs], [1.0, 0.0, 0.0], method="Radau",
                    rtol=1e-11, atol=1e-15, t_eval=[T_obs])
    y = sol.y[:, -1]
    # 零均值输入：不驱动 y1+y2+y3 这个守恒模态，否则 H2/H∞ 合法地发散
    B = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    sys = Descriptor(np.eye(3), J(y), B, np.eye(3))
    return sys, dict(kind="robertson", T_obs=T_obs, y=[float(v) for v in y],
                     kappa_analytic=None, hinf_analytic=None, sigma_y_analytic=None)


def diag_closed_form(sys: Descriptor):
    """A 对角、C=I 时的解析 σ_y²；用于 n=1024 的算例。"""
    W = sys.B @ sys.B.T
    d = np.diag(sys.A)
    if np.any(d >= 0):
        return float("inf")
    p = np.diag(W) / (-2.0 * d)
    from analysis.lyapunov import LyapunovResult
    P = np.diag(p)
    sigma_y2 = float(np.sum(p))
    return LyapunovResult(P, sigma_y2, 0.0, 0.0, True, sys.n, sys.n, 0.0, "diag-closed")


# ==========================================================================
# 主流程
# ==========================================================================
def run_case(name: str, sys: Descriptor, meta: dict, reg=None):
    chk = sys.check()
    lyap = diag_closed_form(sys) if sys.n > 512 else solve_generalized_lyapunov(sys)
    hi = hinf_norm(sys)
    res = resolvent_peak(sys)
    sl = evaluate(sys, HW, lyap=lyap, reg=reg, gamma=res["Gamma"])
    out = dict(
        name=name, n=sys.n, meta=meta, check=chk,
        lyapunov=dict(sigma_y=float(lyap.sigma_y), sigma_y2=float(lyap.sigma_y2),
                      rel_residual=float(lyap.rel_residual), h2_finite=bool(lyap.h2_finite),
                      n_stable=int(lyap.n_stable), method=lyap.method),
        hinf=dict(value=float(hi["value"]), method=hi["method"],
                  sweep=float(hi["sweep"]), rel_diff=float(hi["rel_diff"])),
        resolvent=dict(Gamma=float(res["Gamma"]), omega=float(res["omega"]),
                       sigma_min=float(res["sigma_min"]), method=res["method"]),
        slack=sl.as_dict(),
    )
    if reg is not None:
        out["regularity"] = dict(p=float(reg["p"]), E_tail=float(reg["E_tail"]),
                                 representation=float(reg["representation"]),
                                 shock=bool(reg["shock"]))
    return out


def acceptance(cases: dict) -> list[dict]:
    checks = []

    def add(name, got, exp, tol, kind="log10"):
        if exp is None or not np.isfinite(exp):
            return
        err = abs(np.log10(got / exp)) if (got > 0 and np.isfinite(got)) else float("inf")
        checks.append(dict(name=name, got=float(got), expected=float(exp),
                           log10_err=float(err), tol=float(tol), passed=bool(err <= tol),
                           kind=kind))

    for key in ("diffusion N=64", "diffusion N=1024", "biharmonic N=128"):
        c = cases[key]
        m = c["meta"]
        add("%s κ" % key, c["check"]["kappa"], m["kappa_analytic"], 0.5)
        add("%s Γ" % key, c["resolvent"]["Gamma"], m["hinf_analytic"], 0.15)
        add("%s σ_y" % key, c["lyapunov"]["sigma_y"], m["sigma_y_analytic"], 0.15)
        if c["hinf"]["method"] == "hamiltonian":
            add("%s H∞" % key, c["hinf"]["value"], m["hinf_analytic"], 0.15)
            ham_ok = c["hinf"]["rel_diff"] <= 0.15
            checks.append(dict(name="%s H∞ ham-vs-sweep" % key,
                               got=float(c["hinf"]["rel_diff"]), expected=0.0,
                               log10_err=float(np.log10(max(c["hinf"]["rel_diff"], 1e-16))),
                               tol=0.15, passed=bool(ham_ok), kind="rel"))

    resid = cases["diffusion N=64"]["lyapunov"]["rel_residual"]
    checks.append(dict(name="diffusion N=64 Lyapunov residual",
                       got=float(resid), expected=0.0,
                       log10_err=float(np.log10(max(resid, 1e-16))),
                       tol=1e-10, passed=bool(resid < 1e-10), kind="abs"))

    expected_axis = {
        "diffusion N=64": None,
        "diffusion N=1024": "stiffness",
        "biharmonic N=128": "stiffness",
        "Burgers N=128": "representation",
        "Robertson t=1e4": "stiffness",
    }
    for name, axis in expected_axis.items():
        violated = cases[name]["slack"]["violated"]
        ok = (not violated) if axis is None else (axis in violated)
        checks.append(dict(name="%s verdict" % name, got=",".join(violated) or "-",
                           expected=axis or "-", passed=bool(ok), kind="axis"))
    return checks


def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()

    hr("Phase 0：可行性判定器验收")
    hr("A. 汇总")
    cases = {}
    specs = [
        ("diffusion N=64", build_diffusion(64)),
        ("diffusion N=1024", build_diffusion(1024)),
        ("biharmonic N=128", build_biharmonic(128)),
        ("Burgers N=128", build_burgers(128)),
        ("Robertson t=1e4", build_robertson(1e4)),
    ]
    for name, item in specs:
        if len(item) == 3:
            sys, meta, reg = item
        else:
            sys, meta = item
            reg = None
        cases[name] = run_case(name, sys, meta, reg=reg)

    table(["system", "n", "kappa", "kappa/k_hw", "Gamma", "sigma_y", "B_req[Hz]",
           "Hinf", "method", "violated"],
          [(c["name"], c["n"], c["check"]["kappa"], c["slack"]["values"]["stiffness"],
            c["resolvent"]["Gamma"], c["lyapunov"]["sigma_y"],
            c["check"]["bandwidth_hz"], c["hinf"]["value"], c["hinf"]["method"],
            ",".join(c["slack"]["violated"]) or "-") for c in cases.values()],
          ["", "%d", "%.3e", "%.3f", "%.3e", "%.3e", "%.3e", "%.3e", "", ""])

    hr("B. 六轴 slack（1.0 为红线；n/a = 没有输入）")
    table(["system", "noise", "bandwidth", "stiffness", "robustness", "routing",
           "representation", "status"],
          [(c["name"], _fmt(c["slack"]["values"]["noise"]),
            _fmt(c["slack"]["values"]["bandwidth"]),
            _fmt(c["slack"]["values"]["stiffness"]),
            _fmt(c["slack"]["values"]["robustness"]),
            _fmt(c["slack"]["values"]["routing"]),
            _fmt(c["slack"]["values"]["representation"]),
            c["slack"]["status"]) for c in cases.values()],
          [""] * 8)

    hr("C. 验收对照（κ: ±0.5 decade；H∞/Γ/σ_y: ±3 dB）")
    checks = acceptance(cases)
    table(["check", "got", "expected", "log10 err", "tol", "PASS"],
          [(c["name"], c["got"], c["expected"],
            _fmt(c.get("log10_err")), c.get("tol"), "yes" if c["passed"] else "NO")
           for c in checks],
          ["", "%s", "%s", "", "", ""])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    hr("D. 读法")
    emit("  - diffusion N=1024 的 κ=1.048e6 > 1e6：5% 越界，属于 §2.4 的'贴边'。")
    emit("    判定写 stiffness；若 κ_hw 取 1.05e6 则刚好可行——这正是窗口估计的敏感点。")
    emit("  - biharmonic N=128 的 κ=N⁴=2.68e8，比窗口大 268×，必须降维/换成局部表示。")
    emit("  - Burgers 的失败轴是 representation：p=%.2f、E_tail=%.1e（精确解）。"
         % (cases["Burgers N=128"]["regularity"]["p"],
            cases["Burgers N=128"]["regularity"]["E_tail"]))
    emit("    WENO 捕捉解把 p 抬到 %.2f、E_tail 降到 %.1e，用 ν_num 换掉了谱尾——"
         % (cases["Burgers N=128"]["meta"]["WENO_p"],
            cases["Burgers N=128"]["meta"]["WENO_E_tail"]))
    emit("    所以 representation 轴必须用物理状态算，不能用捕捉格式的输出算。")
    emit("  - Burgers 的 Γ=∞ 是守恒零模态在 s=0 的极点（A 的 resolvent）；")
    emit("    输入已投影到零均值，H∞ 仍是有限的 6.1——Γ 与 ‖H‖∞ 不是同一个对象。")
    emit("  - Robertson 的 κ=%.2e（t=1e4，去掉守恒零模态），判定 stiffness；"
         % cases["Robertson t=1e4"]["check"]["kappa"])
    emit("    10¹¹ 是速率常数比，不是 Jacobian 谱；口径见 §8。")

    _plot(cases)
    emit("\n  json   → results/phase0.json")
    emit("  report → results/phase0_report.txt")
    with open(os.path.join(OUT, "phase0.json"), "w") as fh:
        json.dump(dict(hardware=dict(kappa_hw=HW.kappa_hw, B_hw=HW.B_hw,
                                     etail_req=HW.etail_req),
                       cases=cases, checks=checks), fh, indent=2, default=float)
    with open(os.path.join(OUT, "phase0_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")


def _plot(cases):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = list(cases)
    short = [n.replace("diffusion ", "diff ").replace("biharmonic ", "biharm ")
             .replace("Burgers ", "Burg ").replace("Robertson ", "Rob ") for n in names]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
    x = np.arange(len(names))
    series = [
        ("kappa", [cases[n]["check"]["kappa"] for n in names], 1e6, "κ_t", "κ_hw"),
        ("Gamma", [cases[n]["resolvent"]["Gamma"] for n in names], None, "Γ", None),
        ("sigma_y", [cases[n]["lyapunov"]["sigma_y"] for n in names], None, "σ_y", None),
        ("B_req", [cases[n]["check"]["bandwidth_hz"] for n in names], 10e6, "B_req [Hz]", "B_hw"),
    ]
    for a, (key, vals, hline, title, hname) in zip(ax.ravel(), series):
        vals = np.asarray(vals, dtype=float)
        finite = np.isfinite(vals) & (vals > 0)
        a.bar(x[finite], vals[finite], color="#4c72b0")
        for xi, raw in zip(x[~finite], vals[~finite]):
            a.text(xi, 0.9, "inf", ha="center", va="top", fontsize=8,
                   transform=a.get_xaxis_transform())
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels(short, rotation=18, fontsize=8)
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.3, axis="y")
        if hline is not None:
            a.axhline(hline, color="crimson", ls="--", lw=1.2, label=hname)
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phase0.png"), dpi=130)
    emit("\n  figure → results/phase0.png")


if __name__ == "__main__":
    main()
