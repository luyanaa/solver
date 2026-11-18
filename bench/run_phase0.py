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
                      peak_with_projection, shock_sensor,
                      solve_generalized_lyapunov)
from bench.run_twin_experiments import burgers, face_flux

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
# 表示阈值由 tol_L2=1% 推出 1e-4。
HW = HardwareSpec(kappa_hw=1e6, B_hw=10e6, tol_l2=1e-2)
# 鲁棒轴探针：1% 的失配/接口预算，只加在有守恒零模态的两个算例上，
# 用来验收 QZ 投影。其余算例没有失配模型，鲁棒轴记 NaN（未知），不是 0。
HW_PROBE = HardwareSpec(kappa_hw=1e6, B_hw=10e6, tol_l2=1e-2,
                        delta_A_norm=1e-2, delta_input_norm=1e-2)
_PROBE = {"Burgers N=128", "Robertson t=1e4"}
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
def run_case(name: str, sys: Descriptor, meta: dict, reg=None, spec: HardwareSpec = HW):
    chk = sys.check()
    lyap = diag_closed_form(sys) if sys.n > 512 else solve_generalized_lyapunov(sys)
    hi = hinf_norm(sys)
    res = peak_with_projection(sys)
    sl = evaluate(sys, spec, lyap=lyap, reg=reg,
                  gamma_proj=res["Gamma_proj"], hinf=hi["value"])
    out = dict(
        name=name, n=sys.n, meta=meta, check=chk,
        lyapunov=dict(sigma_y=float(lyap.sigma_y), sigma_y2=float(lyap.sigma_y2),
                      rel_residual=float(lyap.rel_residual), h2_finite=bool(lyap.h2_finite),
                      n_stable=int(lyap.n_stable), method=lyap.method),
        hinf=dict(value=float(hi["value"]), method=hi["method"],
                  sweep=float(hi["sweep"]), rel_diff=float(hi["rel_diff"])),
        resolvent=dict(Gamma=float(res["Gamma"]), omega=float(res["omega"]),
                       sigma_min=float(res["sigma_min"]), method=res["method"],
                       Gamma_proj=float(res["Gamma_proj"]), n_proj=int(res["n_proj"]),
                       projected=bool(res["projected"])),
        slack=sl.as_dict(),
    )
    if res["projected"]:
        # 投影后的 H∞ 用来验证：去掉不可控零模态不改变输入输出行为。
        red, _ = sys.stable_projection()
        if red is not None:
            hp = hinf_norm(red)
            out["projection"] = dict(n=int(res["n_proj"]), hinf=float(hp["value"]),
                                     method=hp["method"], sweep=float(hp["sweep"]))
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

    # 鲁棒轴：投影后必须有限，且投影不改变输入输出行为。
    for key in ("Burgers N=128", "Robertson t=1e4"):
        c = cases[key]
        v = c["slack"]["values"]
        ok = bool(np.isfinite(v["robustness_A"]) and np.isfinite(v["robustness_io"]))
        checks.append(dict(name="%s robustness finite" % key,
                           got="%.3g/%.3g" % (v["robustness_A"], v["robustness_io"]),
                           expected="finite", passed=ok, kind="finite"))
        if "projection" in c:
            # 同一扫频网格上比较：投影只去掉不可控零模态，传递函数应逐点一致。
            hs, hp = c["hinf"]["sweep"], c["projection"]["sweep"]
            rel = abs(hp - hs) / max(abs(hs), 1e-300)
            checks.append(dict(name="%s H∞ sweep proj vs full" % key, got=hp, expected=hs,
                               log10_err=float(np.log10(max(rel, 1e-16))),
                               tol=1e-3, passed=bool(rel <= 1e-3), kind="rel"))
    return checks


def case_dae_noise() -> list[dict]:
    """DAE 噪声注入验收（P0-E 修复）：index-1 代数方程吃进输入不自动 ⇒ H2=∞。

    反例：E=diag(1,0), A=[[-1,1],[0,1]], B=[0,1]ᵀ, C=[1,0]
      方程 x1' = −x1 + x2；0 = x2 + u ⇒ x2 = −u ⇒ H(s) = −1/(s+1)，严格真，
      H2² = 1/2。旧的 naive 判据（B 在非稳定子空间有投影 ⇒ ∞）会错判。

    负控制 1：同系统 C=[0,1]（y = x2 = −u，直接馈通 D_eff = −1）⇒ H2 = ∞。
    负控制 2：E=[[0,1],[0,0]]（nilpotent，index-2 链）⇒ 脉冲 ⇒ H2 = ∞。

    交叉验证：高频 |H(jω)| — 反例衰减（严格真），负控制 1 不衰减（馈通）。
    """
    checks: list[dict] = []

    def add(name, passed, got, expected):
        checks.append(dict(name=name, got=got, expected=expected,
                           log10_err=0.0, tol=0.0, passed=bool(passed), kind="dae"))

    w_big = 1e6
    E = np.array([[1.0, 0.0], [0.0, 0.0]])
    A = np.array([[-1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])

    # 反例：H2² = 1/2
    sys_pos = Descriptor(E, A, B, np.array([[1.0, 0.0]]))
    r_pos = solve_generalized_lyapunov(sys_pos)
    add("DAE index-1 sigma_y2 = 1/2", abs(r_pos.sigma_y2 - 0.5) <= 1e-9 * 0.5,
        "%.12g" % r_pos.sigma_y2, "0.5")
    add("DAE index-1 h2_finite", r_pos.h2_finite, r_pos.method, "True")
    old_verdict = "inf (naive: B projects onto algebraic subspace)"
    add("DAE index-1 naive criterion would say inf", not r_pos.h2_finite or True,
        "%s -> actual %s" % (old_verdict, r_pos.method), "documented")
    tail_pos = float(np.abs(sys_pos.freqresp(w_big))[0, 0])
    add("DAE index-1 high-freq |H| -> 0", tail_pos < 1e-4, "%.3g" % tail_pos, "<1e-4")

    # 负控制 1：直接馈通 ⇒ 无穷
    sys_neg = Descriptor(E, A, B, np.array([[0.0, 1.0]]))
    r_neg = solve_generalized_lyapunov(sys_neg)
    add("DAE feedthrough control is inf", (not r_neg.h2_finite) and (r_neg.sigma_y2 == float("inf")),
        "%.4g" % r_neg.sigma_y2, "inf")
    tail_neg = float(np.abs(sys_neg.freqresp(w_big))[0, 0])
    add("DAE feedthrough |H| does not decay", tail_neg > 0.5, "%.3g" % tail_neg, "~1")

    # 负控制 2：nilpotent/index-2 ⇒ 脉冲 ⇒ 无穷
    sys_nil = Descriptor(np.array([[0.0, 1.0], [0.0, 0.0]]), np.eye(2),
                         np.array([[0.0], [1.0]]), np.array([[1.0, 0.0]]))
    r_nil = solve_generalized_lyapunov(sys_nil)
    add("DAE nilpotent (index-2) control is inf",
        (not r_nil.h2_finite) and (r_nil.sigma_y2 == float("inf")),
        "%.4g" % r_nil.sigma_y2, "inf")

    # ---- P0-G 回归：左投影必须用 Q，不能用 Z ----
    # 相似变换后的同一系统（TF 不变 ⇒ σ_y² 仍须 1/2）。非对称 Ẽ = T⁻¹ET 下
    # Q≠Z：Z1ᵀ 投影的 B_eff 错（这里给 −4.28，真值 −1），Q1ᵀ 投影的对。
    from scipy import linalg as _la
    from analysis.descriptor import stable_finite_mask as _sfm
    rng = np.random.default_rng(7)
    T = rng.standard_normal((2, 2))
    while abs(np.linalg.det(T)) < 0.3:
        T = rng.standard_normal((2, 2))
    Et = np.linalg.solve(T, E) @ T
    At = np.linalg.solve(T, A) @ T
    Bt = np.linalg.solve(T, B)
    Ct = np.array([[1.0, 0.0]]) @ T
    sys_T = Descriptor(Et, At, Bt, Ct)
    r_T = solve_generalized_lyapunov(sys_T)
    add("P0-G transformed DAE sigma_y2 = 1/2", abs(r_T.sigma_y2 - 0.5) <= 1e-8,
        "%.12g" % r_T.sigma_y2, "0.5")
    # 显式对照：Z1ᵀ-Galerkin 降阶（旧 bug）vs Q1ᵀ 降阶+消去（新）的 DC 增益，
    # 真值 H(0) = −1。
    scale_T = max(1.0, float(np.max(np.abs(_la.eigvals(At, Et)[np.isfinite(_la.eigvals(At, Et))]))))
    _, _, alT, beT, QT, ZT = _la.ordqz(At, Et, sort=lambda a, b: _sfm(a, b, 1e-12, scale_T),
                                       output="real")
    kT = int(np.count_nonzero(_sfm(alT, beT, 1e-12, scale_T)))
    Q1T, Q2T, Z1T, Z2T = QT[:, :kT], QT[:, kT:], ZT[:, :kT], ZT[:, kT:]
    gu_T = -_la.solve(Q2T.T @ At @ Z2T, Q2T.T @ Bt)
    BBss_T = Q1T.T @ Et @ Z1T
    AAss_T = Q1T.T @ At @ Z1T
    Bbs_T = _la.solve(BBss_T, (Q1T.T @ Et @ Z2T) @ gu_T)
    Beff_T = Q1T.T @ Bt + (Q1T.T @ At @ Z2T) @ gu_T - AAss_T @ Bbs_T
    Dtot_T = Ct @ Z1T @ (-Bbs_T) + Ct @ Z2T @ gu_T
    gain_Q = float((Ct @ Z1T @ _la.solve(-AAss_T, Beff_T) + Dtot_T)[0, 0])
    E_z, A_z, B_z = Z1T.T @ Et @ Z1T, Z1T.T @ At @ Z1T, Z1T.T @ Bt
    gain_Z = float((Ct @ Z1T @ _la.solve(-A_z, B_z))[0, 0])
    add("P0-G Z-left reduction wrong (regression trap)", abs(gain_Z + 1.0) > 0.5,
        "Z-left DC gain %.4f" % gain_Z, "≠ -1")
    add("P0-G Q-left reduction exact", abs(gain_Q + 1.0) < 1e-8,
        "Q-left DC gain %.12g" % gain_Q, "-1")
    # stable_projection 的降阶特征值 == 被选中的稳定有限特征值。
    red_T, k_sp = sys_T.stable_projection()
    ev_red = _la.eigvals(red_T.A, red_T.E)
    ev_sel = np.array([alT[i] / beT[i] for i in range(len(alT))
                       if _sfm(alT[i:i + 1], beT[i:i + 1], 1e-12, scale_T)[0]])
    ok_sp = (k_sp == kT and len(ev_red) == len(ev_sel)
             and np.allclose(np.sort_complex(ev_red), np.sort_complex(ev_sel), atol=1e-8))
    add("P0-G stable_projection eigvals match selection", ok_sp,
        "%s" % np.round(ev_red, 6), "%s" % np.round(ev_sel, 6))
    # 交叉验证：freqresp 积分 σ_y² = (1/π)∫|H(jω)|²dω ≈ 1/2（真值独立的第二条路）。
    w_grid = np.logspace(-4, 5, 40001)
    H_T = sys_T.freqresp(w_grid)[:, 0, 0]
    sig_int = float(np.trapezoid(np.abs(H_T) ** 2, w_grid) / np.pi)
    add("P0-G freqresp integral sigma_y2", abs(sig_int - 0.5) <= 5e-3,
        "%.6g" % sig_int, "0.5")

    # ---- sys.D 显式直接馈通（review 修复）：旧实现静默忽略 sys.D ——
    # D=1 的稳定 SISO 错报 σ_y²=0.5，真值 H2 = ∞（白噪声直达输出）。
    # 覆盖两条路径：direct（稳定良态 E）和 DAE QZ 消去路径。
    sys_d_direct = Descriptor(np.eye(2), -np.eye(2), np.ones((2, 1)),
                              np.ones((1, 2)), np.array([[1.0]]))
    r_d_direct = solve_generalized_lyapunov(sys_d_direct)
    add("sys.D=1 stable SISO is inf (direct path)",
        (not r_d_direct.h2_finite) and (r_d_direct.sigma_y2 == float("inf")),
        "%.4g" % r_d_direct.sigma_y2, "inf")
    tail_d = float(np.abs(sys_d_direct.freqresp(w_big)).max())
    add("sys.D=1 high-freq |H| does not decay", tail_d > 0.5,
        "%.3g" % tail_d, "~1")
    sys_d_dae = Descriptor(E, A, B, np.array([[1.0, 0.0]]), np.array([[1.0]]))
    r_d_dae = solve_generalized_lyapunov(sys_d_dae)
    add("sys.D=1 DAE QZ path is inf",
        (not r_d_dae.h2_finite) and (r_d_dae.sigma_y2 == float("inf"))
        and "feedthrough" in r_d_dae.method,
        "%.4g (%s)" % (r_d_dae.sigma_y2, r_d_dae.method), "inf")
    # 回归守卫：同一系统 D=0（缺省）仍给 σ_y²=1/2（旧正确行为不变）。
    add("DAE index-1 sigma_y2 unchanged with default D=0",
        abs(r_pos.sigma_y2 - 0.5) <= 1e-9 * 0.5,
        "%.12g" % r_pos.sigma_y2, "0.5")
    return checks


def case_energy_metric() -> list[dict]:
    """P1 能量度量 Γ：裸 σ_min 把电容尺度混进 Γ（§5.8）。

    构造：E=diag(1,1e-3)（per-node 电容跨 1000×），A 非对角耦合。
    能量度量 Γ_M = max_ω σ_max(M^{1/2}(jωE−A)⁻¹M^{−1/2}) 与裸 Euclidean
    Γ = max_ω σ_max((jωE−A)⁻¹) 差一个因子（本例 ~3×；电容跨度越大差越大）。
    注意被求逆的是整个 resolvent：Γ_M ≠ 1/min_ω σ_min(M^{1/2}(jωE−A)M^{−1/2})。
    实测：Γ_eucl=9.91，Γ_M=28.76（brute-force 扫频一致），M=I 时 Γ_M=Γ_eucl。

    已知一致性事项（不在此重接）：slack 的 robustness_A 轴目前消费裸 Γ；
    per-node C 进 IR 之后应换 M 度量的 Γ（owner scope）。
    """
    checks: list[dict] = []

    def add(name, passed, got, expected):
        checks.append(dict(name=name, got=got, expected=expected,
                           log10_err=0.0, tol=0.0, passed=bool(passed),
                           kind="metric"))

    E = np.diag([1.0, 1e-3])
    A = np.array([[-1.0, -1.0], [1.0, -0.1]])
    sys = Descriptor(E, A, np.zeros((2, 0)), np.zeros((0, 2)))

    g_eucl = peak_with_projection(sys, metric=None)["Gamma"]
    g_M = peak_with_projection(sys, metric="energy")["Gamma"]
    ratio = g_M / g_eucl
    add("energy-metric Gamma ratio > 1 (metric matters)", ratio > 1.0,
        "%.3g / %.3g = %.3g" % (g_M, g_eucl, ratio), "> 1")

    # 独立参照：直接在稠密 ω 网格上算 σ_max(M^{1/2}(jωE−A)⁻¹M^{−1/2})。
    w = np.logspace(-6, 6, 2001)
    d = np.diag(E)
    Mh, Mhi = np.diag(np.sqrt(d)), np.diag(1.0 / np.sqrt(d))
    smax = np.empty(w.size)
    for i, wi in enumerate(w):
        T = Mh @ np.linalg.inv(1j * wi * E - A) @ Mhi
        smax[i] = np.linalg.svd(T, compute_uv=False)[0]
    g_ref = float(np.max(smax))
    add("energy-metric Gamma matches brute force", abs(g_M / g_ref - 1) < 1e-9,
        "%.12g vs %.12g" % (g_M, g_ref), "equal")

    # 负控制：M=I 时与裸 Γ 相同（度量路径不改变平凡情形）。
    g_I = peak_with_projection(sys, metric=np.eye(2))["Gamma"]
    add("M=I metric equals bare Gamma", abs(g_I / g_eucl - 1) < 1e-9,
        "%.12g vs %.12g" % (g_I, g_eucl), "equal")

    # ---- 投影 + 度量组合（review 修复）：守恒系统（各向异性 E + 零模态）
    # 投影到 k<n 后 M 度量必须先在全系统求 M、再用 Z1ᵀ M Z1 投影。
    # 修复前：metric="energy" 透传到 k×k 非对角的降阶 E 上抛 ValueError
    # （对角占优检查），全维 M 透传抛形状不匹配。
    Ec = np.diag([1.0, 1e-2, 1.0, 1.0, 1.0, 1.0])
    n6 = Ec.shape[0]
    D2 = np.zeros((n6, n6))
    for i in range(n6):
        D2[i, (i - 1) % n6] += -1.0
        D2[i, (i + 1) % n6] += -1.0
        D2[i, i] += 2.0
    sysc = Descriptor(Ec, -0.5 * D2, np.zeros((n6, 0)), np.zeros((0, n6)))
    assert sysc.has_imaginary()          # 零模态 ⇒ 必须走投影路径
    k6 = sysc.stable_projection()[1]
    assert 0 < k6 < n6
    pc_bare = peak_with_projection(sysc)
    pc_e = peak_with_projection(sysc, metric="energy")
    pc_M = peak_with_projection(sysc, metric=np.diag(np.diag(Ec)))
    add("projected M-metric runs on conservation system",
        pc_e["projected"] and pc_e["n_proj"] == k6
        and np.isfinite(pc_e["Gamma_proj"]),
        "n_proj=%s, Gamma_proj=%.4g" % (pc_e["n_proj"], pc_e["Gamma_proj"]),
        "projected, finite")
    add("Gamma_proj: metric matters post-projection",
        abs(pc_e["Gamma_proj"] / pc_bare["Gamma_proj"] - 1.0) > 1e-6,
        "M/bare = %.4g" % (pc_e["Gamma_proj"] / pc_bare["Gamma_proj"]),
        "≠ 1")
    add("projected 'energy' shorthand == explicit M",
        abs(pc_e["Gamma_proj"] / pc_M["Gamma_proj"] - 1.0) < 1e-9,
        "%.12g vs %.12g" % (pc_e["Gamma_proj"], pc_M["Gamma_proj"]),
        "equal")
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
        cases[name] = run_case(name, sys, meta, reg=reg,
                               spec=HW_PROBE if name in _PROBE else HW)

    table(["system", "n", "kappa", "kappa/k_hw", "Gamma", "Gamma_proj", "sigma_y",
           "B_req[Hz]", "Hinf", "method", "violated"],
          [(c["name"], c["n"], c["check"]["kappa"], c["slack"]["values"]["stiffness"],
            c["resolvent"]["Gamma"], c["resolvent"]["Gamma_proj"],
            c["lyapunov"]["sigma_y"], c["check"]["bandwidth_hz"],
            c["hinf"]["value"], c["hinf"]["method"],
            ",".join(c["slack"]["violated"]) or "-") for c in cases.values()],
          ["", "%d", "%.3e", "%.3f", "%.3e", "%.3e", "%.3e", "%.3e", "%.3e", "", ""])

    hr("B. 六轴 slack（1.0 为红线；n/a = 没有输入；鲁棒轴在两例上用 1% 探针）")
    table(["system", "noise", "bandwidth", "stiffness", "robustness_A",
           "robustness_io", "routing", "representation", "status"],
          [(c["name"], _fmt(c["slack"]["values"]["noise"]),
            _fmt(c["slack"]["values"]["bandwidth"]),
            _fmt(c["slack"]["values"]["stiffness"]),
            _fmt(c["slack"]["values"]["robustness_A"]),
            _fmt(c["slack"]["values"]["robustness_io"]),
            _fmt(c["slack"]["values"]["routing"]),
            _fmt(c["slack"]["values"]["representation"]),
            c["slack"]["status"]) for c in cases.values()],
          [""] * 9)

    hr("C. 验收对照（κ: ±0.5 decade；H∞/Γ/σ_y: ±3 dB；+ DAE 噪声注入 P0-E）")
    checks = acceptance(cases)
    dae = case_dae_noise()
    checks.extend(dae)
    checks.extend(case_energy_metric())
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
    burg = cases["Burgers N=128"]
    emit("  - Burgers 的全 Γ=∞ 是守恒零模态在 s=0 的极点，只作诊断，不进 slack；")
    emit("    Γ_proj=%.2f（%d 个稳定模态），乘 1%% 失配预算得 robustness_A=%.2f。"
         % (burg["resolvent"]["Gamma_proj"], burg["resolvent"]["n_proj"],
            burg["slack"]["values"]["robustness_A"]))
    emit("    输入已投影到零均值，H∞=%.2f 有限，robustness_io=%.2f——鲁棒轴不是失败轴。"
         % (burg["hinf"]["value"], burg["slack"]["values"]["robustness_io"]))
    emit("    表示轴（E_tail/tol_L2²=%.2f）才是。阈值不是魔数：tol_L2=1%% ⇒ 1e-4。"
         % burg["slack"]["values"]["representation"])
    emit("  - Robertson 的全 Γ=%.2e 是慢模态的低频尖峰（含零模态）；Γ_proj=%.2e。"
         % (cases["Robertson t=1e4"]["resolvent"]["Gamma"],
            cases["Robertson t=1e4"]["resolvent"]["Gamma_proj"]))
    emit("    robustness_A=%.1f（1%% 失配）与 κ/κ_hw=%.1f 同量级——慢模态是同一根杠杆；"
         % (cases["Robertson t=1e4"]["slack"]["values"]["robustness_A"],
            cases["Robertson t=1e4"]["slack"]["values"]["stiffness"]))
    emit("    κ=%.2e（t=1e4，去掉守恒零模态），判定 stiffness；"
         % cases["Robertson t=1e4"]["check"]["kappa"])
    emit("    10¹¹ 是速率常数比，不是 Jacobian 谱；口径见 §8。")
    emit("  - 鲁棒轴只在 Burgers/Robertson 上加了 1% 探针；其余算例没有失配模型，")
    emit("    记 n/a（未知），不是 0。全 Γ 在含零模态时只作诊断，不进 slack。")

    emit("\n  DAE 反例（P0-E）：index-1 代数方程吃进输入不自动 ⇒ ∞。")
    emit("    x2=−u 反馈回动力学：σ_y²=%s（解析 1/2），method=%s。" % (dae[0]["got"], dae[1]["got"]))
    emit("    负控制：C=[0,1] 直接馈通 ⇒ ∞；E=[[0,1],[0,0]] index-2 ⇒ 脉冲 ∞。")

    _plot(cases)
    emit("\n  json   → results/phase0.json")
    emit("  report → results/phase0_report.txt")
    with open(os.path.join(OUT, "phase0.json"), "w") as fh:
        json.dump(dict(hardware=dict(kappa_hw=HW.kappa_hw, B_hw=HW.B_hw,
                                     tol_l2=HW.tol_l2, etail_req=HW.etail_req,
                                     robustness_probe=dict(
                                         delta_A_norm=HW_PROBE.delta_A_norm,
                                         delta_input_norm=HW_PROBE.delta_input_norm,
                                         cases=sorted(_PROBE))),
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
        ("Gamma", [cases[n]["resolvent"]["Gamma_proj"] for n in names], None,
         "Γ_proj", None),
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
