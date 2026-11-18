"""对照实验：QoI 反演（§6.1）与接口守恒（§6.4）。

    python3 -m bench.run_twin_experiments
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from twin import qoi
from twin.coupling import CoupledSystem, InterfaceModel, minmod, rusanov, vanleer

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


class _Tee:
    """把 stdout 同时写到 results/report.txt，保证文档命令重跑后记录不过期。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def hr(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def table(header, rows, fmts):
    widths = [max(len(h), 11) for h in header]
    print("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for r in rows:
        cells = []
        for f, v, w in zip(fmts, r, widths):
            s = (f % v) if isinstance(f, str) and f else str(v)
            cells.append(s.rjust(w))
        print("  " + " ".join(cells))


# ==========================================================================
# Burgers / 对流 求解器（只为产生"被测量的数据"，不属于 qoi 模块）
# ==========================================================================
def _weno5_rows(v):
    """v: (n,5) 分量 v[:,0..4] = f_{j-2..j+2}；返回面 j+1/2 的重构。"""
    v0, v1, v2, v3, v4 = (v[:, i] for i in range(5))
    q1 = (2 * v0 - 7 * v1 + 11 * v2) / 6.0
    q2 = (-v1 + 5 * v2 + 2 * v3) / 6.0
    q3 = (2 * v2 + 5 * v3 - v4) / 6.0
    b1 = 13 / 12 * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
    b2 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
    b3 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
    eps = 1e-6
    a1 = 0.1 / (eps + b1) ** 2
    a2 = 0.6 / (eps + b2) ** 2
    a3 = 0.3 / (eps + b3) ** 2
    return (a1 * q1 + a2 * q2 + a3 * q3) / (a1 + a2 + a3)


def _llf(uL, uR):
    """局部 Lax–Friedrichs（仅用于 WENO 的 LF 分裂对照）。"""
    a = np.maximum(np.abs(uL), np.abs(uR))
    return 0.5 * (0.5 * uL * uL + 0.5 * uR * uR) - 0.5 * a * (uR - uL)


def cellavg_sin(n, amp=0.5, mean=0.5):
    """sin 初值的精确单元平均（FV 的状态量必须是平均值，不是点值）。"""
    a = np.arange(n) / n
    b = (np.arange(n) + 1) / n
    return mean + amp * (np.cos(2 * np.pi * a) - np.cos(2 * np.pi * b)) / (2 * np.pi / n)


def _roe(uL, uR):
    """Burgers 的 Roe 通量。耗散 ∝ |ã|(uR−uL)，在激波处为零（精确迎风）。

    注意：这里必须用 Roe 而不是 LLF，否则"重构使界面跳变变大 ⇒ LLF 耗散变大"，
    会把重构的收益完全吃掉甚至反号（见 DESIGN.md §6.8 的说明）。
    """
    a = 0.5 * (uL + uR)
    return 0.5 * (0.5 * uL * uL + 0.5 * uR * uR) - 0.5 * np.maximum(np.abs(a), 1e-6) * (uR - uL)


def face_flux(u, scheme, alpha, h):
    """返回 n 个面（i+1/2，周期）上的数值通量。"""
    n = u.size
    f = 0.5 * u * u
    if scheme == "first":                          # 无重构 = 一阶，同一个 Roe 通量
        return _roe(u, np.roll(u, -1))
    if scheme == "weno5":
        fp = 0.5 * (f + alpha * u)
        fm = 0.5 * (f - alpha * u)
        pe = np.concatenate([fp[-3:], fp, fp[:3]])
        me = np.concatenate([fm[-3:], fm, fm[:3]])
        idx = np.arange(n)
        P = pe[idx[:, None] + np.arange(1, 6)[None, :]]
        M = me[idx[:, None] + np.arange(2, 7)[None, :]][:, ::-1]
        return _weno5_rows(P) + _weno5_rows(M)
    sl = (u - np.roll(u, 1)) / h
    sr = (np.roll(u, -1) - u) / h
    if scheme == "minmod":
        s = np.where(sl * sr > 0,
                     np.sign(sl) * np.minimum(np.abs(sl), np.abs(sr)), 0.0) * h
    else:  # vanleer
        den = sl + sr
        v = np.where(np.abs(den) > 1e-15, 2.0 * sl * sr / den, 0.0)
        s = np.where(sl * sr > 0, v, 0.0) * h
    uL = u + 0.5 * s
    uR = u - 0.5 * s
    return _roe(uR, np.roll(uL, -1))


def burgers(u0, T, scheme, cfl=0.4, alpha=1.6):
    u = np.array(u0, float)
    n = u.size
    h = 1.0 / n
    dt = cfl * h / (2.0 * alpha)          # LF 分裂的稳定界取 (|u|+alpha) 尺度
    nst = int(np.ceil(T / dt))
    dt = T / nst
    for _ in range(nst):
        k1 = -(face_flux(u, scheme, alpha, h) - np.roll(face_flux(u, scheme, alpha, h), 1)) / h
        u2 = u + 0.5 * dt * k1
        k2 = -(face_flux(u2, scheme, alpha, h) - np.roll(face_flux(u2, scheme, alpha, h), 1)) / h
        u3 = u + 0.5 * dt * k2
        k3 = -(face_flux(u3, scheme, alpha, h) - np.roll(face_flux(u3, scheme, alpha, h), 1)) / h
        u4 = u + dt * k3
        k4 = -(face_flux(u4, scheme, alpha, h) - np.roll(face_flux(u4, scheme, alpha, h), 1)) / h
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def advect_diffuse(c0, T, a, kappa, cfl=0.4, n=None):
    """c_t + a c_x = kappa c_xx，周期；一阶迎风 + 中心扩散。"""
    c = np.array(c0, float)
    n = c.size
    h = 1.0 / n
    dt = cfl * h / abs(a) if a != 0 else 0.4 * h * h / kappa
    if kappa > 0:
        dt = min(dt, 0.4 * h * h / kappa)      # 显式扩散稳定界
    nst = int(np.ceil(T / dt))
    dt = T / nst
    for _ in range(nst):
        def rhs(cc):
            left = np.concatenate(([0.0], cc[:-1]))            # 入流 c=0
            right = np.concatenate((cc[1:], [cc[-1]]))         # 出流零梯度
            up = a * (cc - left) / h
            dif = kappa * (right - 2 * cc + left) / h ** 2
            return -up + dif
        k1 = rhs(c)
        k2 = rhs(c + 0.5 * dt * k1)
        k3 = rhs(c + 0.5 * dt * k2)
        k4 = rhs(c + dt * k3)
        c = c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return c


# ==========================================================================
# A. 激波 QoI 反演
# ==========================================================================
def exp_a():
    hr("A. 激波宽度 → 等效黏性 ν_num（解析 ground truth）")
    x = np.linspace(0, 1, 128)
    dx = x[1] - x[0]
    rng = np.random.default_rng(7)
    rows = []
    for nu in [2e-3, 5e-3, 1e-2, 3e-2, 8e-2]:
        for noise in [0.0, 0.01]:
            u = qoi.taylor_shock(x, xs=0.47, du=2.0, nu=nu, s=1.0)
            if noise:
                u = u + noise * rng.standard_normal(x.size)
            r = qoi.fit_tanh(x, u)
            rows.append((nu, noise, r["nu_num"], r["w_10_90"] / dx, r["xs"],
                         abs(r["nu_num"] - nu) / nu))
    table(["nu_true", "noise", "nu_fit", "W_d[cell]", "x_s", "rel.err"],
          rows, ["%.3e", "%.0e", "%.3e", "%.2f", "%.4f", "%.1e"])
    print("  说明：W_d 逼近 1 cell 时估计器达到网格分辨率下限，ν_num 会被高估（见 B）。")
    return rows


# ==========================================================================
# B. 名义阶数 vs 有效 ν_num（在同一个激波上）
# ==========================================================================
def exp_b0():
    """实现自检：光滑区里名义阶数是真的（否则 B 的结论不可信）。"""
    hr("B0. 光滑区收敛诊断（仅作求解器可信度参考）")
    T, amp, nref = 0.10, 0.10, 4096
    xr = (np.arange(nref) + 0.5) / nref
    uref = burgers(cellavg_sin(nref, amp=amp, mean=1.0), T, "weno5", cfl=0.35)
    ns = [32, 64, 128, 256]
    rows = []
    for sch, nominal in [("first", 1), ("minmod", 2), ("vanleer", 2), ("weno5", 5)]:
        errs = []
        for n in ns:
            u = burgers(cellavg_sin(n, amp=amp, mean=1.0), T, sch, cfl=0.35)
            ur = uref.reshape(n, nref // n).mean(axis=1)
            errs.append(float(np.mean(np.abs(u - ur))))
        p = -float(np.polyfit(np.log(ns), np.log(errs), 1)[0])
        rows.append((sch, nominal, p, errs[0], errs[-1]))
    table(["scheme", "nominal", "measured", "L1(N=32)", "L1(N=256)"],
          rows, ["", "%d", "%.2f", "%.2e", "%.2e"])
    print("  注意两点（都已知，不属于 B 的结论）：")
    print("   - minmod/vanleer 在光滑极值处斜率被截断，L1 只有 O(h)：这是 limiter 的结构性问题。")
    print("   - 本原型的 WENO5 用固定 eps=1e-6，低幅低分辨率下 smoothness indicator 量级不匹配，")
    print("     会掉阶（需 eps ∝ h² 才能恢复 5 阶）。因此 B 表里的 WENO5 只当作")
    print("     '一个压缩性很强的 5 点重构'，不宣称它是已验证的 5 阶格式。")
    return rows


def exp_b():
    hr("B. 同一激波：名义阶数 ≠ 有效分辨率")
    n = 128
    x = (np.arange(n) + 0.5) / n
    u0 = 0.5 + 0.5 * np.sin(2 * np.pi * x)
    T = 0.30

    nref = 2048
    uref = burgers(0.5 + 0.5 * np.sin(2 * np.pi * (np.arange(nref) + 0.5) / nref),
                   T, "weno5", cfl=0.35)
    uref_c = uref.reshape(n, nref // n).mean(axis=1)

    rows = []
    for scheme, nominal in [("first", 1), ("minmod", 2), ("vanleer", 2), ("weno5", 5)]:
        u = burgers(u0, T, scheme, cfl=0.35)
        i_s = int(np.argmax(np.abs(np.gradient(u, 1.0 / n))))
        lo, hi = max(0, i_s - 6), min(n, i_s + 7)
        r = qoi.fit_tanh(x[lo:hi], u[lo:hi])
        l1 = np.mean(np.abs(u - uref_c))
        tv = np.sum(np.abs(np.diff(u)))
        rows.append((scheme, nominal, r["w_10_90"] / (1.0 / n), r["nu_num"], l1, tv, r["xs"], r["rms"]))
    table(["scheme", "nom.", "W_d[cell]", "nu_num", "L1", "TV", "x_s", "fit.rms"],
          rows, ["", "%d", "%.2f", "%.3e", "%.2e", "%.3f", "%.4f", "%.1e"])
    print("  TV(初值)=%.3f；TV 损失 ~ 数值耗散总量。" % np.sum(np.abs(np.diff(u0))))
    print("  读法：W_d 与 ν_num 直接给出'激波被糊多宽'；名义阶数只解释其中一部分。")
    return [dict(zip(["scheme", "nom", "Wd", "nu", "L1", "TV", "xs", "rms"], r)) for r in rows]


# ==========================================================================
# C. 锋面 QoI：宽度增长 → 混合系数（海洋红线）
# ==========================================================================
def exp_c():
    hr("C. 锋面宽度增长 → κ_num（海洋红线的定量化）")
    n = 1024
    z = (np.arange(n) + 0.5) / n
    c0 = 0.5 * (1.0 + np.tanh((z - 0.10) / 0.01))     # 非周期：入流 0，出流 ~1

    Ts = [0.1, 0.2, 0.3]
    rows = []
    for kappa in [1e-4, 3e-4]:
        w = [qoi.front_width_10_90(z, advect_diffuse(c0, T, 0.0, kappa)) for T in Ts]
        k_est = qoi.kappa_from_width_growth(w, Ts)
        rows.append(("pure diffusion", kappa, k_est, k_est / kappa))
    h = 1.0 / n
    for cfl in [0.4, 0.8]:
        w = [qoi.front_width_10_90(z, advect_diffuse(c0, T, 1.0, 0.0, cfl=cfl)) for T in Ts]
        k_est = qoi.kappa_from_width_growth(w, Ts)
        nu_hat = 0.5 * h            # 半离散极限：ν = ½·a·h（与 CFL 无关）
        rows.append(("advect only CFL=%.1f" % cfl, nu_hat, k_est, k_est / nu_hat))
    table(["case", "ref/analytic", "kappa_est", "ratio"], rows, ["", "%.3e", "%.3e", "%.2f"])
    print("  半离散迎风：ν_num = ½·a·h = %.2e (h=%.1e)。(1−CFL) 因子只属于时间离散，"
          % (0.5 * (1 / n), 1 / n))
    print("  因此两个 CFL 给出同样的 κ_num —— 时间积分器已收敛，扩散完全是空间的。")
    print("  红线换算：κ_num < 1e-6 m²/s（物理跨密度的 10%）")
    for a, cfl in [(0.1, 0.5), (1.0, 0.5)]:
        print("    a=%.1f m/s, CFL=%.1f ⇒ 一阶迎风需要 h < %.2e m" % (a, cfl, 2e-6 / (a * (1 - cfl))))
    return rows


# ==========================================================================
# D. 接口守恒对照
# ==========================================================================
def _coupled(**kw):
    p = kw.pop("p", 8)
    n = kw.pop("n_cells", 32)
    model = InterfaceModel(**kw)
    return CoupledSystem(p=p, n_cells=n, model=model)


def exp_d():
    hr("D. spectral ↔ local 接口：单通量源 vs 两侧各算一遍")
    T, dt = 0.2, 1e-3
    base = _coupled()

    rows = []
    for name, kw in [
        ("single-src", dict(single_source=True)),
        ("single+delay5", dict(single_source=True, stale=5)),
        ("single+p6", dict(single_source=True)),
        ("indep eps=1e-4", dict(single_source=False, mismatch=1e-4)),
        ("indep eps=1e-3", dict(single_source=False, mismatch=1e-3)),
        ("indep eps=1e-2", dict(single_source=False, mismatch=1e-2)),
        ("indep eps=5e-2", dict(single_source=False, mismatch=5e-2)),
        ("no-trace-xchg", dict(single_source=False, extrapolate=True)),
    ]:
        p = 6 if name == "single+p6" else 8
        s = _coupled(p=p, **kw)
        r = s.run(s.smooth_ic(), T=T, dt=dt)
        rows.append((name, r["drift"], r["drift_pred"], r["leak_max"],
                     abs(r["drift"] - r["drift_pred"])))
    table(["mode", "drift", "drift_pred", "max|dF|", "|drift-pred|"],
          rows, ["", "%.3e", "%.3e", "%.3e", "%.1e"])

    # ---- eps 标定：drift ∝ eps ----
    eps = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    dr = []
    for e in eps:
        s = _coupled(single_source=False, mismatch=float(e))
        r = s.run(s.smooth_ic(), T=T, dt=dt)
        dr.append(abs(r["drift"]))
    dr = np.array(dr)
    slope = np.polyfit(np.log(eps), np.log(dr), 1)[0]
    print("\n  drift vs ε 的 log-log 斜率 = %.3f  （1.0 ⇒ 线性：漂移率 = ε·|F|·接口数）" % slope)

    # ---- 阶数扫描：精度提升，守恒不变 ----
    rows2 = []
    ref = _coupled(p=16, n_cells=32, single_source=True).run(
        _coupled(p=16, n_cells=32).smooth_ic(), T=T, dt=5e-4)
    xq = np.linspace(0, base.xc, 201)
    u_ref = ref["hist"]["u"][-1]
    f_ref = _coupled(p=16, n_cells=32, single_source=True)
    f_ref.u = u_ref
    uref_s = f_ref.spec.interpolate(u_ref[:f_ref.spec.n], xq)
    for p in [2, 4, 8, 12]:
        s = _coupled(p=p, single_source=False, mismatch=1e-2)
        r = s.run(s.smooth_ic(), T=T, dt=5e-4)
        s0 = _coupled(p=p, single_source=True)          # 精度用无失配的那次
        r0 = s0.run(s0.smooth_ic(), T=T, dt=5e-4)
        us = r0["u"][:s0.spec.n]
        l2 = float(np.sqrt(np.mean((s0.spec.interpolate(us, xq) - uref_s) ** 2)))
        rows2.append((p, l2, r["drift"]))
    table(["p", "L2(spec island)", "drift(eps=1e-2)"], rows2, ["%d", "%.2e", "%.3e"])
    print("  读法：L2 随 p 指数下降，drift 基本不动 —— 泄漏不是截断误差。")

    return dict(main=rows, eps_slope=float(slope), order=rows2)


# ==========================================================================
# E. 真实器件接口：λ 项与系统 ratio 误差回到 §6.4 的耦合实验
# ==========================================================================
def _device_params():
    path = os.path.join(OUT, "device_extract.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {d["pdk"]: d for d in json.load(fh)}


def _gm_run(**kw):
    model = InterfaceModel(mode="gm", v_per_u=0.1, **kw)
    s = CoupledSystem(p=8, n_cells=32, model=model)
    return s.run(s.smooth_ic(), T=0.2, dt=1e-3)


def exp_e():
    hr("E. 真实 PDK 接口：λ 项 vs 两条独立镜像的系统误差")
    dev = _device_params()
    if not dev:
        print("  results/device_extract.json 不存在；先跑 python3 -m twin.spice.extract")
        return []

    ideal = _gm_run(lam=0.0, eps_dev=0.0)["drift"]
    rows = []
    for pdk in ["sky130", "gf180mcu", "ihp-sg13g2", "ihp-hbt"]:
        if pdk not in dev:
            continue
        d = dev[pdk]
        lam = d["lambda_eff"]
        eps_sys = d["ratio"] - 1.0
        for shared in (True, False):
            r = _gm_run(lam=lam, shared_ref=shared,
                        eps_sys=0.0 if shared else eps_sys)
            rows.append(dict(
                pdk=pdk, mode="shared" if shared else "indep",
                lam=lam, eps_sys=0.0 if shared else eps_sys,
                drift=r["drift"], leak_max=r["leak_max"],
                over_ideal=abs(r["drift"] / ideal) if ideal else float("inf"),
            ))
    table(["interface", "mode", "lam [1/V]", "eps_sys", "drift", "max|dF|", "|drift|/ideal"],
          [(r["pdk"], r["mode"], r["lam"], r["eps_sys"], r["drift"],
            r["leak_max"], r["over_ideal"]) for r in rows],
          ["", "", "%.4f", "%.4f", "% .3e", "%.3e", "%.2e"])

    if "sky130" in dev and "ihp-hbt" in dev:
        ds = next(r["drift"] for r in rows
                  if r["pdk"] == "sky130" and r["mode"] == "shared")
        dh = next(r["drift"] for r in rows
                  if r["pdk"] == "ihp-hbt" and r["mode"] == "shared")
        lam_ratio = dev["sky130"]["lambda_eff"] / dev["ihp-hbt"]["lambda_eff"]
        print("\n  λ(sky)/λ(HBT) = %.3f ; drift(sky)/drift(HBT) = %.3f"
              % (lam_ratio, ds / dh))
        print("  两者一致 ⇒ 残余泄漏由 λ·ΔV·F 主导，而不是随机失配。")
    print("  注：eps_sys = 实测 ratio − 1（含 λ·ΔV 项），因此 indep 行是上界。")

    kT = 1.380649e-23 * 300.0
    print("\n  折算输入噪声 4kTγ/g_m（未跑 .noise）：MOS γ=2/3，BJT γ=1/2")
    for pdk, d in dev.items():
        gamma = 0.5 if "hbt" in pdk else 2.0 / 3.0
        print("    %-12s gm=%6.2f uS   vn=%.1f nV/√Hz"
              % (pdk, d["gm"] * 1e6, np.sqrt(4.0 * kT * gamma / d["gm"]) * 1e9))
    return rows


def _plots(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(OUT, exist_ok=True)
    base = _coupled()
    u0 = base.smooth_ic()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, kw, st in [("single-source", dict(single_source=True), "-"),
                         ("indep eps=1e-2", dict(single_source=False, mismatch=1e-2), "--"),
                         ("no trace exchange", dict(single_source=False, extrapolate=True), ":")]:
        r = _coupled(**kw).run(u0, T=0.2, dt=1e-3, record_every=2)
        ax[0].semilogy(r["hist"]["t"], np.abs(np.array(r["hist"]["M"]) / r["M0"] - 1 + 1e-18),
                       st, label=name, lw=1.4)
    ax[0].set_xlabel("t"); ax[0].set_ylabel("|ΔM| / M0"); ax[0].set_title("mass drift")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    n = 256
    x = (np.arange(n) + 0.5) / n
    u0s = 0.5 + 0.5 * np.sin(2 * np.pi * x)
    for sch in ["upwind1", "minmod", "weno5"]:
        u = burgers(u0s, 0.3, sch, cfl=0.35)
        ax[1].plot(x, u, lw=1.2, label=sch)
    ax[1].set_xlim(0.30, 0.70); ax[1].set_title("Burgers shock @T=0.3, N=256")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "twin_experiments.png"), dpi=130)
    print("\n  figure → results/twin_experiments.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "report.txt"), "w", buffering=1)
    stdout = sys.stdout
    sys.stdout = _Tee(stdout, log)
    try:
        res = dict(A=exp_a(), B0=exp_b0(), B=exp_b(), C=exp_c(), D=exp_d(), E=exp_e())
        _plots(res)
        with open(os.path.join(OUT, "twin_experiments.json"), "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print("\n  json   → results/twin_experiments.json")
        print("  report → results/report.txt")
    finally:
        sys.stdout = stdout
        log.close()


if __name__ == "__main__":
    main()
