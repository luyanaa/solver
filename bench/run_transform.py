"""Phase 4.5 验收：adaptive representation 与 TransformContract（DESIGN.md §5.7–§5.9）。

四组，对应 DESIGN 的 4.5A–4.5E：

  A  WENO 三算子：sensor → σ → conservative face flux；守恒与无 chatter
  A' Riemann 精确解参考（P0-F）：L1/L2/x_s 收敛、熵产生率 = (uL−uR)³/12、
     熵不等式取代 "ΣV_i² 守恒"（shock 形成后守恒是错误判据）
  B  lifting wavelet：W / W_inv / W_star；isometry、PR、adjoint 分开验证
  C  transposeable edge：A↔Aᵀ 与离散 adjoint 梯度
  D  scale-dependent poles：mode C 与 mode g_m 的 g_m/噪声 trade-off
  E  Gram calibration：G_T = T̃*T̃，G_T^{-1/2} 白化

运行：python3 -m bench.run_transform
输出：results/transform.json、results/transform_report.txt
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from twin.qoi import fit_tanh
from twin.transform import (
    LiftingWavelet, SigmaTracker, TransposeableEdge,
    adjoint_ode_gradient, blend_face_flux, central_face_flux,
    conservation_drift, count_threshold_crossings, edge_matrix,
    finite_difference_gradient, gram_calibrate, isometry_error,
    muscl_face_flux, quadratic_gradient, rhs_from_face_flux, rk4_matrix,
    scale_time_error, sigma_from_indicator, smoothness_indicator,
    stationary_haar, stationary_haar_inverse, tracker_rate_bounds,
    weno5_face_flux,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
_LINES = []
CASES = {}


def emit(s=""):
    print(s)
    _LINES.append(s)


def hr(title):
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    emit(fmt.format(*headers))
    emit("  ".join("-" * w for w in widths))
    for row in rows:
        emit(fmt.format(*[str(v) for v in row]))


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _rk4(u, dt, rhs):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# 4.5A WENO 三算子
# ---------------------------------------------------------------------------
def case_weno():
    n = 256
    length = 1.0
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    u0 = 0.5 + 0.5 * np.sin(2.0 * np.pi * x)
    flux = lambda v: 0.5 * v * v

    t_end = 0.5
    cfl = 0.2
    dt = cfl * dx / max(abs(u0))
    nsteps = int(np.ceil(t_end / dt))
    dt = t_end / nsteps

    def rhs_ref(u):
        a = max(float(np.max(np.abs(u))), 1e-12)
        return rhs_from_face_flux(weno5_face_flux(u, flux, a), dx)

    def rhs_adapt(u, sig):
        a = max(float(np.max(np.abs(u))), 1e-12)
        Fs = central_face_flux(u, flux)
        Fw = muscl_face_flux(u, flux, a)
        return rhs_from_face_flux(blend_face_flux(Fs, Fw, sig), dx)

    u_ref = u0.copy()
    u_ad = u0.copy()
    tracker = SigmaTracker(beta0=0.15, k=16.0, hysteresis=0.08,
                           max_rate=400.0, release_rate=10.0)
    sig_hist = []
    # 熵不等式（P0-F 修正）：周期域上 d/dt ∫u²/2 dx = −Σ D_shock ≤ 0 严格成立
    #（熵对 U=u²/2，熵流 q=u³/3；周期边界熵流为零）。
    # 原来的 "ΣV_i² 守恒" 检查在 shock 形成后是错的——shock 处必须有严格耗散。
    E_ad = float(np.sum(0.5 * u_ad * u_ad) * dx)
    E_ref = float(np.sum(0.5 * u_ref * u_ref) * dx)
    dE_ad, dE_ref = [], []
    for _ in range(nsteps):
        u_ref = _rk4(u_ref, dt, rhs_ref)
        E_ref_new = float(np.sum(0.5 * u_ref * u_ref) * dx)
        dE_ref.append(E_ref_new - E_ref)
        E_ref = E_ref_new
        s = smoothness_indicator(u_ad)
        sig = tracker.update(s, dt)
        sig_hist.append(sig.copy())
        u_ad = _rk4(u_ad, dt, lambda v: rhs_adapt(v, sig))
        E_ad_new = float(np.sum(0.5 * u_ad * u_ad) * dx)
        dE_ad.append(E_ad_new - E_ad)
        E_ad = E_ad_new
    dE_ad = np.asarray(dE_ad)
    dE_ref = np.asarray(dE_ref)

    s_final = smoothness_indicator(u_ad)
    sig_final = sigma_from_indicator(s_final, beta0=0.15, k=16.0)
    i_ad = int(np.argmax(np.abs(np.diff(u_ad))))
    i_ref = int(np.argmax(np.abs(np.diff(u_ref))))
    x_ad = (i_ad + 0.5) * dx
    x_ref = (i_ref + 0.5) * dx

    # 静态 sensor：理想 step 与光滑 sine 各测一次；避开周期边界
    u_step = (x > 0.5).astype(float)
    s_step = smoothness_indicator(u_step)
    sig_step = sigma_from_indicator(s_step, beta0=0.15, k=16.0)
    i_step = n // 2

    u_sine = np.sin(2.0 * np.pi * x)
    s_sine = smoothness_indicator(u_sine)
    sig_sine = sigma_from_indicator(s_sine, beta0=0.15, k=16.0)
    smooth_mask = (np.abs(x - 0.5) > 0.1) & (np.abs(x - 0.5) < 0.45)

    hist = np.asarray(sig_hist)
    crossings = count_threshold_crossings(hist, 0.5)

    return dict(
        n=n, nsteps=nsteps, dt=dt,
        cfl=cfl,
        shock_sigma_static=float(sig_step[i_step]),
        smooth_sigma_static=float(np.max(sig_sine[smooth_mask])),
        shock_sigma_final=float(sig_final[i_ad]),
        smooth_sigma_final=float(np.max(sig_final[np.abs(x - x_ad) > 0.15])),
        conservation_drift=float(conservation_drift(u_ad, u0, dx)),
        conservation_drift_ref=float(conservation_drift(u_ref, u0, dx)),
        shock_x_ad=x_ad, shock_x_ref=x_ref,
        shock_x_error=abs(x_ad - x_ref),
        l2_vs_weno5=float(np.sqrt(np.mean((u_ad - u_ref) ** 2))),
        max_cell_crossings=int(np.max(crossings)),
        mean_cell_crossings=float(np.mean(crossings)),
        # 熵不等式：每步 ΔE ≤ tol（周期域，shock 形成后应有严格耗散）
        entropy_dE_max_ad=float(np.max(dE_ad)),
        entropy_dE_min_ad=float(np.min(dE_ad)),
        entropy_dE_max_ref=float(np.max(dE_ref)),
        entropy_dE_min_ref=float(np.min(dE_ref)),
        entropy_total_dE_ad=float(np.sum(dE_ad)),
        entropy_total_dE_ref=float(np.sum(dE_ref)),
    )


# ---------------------------------------------------------------------------
# 4.5A' Riemann 精确解参考（review P0-F：两套实现一致 ≠ 接近真解）
# ---------------------------------------------------------------------------
# 扩展域上的 inviscid Burgers Riemann 问题：uL=1（左），uR=0（右）。
# 精确熵解：单一 shock，x_s(t) = x0 + s·t，s = (uL+uR)/2（Rankine–Hugoniot）。
# 熵对 U=u²/2，熵流 q=u³/3。shock 耗散率
#   D = s·[U] − [q] = (uL−uR)³/12 = 1/12   ([g] ≡ g_R − g_L)
# 推导：s(uR²−uL²)/2 − (uR³−uL³)/3
#      = (uR−uL)·[(uL+uR)²/4 − (uL²+uLuR+uR²)/3] = (uL−uR)³/12。
# 扩展域（左右边界固定在常数态）上 d/dt∫U dx = q(uL) − q(uR) − D；
# 离散检查：每步熵产生 P_step = (q_L − q_R)·dt − ΔE ≈ D·dt ≥ 0。
# 注意：周期域上 uL>uR 的 Riemann 问题不含单一 shock（回绕点是稀疏波），
# 故用扩展域 + f(uL)/f(uR) 边界通量。
def _entropy_of(u, dx):
    return float(np.sum(0.5 * np.asarray(u) ** 2) * dx)


def _riemann_rhs(u, scheme, dx, uL, uR, sig=None):
    """扩展域 rhs：内部 face 用所选格式，边界通量固定为 f(uL)/f(uR)。"""
    flux = lambda v: 0.5 * v * v
    a = max(float(np.max(np.abs(u))), 1e-12)
    if scheme == "weno5":
        F = weno5_face_flux(u, flux, a)
    elif scheme == "adaptive":
        Fs = central_face_flux(u, flux)
        Fw = muscl_face_flux(u, flux, a)
        F = blend_face_flux(Fs, Fw, sig)
    else:
        raise ValueError(scheme)
    Fi = F[:-1]          # 内部 face：face i 在 cell i 与 i+1 之间
    n = u.size
    r = np.empty(n)
    r[0] = (0.5 * uL * uL - Fi[0]) / dx
    r[1:-1] = -(Fi[1:] - Fi[:-1]) / dx
    r[-1] = (Fi[-1] - 0.5 * uR * uR) / dx
    return r


def _run_riemann(n, scheme, t_end=0.5, x0=0.25, uL=1.0, uR=0.0, cfl=0.2):
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    s = 0.5 * (uL + uR)
    xs_exact = x0 + s * t_end
    u = np.where(x < x0, uL, uR)
    dt = cfl * dx / max(abs(uL), abs(uR), 1e-12)
    nsteps = int(np.ceil(t_end / dt))
    dt = t_end / nsteps

    tracker = SigmaTracker(beta0=0.15, k=16.0, hysteresis=0.08,
                           max_rate=400.0, release_rate=10.0)
    E0 = _entropy_of(u, dx)
    E = E0
    dE = []
    for _ in range(nsteps):
        sig = None
        if scheme == "adaptive":
            sig = tracker.update(smoothness_indicator(u), dt)
        u = _rk4(u, dt, lambda v: _riemann_rhs(v, scheme, dx, uL, uR, sig))
        E_new = _entropy_of(u, dx)
        dE.append(E_new - E)
        E = E_new
    dE = np.asarray(dE)

    qL, qR = uL ** 3 / 3.0, uR ** 3 / 3.0
    P_step = (qL - qR) * dt - dE          # 每步熵产生，期望 ≈ D·dt ≥ 0
    P_rate = P_step / dt                  # 每步熵产生率，期望 ≈ D
    D = (uL - uR) ** 3 / 12.0
    P_rate_avg = ((qL - qR) * t_end - (E - E0)) / t_end

    # 误差与收敛量（相对精确熵解）
    ue = np.where(x < xs_exact, uL, uR)
    l1 = float(np.sum(np.abs(u - ue)) * dx)
    l2 = float(np.sqrt(np.sum((u - ue) ** 2) * dx))
    i_sh = int(np.argmax(np.abs(np.diff(u))))
    x_sh = (i_sh + 1) * dx                # jump 中点
    sh_err_cells = abs(x_sh - xs_exact) / dx
    # 扩展域质量不是守恒量：d/dt∫u dx = f(uL) − f(uR)（边界通量）。
    # 正确检查是质量随边界通量精确演化（RK4 时间积分误差 + roundoff 级）。
    mass0 = uL * x0 + uR * (1.0 - x0)
    mass_pred = mass0 + (0.5 * uL ** 2 - 0.5 * uR ** 2) * t_end
    mass_drift = abs(float(np.sum(u) * dx) - mass_pred) / max(abs(mass0), 1e-30)

    # 光滑区 L1（排除 shock 邻域 ±8 cells）
    smooth = np.ones(n, dtype=bool)
    smooth[max(i_sh - 8, 0):i_sh + 9] = False
    l1_smooth = float(np.sum(np.abs((u - ue)[smooth])) * dx)

    # ν_num：tanh 拟合 shock 剖面（twin.qoi 现成机制）
    win = np.abs(x - xs_exact) < 0.25
    try:
        fit = fit_tanh(x[win], u[win])
        nu_num = float(fit["nu_num"]) if fit["ok"] else float("nan")
        w_1090 = float(fit["w_10_90"]) if fit["ok"] else float("nan")
        tanh_ok = bool(fit["ok"])
    except Exception:
        nu_num = w_1090 = float("nan")
        tanh_ok = False

    return dict(
        n=n, scheme=scheme, nsteps=nsteps, dt=dt,
        l1=l1, l2=l2, l1_smooth=l1_smooth,
        shock_x=x_sh, shock_x_exact=xs_exact, shock_err_cells=sh_err_cells,
        mass_drift=mass_drift,
        entropy_rate_avg=P_rate_avg, entropy_rate_min=float(np.min(P_rate)),
        entropy_rate_max=float(np.max(P_rate)), entropy_analytic=D,
        nu_num=nu_num, width_10_90=w_1090, tanh_ok=tanh_ok,
        u_final=u, x=x,
    )


def case_riemann():
    uL, uR = 1.0, 0.0
    # 熵常数交叉验证（代数式 vs 展开式）
    s = 0.5 * (uL + uR)
    D_cross = abs((s * (uR ** 2 - uL ** 2) / 2.0 - (uR ** 3 - uL ** 3) / 3.0)
                  - (uL - uR) ** 3 / 12.0)

    runs = {}
    for scheme in ("weno5", "adaptive"):
        for n in (128, 256, 512):
            runs[(scheme, n)] = _run_riemann(n, scheme, uL=uL, uR=uR)

    rates = {}
    for scheme in ("weno5", "adaptive"):
        r1 = runs[(scheme, 128)]
        r2 = runs[(scheme, 256)]
        r4 = runs[(scheme, 512)]
        rates[scheme] = dict(
            l1=float(np.log2(r1["l1"] / r2["l1"])) if r2["l1"] > 0 else float("nan"),
            l1_2=float(np.log2(r2["l1"] / r4["l1"])) if r4["l1"] > 0 else float("nan"),
            l1_smooth=float(np.log2(r1["l1_smooth"] / r2["l1_smooth"]))
            if r2["l1_smooth"] > 0 else float("nan"),
            l1_smooth_2=float(np.log2(r2["l1_smooth"] / r4["l1_smooth"]))
            if r4["l1_smooth"] > 0 else float("nan"),
        )

    return dict(uL=uL, uR=uR, D_cross=D_cross, runs=runs, rates=rates,
                D_analytic=(uL - uR) ** 3 / 12.0)


# ---------------------------------------------------------------------------
# 4.5F tracker rate bounds（外部 review 的可证伪预测）
# ---------------------------------------------------------------------------
# review 预测：r_attack ≳ |s|/Δx（advection 下界），
#              r_release ≤ π²·ν_num·Δσ_hyst/Δx²（耗散上界）。
# 对 N=256 Burgers（Δx=1/256, |s|≈0.5, ν_num≈3e-3）review 给出
# r_attack ≳ 300/s, r_release ≲ 15/s — 声称 bracket 住实测最优 400/10。
# 本 case 用实测 ν_num（Riemann run 的 tanh 拟合）重算边界，并在
# (r_attack, r_release) 网格上扫 SigmaTracker：违反边界的组合应出现
# 更多 σ 阈值穿越（chatter）。若整个网格穿越数无响应，边界是空洞的，
# 必须如实报告。
def _run_adaptive_burgers(n, t_end, max_rate, release_rate,
                          beta0=0.15, k=16.0, hysteresis=0.08, cfl=0.2):
    """case_weno 的可参数化版本（短 T），返回 crossings / drift / 剖面。"""
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    u0 = 0.5 + 0.5 * np.sin(2.0 * np.pi * x)
    flux = lambda v: 0.5 * v * v
    dt0 = cfl * dx / max(abs(u0))
    nsteps = int(np.ceil(t_end / dt0))
    dt = t_end / nsteps

    def rhs_adapt(u, sig):
        a = max(float(np.max(np.abs(u))), 1e-12)
        Fs = central_face_flux(u, flux)
        Fw = muscl_face_flux(u, flux, a)
        return rhs_from_face_flux(blend_face_flux(Fs, Fw, sig), dx)

    u = u0.copy()
    tracker = SigmaTracker(beta0=beta0, k=k, hysteresis=hysteresis,
                           max_rate=max_rate, release_rate=release_rate)
    sig_hist = []
    E = float(np.sum(0.5 * u * u) * dx)
    dE = []
    for _ in range(nsteps):
        s = smoothness_indicator(u)
        sig = tracker.update(s, dt)
        sig_hist.append(sig.copy())
        u = _rk4(u, dt, lambda v: rhs_adapt(v, sig))
        E_new = float(np.sum(0.5 * u * u) * dx)
        dE.append(E_new - E)
        E = E_new
    crossings = count_threshold_crossings(np.asarray(sig_hist), 0.5)
    return dict(
        n=n, nsteps=nsteps, dt=dt,
        max_crossings=int(np.max(crossings)),
        mean_crossings=float(np.mean(crossings)),
        n_cells_above_median=int(np.sum(crossings > np.median(crossings))),
        conservation_drift=float(conservation_drift(u, u0, dx)),
        entropy_dE_max=float(np.max(dE)),
        # 与 review 预测相关的状态量：实测最大特征波速作 s_shock
        s_max=float(np.max(np.abs(u))),
        u_final=u, x=x,
    )


def case_tracker_bounds(n=256):
    # ν_num：用现有 Riemann adaptive run 的 tanh 拟合实测值；不可用时退回
    # 理论 ½·alpha·h（alpha = max|u|，Rusanov/MUSCL 的实际取法）。
    riemann = CASES.get("riemann", {}).get("runs", {})
    ref = riemann.get(("adaptive", n))
    if ref is not None and ref["tanh_ok"] and np.isfinite(ref["nu_num"]):
        nu_num = float(ref["nu_num"])
        nu_source = "fit_tanh (riemann adaptive N=256)"
    else:
        nu_num = 0.5 * 1.0 * (1.0 / n)
        nu_source = "theory 0.5*alpha*h (alpha=1)"
    dx = 1.0 / n
    s_shock = 0.5  # Rankine–Hugoniot for uL=1, uR=0（bench 激波）
    dsigma_hyst = 0.08  # bench tracker 实际配置（非默认 0.01）

    r_att_min, r_rel_max = tracker_rate_bounds(dx, nu_num, s_shock, dsigma_hyst)

    # review 原文数字（Δσ_hyst≈0.008 的假设）——保留作对照
    r_rel_max_review = tracker_rate_bounds(dx, nu_num, s_shock, 0.008)[1]

    r_attack_grid = [50, 200, 400, 800]
    r_release_grid = [2, 10, 40, 160]
    sweep = {}
    for ra in r_attack_grid:
        for rr in r_release_grid:
            d = _run_adaptive_burgers(n, t_end=0.5, max_rate=ra,
                                      release_rate=rr)
            d["infeasible"] = bool(ra < r_att_min or rr > r_rel_max)
            sweep[(ra, rr)] = d
    opt = sweep[(400, 10)]

    return dict(
        n=n, dx=dx, s_shock=s_shock, dsigma_hyst=dsigma_hyst,
        nu_num=nu_num, nu_source=nu_source,
        r_attack_min=r_att_min, r_release_max=r_rel_max,
        r_release_max_review_dsigma008=r_rel_max_review,
        r_attack_grid=r_attack_grid, r_release_grid=r_release_grid,
        sweep={f"{ra}/{rr}": d for (ra, rr), d in sweep.items()},
        opt_max_crossings=opt["max_crossings"],
        opt_mean_crossings=opt["mean_crossings"],
        opt_drift=opt["conservation_drift"],
    )


# ---------------------------------------------------------------------------
# 4.5B lifting wavelet contract
# ---------------------------------------------------------------------------
def case_lifting():
    rng = np.random.default_rng(7)
    n = 64
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    w_haar = LiftingWavelet(n, "haar", levels=3)
    w_cdf = LiftingWavelet(n, "cdf53", levels=3)

    a, details = stationary_haar(x, levels=3)
    a_shift, details_shift = stationary_haar(np.roll(x, 5), levels=3)
    trans_err = max(
        float(np.max(np.abs(np.roll(details[j], 5) - details_shift[j])))
        for j in range(3)
    )
    rec = stationary_haar_inverse(a, details)

    return dict(
        n=n,
        haar_isometry=w_haar.isometry_error(),
        haar_pr=w_haar.pr_error(),
        haar_dot=w_haar.dot_error(x, y),
        haar_noise_gain=w_haar.noise_gain,
        haar_condition=w_haar.condition,
        cdf_isometry=w_cdf.isometry_error(),
        cdf_pr=w_cdf.pr_error(),
        cdf_adjoint_error=w_cdf.adjoint_error(),
        cdf_condition=w_cdf.condition,
        stationary_translation_error=trans_err,
        stationary_inverse_error=float(np.max(np.abs(rec - x))),
        stationary_redundancy=len(details) + 1,
    )


# ---------------------------------------------------------------------------
# 4.5C transposeable edge / adjoint
# ---------------------------------------------------------------------------
def case_adjoint():
    rng = np.random.default_rng(11)
    n = 8
    edges = [TransposeableEdge(int(i), int(j), float(g))
             for i, j, g in zip(
                 rng.integers(0, n, 24),
                 rng.integers(0, n, 24),
                 rng.normal(0.0, 1.0, 24))]
    A = edge_matrix(edges, n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    lhs = float(np.dot(A @ x, y))
    rhs = float(np.dot(x, A.T @ y))
    trans_err = abs(lhs - rhs) / max(abs(lhs) + abs(rhs), 1e-30)

    b = rng.standard_normal(n)
    g_an = quadratic_gradient(A, b, x)
    g_fd = finite_difference_gradient(
        lambda z: 0.5 * float(np.sum((A @ z - b) ** 2)), x)
    grad_err = float(np.max(np.abs(g_an - g_fd))
                     / max(np.max(np.abs(g_fd)), 1e-30))

    Aode = -np.diag([1.0, 2.0, 3.0]) + 0.1 * rng.standard_normal((3, 3))
    x0 = rng.standard_normal(3)
    C = rng.standard_normal((2, 3))
    y2 = rng.standard_normal(2)
    T = 1.0
    nsteps = 100
    adj = adjoint_ode_gradient(Aode, x0, T, nsteps, C, y2)

    M = rk4_matrix(Aode, T / nsteps)

    def cost(z):
        xT = np.linalg.matrix_power(M, nsteps) @ z
        return 0.5 * float(np.sum((C @ xT - y2) ** 2))

    g_ode_fd = finite_difference_gradient(cost, x0)
    ode_err = float(np.max(np.abs(adj["grad"] - g_ode_fd))
                    / max(np.max(np.abs(g_ode_fd)), 1e-30))

    return dict(
        n=n, n_edges=len(edges),
        transpose_identity_error=trans_err,
        quadratic_grad_fd_error=grad_err,
        adjoint_ode_fd_error=ode_err,
        checkpoint_bytes=adj["checkpoint_bytes"],
        checkpoint_states=adj["checkpoint_states"],
    )


# ---------------------------------------------------------------------------
# 4.5D scale-dependent poles
# ---------------------------------------------------------------------------
def case_scale():
    levels = 6
    pC = scale_time_error(levels, mode="C")
    pgm = scale_time_error(levels, mode="gm")
    pn = scale_time_error(levels, mode="none")

    def pole_err(p):
        return float(np.max(np.abs(p["omega"] / p["lambda_model"] - 1.0)))

    return dict(
        levels=levels,
        C_pole_spread=pC["pole_spread"],
        gm_pole_spread=pgm["pole_spread"],
        C_pole_error=pole_err(pC),
        gm_pole_error=pole_err(pgm),
        C_model_time_error=pC["max_rel_error"],
        gm_model_time_error=pgm["max_rel_error"],
        none_model_time_error=pn["max_rel_error"],
        C_gm_spread=pC["gm_spread"],
        C_noise_spread=pC["noise_spread"],
        gm_gm_spread=pgm["gm_spread"],
        gm_noise_spread=pgm["noise_spread"],
        C_omega=pC["omega"],
        gm_omega=pgm["omega"],
    )


# ---------------------------------------------------------------------------
# 4.5E Gram calibration
# ---------------------------------------------------------------------------
def case_gram():
    rng = np.random.default_rng(19)
    n = 64
    w_haar = LiftingWavelet(n, "haar", levels=3)
    w_cdf = LiftingWavelet(n, "cdf53", levels=3)

    W_haar_hw = w_haar.W + 0.002 * rng.standard_normal((n, n))
    g_haar = gram_calibrate(W_haar_hw)
    # 同一个右 Gram 的 K，如果错误地左乘，不会正交化；正确的左标定要用左 Gram
    naive_left = g_haar["K"] @ W_haar_hw
    g_haar_left = gram_calibrate(W_haar_hw, side="left")

    W_cdf_hw = w_cdf.W + 0.01 * rng.standard_normal((n, n))
    g_cdf = gram_calibrate(W_cdf_hw)

    return dict(
        haar_isometry_before=g_haar["isometry_before"],
        haar_isometry_after=g_haar["isometry_after"],
        haar_pr_after=g_haar["pr_after"],
        haar_noise_gain=g_haar["noise_gain"],
        haar_cond=g_haar["cond"],
        haar_adjoint_before=g_haar["adjoint_before"],
        haar_adjoint_after=g_haar["adjoint_after"],
        naive_left_isometry=isometry_error(naive_left),
        haar_left_isometry_after=g_haar_left["isometry_after"],
        cdf_isometry_before=g_cdf["isometry_before"],
        cdf_isometry_after=g_cdf["isometry_after"],
        cdf_pr_after=g_cdf["pr_after"],
        cdf_noise_gain=g_cdf["noise_gain"],
        cdf_cond=g_cdf["cond"],
        cdf_adjoint_before=g_cdf["adjoint_before"],
        cdf_adjoint_after=g_cdf["adjoint_after"],
    )


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------
def acceptance():
    checks = []

    def chk(name, ok, value, threshold, case):
        checks.append(dict(name=name, ok=bool(ok), value=_jsonable(value),
                           threshold=threshold, case=case))
        emit(f"[{'PASS' if ok else 'FAIL'}] {name}: {value} "
             f"(threshold {threshold})")

    w = CASES["weno"]
    emit("4.5A WENO 三算子")
    chk("static shock σ", w["shock_sigma_static"] > 0.8,
        f"{w['shock_sigma_static']:.4f}", ">0.8", "weno")
    chk("static smooth σ", w["smooth_sigma_static"] < 0.2,
        f"{w['smooth_sigma_static']:.4f}", "<0.2", "weno")
    chk("adaptive conservation drift", w["conservation_drift"] < 1e-8,
        f"{w['conservation_drift']:.3e}", "<1e-8", "weno")
    chk("shock position error", w["shock_x_error"] < 0.03,
        f"{w['shock_x_error']:.4f}", "<0.03", "weno")
    chk("L2 vs WENO5", w["l2_vs_weno5"] < 0.05,
        f"{w['l2_vs_weno5']:.3e}", "<0.05", "weno")
    chk("max cell crossings", w["max_cell_crossings"] <= 3,
        w["max_cell_crossings"], "<=3", "weno")
    chk("final smooth σ", w["smooth_sigma_final"] < 0.3,
        f"{w['smooth_sigma_final']:.4f}", "<0.3", "weno")
    # P0-F 修正：熵不等式取代 "ΣV_i² 守恒"。周期域上 d/dt∫u²/2 = −ΣD_shock ≤ 0
    # 严格成立（shock 形成后必须有耗散，守恒是错误判据）。
    chk("entropy inequality (adaptive)", w["entropy_dE_max_ad"] <= 1e-10,
        f"{w['entropy_dE_max_ad']:.3e}", "<=1e-10 per step", "weno")
    chk("entropy inequality (weno5 ref)", w["entropy_dE_max_ref"] <= 1e-10,
        f"{w['entropy_dE_max_ref']:.3e}", "<=1e-10 per step", "weno")

    r = CASES["riemann"]
    emit()
    emit("4.5A' Riemann exact reference (P0-F)")
    chk("entropy constant cross-check", r["D_cross"] < 1e-14,
        f"{r['D_cross']:.3e}", "<1e-14", "riemann")
    rw = r["runs"]
    for scheme in ("weno5", "adaptive"):
        rf = rw[(scheme, 512)]
        rate = rf["entropy_rate_avg"]
        chk(f"entropy production rate ({scheme})",
            abs(rate - r["D_analytic"]) < 0.1 * r["D_analytic"],
            f"{rate:.6f} vs {r['D_analytic']:.6f}",
            "±10% of (uL-uR)^3/12", "riemann")
        chk(f"entropy inequality direction ({scheme})",
            rf["entropy_rate_min"] > -1e-9,
            f"min {rf['entropy_rate_min']:.3e}", "> -1e-9", "riemann")
        chk(f"mass conservation ({scheme})",
            rf["mass_drift"] < 1e-10,
            f"{rf['mass_drift']:.3e}", "<1e-10", "riemann")
        chk(f"shock position ({scheme})",
            rf["shock_err_cells"] < 3.0,
            f"{rf['shock_err_cells']:.2f} cells", "<3 cells", "riemann")
    ra = r["rates"]
    emit()
    emit(f"measured L1 convergence (overall, shock-dominated): "
         f"weno5 {ra['weno5']['l1']:.3f}/{ra['weno5']['l1_2']:.3f}, "
         f"adaptive {ra['adaptive']['l1']:.3f}/{ra['adaptive']['l1_2']:.3f} "
         "(expect ~1 for a monotone scheme at a shock)")
    for scheme in ("weno5", "adaptive"):
        p = ra[scheme]
        chk(f"L1 refinement reduces error ({scheme})",
            p["l1"] > 0 and p["l1_2"] > 0,
            f"{p['l1']:.3f}, {p['l1_2']:.3f}", ">0 (reported)", "riemann")
    chk("smooth-region L1 monotone (weno5)",
        rw[("weno5", 512)]["l1_smooth"] < rw[("weno5", 128)]["l1_smooth"],
        f"{rw[('weno5', 128)]['l1_smooth']:.3e} -> "
        f"{rw[('weno5', 512)]['l1_smooth']:.3e}", "decreasing", "riemann")
    chk("smooth-region L1 monotone (adaptive)",
        rw[("adaptive", 512)]["l1_smooth"] < rw[("adaptive", 128)]["l1_smooth"],
        f"{rw[('adaptive', 128)]['l1_smooth']:.3e} -> "
        f"{rw[('adaptive', 512)]['l1_smooth']:.3e}", "decreasing", "riemann")

    l = CASES["lifting"]
    emit()
    emit("4.5B lifting wavelet")
    chk("Haar isometry", l["haar_isometry"] < 5e-3,
        f"{l['haar_isometry']:.3e}", "<0.5%", "lifting")
    chk("Haar PR", l["haar_pr"] < 2e-3,
        f"{l['haar_pr']:.3e}", "<0.2%", "lifting")
    chk("Haar adjoint dot", l["haar_dot"] < 1e-3,
        f"{l['haar_dot']:.3e}", "<1e-3", "lifting")
    chk("CDF53 PR", l["cdf_pr"] < 2e-3,
        f"{l['cdf_pr']:.3e}", "<0.2%", "lifting")
    chk("CDF53 W_inv != W_star", l["cdf_adjoint_error"] > 1e-2,
        f"{l['cdf_adjoint_error']:.3e}", ">0.01", "lifting")
    chk("stationary translation", l["stationary_translation_error"] < 1e-12,
        f"{l['stationary_translation_error']:.3e}", "<1e-12", "lifting")
    chk("stationary inverse", l["stationary_inverse_error"] < 1e-12,
        f"{l['stationary_inverse_error']:.3e}", "<1e-12", "lifting")

    a = CASES["adjoint"]
    emit()
    emit("4.5C transposeable edge / adjoint")
    chk("A↔Aᵀ identity", a["transpose_identity_error"] < 1e-12,
        f"{a['transpose_identity_error']:.3e}", "<1e-12", "adjoint")
    chk("quadratic grad vs FD", a["quadratic_grad_fd_error"] < 1e-6,
        f"{a['quadratic_grad_fd_error']:.3e}", "<1e-6", "adjoint")
    chk("adjoint ODE grad vs FD", a["adjoint_ode_fd_error"] < 1e-6,
        f"{a['adjoint_ode_fd_error']:.3e}", "<1e-6", "adjoint")
    chk("checkpoint reported", a["checkpoint_bytes"] > 0,
        a["checkpoint_bytes"], ">0", "adjoint")

    s = CASES["scale"]
    emit()
    emit("4.5D scale-dependent poles")
    chk("mode C pole spread", abs(s["C_pole_spread"] - 4 ** (s["levels"] - 1)) < 1e-9,
        f"{s['C_pole_spread']:.1f}", f"{4 ** (s['levels'] - 1)}", "scale")
    chk("mode gm pole spread", abs(s["gm_pole_spread"] - 4 ** (s["levels"] - 1)) < 1e-9,
        f"{s['gm_pole_spread']:.1f}", f"{4 ** (s['levels'] - 1)}", "scale")
    chk("mode C pole error", s["C_pole_error"] < 0.05,
        f"{s['C_pole_error']:.3e}", "<5%", "scale")
    chk("mode gm pole error", s["gm_pole_error"] < 0.05,
        f"{s['gm_pole_error']:.3e}", "<5%", "scale")
    chk("mode C model time", s["C_model_time_error"] < 1e-12,
        f"{s['C_model_time_error']:.3e}", "<1e-12", "scale")
    chk("mode gm model time", s["gm_model_time_error"] < 1e-12,
        f"{s['gm_model_time_error']:.3e}", "<1e-12", "scale")
    chk("no-normalization fails", s["none_model_time_error"] > 0.1,
        f"{s['none_model_time_error']:.3f}", ">0.1", "scale")
    chk("mode C trades noise", abs(s["C_gm_spread"] - 1.0) < 1e-12
        and abs(s["C_noise_spread"] - 4 ** (s["levels"] - 1)) < 1e-9,
        f"gm {s['C_gm_spread']:.1f}, noise {s['C_noise_spread']:.1f}",
        "gm=1, noise=4^L", "scale")
    chk("mode gm trades g_m", abs(s["gm_gm_spread"] - 4 ** (s["levels"] - 1)) < 1e-9
        and abs(s["gm_noise_spread"] - 1.0) < 1e-12,
        f"gm {s['gm_gm_spread']:.1f}, noise {s['gm_noise_spread']:.1f}",
        "gm=4^L, noise=1", "scale")

    g = CASES["gram"]
    emit()
    emit("4.5F tracker rate bounds（review 可证伪预测）")
    b = CASES["tracker"]
    emit(f"ν_num={b['nu_num']:.3e} ({b['nu_source']}), Δx={b['dx']:.3e}, "
         f"s={b['s_shock']}, Δσ_hyst={b['dsigma_hyst']}")
    emit(f"predicted: r_attack_min={b['r_attack_min']:.1f}/s, "
         f"r_release_max={b['r_release_max']:.1f}/s "
         f"(review's Δσ=0.008 variant: {b['r_release_max_review_dsigma008']:.1f}/s)")
    rows = []
    for key, d in b["sweep"].items():
        ra, rr = key.split("/")
        rows.append([f"{ra}/{rr}", "yes" if d["infeasible"] else "no",
                     d["max_crossings"], f"{d['mean_crossings']:.2f}",
                     f"{d['conservation_drift']:.2e}",
                     f"{d['entropy_dE_max']:.2e}"])
    table(["r_attack/r_release", "infeasible", "max crossings",
           "mean crossings", "mass drift", "max ΔE/step"], rows)
    chk("r_attack_min O(100)", 30.0 < b["r_attack_min"] < 1000.0,
        f"{b['r_attack_min']:.1f}", "(30, 1000)/s", "tracker")
    chk("r_release_max O(10..100)", 1.0 < b["r_release_max"] < 300.0,
        f"{b['r_release_max']:.1f}", "(1, 300)/s", "tracker")
    chk("optimum (400,10) inside feasible box",
        b["r_attack_min"] < 400.0 and 10.0 < b["r_release_max"],
        f"attack_min {b['r_attack_min']:.1f} < 400, "
        f"release_max {b['r_release_max']:.1f} > 10", "brackets 400/10",
        "tracker")
    infeas = {k: d for k, d in b["sweep"].items() if d["infeasible"]}
    feas = {k: d for k, d in b["sweep"].items() if not d["infeasible"]}
    if infeas:
        worst = max(infeas.items(), key=lambda kv: kv[1]["max_crossings"])
        chk("optimum no worse than every infeasible corner (max crossings)",
            all(b["opt_max_crossings"] <= d["max_crossings"]
                for d in infeas.values()),
            f"opt {b['opt_max_crossings']} vs infeasible max "
            f"{worst[1]['max_crossings']} at {worst[0]}",
            "opt <= all infeasible", "tracker")
        elevated = [k for k, d in infeas.items()
                    if d["max_crossings"] > b["opt_max_crossings"]]
        chk("falsifiability: some infeasible corner elevates crossings",
            len(elevated) > 0,
            f"{len(elevated)}/{len(infeas)} infeasible corners above "
            f"optimum ({', '.join(elevated) if elevated else 'none'})",
            ">=1 elevated corner", "tracker")
    else:
        chk("falsifiability: some infeasible corner elevates crossings",
            False, "no infeasible corner in grid — grid too small",
            ">=1 elevated corner", "tracker")
    drift_ok = all(d["conservation_drift"] < 1e-8 for d in b["sweep"].values())
    chk("sweep conservation drift (all rates)", drift_ok,
        f"max {max(d['conservation_drift'] for d in b['sweep'].values()):.2e}",
        "<1e-8", "tracker")

    g = CASES["gram"]
    emit()
    emit("4.5E Gram calibration")
    chk("Haar isometry before", g["haar_isometry_before"] > 5e-3,
        f"{g['haar_isometry_before']:.3e}", ">0.5%", "gram")
    chk("Haar isometry after", g["haar_isometry_after"] < 5e-3,
        f"{g['haar_isometry_after']:.3e}", "<0.5%", "gram")
    chk("Haar PR after", g["haar_pr_after"] < 2e-3,
        f"{g['haar_pr_after']:.3e}", "<0.2%", "gram")
    chk("Haar noise gain", g["haar_noise_gain"] < 1.05,
        f"{g['haar_noise_gain']:.4f}", "<1.05", "gram")
    chk("Haar adjoint after", g["haar_adjoint_after"] < 5e-3,
        f"{g['haar_adjoint_after']:.3e}", "<0.5%", "gram")
    chk("naive left fails", g["naive_left_isometry"] > 1e-2,
        f"{g['naive_left_isometry']:.3f}", ">0.01", "gram")
    chk("left calibration after", g["haar_left_isometry_after"] < 5e-3,
        f"{g['haar_left_isometry_after']:.3e}", "<0.5%", "gram")
    chk("CDF adjoint reduced", g["cdf_adjoint_after"] < g["cdf_adjoint_before"] / 10.0,
        f"{g['cdf_adjoint_before']:.3e} -> {g['cdf_adjoint_after']:.3e}",
        ">10x", "gram")

    return checks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    hr("Phase 4.5 — adaptive representation / TransformContract")

    CASES["weno"] = case_weno()
    hr("4.5A WENO 三算子：sensor → σ → face flux")
    table(["quantity", "value"], [
        ["N / steps / dt", f"{CASES['weno']['n']} / "
                           f"{CASES['weno']['nsteps']} / "
                           f"{CASES['weno']['dt']:.2e}"],
        ["σ at static shock", f"{CASES['weno']['shock_sigma_static']:.4f}"],
        ["σ smooth (static)", f"{CASES['weno']['smooth_sigma_static']:.4f}"],
        ["σ at final shock", f"{CASES['weno']['shock_sigma_final']:.4f}"],
        ["σ smooth (final)", f"{CASES['weno']['smooth_sigma_final']:.4f}"],
        ["conservation drift", f"{CASES['weno']['conservation_drift']:.3e}"],
        ["shock x (adaptive)", f"{CASES['weno']['shock_x_ad']:.4f}"],
        ["shock x (WENO5)", f"{CASES['weno']['shock_x_ref']:.4f}"],
        ["L2 vs WENO5", f"{CASES['weno']['l2_vs_weno5']:.3e}"],
        ["max cell crossings", f"{CASES['weno']['max_cell_crossings']}"],
    ])

    CASES["riemann"] = case_riemann()
    hr("4.5A' Riemann 精确解参考：uL=1, uR=0, x0=0.25, t=0.5")
    emit("熵常数 D=(uL−uR)³/12=1/12；扩展域 d/dt∫U = q(uL)−q(uR)−D = 1/4")
    emit("P0-F：ΣV_i² 守恒是错误判据；shock 形成后判据为熵产生率 ∈ [D(1±tol), ∞)")
    rows = []
    for scheme in ("weno5", "adaptive"):
        for n in (128, 256, 512):
            d = CASES["riemann"]["runs"][(scheme, n)]
            rows.append([
                scheme, n,
                f"{d['l1']:.3e}", f"{d['l2']:.3e}",
                f"{d['shock_err_cells']:.2f}",
                f"{d['entropy_rate_avg']:.6f}",
                f"{d['nu_num']:.3e}" if d["tanh_ok"] else "n/a",
            ])
    table(["scheme", "N", "L1", "L2", "x_s err (cells)",
           "entropy rate (D=1/12)", "nu_num"], rows)
    for scheme in ("weno5", "adaptive"):
        p = CASES["riemann"]["rates"][scheme]
        emit(f"{scheme} L1 rate: {p['l1']:.3f} (128→256), "
             f"{p['l1_2']:.3f} (256→512); smooth-region: "
             f"{p['l1_smooth']:.3f}, {p['l1_smooth_2']:.3f}")

    CASES["lifting"] = case_lifting()
    hr("4.5B lifting wavelet：W / W_inv / W_star")
    table(["quantity", "value"], [
        ["Haar isometry", f"{CASES['lifting']['haar_isometry']:.3e}"],
        ["Haar PR", f"{CASES['lifting']['haar_pr']:.3e}"],
        ["Haar adjoint dot", f"{CASES['lifting']['haar_dot']:.3e}"],
        ["Haar noise gain", f"{CASES['lifting']['haar_noise_gain']:.4f}"],
        ["CDF53 isometry", f"{CASES['lifting']['cdf_isometry']:.3e}"],
        ["CDF53 PR", f"{CASES['lifting']['cdf_pr']:.3e}"],
        ["CDF53 W_inv != W_star", f"{CASES['lifting']['cdf_adjoint_error']:.3e}"],
        ["stationary translation", f"{CASES['lifting']['stationary_translation_error']:.3e}"],
        ["stationary inverse", f"{CASES['lifting']['stationary_inverse_error']:.3e}"],
    ])

    CASES["adjoint"] = case_adjoint()
    hr("4.5C transposeable edge / adjoint")
    table(["quantity", "value"], [
        ["A↔Aᵀ identity", f"{CASES['adjoint']['transpose_identity_error']:.3e}"],
        ["quadratic grad vs FD", f"{CASES['adjoint']['quadratic_grad_fd_error']:.3e}"],
        ["adjoint ODE grad vs FD", f"{CASES['adjoint']['adjoint_ode_fd_error']:.3e}"],
        ["checkpoint bytes", f"{CASES['adjoint']['checkpoint_bytes']}"],
    ])

    CASES["scale"] = case_scale()
    hr("4.5D scale-dependent poles：mode C vs mode g_m")
    table(["mode", "pole spread", "gm spread", "noise spread",
           "pole err", "model-time err"], [
        ["C", f"{CASES['scale']['C_pole_spread']:.1f}",
         f"{CASES['scale']['C_gm_spread']:.1f}",
         f"{CASES['scale']['C_noise_spread']:.1f}",
         f"{CASES['scale']['C_pole_error']:.2e}",
         f"{CASES['scale']['C_model_time_error']:.2e}"],
        ["gm", f"{CASES['scale']['gm_pole_spread']:.1f}",
         f"{CASES['scale']['gm_gm_spread']:.1f}",
         f"{CASES['scale']['gm_noise_spread']:.1f}",
         f"{CASES['scale']['gm_pole_error']:.2e}",
         f"{CASES['scale']['gm_model_time_error']:.2e}"],
        ["none", "-", "1.0", "1.0", "-",
         f"{CASES['scale']['none_model_time_error']:.2e}"],
    ])

    CASES["tracker"] = case_tracker_bounds(n=256)
    # (4.5F table printed in acceptance())

    CASES["gram"] = case_gram()
    hr("4.5E Gram calibration：G_T = T̃*T̃")
    table(["quantity", "Haar", "CDF53"], [
        ["isometry before", f"{CASES['gram']['haar_isometry_before']:.3e}",
         f"{CASES['gram']['cdf_isometry_before']:.3e}"],
        ["isometry after", f"{CASES['gram']['haar_isometry_after']:.3e}",
         f"{CASES['gram']['cdf_isometry_after']:.3e}"],
        ["PR after", f"{CASES['gram']['haar_pr_after']:.3e}",
         f"{CASES['gram']['cdf_pr_after']:.3e}"],
        ["adjoint before", f"{CASES['gram']['haar_adjoint_before']:.3e}",
         f"{CASES['gram']['cdf_adjoint_before']:.3e}"],
        ["adjoint after", f"{CASES['gram']['haar_adjoint_after']:.3e}",
         f"{CASES['gram']['cdf_adjoint_after']:.3e}"],
        ["naive left isometry", f"{CASES['gram']['naive_left_isometry']:.3f}",
         "-"],
        ["left calibration after", f"{CASES['gram']['haar_left_isometry_after']:.3e}",
         "-"],
        ["cond(G_T)", f"{CASES['gram']['haar_cond']:.3f}",
         f"{CASES['gram']['cdf_cond']:.3f}"],
        ["noise gain", f"{CASES['gram']['haar_noise_gain']:.4f}",
         f"{CASES['gram']['cdf_noise_gain']:.4f}"],
    ])

    hr("ACCEPTANCE")
    checks = acceptance()
    passed = sum(c["ok"] for c in checks)
    emit()
    emit(f"RESULT: {passed}/{len(checks)} checks passed")

    os.makedirs(OUT, exist_ok=True)
    payload = dict(
        cases=CASES,
        checks=checks,
        passed=passed,
        total=len(checks),
    )
    with open(os.path.join(OUT, "transform.json"), "w") as f:
        json.dump(_jsonable(payload), f, indent=2)
    with open(os.path.join(OUT, "transform_report.txt"), "w") as f:
        f.write("\n".join(_LINES) + "\n")

    emit()
    emit(f"wrote {os.path.abspath(os.path.join(OUT, 'transform.json'))}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
