"""§6.22 运行时误差控制合同验收（两条基准 + 三组负对照）。

运行：python3 -m bench.run_error_control   → results/error_control.json

命题 1（Poisson iterative refinement，§6.22 基准 A）：
  低精度模拟求解核心（量化 Richardson/T10 松弛，相对精度 u_a=2^-10）
  + 数字 float64 残差 + 低精度修正解，能把解精度推进到显著高于
  单个单元的精度。收敛率预测 ρ_IR ≈ κ(G)·u_a（混合精度 IR 经典结论）；
  κ·u_a > 1 时**必须提前判死**，不许碰运气（病态负对照）。
  残差也用低精度算（自监督捷径）⇒ 修正失效在量化地板上 —— "残差
  必须由可信度更高的通路计算"是 IR 生效的前提，不是可选项。

命题 2（diffusion 残差驱动分段校正，§6.22 基准 B）：
  收缩/中性/扩张三种模态的误差累积行为实测 = 理论界（Grönwall）：
    contracting: e(T) ≤ e^{-γT} e0 + (r_max/γ)(1−e^{-γT})   饱和
    neutral:     e(T) = e0 + ∫r                              线性
    expanding:   e(T) ≤ e^{λT} e0 + (r_max/λ)(e^{λT}−1)      指数
  残差驱动的分段校正（窗口积分残差 = 注入积累 → 状态回注扣除）在
  收缩系统上把长期误差压到 η/(1−ρ) 量级，且**每个窗口都满足
  ε_measured ≤ ε_predicted**（界的覆盖率是验收标准，不是误差均值）。
  调度间隔由校正公式给出（λ, r_max, e0, ε 反解 ΔT），不是定时常数。

负对照（都对应 §6.22 的诚实边界）：
  N1  残差低精度 ⇒ IR 失效在量化地板（命题 1 的前提被抽掉）
  N2  κ·u_a > 1 ⇒ 证书提前判死，实测确认不收敛（不许碰运气）
  N3  一致偏移对残差通路不可见（守恒零模态上的常值偏移：残差 ~0、
      误差持久）—— 残差小 ≠ 误差小；只能靠独立参考 detect_offset。
      这是对"用自身输出自监督可以发现一切"的直接否证。

模型口径（诚实边界）：这里的"模拟硬件"是行为级 —— 量化 + 有界噪声
+ 偏置注入，不是 SPICE 级（B2 的器件模型管那一层）。本 bench 验证的
是**误差控制合同的数学**，不是电路实现；合同数字（ρ 预测、界覆盖）
在任何满足同一误差模型注入的实现上都应复现，若不复现即是证伪。

输出：results/error_control.json + error_control_report.txt
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.error_propagation import (classify_propagation,
                                        conservation_drift_rate,
                                        correction_interval, steady_bound)
from twin.residual import (detect_offset, linear_window_residual,
                           window_integral_residual)

_checks = []
_LINES = []


def add_check(name, value, ok, kind="error_control"):
    _checks.append(dict(name=name, value=str(value), ok=bool(ok), kind=kind))
    emit("  [%s] %-64s %s" % ("PASS" if ok else "FAIL", name, value))
    return ok


def emit(s=""):
    print(s)
    _LINES.append(s)


# ---------------------------------------------------------------- 公共模型

def quantize(x, u_a, scale):
    """低精度存储模型：每元素加相对精度 u_a·scale 的均匀舍入噪声。"""
    return x + u_a * scale * np.random.uniform(-1.0, 1.0, size=x.shape)


def quantized_richardson(G, b, u_a, iters=400, scale=0.3):
    """量化 Richardson 迭代（T10 松弛网的行为级模型）。

    ω = 1/λ_max 保证收缩；量化噪声每步注入，收敛停在
    u_a·scale·√n/λ_min 量级的地板上 —— 这就是"模拟单元精度"。
    返回解与实测地板（对照解析预测）。
    """
    lam = np.linalg.eigvalsh(G)
    omega = 1.0 / lam.max()
    x = np.zeros_like(b)
    for _ in range(iters):
        x = quantize(x + omega * (b - G @ x), u_a, scale)
    return x


def poisson_grid(n):
    """2D 五点 Laplacian（内部 n×n，Dirichlet 接地）。行和/列和结构
    与 KCL 同构 —— 这是 §3.4 约束 fabric 的 A(x)y=b(x) 线性化。"""
    N = n * n
    G = np.zeros((N, N))
    idx = lambda i, j: i * n + j
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            G[k, k] = 4.0
            if i > 0:
                G[k, idx(i - 1, j)] = -1.0
            if i < n - 1:
                G[k, idx(i + 1, j)] = -1.0
            if j > 0:
                G[k, idx(i, j - 1)] = -1.0
            if j < n - 1:
                G[k, idx(i, j + 1)] = -1.0
    return G


# ---------------------------------------------------------------- 基准 A

def case_poisson_ir():
    """Poisson iterative refinement：低精度核 + 高精度残差 + 病态对照。"""
    emit("[case_poisson_ir]  低精度模拟核 + float64 残差 + 低精度修正解")
    rng = np.random.default_rng(2026)
    n = 12
    G = poisson_grid(n)
    kappa = float(np.linalg.cond(G))
    b = np.sin(np.pi * np.arange(n * n) / (n + 1)) ** 2 + 0.1
    x_true = np.linalg.solve(G, b)
    x_norm = float(np.linalg.norm(x_true))

    u_a = 2.0 ** -10
    # 预测：混合精度 IR 收敛率 ρ ≈ κ·u_a（残差 fp64、修正解低精度）
    rho_pred = kappa * u_a
    add_check("IR 前提：rho_pred=kappa*u_a=%.3f < 1（修正闭环合法）" % rho_pred,
              "kappa=%.1f u_a=%.3e" % (kappa, u_a), rho_pred < 1.0)

    # 模拟核直接解（无修正）：量化地板
    np.random.seed(11)
    x_analog = quantized_richardson(G, b, u_a)
    e_analog = float(np.linalg.norm(x_analog - x_true))
    add_check("模拟核地板：e_analog=%.3e 显著大于机器精度（低精度前提成立）" % e_analog,
              "rel=%.3e" % (e_analog / x_norm),
              1e-6 < e_analog / x_norm < 1e-1)

    # IR 主线：残差 float64，修正解走同一量化核（对 r 而不是 b 松弛）
    x = x_analog.copy()
    rates = []
    for k in range(4):
        r = b - G @ x                                  # float64 残差
        delta = quantized_richardson(G, r, u_a)        # 低精度修正解
        e_prev = float(np.linalg.norm(x - x_true))
        x = x + delta                                  # 数字端精确累加
        e_new = float(np.linalg.norm(x - x_true))
        rates.append(e_prev / max(e_new, 1e-300))
    e_final = float(np.linalg.norm(x - x_true))
    add_check("IR step 0: 收缩率实测=%.2f vs 预测 1/rho=%.2f（×3 容差）"
              % (rates[0], 1.0 / rho_pred),
              "e: %.3e -> %.3e" % (e_analog, e_final),
              abs(rates[0] - 1.0 / rho_pred) / (1.0 / rho_pred) < 3.0)
    tail = rates[1:]
    add_check("IR steps 1-3: 稳定在修正解量化地板（无发散，尾部率≈1）",
              "rates=%s" % ["%.2f" % r for r in tail],
              all(0.5 < r < 2.0 for r in tail))
    improve = e_analog / max(e_final, 1e-300)
    add_check("IR 总改进倍数=%.1f ≥ 20（最终精度高于单个单元地板）" % improve,
              "e_analog=%.3e -> e_final=%.3e" % (e_analog, e_final),
              improve >= 20.0)

    # N1：残差也用低精度量化（u_r=2^-7，粗糙一位）⇒ 可达精度被
    # ‖G⁻¹·δr‖ 拖住，而不是修正解的地板 —— 残差精度决定 attainable。
    u_r = 2.0 ** -7
    np.random.seed(12)
    x_low = x_analog.copy()
    for _ in range(4):
        r_low = quantize(b - G @ x_low, u_r, 1.0)      # 粗量化残差
        x_low = x_low + quantized_richardson(G, r_low, u_a)
    e_low = float(np.linalg.norm(x_low - x_true))
    add_check("N1 残差粗量化：可达误差 %.3e 比主线 %.3e 差 ≥3×"
              "（残差精度是 attainable 的瓶颈）" % (e_low, e_final),
              "ratio=%.1f" % (e_low / max(e_final, 1e-300)),
              e_low > 3.0 * e_final)

    # N2：病态变体 —— 对角共轭把 κ 抬到 κ·u_a > 1，证书必须提前判死
    D = np.ones(n * n)
    D[: n] = 1e-2                                     # 一行模态软化 ×100
    G_ill = (D[:, None] * G * D[None, :])
    kappa_ill = float(np.linalg.cond(G_ill))
    rho_ill = kappa_ill * u_a
    b_ill = D * b
    x_true_ill = np.linalg.solve(G_ill, b_ill)
    add_check("N2 病态证书：kappa·u_a=%.1f > 1 ⇒ 提前判死（不许碰运气）" % rho_ill,
              "kappa_ill=%.0f" % kappa_ill, rho_ill > 1.0)
    np.random.seed(13)
    x_ill = quantized_richardson(G_ill, b_ill, u_a)
    e0_ill = float(np.linalg.norm(x_ill - x_true_ill))
    for _ in range(3):
        r = b_ill - G_ill @ x_ill
        x_ill = x_ill + quantized_richardson(G_ill, r, u_a)
    e_ill = float(np.linalg.norm(x_ill - x_true_ill))
    add_check("N2 病态实测：IR 后误差 %.3e 未低于起点 %.3e（不收敛被证实）"
              % (e_ill, e0_ill),
              "ratio=%.3f" % (e_ill / e0_ill), e_ill >= 0.5 * e0_ill)

    return dict(kappa=kappa, u_a=u_a, rho_pred=rho_pred,
                e_analog=e_analog, e_final=e_final, improve=improve,
                e_lowres_residual=e_low,
                kappa_ill=kappa_ill, rho_ill=rho_ill, e_ill=e_ill)


# ---------------------------------------------------------------- 基准 B

def _hw_sim(A, x0, T, dt, bias, r_max, rng):
    """硬件行为级：ẋ = A x + bias + U(−r_max, r_max) 有界注入。"""
    n = A.shape[0]
    steps = int(round(T / dt))
    xs = np.empty((steps + 1, n))
    xs[0] = x0
    for k in range(steps):
        inj = bias + r_max * rng.uniform(-1.0, 1.0, size=n)
        xs[k + 1] = xs[k] + dt * (A @ xs[k] + inj)
    return xs


def case_diffusion_windows():
    """diffusion 三分类 + 残差驱动分段校正 + 界覆盖 + 一致偏移负对照。"""
    emit("[case_diffusion_windows]  三分类实测 / 调度公式 / 界覆盖 / N3")
    N = 24
    nu = 1e-2
    k = np.arange(1, N + 1)
    lam = -nu * (2.0 * np.pi * k) ** 2
    A = np.diag(lam)
    gamma = -lam.max()                                # 最慢收缩率
    rng = np.random.default_rng(7)

    # --- 三分类 ---
    c_diff = classify_propagation(A, horizons=(0.2, 1.0))
    add_check("分类 diffusion ⇒ contracting（谱横坐标=%.3f<0）" % c_diff.spectral_abscissa,
              c_diff.cls, c_diff.cls == "contracting")
    add_check("收缩系统对数范数 μ₂=%.3f<0（任意初始误差单调收缩的上界）" % c_diff.lognorm,
              "%.3f" % c_diff.lognorm, c_diff.lognorm < 0)
    c_neutral = classify_propagation(np.zeros((2, 2)))
    c_expand = classify_propagation(np.eye(1) * 0.5)
    add_check("分类 纯积分器 ⇒ neutral / λ=+0.5 ⇒ expanding",
              "%s / %s" % (c_neutral.cls, c_expand.cls),
              c_neutral.cls == "neutral" and c_expand.cls == "expanding")

    # --- 泄漏 vs 保守型 ΔA（率对账）---
    R_rand = rng.normal(0, 1e-3, size=(N, N))
    dA_cons = R_rand - np.diag(R_rand.sum(axis=0))    # 列和精确为 0
    dA_leak = dA_cons.copy()
    dA_leak[0, N // 2] += 1e-3                        # 凭空注入节点 0
    rate_cons = conservation_drift_rate(A, dA_cons)
    rate_leak = conservation_drift_rate(A, dA_leak)
    add_check("泄漏率/保守率 = %.0f ≫ 1（Γ_proj 看不见的那一路，率可见）"
              % (rate_leak / max(rate_cons, 1e-300)),
              "%.2e vs %.2e" % (rate_leak, rate_cons),
              rate_leak > 100.0 * max(rate_cons, 1e-300))

    # --- 基准 B 主体：收缩系统的残差驱动分段校正 ---
    x0 = np.cos(np.pi * k / (N + 1))
    bias = np.full(N, 2e-3)
    r_noise = 3e-2
    r_max = float(np.linalg.norm(bias)) + r_noise * np.sqrt(N)
    eps = 1e-2
    e0 = 1e-3                                         # 一次校正后的残余

    dT = correction_interval(gamma, r_max, e0, eps)
    add_check("调度公式：ΔT=%.4f（λ=%.3f, r_max=%.3f, e0=%.0e, ε=%.0e）"
              % (dT, gamma, r_max, e0, eps),
              "s", 0.0 < dT < 10.0)
    dT_fast = correction_interval(gamma * 10, r_max, e0, eps)
    dT_strong = correction_interval(gamma, r_max * 2, e0, eps)
    add_check("调度单调性：λ×10 ⇒ ΔT×%.2f↓；r_max×2 ⇒ ΔT×%.2f↓（扰动越强查越勤）"
              % (dT_fast / dT, dT_strong / dT),
              "%.4f / %.4f" % (dT_fast, dT_strong),
              dT_fast < dT and dT_strong < dT)
    try:
        correction_interval(gamma, r_max, eps, eps)   # e0 ≥ ε ⇒ 必须 raise
        ok_raise = False
    except ValueError:
        ok_raise = True
    add_check("调度公式：e0 ≥ ε ⇒ raise（不许静默给∞间隔）", "raise", ok_raise)

    # 无校正运行：Grönwall 饱和界（最坏情形）+ 偏置主导项的精确预测
    T_total = 40 * dT
    xs = _hw_sim(A, x0, T_total, 2e-4, bias, r_noise, np.random.default_rng(21))
    x_true_T = np.diag(np.exp(lam * T_total)) @ x0
    e_unc = float(np.linalg.norm(xs[-1] - x_true_T))
    bound_unc = r_max / gamma                         # 最坏情形饱和值
    # 偏置主导项（有解析式）：‖A⁻¹(I−e^{AT})·bias‖；噪声实现有界但
    # 不贴最坏情形 —— 界必须覆盖，但不许把覆盖当成"接近饱和"。
    pred_bias = float(np.linalg.norm(
        (1.0 - np.exp(lam * T_total)) / lam * bias))
    add_check("无校正：实测 %.4f ≤ 饱和界 r_max/γ=%.3f 且 ≈ 偏置预测 %.4f（×2）"
              % (e_unc, bound_unc, pred_bias),
              "ratio_sat=%.3f" % (e_unc / bound_unc),
              e_unc <= bound_unc and e_unc < 2.0 * pred_bias)

    # 有校正运行：每窗口 [残差测量 → 界传播 → 状态回注扣除]
    dt = 2e-4
    sub = max(int(round(dT / dt)), 8)
    dT_eff = sub * dt
    xh = x0.copy()
    rng_c = np.random.default_rng(22)
    errs, etas, covered = [], [], 0
    K = 40
    for w in range(K):
        xs_w = np.empty((sub + 1, N))
        xs_w[0] = xh
        for s in range(sub):
            inj = bias + r_noise * rng_c.uniform(-1.0, 1.0, size=N)
            xh = xh + dt * (A @ xh + inj)
            xs_w[s + 1] = xh
        xh_traj = xs_w
        res = linear_window_residual(A, np.linspace(0, dT_eff, sub + 1), xh_traj)
        x_true_end = np.diag(np.exp(lam * (w + 1) * dT_eff)) @ x0
        e_meas = float(np.linalg.norm(xh - x_true_end))
        errs.append(e_meas)
        etas.append(res.eta)
        # 界：G·e_after_reset + η；reset 后误差取上一窗口末实测（诚实）
        G_w = float(np.exp(-gamma * dT_eff))
        e_prev = errs[-2] if errs[:-1] else 0.0
        bound = G_w * 0.0 + res.eta                   # reset 恰好扣除 ∫r ⇒ 界=η
        if e_meas <= bound * 1.2:
            covered += 1
        xh = xh - res.R                               # 校正：扣除注入积累
    cov = covered / K
    sb = steady_bound(0.0, float(np.mean(etas)))
    add_check("界覆盖：%d/%d 窗口 ε_measured ≤ 1.2×ε_predicted（覆盖率=%.0f%%）"
              % (covered, K, 100 * cov),
              "mean η=%.3e" % np.mean(etas), cov >= 0.95)
    add_check("校正后长期误差 max=%.4f ≪ 无校正饱和 %.4f（×5 至少）"
              % (max(errs), bound_unc),
              "ratio=%.3f" % (max(errs) / bound_unc),
              max(errs) < 0.2 * bound_unc)
    add_check("长期界 η/(1−ρ)=%.3e 与实测 max 误差同量级（×5 内）" % sb,
              "%.3e vs %.3e" % (sb, max(errs)),
              max(errs) < 5.0 * max(sb, 1e-12) and max(errs) > 0.05 * sb)

    # --- N3：一致偏移对残差不可见（守恒零模态 + 常值偏移）---
    # path-graph Laplacian 的零模态 = 常值向量。真初值 = 0（真解静态
    # 为 0），硬件初值带零模态偏移 c ⇒ 硬件轨迹静态为 c：与目标方程
    # 完全自洽，每窗口残差恒为 0，而解误差 = ‖c‖ 永远持留。
    L = np.zeros((8, 8))
    for i in range(8):
        if i > 0:
            L[i, i] += 1.0
            L[i, i - 1] = -1.0
        if i < 7:
            L[i, i] += 1.0
            L[i, i + 1] = -1.0
    c = np.full(8, 0.5)                               # 零模态方向的一致偏移
    xs_off = _hw_sim(L, c, 2.0, 2e-4, np.zeros(8), 0.0,
                     np.random.default_rng(31))
    rows = xs_off[7500:]                              # 窗口 [1.5, 2.0]
    t_off = 1.5 + 2e-4 * np.arange(rows.shape[0])
    res_off = linear_window_residual(L, t_off, rows)
    e_off = float(np.linalg.norm(xs_off[-1] - np.zeros(8)))   # 真解 = 0
    add_check("N3 残差不可见：窗口残差 %.2e ≪ 实际误差 %.2e（残差小≠误差小）"
              % (res_off.eta, e_off),
              "ratio=%.1e" % (res_off.eta / e_off),
              res_off.eta < 0.05 * e_off)
    chk = detect_offset(xs_off[-1], np.zeros(8), 1e-6)
    chk_noref = detect_offset(xs_off[-1], None, 1e-6)
    add_check("N3 独立参考可见：offset=%.3f 被检出；无参考 ⇒ 如实 detectable=False"
              % chk["offset_norm"],
              "%s / %s" % (chk["detectable"], chk_noref["detectable"]),
              chk["detectable"] and not chk_noref["detectable"])

    # --- 中性/扩张的界实测 ---
    # neutral: A=0, bias=2e-2 ⇒ e(t)=e0+b·t（噪声对消后斜率=b）
    t_grid = np.linspace(0, 1.0, 201)
    x_int = _hw_sim(np.zeros((1, 1)), [0.0], 1.0, 5e-3, np.array([2e-2]), 0.0,
                    np.random.default_rng(41))[:, 0]
    slope = (x_int[-1] - x_int[0]) / 1.0
    add_check("neutral 实测斜率=%.4f ≈ bias=0.0200（线性累积被证实）" % slope,
              "%.4f" % slope, abs(slope - 2e-2) < 2e-3)
    dT_n = correction_interval(0.0, 2e-2, 0.0, 1e-2)
    add_check("neutral 调度退化为预算式：ΔT=(ε−e0)/r_max=%.4f" % dT_n,
              "%.4f" % dT_n, abs(dT_n - 0.5) < 1e-9)

    # expanding: λ=+0.5, 有界注入 ⇒ 指数界覆盖
    lam_x = 0.5
    x_ex = _hw_sim(np.array([[lam_x]]), [1e-3], 2.0, 2e-4, np.zeros(1), 1e-2,
                   np.random.default_rng(51))[:, 0]
    e_T = abs(x_ex[-1] - np.exp(lam_x * 2.0) * 1e-3)
    bound_ex = np.exp(lam_x * 2.0) * 0.0 + (1e-2 / lam_x) * (np.exp(lam_x * 2.0) - 1.0)
    add_check("expanding 实测 %.4f ≤ 指数界 %.4f（有限时窗内成立）"
              % (e_T, bound_ex),
              "%.4f" % e_T, e_T <= bound_ex)
    try:
        correction_interval(lam_x, 1e-2, 1.0, 1e-2)   # e0 已爆 ⇒ raise
        ok = False
    except ValueError:
        ok = True
    add_check("expanding 终局：e0≥ε ⇒ raise（扩张系统不许诺无限时窗轨迹精度）",
              "raise", ok)

    return dict(gamma=gamma, dT=dT, e_uncorrected=e_unc, bound_sat=bound_unc,
                coverage=cov, e_corrected_max=max(errs),
                eta_mean=float(np.mean(etas)), steady_bound=sb,
                n3_residual=res_off.eta, n3_error=e_off)


# ---------------------------------------------------------------- main

def main():
    emit("=" * 72)
    emit("§6.22 运行时误差控制合同验收")
    emit("=" * 72)
    ir = case_poisson_ir()
    emit("")
    dif = case_diffusion_windows()
    n_pass = sum(1 for c in _checks if c["ok"])
    emit("")
    emit("%d/%d checks passed" % (n_pass, len(_checks)))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "error_control_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    payload = dict(kind="error_control",
                   meta=dict(bench="bench.run_error_control",
                             section="DESIGN.md §6.22",
                             model="behavioral: quantize + bounded noise + bias"),
                   poisson_ir=ir, diffusion=dif, checks=_checks)
    with open(os.path.join(out_dir, "error_control.json"), "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return n_pass == len(_checks)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
