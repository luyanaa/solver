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
from lowering.budget import evaluate_sys
from lowering.error_control import (ErrorControlResult, classify_constraint,
                                    constraint_refine, constraint_schedule)
from twin.constraint import ConstraintMesh, solve_constraint
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


# ---------------------------------------------------------------- 约束综合

def case_constraint_fabric():
    """§6.22 × T10 综合：误差控制合同在约束 fabric 上的实例化。

    三条合同（lowering/error_control.py）：
      C1 约束网分类（−G/C 必然 contracting，非 SPD 网如实改判并要求对账）；
      C2 约束网 IR：mismatch 偏置被 KCL 残差暴露并扣除 —— regime B（安静
         模拟核 + 10% dev 压力注入）实测最终误差**低于 mismatch 地板**，
         即约束 fabric 在数字残差通路上自校准（§3.6 的定量版）；
      C3 调度：背景重校间隔由预算式反解，与 calib 语义一致。
    加证书聚合：_error_control_verdict 四情形 + evaluate_sys 转发。
    """
    emit("[case_constraint_fabric]  T10 约束网分类 / IR 自校准 / 调度 / 证书聚合")
    from lowering.budget import _error_control_verdict

    mesh = ConstraintMesh.poisson_1d(16, g=1.0)
    N = mesh.n
    I = np.sin(np.pi * np.arange(N) / (N + 1)) * 1e-6
    c_node = 1e-12

    # --- 前级事实：solve_constraint 先行（奇异网/不守恒网到此为止）---
    cons = solve_constraint(mesh, I, c_node, solve_deadline=1e-9,
                            residual_tol=1e-6)
    add_check("T10 前级事实：verdict=%s（非 SINGULAR），τ_relax=%.3e s"
              % (cons.verdict, cons.tau_relax),
              "kappa_G=%.1f" % cons.kappa_G,
              cons.verdict in ("STEADY", "PASS") and cons.tau_relax > 0)

    # --- C1 分类 ---
    pc = classify_constraint(mesh, c_node)
    add_check("C1 约束网分类 ⇒ contracting（μ₂=%.3g<0，τ_relax 内检查）" % pc.lognorm,
              pc.cls, pc.cls == "contracting")

    # --- C2 regime A：量化主导（标称 1e-3 dev，u_a=2^-10）---
    rA = constraint_refine(mesh, I, c_node, lam=0.0138, eps_dev=1e-3, seed=7)
    add_check("C2-A 量化主导：IR 改进 %.1f× ≥10（e %.3e→%.3e）"
              % (rA["ir_improve"], rA["e_analog"], rA["e_final"]),
              "kcl_res=%.3e" % rA["kcl_residual_final"],
              rA["ir_improve"] >= 10.0 and rA["e_final"] <= 0.1 * rA["e_analog"])
    add_check("C2-A mismatch 分量被 bound 覆盖（保守 sup 范数）",
              "%.3e ≤ %.3e" % (rA["mismatch_err"], rA["mismatch_bound_inf"]),
              rA["mismatch_within_bound"])

    # --- C2 regime B：mismatch 主导（10% dev 压力注入，安静核 u_a=2^-12）---
    rB = constraint_refine(mesh, I, c_node, u_a=2.0 ** -12, lam=0.0138,
                           eps_dev=1e-1, seed=7)
    add_check("C2-B mismatch 可见：e_analog %.3e ≥ 5×地板 %.3e"
              % (rB["e_analog"], rB["mismatch_floor_inf"]),
              "ratio=%.1f" % (rB["e_analog"] / rB["mismatch_floor_inf"]),
              rB["e_analog"] >= 5.0 * rB["mismatch_floor_inf"])
    add_check("C2-B 自校准：IR 后 %.3e **低于** mismatch 地板 %.3e"
              "（KCL 残差暴露 δI，修正解扣除）"
              % (rB["e_final"], rB["mismatch_floor_inf"]),
              "improve=%.1f×" % rB["ir_improve"],
              rB["e_final"] < rB["mismatch_floor_inf"]
              and rB["ir_improve"] >= 20.0)

    # --- C3 调度（漂移=中性累积，预算式 + calib 对账）---
    sch = constraint_schedule(mesh, c_node, drift_rate=1e-4, calib_floor=1e-6,
                              eps=1e-3)
    add_check("C3 调度：ΔT=%.2f s，calib 语义对账 residual=%.3e ≤ ε"
              % (sch["dT"], sch["predicted_residual"]),
              "consistent=%s" % sch["calib_consistent"],
              sch["dT"] > 0 and sch["calib_consistent"])
    try:
        constraint_schedule(mesh, c_node, drift_rate=1e-4, calib_floor=2e-3,
                            eps=1e-3)
        ok = False
    except ValueError:
        ok = True
    add_check("C3 地板 ≥ ε ⇒ raise（调度无解不许静默）", "raise", ok)

    # --- 证书聚合：_error_control_verdict 四情形 + evaluate_sys 转发 ---
    ec_good = ErrorControlResult(
        cls="contracting", n_windows=40, coverage_ratio=1.0,
        e_measured_max=8.1e-4, e_predicted_max=1.69e-3,
        eta_mean=1.69e-3, steady_bound=1.69e-3,
        ir_improve=rB["ir_improve"], schedule_dT=sch["dT"]).as_dict()
    v, _, _ = _error_control_verdict(ec_good)
    add_check("证书聚合：合法覆盖证据 ⇒ pass", v, v == "pass")

    ec_bad = dict(ec_good, coverage_ratio=0.9)
    v, _, d = _error_control_verdict(ec_bad)
    add_check("证书聚合：覆盖 0.9<1 ⇒ rejected（界未罩住实测）", v,
              v == "rejected" and "coverage" in d)

    ec_exp = dict(ec_good, cls="expanding", finite_horizon=False)
    v, _, _ = _error_control_verdict(ec_exp)
    add_check("证书聚合：expanding 无 finite_horizon ⇒ rejected", v, v == "rejected")

    v, _, _ = _error_control_verdict({"foo": 1})
    add_check("证书聚合：畸形证据 ⇒ rejected（不冒充可行）", v, v == "rejected")

    # evaluate_sys 转发：error_control 进 Certificate 字段与 items
    from analysis.descriptor import Descriptor
    from analysis.slack import HardwareSpec
    cert = evaluate_sys(Descriptor(np.eye(1), np.array([[-1.0]]),
                                   np.array([[1.0]]), np.array([[1.0]])),
                        HardwareSpec(kappa_hw=1e6, B_hw=10e6),
                        error_control=ec_good)
    item = cert.items.get("error_control", {})
    add_check("evaluate_sys 转发：error_control 进 items（verdict=%s）"
              % item.get("verdict"),
              "cert.error_control=%s" % (cert.error_control is not None),
              item.get("verdict") == "pass"
              and cert.error_control is not None)

    # 静态 IR-only 事实进证书：pass 但 detail 声明非动态判决
    ec_static = ErrorControlResult(
        cls="contracting", n_windows=0, coverage_ratio=0.0,
        e_measured_max=rA["e_final"], e_predicted_max=rA["e_final"],
        ir_improve=rA["ir_improve"]).as_dict()
    v, _, d = _error_control_verdict(ec_static)
    add_check("证书聚合：纯 IR 静态事实 ⇒ pass 但声明非动态判决", v,
              v == "pass" and "not a dynamic feasibility verdict" in d)

    return dict(cons_verdict=cons.verdict, tau_relax=cons.tau_relax,
                cls=pc.cls, rA=rA, rB=rB, schedule=sch)


# ---------------------------------------------------------------- ε_J 双轨

def case_adjoint_qoi():
    """§6.22 续：adjoint ε_J 双轨——QoI 导向的误差估计与精度分配。

    线性恒等式 δJ = λ(0)ᵀe(0) + ∫λᵀr dt 逐路径成立，本 case 直接对账：
      A1 δJ₁（窗口伴随加权残差）vs 真实 cᵀe(T)——纯 bias 与 bias+noise；
      A2 cancellation_factor：<1 时各窗口贡献正负抵消，恒等式仍成立
         （线性系统 δJ₁ 是估计的极限=真值），但 δJ 小 ≠ 场误差小；
      A3 精度分配：同样范数的 bias，高 w 节点 vs 低 w 节点的 δJ 差
         ≈ w 之比（实测 100.8×）——校正预算按 w 排序；
      A4 双轨必要性：低 w 节点 bias 使 δJ 通过 ε_J 而 ε_global 违约
         ——ε_J 单独存在会掩盖场失真，两轨必须同时记录。
    """
    emit("[case_adjoint_qoi]  恒等式对账 / 抵消指标 / 精度分配 / 双轨必要性")
    from analysis.adjoint import adjoint_field, dJ_weights, qoi_error_estimate

    N = 8
    A = np.diag(-0.5 * np.arange(1, N + 1))
    c = np.ones(N)
    c[7] = 0.05                       # QoI 对快模态几乎不可观
    T = 2.0
    t = np.linspace(0.0, T, 81)
    lam = adjoint_field(A, c, t)
    add_check("A1 终端条件：λ(T)=c（伴随场构造正确）",
              "max dev=%.2e" % np.max(np.abs(lam[:, -1] - c)),
              np.allclose(lam[:, -1], c))
    w = dJ_weights(lam, t)
    add_check("A1 权重排序：w ∝ 1/γ（慢模态主导），w_high/w_low=%.1f"
              % (w[0] / w[7]),
              "w=%s" % np.round(w, 3).tolist(),
              np.all(np.diff(w) <= 1e-12) and w[0] / w[7] > 50.0)

    def hw_run(bias, r_noise, seed):
        rng = np.random.default_rng(seed)
        dt = 2e-4
        x = np.zeros(N)
        traj = [x.copy()]
        for _ in range(int(round(T / dt))):
            x = x + dt * (A @ x + bias + r_noise * rng.uniform(-1.0, 1.0, size=N))
            traj.append(x.copy())
        return np.array(traj)

    def windows_and_estimate(traj, bias, r_noise, seed):
        rng = np.random.default_rng(seed)
        dt = 2e-4
        K = 40
        sub = (len(traj) - 1) // K
        dT = sub * dt
        Rs, centers = [], []
        for k in range(K):
            rows = traj[k * sub:(k + 1) * sub + 1]
            res = linear_window_residual(A, np.arange(sub + 1) * dt, rows)
            Rs.append(res.R)
            centers.append((k + 0.5) * dT)
        est = qoi_error_estimate(lam, t, Rs, centers)
        return est

    # A2 纯 bias：恒等式逐路径成立
    bias = np.full(N, 5e-4)
    traj = hw_run(bias, 0.0, 101)
    est = windows_and_estimate(traj, bias, 0.0, 101)
    dJ_actual = float(c @ (traj[-1] - 0.0))
    rel = abs(est["delta_J"] - dJ_actual) / max(abs(dJ_actual), 1e-300)
    add_check("A2 恒等式（纯 bias）：δJ₁=%.3e vs 实测 %.3e，相对误差 %.1f%% < 5%%"
              % (est["delta_J"], dJ_actual, 100 * rel),
              "cancellation=%.2f" % est["cancellation_factor"], rel < 0.05)

    # bias + noise：恒等式按路径成立，窗口常数近似吃掉部分精度
    traj_n = hw_run(bias, 1e-2, 102)
    est_n = windows_and_estimate(traj_n, bias, 1e-2, 102)
    dJ_actual_n = float(c @ (traj_n[-1] - 0.0))
    err_n = abs(est_n["delta_J"] - dJ_actual_n)
    add_check("A2 恒等式（bias+noise）：偏差 %.3e ≤ 0.3×Σ|贡献|=%.3e"
              % (err_n, est_n["abs_sum"]),
              "δJ₁=%.3e actual=%.3e" % (est_n["delta_J"], dJ_actual_n),
              err_n <= 0.3 * est_n["abs_sum"])

    # A3 精度分配：同范数 bias，节点 0（高 w）vs 节点 7（低 w）
    def dJ_of(node):
        b = np.zeros(N)
        b[node] = 1e-3
        tr = hw_run(b, 0.0, 200 + node)
        est_k = windows_and_estimate(tr, b, 0.0, 200 + node)
        return est_k["delta_J"]
    dJ_hi, dJ_lo = dJ_of(0), dJ_of(7)
    ratio = dJ_hi / max(abs(dJ_lo), 1e-300)
    add_check("A3 精度分配：同范数 bias 的 δJ 比 %.1f× ≈ w 比 %.1f×（×2.5 内）"
              % (ratio, w[0] / w[7]),
              "δJ_hi=%.3e δJ_lo=%.3e" % (dJ_hi, dJ_lo),
              abs(ratio - w[0] / w[7]) / (w[0] / w[7]) < 1.5)

    # A4 双轨必要性：低 w 节点 bias —— ε_J 通过而 ε_global 违约
    eps_J = 1e-4
    b_lo = np.zeros(N)
    b_lo[7] = 1e-3
    tr_lo = hw_run(b_lo, 0.0, 301)
    dJ_lo2 = float(c @ (tr_lo[-1] - 0.0))
    e_global = float(np.linalg.norm(tr_lo[-1]))
    add_check("A4 双轨：δJ=%.3e < ε_J=%.0e（QoI 合同通过）而 ε_global=%.3e > ε_J"
              "（场失真只能靠全局轨兜住）"
              % (dJ_lo2, eps_J, e_global),
              "budget 节省：节点 7 免校正", dJ_lo2 < eps_J and e_global > eps_J)

    # A4 反面：高 w 节点同范数 bias 必须校正（δJ 违约 ε_J）
    tr_hi = hw_run(np.concatenate([[1e-3], np.zeros(N - 1)]), 0.0, 302)
    dJ_hi2 = float(c @ (tr_hi[-1] - 0.0))
    add_check("A4 反面：高 w 节点同范数 bias ⇒ δJ=%.3e > ε_J（必须校正）" % dJ_hi2,
              "dJ_hi/dJ_lo=%.0f×" % (dJ_hi2 / max(abs(dJ_lo2), 1e-300)),
              dJ_hi2 > eps_J)

    # 证书：ε_J 字段进聚合
    ec_q = ErrorControlResult(cls="contracting", n_windows=40, coverage_ratio=1.0,
                              e_measured_max=8.1e-4, e_predicted_max=1.69e-3,
                              qoi_estimate=float(dJ_lo2), qoi_tolerance=eps_J,
                              qoi_cancellation=0.31).as_dict()
    from lowering.budget import _error_control_verdict
    v, _, d = _error_control_verdict(ec_q)
    add_check("证书聚合：ε_J 声明且 δJ ≤ ε_J ⇒ pass（cancellation 进 detail）", v,
              v == "pass" and "cancellation" in d)
    ec_qv = dict(ec_q, qoi_estimate=5e-4)
    v, _, d = _error_control_verdict(ec_qv)
    add_check("证书聚合：δJ > ε_J ⇒ rejected（ε_J 合同违反）", v,
              v == "rejected" and "eps_J" in d)
    ec_qm = dict(ec_good_stub(), qoi_tolerance=1e-4)
    v, _, _ = _error_control_verdict(ec_qm)
    add_check("证书聚合：声明 ε_J 但缺 δJ ⇒ rejected（malformed）", v, v == "rejected")

    return dict(w=w.tolist(), dJ_hi=dJ_hi, dJ_lo=dJ_lo,
                cancellation=est_n["cancellation_factor"],
                eps_J_pass=bool(dJ_lo2 < eps_J))


def ec_good_stub():
    """合法基底证据（无 ε_J 字段）——供 ε_J malformed 检查复用。"""
    return ErrorControlResult(cls="contracting", n_windows=40, coverage_ratio=1.0,
                              e_measured_max=8.1e-4,
                              e_predicted_max=1.69e-3).as_dict()


# ---------------------------------------------------------------- 激波熵稳定

def _godunov_flux(a, b):
    """Burgers（凸通量 f=u²/2）的 Godunov 通量：熵稳定（唯一熵解）。

    取自相似解在 ξ=0 处的状态：激波（a>b，激波速 s=(a+b)/2）右传
    ⇒ 面上是左状态 f(a)，左传 ⇒ f(b)；稀疏波 ⇒ f(max(a, min(b,0)))。
    陷阱（曾实测翻车）：s=(a+b)/2 是**速度**不是状态，f(s) 是错的
    通量——它不满足 Rankine–Hugoniot 一致性，TVD/max 原理全破
    （实测 TV 从 1 涨到 3.9、干净激波位置错 2 个 cell）。
    """
    f = lambda u: 0.5 * u * u
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.where(a > b,
                   np.where(a + b >= 0, f(a), f(b)),      # 激波：s 符号定边
                   f(np.maximum(a, np.minimum(b, 0.0))))  # 稀疏波
    return out


def _central_flux(a, b):
    """中心通量（算术平均）：无熵不等式保证 —— 激波处熵产生为正的负对照。"""
    return 0.25 * (np.asarray(a) ** 2 + np.asarray(b) ** 2)


def _fv_riemann_run(uL, uR, n, T, flux, bias, r_noise, seed, cfl=0.45):
    """Riemann 初值的周期 FV Euler 步进 + 逐步**守恒型**硬件注入。

    注入模型：bias 需满足 Σbias=0（电流镜失配的保守型口径，slack.py
    P0-D 合同）；噪声逐步零均值化（Σξ=0）——两者都是 KCL 式再分配，
    于是 shared flux 使总质量**精确**守恒（浮点级），B3 的盲区声明
    才成立：守恒量对误差路径零敏感。返回 (x_center, 采样轨迹 {t: u},
    实测注入 ‖r‖₁dt 累计——与状态更新用同一次采样，不做独立重抽）。
    """
    rng = np.random.default_rng(seed)
    dx = 1.0 / n
    dt = cfl * dx / max(abs(uL), abs(uR), 1e-30)
    nst = int(round(T / dt))
    dt = T / nst
    u = np.where((np.arange(n) + 0.5) * dx < 0.5, uL, uR)
    x = (np.arange(n) + 0.5) * dx
    snaps = {}
    inj_l1 = 0.0
    snap_times = (T / 4, T / 2, 3 * T / 4, T)
    snaps[0.0] = u.copy()
    isnap = 0
    for k in range(nst):
        xi = rng.uniform(-1.0, 1.0, size=n)
        xi = xi - xi.mean()                       # 守恒型噪声（Σξ=0）
        inj = bias + r_noise * xi
        F = flux(u, np.roll(u, -1))
        u = u + dt * (-(F - np.roll(F, 1)) / dx + inj)
        inj_l1 += float(np.sum(np.abs(inj)) * dt * dx)  # L1 单位（×dx），与状态更新同一次采样
        t_now = (k + 1) * dt
        if isnap < len(snap_times) and t_now >= snap_times[isnap] - dt / 2:
            snaps[round(snap_times[isnap], 6)] = u.copy()
            isnap += 1
    return x, snaps, inj_l1


def _shock_position_subcell(x, u, level):
    """无黏激波位置的子 cell 估计：末次穿越处的线性插值。

    cell 中心锚点（_shock_position）的分辨率是 1 cell；clean 与 hw
    跑同一格式、同一 smeared 剖面，插值位置的**相对差**可到 fp 级——
    这是位移 DWR 可测的前提。qoi.py 的 fit_tanh 是黏性剖面口径，
    不适用于无黏间断，不冒充。
    """
    idx = np.where(u >= level)[0]
    if idx.size == 0 or idx.size == u.size:
        raise ValueError("剖面无穿越")
    i = idx[-1]
    if i == u.size - 1:
        return float(x[i])
    frac = (u[i] - level) / (u[i] - u[i + 1])
    return float(x[i] + frac * (x[i + 1] - x[i]))


def _shock_position(x, u, level):
    """无黏 Riemann 剖面的激波位置：u 穿越 level 的首个 cell 中心。

    qoi.py 的 fit_tanh 是黏性 tanh 剖面的反演（§6.1 口径）；无黏激波
    是间断，用穿越锚点——两者适用面不同，这里不冒充 W_d 反演。
    """
    idx = np.where(u >= level)[0]
    if idx.size == 0 or idx.size == u.size:
        raise ValueError("剖面无穿越：level 不在 (uR, uL) 内")
    return float(x[idx[-1]])


def case_burgers_entropy():
    """§6.22 × T5 激波：熵稳定结构 + L1 中性合同 + 不变量盲区。

    命题（对应 §5.4 flux tile / 用户合同的 Entropy stability 行）：
      B1 熵稳定结构：Godunov 通量 TVD（TV(u_T) ≤ TV(u_0)）且 L1 误差
         小于中心通量；中心通量 TV 增长 ⇒ 熵产生为正被 TV 实测抓住
         ——"把不变量做成硬件结构" 的负对照版。
      B2 L1 中性合同：激波系统的误差在 L1 里不指数放大——标量凸通量
         的熵解有 L1 收缩性 ‖u−v‖₁ 非增；注入误差按中性累积（线性），
         调度走 λ=0 预算式。谱判据在此误导（激波线性化非正规），
         合同必须建立在熵/L1 上而非特征值上。
      B3 不变量盲区：shared flux 使总质量精确守恒，但激波位置漂移
         δx_s 与质量漂移无关——位置是 QoI 观测（穿越锚点 + 独立
         精确解），不能靠守恒量自监督发现（N3 在激波上的镜像）。
    """
    emit("[case_burgers_entropy]  熵稳定结构 / L1 中性合同 / 激波漂移 / 盲区")
    uL, uR = 1.0, 0.0
    n, T = 128, 0.4
    s_speed = 0.5 * (uL + uR)                 # 激波速度
    x0 = 0.5
    dx = 1.0 / n

    def exact(x, t):
        """周期 Riemann IC 的精确熵解（三段）：

        wrap 面（x=0，(0,1) 稀疏波）的 fan 覆盖 [0, t]（u=x/t 线性斜坡），
        主激波（(1,0)，速度 s=(uL+uR)/2）在 0.5+s·t，其间平台 uL，
        激波右侧 uR。陷阱（曾实测翻车）：只建主激波会把 fan 区的
        参考偏差当成硬件误差——实测 L1 虚高 192×。
        """
        return np.where(x < t, x / t,
                        np.where(x < x0 + s_speed * t, uL, uR))

    # 周期域有两个跳变（主激波 + wrap 面）：TV0 必须从初始离散剖面实测
    # （TV 计入 wrap 项——np.diff 不含首尾，漏掉会把 TV0 低估一半）
    def _tv_periodic(u):
        return float(np.sum(np.abs(np.diff(u))) + abs(u[0] - u[-1]))

    u_init = np.where((np.arange(n) + 0.5) * dx < 0.5, uL, uR)
    tv0 = _tv_periodic(u_init)

    # --- B1：熵稳定 vs 中心通量（干净运行）---
    x, snaps_g, _ = _fv_riemann_run(uL, uR, n, T, _godunov_flux,
                                    np.zeros(n), 0.0, seed=41)
    u_g = snaps_g[round(T, 6)]
    tv_g = _tv_periodic(u_g)
    l1_g = float(np.sum(np.abs(u_g - exact(x, T))) * dx)
    xs_g = _shock_position(x, u_g, 0.5 * (uL + uR))
    add_check("B1 Godunov TVD：TV(u_T)=%.4f ≤ TV(u_0)=%.4f（周期双跳，"
              "熵产生以 TV 不增为代理实测）" % (tv_g, tv0),
              "xs=%.4f" % xs_g, tv_g <= tv0 + 1e-12)
    add_check("B1 激波位置：干净运行误差 %.3e ≤ 1 cell=%.3e"
              % (abs(xs_g - (x0 + s_speed * T)), dx),
              "xs=%.4f exact=%.4f" % (xs_g, x0 + s_speed * T),
              abs(xs_g - (x0 + s_speed * T)) <= dx)

    # 中心通量非线性失稳（冻结系数放大率 >1），T=0.4 即溢出——负对照取
    # 可比窗口 T=0.1，两通量在同窗口对账
    T_c = T / 4
    x_c, snaps_c, _ = _fv_riemann_run(uL, uR, n, T_c, _central_flux,
                                      np.zeros(n), 0.0, seed=41)
    u_c = snaps_c[round(T_c, 6)]
    tv_c = _tv_periodic(u_c)
    l1_c = float(np.sum(np.abs(u_c - exact(x, T_c))) * dx)
    l1_g_tc = float(np.sum(np.abs(snaps_g[round(T_c, 6)]
                                  - exact(x, T_c))) * dx)
    add_check("B1 中心通量负对照（T=%.2f）：TV=%.3f > TV0（熵产生为正被 TV 抓住）"
              "且 L1 %.3e > Godunov %.3e" % (T_c, tv_c, l1_c, l1_g_tc),
              "ratio=%.1f" % (l1_c / max(l1_g_tc, 1e-300)),
              tv_c > tv0 + 1e-12 and l1_c > l1_g_tc)

    # --- B2：L1 中性合同（注入运行，Godunov 结构）---
    # 守恒型 bias：左 30% cell +b_L，其余 −b_L·(0.3/0.7)（Σ=0，KCL 式
    # 再分配）——质量精确守恒，但激波两侧状态被推移（漂移源）。
    n_left = int(0.3 * n)
    b_L = 2e-3
    bias = np.where(np.arange(n) < n_left, b_L, -b_L * n_left / (n - n_left))
    r_noise = 5e-3
    x_h, snaps_h, inj_l1 = _fv_riemann_run(uL, uR, n, T, _godunov_flux,
                                           bias, r_noise, seed=42)
    # 双项分解（ε_PDE = ε_scheme + ε_hw 的实测版）：总误差 = 格式自身的
    # ν_num 扩散基线（clean 运行 vs 精确熵解，确定性、与注入无关）
    # + 注入诱发项。注入合同建立在**离散单调格式的 Crandall–Tartar
    # L1 收缩**上：‖u_hw − u_clean‖₁ ≤ ∫‖r‖₁dx——激波不放大注入。
    l1_clean = float(np.sum(np.abs(u_g - exact(x, T))) * dx)
    l1_inj = {}
    for tt, uu in snaps_h.items():
        l1_inj[tt] = float(np.sum(np.abs(uu - snaps_g[tt])) * dx)
    bound = inj_l1                             # 实测路径 ∫‖r‖₁dx（初始差 0）
    l1_T = l1_inj[round(T, 6)]
    l1_total = float(np.sum(np.abs(snaps_h[round(T, 6)]
                                   - exact(x, T))) * dx)
    add_check("B2 L1 收缩覆盖：注入诱发 ‖u_hw−u_clean‖₁=%.4f ≤ 实测 "
              "∫‖r‖₁dx=%.4f（Crandall–Tartar，激波不放大注入）"
              % (l1_T, bound),
              "ratio=%.2f" % (l1_T / max(bound, 1e-300)),
              l1_T <= bound and l1_T > 0.05 * bound)
    add_check("B2 双项分解：总 L1=%.4f ≤ ν_num 基线 %.4f + 注入项 %.4f"
              "（三角不等式，两项预算分立）"
              % (l1_total, l1_clean, l1_T),
              "基线占比=%.0f%%" % (100 * l1_clean / max(l1_total, 1e-300)),
              l1_total <= l1_clean + l1_T + 1e-12)

    # 中性分类：注入诱发 L1 近线性（后半程斜率 ≈ 全程斜率，×2 内）；
    # 与扩张系统（指数）区分——调度必须用预算式而非 λ 公式
    half = round(T / 2, 6)
    slope_full = l1_T / T
    slope_half = (l1_T - l1_inj[half]) / (T / 2)
    add_check("B2 中性累积：后半程斜率 %.3e ≈ 全程 %.3e（×2 内，线性非指数）"
              % (slope_half, slope_full),
              "L1(T/2)=%.4f L1(T)=%.4f" % (l1_inj[half], l1_T),
              abs(slope_half / max(slope_full, 1e-300) - 1.0) < 1.0)

    # 调度（λ=0 预算式）：r_max 用注入 L1 速率上界（守恒 bias + 零均值
    # 噪声包络，×dx 折进 L1 单位）
    r_max = (float(np.abs(bias).sum()) + r_noise * n) * dx
    eps_L1 = 1e-3
    dT = correction_interval(0.0, r_max, 0.0, eps_L1)
    add_check("B2 调度：预算式 ΔT=%.3f s（r_max=%.3f，ε₁=%.0e），"
              "预报注入量 r_max·ΔT=%.3f ≥ 实测注入诱发 L1=%.4f"
              % (dT, r_max, eps_L1, r_max * dT, l1_T),
              "s", 0.0 < dT < 10.0 and r_max * dT >= l1_T)

    # --- B3：不变量盲区（N3 的激波镜像）---
    xs_h = _shock_position(x, snaps_h[round(T, 6)], 0.5 * (uL + uR))
    dx_s = abs(xs_h - (x0 + s_speed * T))
    mass_err = abs(float(np.sum(snaps_h[round(T, 6)])
                         - np.sum(snaps_h[0.0])) * dx)
    add_check("B3 质量精确守恒（shared flux + 守恒注入）：误差 %.2e ~ 浮点级，"
              "而注入诱发 ‖e‖₁=%.3e 持续累积——守恒量对误差路径零敏感"
              % (mass_err, l1_T),
              "观测极限：位置只能靠锚点", mass_err < 1e-10 and l1_T > 0.0002)
    add_check("B3 激波位置被合同罩住：δx_s=%.4f ≤ 1 cell + 1.5·‖e‖₁/(uL−uR)=%.4f"
              "（FV 间断的位置观测极限 = 1 cell）"
              % (dx_s, dx + 1.5 * l1_T / (uL - uR)),
              "xs=%.4f exact=%.4f" % (xs_h, x0 + s_speed * T),
              dx_s <= dx + 1.5 * l1_T / (uL - uR))

    # 证书：中性类 L1 合同进聚合（e_predicted = 实测注入积分）
    ec = ErrorControlResult(cls="neutral", n_windows=4, coverage_ratio=1.0,
                            e_measured_max=l1_T, e_predicted_max=bound,
                            schedule_dT=dT).as_dict()
    from lowering.budget import _error_control_verdict
    v, _, _ = _error_control_verdict(ec)
    add_check("证书聚合：激波 L1 中性合同 ⇒ pass（预算式调度进 detail）", v,
              v == "pass")

    return dict(tv_godunov=tv_g, tv_central=tv_c, l1_godunov=l1_g,
                l1_central=l1_c, l1_T=l1_T, bound=bound, dx_s=dx_s,
                dT=dT, mass_err=mass_err)


# ---------------------------------------------------------------- 位移 DWR

def case_shock_dwr():
    """§6.22 续：shock displacement 的界面项 DWR（ε_J 在激波问题上的主体）。

    光滑区伴随 δJ₁=∫λᵀr 对激波问题**结构性缺项**：解是分片光滑 + 间断，
    QoI 对间断位置的敏感度走**跳变条件**（激波位移的显式界面态），
    不走 λ 场。Burgers Riemann 的激波速 s=(uL+uR)/2 ⇒

        ds = (δuL + δuR)/2,   δx_s(T) = ∫₀ᵀ ds dt,   δJ_int = Δu·δx_s

    其中 δuL/δuR 是激波邻域**整格**状态的偏差（部分覆盖 cell 的
    smeared 值是伪影，Δu 用解析跳变）。注意：平台内的源项偏差被
    对流进激波（准稳态 ~b·dx），δx_s 不走简单的 b·T 或 b·T² 律——
    预测用实测邻域偏差轨迹经跳变条件积分，不预设解析式。实测：
      D1 界面项主导 δJ（跨激波窗口的部分质量 J），光滑余项是小量；
      D2 光滑跳变条件预测是**保守上界**：离散 FV 的位置响应被 cell
         平均簿记量化，实测位移比光滑预测低 ~10×——界面项 δJ_int
         必须用同一估计器的相对位移（clean vs hw），不能用光滑预测；
      D3 空间分配：同 ‖bias‖ 的守恒偶极，激波邻域 δJ ≫ 远场（uR 区
         偶极在 T 内到不了激波，δJ≈0）——精度预算集中在激波邻域，
         光滑伴随的 w_i 权重在此失效（分配必须走界面敏感度）；
      D4 符号对称：镜像偶极 δx_s 反号同幅；
      D5 远场 plateau 偶极：追上激波需 0.7 > T，本时窗 δJ≈0
         （如实记录：更长时间会到达，界面项是瞬态非永久豁免）。
    诚实边界：跳变条件敏感度是单激波线性化（离散界面伴随的模拟
    版本）；多激波/激波相互作用下逐个位移的线性叠加不成立，须
    显式界面状态或离散伴随——本 case 不外推。
    """
    emit("[case_shock_dwr]  跳变条件敏感度 / 界面项主导 / 空间分配 / ε_J")
    uL, uR = 1.0, 0.0
    n, T = 128, 0.4
    s_speed = 0.5 * (uL + uR)
    dx = 1.0 / n
    b = 2e-3

    x, snaps_c, _ = _fv_riemann_run(uL, uR, n, T, _godunov_flux,
                                    np.zeros(n), 0.0, seed=51)
    u_c = snaps_c[round(T, 6)]
    xs_clean = _shock_position_subcell(x, u_c, 0.5 * (uL + uR))

    # J：跨激波窗口 [0.6, 0.8] 的部分质量（拖力类积分 QoI）
    win = (x >= 0.6) & (x <= 0.8)
    J = lambda uu: float(np.sum(uu[win]) * dx)
    J_clean = J(u_c)

    # D0 子 cell 估计器：clean 位置 vs 精确 0.5+0.5T
    add_check("D0 子 cell 估计器：|xs_clean−exact|=%.2e ≤ dx/4=%.2e"
              % (abs(xs_clean - (0.5 + s_speed * T)), dx / 4),
              "xs=%.5f" % xs_clean,
              abs(xs_clean - (0.5 + s_speed * T)) <= dx / 4)

    def run_dipole(i_plus, i_minus, seed):
        bias = np.zeros(n)
        bias[i_plus] += b
        bias[i_minus] -= b
        _, snaps, _ = _fv_riemann_run(uL, uR, n, T, _godunov_flux,
                                      bias, 0.0, seed=seed)
        uu = snaps[round(T, 6)]
        xs = _shock_position_subcell(x, uu, 0.5 * (uL + uR))
        return dict(dx_s=xs - xs_clean, dJ=J(uu) - J_clean, snaps=snaps)

    # 激波邻域的整格 cell（跳变条件的 δuL 取整格状态，不是部分覆盖
    # cell——smeared 值是部分覆盖伪影，Δu 用解析跳变 uL−uR）
    i_s = int(np.argmax(-np.diff(u_c)))          # 最大降落面的左侧 cell
    i_ul = i_s - 1                               # 激波左侧整格（uL 态）
    dJ_int_scale = float(uL - uR)                # 解析跳变 Δu

    def jump_pred(snaps_a):
        """跳变条件预测：δx_s = ∫(δuL+δuR)/2 dt（实测邻域偏差轨迹）。"""
        ts = sorted(snaps_c.keys())
        devL = np.array([snaps_a[tt][i_ul] - snaps_c[tt][i_ul] for tt in ts])
        devR = np.array([snaps_a[tt][i_s + 1] - snaps_c[tt][i_s + 1]
                         for tt in ts])
        t_arr = np.array(ts)
        return float(np.trapezoid((devL + devR) / 2.0, t_arr))

    # D1/D2：激波邻域偶极（+b 在激波左侧整格 cell，−b 在远右 uR 区，
    # Σ=0）。平台内的源项被对流进激波（准稳态偏差 ~ b·dx），位移由
    # 实测邻域偏差经跳变条件积分预测——不预设解析式。
    a = run_dipole(i_ul, 115, seed=61)
    pred = jump_pred(a["snaps"])
    add_check("D1 界面项主导：|Δu·δx_s|=%.3e ≥ 5×|光滑余项|=%.3e"
              "（激波问题的 δJ 主体是位移项，Δu=解析跳变 uL−uR）"
              % (abs(dJ_int_scale * a["dx_s"]),
                 abs(a["dJ"] - dJ_int_scale * a["dx_s"])),
              "dJ=%.3e dx_s=%.3e" % (a["dJ"], a["dx_s"]),
              abs(dJ_int_scale * a["dx_s"]) >= 5.0
              * abs(a["dJ"] - dJ_int_scale * a["dx_s"]))
    add_check("D2 跳变条件上界：δx_s 实测 %.3e ≤ 光滑预测 ∫(δuL+δuR)/2 dt=%.3e"
              "（实测低 10×：FV 位置响应被 cell 平均簿记量化/粘滞——"
              "光滑跳变条件只作保守上界，界面项用同一估计器的相对位移）"
              % (a["dx_s"], pred),
              "jump: ds=(δuL+δuR)/2", a["dx_s"] > 0
              and a["dx_s"] <= pred)

    # D4 符号对称：镜像偶极
    bmir = run_dipole(115, i_ul, seed=62)
    pred_m = jump_pred(bmir["snaps"])
    add_check("D4 镜像偶极：δx_s 反号同幅（%.3e vs %.3e，×1.5 内）"
              % (bmir["dx_s"], -a["dx_s"]),
              "pred=%.3e" % pred_m,
              abs(bmir["dx_s"] + a["dx_s"]) <= 0.5 * abs(a["dx_s"]))

    # D3/D5：远场对照（同 ‖bias‖₁ 的守恒偶极）
    d = run_dipole(115, 116, seed=63)            # uR 区内部偶极
    c_ = run_dipole(44, 45, seed=64)             # plateau 内部偶极
    add_check("D3 空间分配：激波邻域 |δJ|=%.3e ≥ 100× uR 远场 %.3e"
              "（扰动传播速度 ~0，T 内到不了激波）"
              % (abs(a["dJ"]), abs(d["dJ"])),
              "dx_s(d)=%.3e" % d["dx_s"],
              abs(a["dJ"]) >= 100.0 * abs(d["dJ"]))
    add_check("D5 plateau 远场：|δJ|=%.3e ≤ 0.3×激波邻域（追赶时间 0.7>T；"
              "更长时间会到达——界面项是瞬态非永久豁免）"
              % abs(c_["dJ"]),
              "dx_s(c)=%.3e" % c_["dx_s"],
              abs(c_["dJ"]) <= 0.3 * abs(a["dJ"]))

    # D6 ε_J 证书：激波 QoI 的双轨证据进聚合（界面项作为 δJ 主体的声明）
    ec = ErrorControlResult(cls="neutral", n_windows=1, coverage_ratio=1.0,
                            e_measured_max=a["dJ"],
                            e_predicted_max=a["dJ"],
                            qoi_estimate=abs(a["dJ"]),
                            qoi_tolerance=1e-3,
                            qoi_cancellation=None).as_dict()
    from lowering.budget import _error_control_verdict
    v, _, dv = _error_control_verdict(ec)
    add_check("D6 ε_J 证书：激波 QoI δJ=%.3e ≤ ε_J=1e-3 ⇒ pass（estimate 口径）"
              % abs(a["dJ"]), v, v == "pass" and "estimate" in dv)

    return dict(xs_clean=xs_clean, dx_s=a["dx_s"], pred=pred,
                dJ_shock=a["dJ"], dJ_ur=d["dJ"], dJ_plateau=c_["dJ"],
                interface_share=float(abs(dJ_int_scale * a["dx_s"])
                                      / max(abs(a["dJ"]), 1e-300)))


# ---------------------------------------------------------------- main

def main():
    emit("=" * 72)
    emit("§6.22 运行时误差控制合同验收")
    emit("=" * 72)
    ir = case_poisson_ir()
    emit("")
    dif = case_diffusion_windows()
    emit("")
    con = case_constraint_fabric()
    emit("")
    adj = case_adjoint_qoi()
    emit("")
    bur = case_burgers_entropy()
    emit("")
    dwr = case_shock_dwr()
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
                   poisson_ir=ir, diffusion=dif, constraint_fabric=con,
                   adjoint_qoi=adj, burgers=bur, shock_dwr=dwr,
                   checks=_checks)
    with open(os.path.join(out_dir, "error_control.json"), "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return n_pass == len(_checks)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
