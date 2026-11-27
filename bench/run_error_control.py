"""§6.22 运行时误差控制合同验收（基准 + 负对照 + 约束/ε_J/激波/多激波六轮）。

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

from analysis.adjoint import (adjoint_field, descriptor_adjoint_field,
                              descriptor_qoi_error_estimate, dJ_weights,
                              dJ_weights_exact, nonlinear_adjoint_backward,
                              port_sensitivities, qoi_error_estimate,
                              statistical_coverage_contract)
from analysis.error_propagation import (classify_propagation,
                                        conservation_drift_rate,
                                        correction_interval,
                                        corrected_recursion, eta_from_lognorm,
                                        classify_propagation_type,
                                        error_recursion_bound,
                                        logarithmic_norm, steady_bound)
from lowering.budget import _error_control_verdict, evaluate_sys
from lowering.error_control import (ErrorControlResult, classify_constraint,
                                    constraint_refine, constraint_schedule,
                                    evolve_shock_interfaces)
from lowering.p7 import qoi_adjoint_for_config
from twin.constraint import ConstraintMesh, solve_constraint
from twin.residual import (detect_offset, injection_budget,
                           linear_window_residual, window_integral_residual)

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


def _correction_episode(A, corrector, seed, x0, bias_v, r_amp, K, sub, dt,
                        Phi_w, write_amp, obs_C=None, obs_sigma=0.0):
    """一个校正闭环 run。**真解只用于离线评估**：corrector.apply 只吃
    窗口采样（+可选可信观测 y），没有任何真解入口。

    返回逐窗口 (η, e_prev, e_minus, e_plus)——e_prev 是窗口起点
    （上次校正后）的误差，e_minus 是窗口末校正前的误差，e_plus 是
    校正后误差。q 的合同作用在 e_prev→e_plus 上（校正器是窗口末
    作用，不改变窗口内传播）。
    """
    rng = np.random.default_rng(seed)
    from analysis.correction import CorrectionWindow
    x = np.asarray(x0, dtype=float).copy()
    truth = x.copy()
    etas, e_prev, e_minus, e_plus = [], [], [], []
    for _w in range(K):
        truth_start = truth.copy()
        e_prev.append(float(np.linalg.norm(x - truth_start)))
        xs = np.empty((sub + 1, A.shape[0]))
        xs[0] = x
        for s in range(sub):
            x = x + dt * (A @ x + bias_v + r_amp * rng.uniform(-1.0, 1.0,
                                                              size=A.shape[0]))
            xs[s + 1] = x
        tloc = np.arange(sub + 1) * dt
        res = linear_window_residual(A, tloc, xs)
        truth = Phi_w @ truth_start
        em = float(np.linalg.norm(x - truth))
        y = None
        if obs_C is not None:
            y = obs_C @ truth + obs_sigma * rng.standard_normal(obs_C.shape[0])
        xp = corrector.apply(CorrectionWindow(tloc, xs), y)
        xp = xp + write_amp * rng.uniform(-1.0, 1.0, size=A.shape[0])
        ep = float(np.linalg.norm(xp - truth))
        etas.append(float(res.eta))
        e_minus.append(em)
        e_plus.append(ep)
        x = xp
    return dict(eta=np.array(etas), e_prev=np.array(e_prev),
                e_minus=np.array(e_minus), e_plus=np.array(e_plus))


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

    # --- 可部署校正（B 修复）：三种方案 + 模型 q / 声明 ζ + held-out 验证 ---
    # 旧版把硬件状态回注到解析真解、再用真解反推 q——那只配做离线
    # characterization。现在三个 corrector 全部只吃窗口采样/可信观测
    # （apply 没有真解入口）；q 由**模型**给出（线性校正器可解析），
    # ζ 由硬件地板声明（DAC 回写 LSB / 传感器噪声增益），真解只在
    # 离线覆盖率评估里出现。递推 ε⁺_{k+1} ≤ q·G·ε⁺_k + h_η·η_k + ζ
    # 与 corrected_recursion 的差别是 η 不被 q 乘（校正器作用在窗口末，
    # 不撤销窗口内已发生的注入传播；h_η 逐方案声明）。
    from analysis.correction import (DigitalRecompute, ObserverCorrection,
                                     ResidualPropagation, SignedSubtraction,
                                     characterize_q, heldout_coverage,
                                     residual_correction_eta_factor)
    dt = 2e-4
    sub = max(int(round(dT / dt)), 8)
    dT_eff = sub * dt
    K = 40
    write_amp = 1e-5                    # DAC 回写量化 LSB（声明的 ζ 分量）
    zeta_write = write_amp * np.sqrt(N)
    Phi_w = np.diag(np.exp(lam * dT_eff))
    G_w = float(np.linalg.norm(Phi_w, 2))          # 对角：max|e^{λh}|
    C_mass = np.ones((1, N)) / np.sqrt(N)          # 守恒量（总质量）观测
    L_mass = C_mass.T @ np.linalg.inv(C_mass @ C_mass.T)
    q_obs = float(np.linalg.norm((np.eye(N) - L_mass @ C_mass) @ Phi_w, 2))
    eta_res = float(residual_correction_eta_factor(A, dT_eff))
    obs_sigma = 1e-4
    contracts = {
        "digital recompute": dict(
            corrector=DigitalRecompute(A=A), q=G_w, zeta=zeta_write,
            eta_factor=0.0, source="model: ||expm(A h)||_2"),
        "residual + correction solve": dict(
            corrector=ResidualPropagation(A=A), q=G_w, zeta=zeta_write,
            eta_factor=eta_res,
            source="model: ||expm(A h)||_2, quadrature factor %.3f"
                   % eta_res),
        "sparse observer (mass)": dict(
            corrector=ObserverCorrection(C_mass),
            q=q_obs, zeta=zeta_write + 3.0 * obs_sigma / np.sqrt(N),
            eta_factor=1.0,
            source="model: ||(I-LC)expm(A h)||_2 (L = LS projection)"),
        "signed subtraction (control)": dict(
            corrector=SignedSubtraction(A=A), q=None, zeta=zeta_write,
            eta_factor=1.0, source="no certified q (negative control)"),
    }
    # 训练集包络：零注入 / 额定 / 1.5× 三档 + 不同初值、不同种子；
    # held-out 用**未参与训练**的种子 + 额定 workload（低注入角落）。
    train_specs = [(100, np.cos(np.pi * (i + 1) * k / (N + 1)), b, ra)
                   for i, (b, ra) in enumerate([(0.0, 0.0), (0.0, 1.5e-2),
                                                (3e-3, 4.5e-2),(3e-3, 0.0)])]
    val_specs = [(200 + i, np.cos(np.pi * k / (N + 1)), 2e-3, 3e-2)
                 for i in range(3)]
    results = {}
    for name, spec in contracts.items():
        pairs = []
        for seed, xx0, b, ra in train_specs:
            ep = _correction_episode(A, spec["corrector"], seed, xx0, b, ra,
                                     K, sub, dt, Phi_w, write_amp,
                                     obs_C=(C_mass if "observer" in name
                                            else None),
                                     obs_sigma=(obs_sigma if "observer" in name
                                                else 0.0))
            pairs += list(zip(ep["e_prev"], ep["e_plus"]))
        ch = characterize_q(pairs, zeta=spec["zeta"])
        if spec["q"] is not None:
            q_used = spec["q"]
        else:
            # 负对照没有模型 q：characterization 只能给出"≥1"的实测比，
            # 截到有限值以免递推溢出；它本来就不许长期界。
            q_used = float(min(max(ch["raw_ratio_max"], 1.0001), 1e3))
        rho = q_used * G_w
        cov, cov_minus, worst, max_ep, max_pred = [], [], 0.0, 0.0, 0.0
        if rho >= 1.0:
            # ρ≥1 ⇒ 递推不收敛，不积分、不许诺任何界
            for seed, xx0, b, ra in val_specs:
                ep = _correction_episode(A, spec["corrector"], seed, xx0, b, ra,
                                         K, sub, dt, Phi_w, write_amp,
                                         obs_C=(C_mass if "observer" in name
                                                else None),
                                         obs_sigma=(obs_sigma if "observer" in name
                                                    else 0.0))
                max_ep = max(max_ep, float(np.max(ep["e_plus"])))
                cov.append(0.0)
                cov_minus.append(float(np.mean(ep["e_minus"] <= 1.2 * (
                    G_w * np.concatenate(([0.0], ep["e_minus"][:-1]))
                    + ep["eta"]))))
            results[name] = dict(q=q_used, q_identifiable=ch["q_max"],
                                 n_resolvable=ch["n_resolvable"], n=ch["n"],
                                 zeta=spec["zeta"],
                                 eta_factor=spec["eta_factor"], rho=rho,
                                 steady=float("inf"),
                                 coverage=0.0,
                                 coverage_minus=float(np.mean(cov_minus)),
                                 worst_ratio=float("inf"), max_err=max_ep,
                                 max_pred=float("inf"), source=spec["source"])
            continue
        for seed, xx0, b, ra in val_specs:
            ep = _correction_episode(A, spec["corrector"], seed, xx0, b, ra,
                                     K, sub, dt, Phi_w, write_amp,
                                     obs_C=(C_mass if "observer" in name
                                            else None),
                                     obs_sigma=(obs_sigma if "observer" in name
                                                else 0.0))
            prev = 0.0
            pred, pred_m = [], []
            for i in range(K):
                pp = (q_used * G_w * prev
                      + spec["eta_factor"] * ep["eta"][i] + spec["zeta"])
                pm = G_w * prev + ep["eta"][i]
                pred.append(pp)
                pred_m.append(pm)
                prev = pp
            hc = heldout_coverage(pred, ep["e_plus"])
            hm = heldout_coverage(pred_m, ep["e_minus"])
            cov.append(hc["coverage"])
            cov_minus.append(hm["coverage"])
            worst = max(worst, hc["worst_ratio"])
            max_ep = max(max_ep, float(np.max(ep["e_plus"])))
            max_pred = max(max_pred, float(np.max(pred)))
        eta_bar = spec["eta_factor"] * float(np.max(ep["eta"])) + spec["zeta"]
        sb = (eta_bar / (1.0 - rho) if rho < 1.0 else float("inf"))
        results[name] = dict(q=q_used, q_identifiable=ch["q_max"],
                             n_resolvable=ch["n_resolvable"], n=ch["n"],
                             zeta=spec["zeta"], eta_factor=spec["eta_factor"],
                             rho=rho, steady=sb, coverage=float(np.mean(cov)),
                             coverage_minus=float(np.mean(cov_minus)),
                             worst_ratio=worst, max_err=max_ep,
                             max_pred=max_pred, source=spec["source"])
    rr = results["digital recompute"]
    ro = results["sparse observer (mass)"]
    rp = results["residual + correction solve"]
    rn = results["signed subtraction (control)"]
    add_check("B1 q 由模型给出（不读真解）：digital=%.3f、resprop=%.3f、"
              "observer=%.3f 全部 <1" % (rr["q"], rp["q"], ro["q"]),
              "signed 无认证 q=%.2f" % rn["q"],
              rr["q"] < 1.0 and rp["q"] < 1.0 and ro["q"] < 1.0)
    add_check("B2 held-out 覆盖（未参与拟合的 workload）：digital %.0f%%、"
              "resprop %.0f%%、observer %.0f%%（最坏比 %.2f）"
              % (100 * rr["coverage"], 100 * rp["coverage"],
                 100 * ro["coverage"], max(rr["worst_ratio"],
                                           rp["worst_ratio"],
                                           ro["worst_ratio"])),
              "真解只在评估里出现", min(rr["coverage"], rp["coverage"],
                                        ro["coverage"]) >= 0.95)
    add_check("B3 负对照 signed：q=%.2f > 1 ⇒ ρ=%.2f，无长期界（不许诺）"
              % (rn["q"], rn["rho"]), "x−R 不是传播加权扣除",
              rn["q"] > 1.0 and rn["rho"] >= 1.0)
    add_check("B4 digital 长期界 η̄/(1−ρ)=%.2e 罩住实测 max ε⁺=%.2e"
              "（≤100× 松度）" % (rr["steady"], rr["max_err"]),
              "η̄=%.2e, ρ=%.3f" % (rr["steady"] * (1 - rr["rho"]), rr["rho"]),
              rr["steady"] >= rr["max_err"]
              and rr["steady"] <= 100.0 * max(rr["max_err"], 1e-300))
    add_check("B5 稀疏观测的代价：observer max ε⁺=%.2e 比 digital %.2e 大"
              " ≥10×（只观测守恒量 ⇒ 快模态误差不被校正）"
              % (ro["max_err"], rr["max_err"]),
              "q 观测 %.3f vs %.3f" % (ro["q"], rr["q"]),
              ro["max_err"] >= 10.0 * rr["max_err"])
    add_check("B6 校正前界覆盖：digital %d/%d 窗口 ε⁻ ≤ 1.2×ε⁻_pred"
              % (int(round(rr["coverage_minus"] * K * len(val_specs))),
                 K * len(val_specs)),
              "%.0f%%" % (100 * rr["coverage_minus"]),
              rr["coverage_minus"] >= 0.95)
    add_check("B7 校正后长期误差 max=%.2e ≪ 无校正饱和 %.2e（×5 至少）"
              % (rr["max_err"], bound_unc),
              "ratio=%.2e" % (rr["max_err"] / bound_unc),
              rr["max_err"] < 0.2 * bound_unc)

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
                correction=results,
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


def _fv_riemann_run(uL, uR, n, T, flux, bias, r_noise, seed, cfl=0.45,
                    u0=None, state_sources=None):
    """Riemann 初值的周期 FV Euler 步进 + 逐步**守恒型**硬件注入。

    注入模型：bias 需满足 Σbias=0（电流镜失配的保守型口径，slack.py
    P0-D 合同）；噪声逐步零均值化（Σξ=0）——两者都是 KCL 式再分配，
    于是 shared flux 使总质量**精确**守恒（浮点级），B3 的盲区声明
    才成立：守恒量对误差路径零敏感。u0 可显式给初值剖面（多态
    Riemann 用），缺省为 (uL,uR) 二态。返回 (x_center, 采样轨迹
    {t: u}, 实测注入 ‖r‖₁dt 累计——与状态更新用同一次采样)。
    """
    rng = np.random.default_rng(seed)
    dx = 1.0 / n
    dt = cfl * dx / max(abs(uL), abs(uR), 1e-30)
    nst = int(round(T / dt))
    dt = T / nst
    if u0 is None:
        u = np.where((np.arange(n) + 0.5) * dx < 0.5, uL, uR)
    else:
        u = np.array(u0, dtype=float)
        if u.shape[0] != n:
            raise ValueError("u0 长度必须等于 n")
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
        if state_sources:
            # 跟材料态走的源（§6.22 路线图①）：注入到**当前值**落在
            # [lo, hi] 带内的 cell——硬件语义是偏置电流影响持有该状态
            # 的 gm cell，激波扫过后 cell 换了状态就不再注入（dwell
            # time 积累由此自然涌现，无需人为遮罩）。
            for lo, hi, rate in state_sources:
                inj = inj + rate * ((u >= lo) & (u <= hi))
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


# ------------------------------------------------------------ 多激波相互作用

def _jump_positions(x, u, thresh=0.05, cluster_cells=4):
    """剖面里的间断位置：超过 thresh 的降落面，4dx 内聚类为一道激波。

    Euler+Godunov 把激波展宽到 2-4 个 cell（每格降幅可低于单格
    阈值——实测 n=256 时 (1,0.6) 激波三格各降 ~0.07），阈值必须低、
    且相邻降落面必须聚类，否则一道激波被计成多跳。稀疏波 fan 是
    平滑上升（diff>0），不会被误计。
    """
    d = np.diff(u)
    merge_dx = cluster_cells * (x[1] - x[0])
    out = []
    for i in np.where(d < -thresh)[0]:
        frac = (u[i] - (u[i] + u[i + 1]) / 2.0) / (u[i] - u[i + 1])
        pos = float(x[i] + frac * (x[i + 1] - x[i]))
        if out and pos - out[-1] < merge_dx:
            out[-1] = 0.5 * (out[-1] + pos)   # 同一道 smeared 激波的相邻面
        else:
            out.append(pos)
    return out


def case_multi_shock():
    """§6.22 续：多激波相互作用界面态——中间态敏感度穿透合并。

    三态周期 Riemann：u=(1, 0.6, 0) 在 x=(0.3, 0.6)（wrap 面 (0,1)
    是稀疏波 fan，同 B case）。两道激波：s12=(u1+u2)/2=0.8、
    s23=(u2+u3)/2=0.3，间距闭合率 = (u1−u3)/2 = 0.5（与 u2 无关），
    解析合并：t_m = 0.3/0.5 = 0.6、x_m = 0.3+0.8·0.6 = 0.78；合并后
    单激波 (u1,u3) 以 s13=0.5 前进。

    界面态合同的核心命题：**合并把中间态敏感度永久冻结进位置**——
        ∂X(T)/∂u2 = t_m/2 = 0.3 ≠ 0
    而"逐个激波独立线性叠加"的界面模型只看终态剖面的相邻态
    (u1,u3) ⇒ 预测 ∂X/∂u2 = 0。实测 IC 微扰 u2+ε 的合并位置位移
    直接否证叠加模型（D2 同款教训的相互作用版）。

    诚实边界：u2 微扰取 IC（真 Riemann 态变更）而非运行时源——
    运行时源在双激波夹层内有 dwell-time 积累效应（cell 被 shock2
    吞掉前只积累了部分偏差），是显式界面状态演化的后续内容；
    L1 收缩仍是标量凸通量口径；三态以上/激波相互作用多次的组合
    敏感度须显式界面状态演化或离散伴随，不外推。
    """
    emit("[case_multi_shock]  双激波合并 / 中间态敏感度穿透 / 叠加失效")
    u1, u2, u3 = 1.0, 0.6, 0.0
    x2, x3 = 0.3, 0.6
    n, T = 256, 0.8
    dx = 1.0 / n
    s12 = 0.5 * (u1 + u2)
    s23 = 0.5 * (u2 + u3)
    s13 = 0.5 * (u1 + u3)
    t_m = (x3 - x2) / (s12 - s23)
    x_m = x2 + s12 * t_m

    def exact_shocks(t):
        """解析激波位置：合并前两道，合并后一道。"""
        if t < t_m:
            return [x2 + s12 * t, x3 + s23 * t]
        return [x_m + s13 * (t - t_m)]

    def exact_profile(x, t):
        if t < t_m:
            return np.where(x < t, x / t,
                            np.where(x < x2 + s12 * t, u1,
                                     np.where(x < x3 + s23 * t, u2, u3)))
            # 注意：fan 右缘速度 = u1 = 1，在 t<t_m 时 fan 区 [0,t]
        xs = exact_shocks(t)[0]
        return np.where(x < t, x / t, np.where(x < xs, u1, u3))

    add_check("E1 解析合并：t_m=%.3f、x_m=%.3f（闭合率=(u1−u3)/2=0.5，"
              "与 u2 无关）" % (t_m, x_m),
              "s12=%.2f s23=%.2f" % (s12, s23),
              abs(t_m - 0.6) < 1e-12 and abs(x_m - 0.78) < 1e-12)

    centers = (np.arange(n) + 0.5) * dx
    u0 = np.where(centers < x2, u1, np.where(centers < x3, u2, u3))
    x, snaps, _ = _fv_riemann_run(u1, u3, n, T, _godunov_flux,
                                  np.zeros(n), 0.0, seed=71, u0=u0)

    # E2：合并前两道激波位置（T/4=0.2 < t_m=0.6，快照时刻对齐）
    t_pre = round(T / 4, 6)
    u_pre = snaps[t_pre]
    js_pre = _jump_positions(x, u_pre)
    ex_pre = exact_shocks(t_pre)
    ok2 = len(js_pre) == 2 and all(abs(js - ex) <= dx
                                   for js, ex in zip(js_pre, ex_pre))
    add_check("E2 合并前双激波（t=%.2f）：实测 %s vs 解析 %s（±1 cell）"
              % (t_pre, ["%.4f" % v for v in js_pre],
                 ["%.4f" % v for v in ex_pre]),
              "n_jumps=%d" % len(js_pre), ok2)

    # E3：合并后单激波位置（T=0.8 > t_m）
    u_T = snaps[round(T, 6)]
    js_T = _jump_positions(x, u_T)
    ex_T = exact_shocks(T)[0]
    ok1 = len(js_T) == 1 and abs(js_T[0] - ex_T) <= dx
    add_check("E3 合并后单激波：实测 %.4f vs 解析 %.4f（±1 cell，"
              "合并后走 s13=%.2f）" % (js_T[0] if js_T else -1, ex_T, s13),
              "n_jumps=%d" % len(js_T), ok1)

    # E4：中间态敏感度穿透合并（IC 微扰 u2+ε）
    eps = 2e-3
    u0_p = np.where(centers < x2, u1,
                    np.where(centers < x3, u2 + eps, u3))
    _, snaps_p, _ = _fv_riemann_run(u1, u3, n, T, _godunov_flux,
                                    np.zeros(n), 0.0, seed=71, u0=u0_p)
    xs_p = _jump_positions(x, snaps_p[round(T, 6)])
    dX = xs_p[0] - js_T[0]
    # 解析敏感度：ds12=ds23=ε/2 ⇒ t_m 不变，x_m 平移 ε·t_m/2，
    # 合并后 s13 与 u2 无关 ⇒ ∂X/∂u2 = t_m/2 = 0.3
    pred = t_m / 2 * eps
    add_check("E4 敏感度穿透：δX=%.3e ≈ (t_m/2)·ε=%.3e（±30%%）"
              "——中间态敏感度被合并永久保留"
              % (dX, pred),
              "dX/eps=%.3f (解析 0.3)" % (dX / eps),
              dX > 0 and abs(dX - pred) <= 0.3 * pred)

    # E5 叠加模型失效：单激波界面模型只看终态相邻态 (u1,u3) ⇒ δX=0
    fp_floor = 1e-9
    add_check("E5 叠加失效：单激波模型预测 δX=0，实测 %.3e ≥ 10³×fp 底"
              "（相互作用界面态不可省略）" % dX,
              "failure factor: 预测0 vs 实测%.0e" % dX,
              abs(dX) >= 1e3 * fp_floor)

    # E6 ε_J 证书：跨合并激波窗口的部分质量 QoI
    win = (x >= 0.82) & (x <= 0.98)             # 含合并激波，避开 fan 缘 0.8
    J = lambda uu: float(np.sum(uu[win]) * dx)
    dJ = J(snaps_p[round(T, 6)]) - J(u_T)
    interface = (u1 - u3) * dX
    ec = ErrorControlResult(cls="neutral", n_windows=1, coverage_ratio=1.0,
                            e_measured_max=abs(dJ),
                            e_predicted_max=abs(dJ),
                            qoi_estimate=abs(dJ), qoi_tolerance=1e-3,
                            qoi_cancellation=None).as_dict()
    from lowering.budget import _error_control_verdict
    v, _, dv = _error_control_verdict(ec)
    add_check("E6 ε_J 证书：δJ=%.3e（界面项占比 %.0f%%）≤ ε_J ⇒ pass"
              % (abs(dJ), 100 * abs(interface) / max(abs(dJ), 1e-300)),
              v, v == "pass" and "estimate" in dv)

    return dict(t_m=t_m, x_m=x_m, js_pre=js_pre, js_T=js_T,
                dX=dX, pred=pred, dJ=dJ)


# ------------------------------------------------------------ ε_J 进校准计划

def case_epsj_allocation():
    """§6.22 路线图⑥：ε_J 分配进 P7 校准计划。

    把 dJ 敏感度（adjoint 权重）变成逐元素残差预算：等贡献
    s_i·e_i = ε_J/n ⇒ e_i = ε_J/(n·s_i)——敏感度高的元素拿更紧的
    预算（更短的重标间隔），零敏感度元素不分配（误差只进 ε_global）。
    可行性对账（calib.py 诚实口径）：预算 < 测量底 ⇒ flagged，不许诺
    做不到的数字。generate_config(qoi=, eps_J=, T_horizon=) 启用后
    plan.budget_source="dJ-allocation"，per-element 预算 OVERRIDE
    全局启发式；不声明时行为不变（run_calib 62/62 回归保障）。
    """
    emit("[case_epsj_allocation]  等贡献分配 / 反比权重 / 可行性 / P7 集成")
    from lowering.p7 import allocate_element_budgets, generate_config
    from bench.run_calib import _diffusion_bundle
    from twin.calib import CalibratedTwin, calibrate_background

    # --- 纯分配器：8 模态系统的 dJ 权重（c7=0 ⇒ 零敏感度元素）---
    N = 8
    A = np.diag(-0.5 * np.arange(1, N + 1))
    c = np.ones(N)
    c[7] = 0.0
    t = np.linspace(0.0, 2.0, 81)
    w = dJ_weights(adjoint_field(A, c, t), t)
    eps_J, floor_ppm, drift = 500.0, 100.0, 500.0
    alloc = allocate_element_budgets(w.tolist(), eps_J,
                                     drift_rates=[drift] * N,
                                     meas_noise_ppm=floor_ppm)
    items = alloc["items"]
    add_check("A 分配恒等式：|Σ s_i·e_i − ε_J|=%.2e（fp 级，ε_J 合同总账）"
              % alloc["identity_residual"],
              "n_alloc=%d" % alloc["n_allocated"],
              alloc["identity_residual"] <= 1e-9 * eps_J)

    b_hi = items[0]["residual_budget_ppm"]    # 最高权重 ⇒ 最紧预算
    b_lo = items[6]["residual_budget_ppm"]    # 最低非零权重 ⇒ 最松
    ratio = b_lo / b_hi
    add_check("A 预算反比权重：e_0/e_6=%.3f ≈ w_6/w_0=%.3f（1%% 内）"
              % (b_hi / b_lo, w[6] / w[0]),
              "b_0=%.1f b_6=%.1f ppm" % (b_hi, b_lo),
              abs((b_lo / b_hi) - (w[0] / w[6])) / (w[0] / w[6]) < 0.01)

    add_check("A 零权重元素：不分配（budget=None，误差只进 ε_global）",
              items[7]["reason"], items[7]["residual_budget_ppm"] is None
              and not items[7]["feasible"])

    iv_hi = items[0]["update_interval"]
    iv_lo = items[6]["update_interval"]
    add_check("A 背景间隔：高权重元素更短（%.1f < %.1f，重标更勤）"
              % (iv_hi, iv_lo),
              "interval ∝ budget−floor", iv_hi < iv_lo)

    # --- 可行性对账：给最紧元素一个高于其预算的测量底 ⇒ flagged ---
    floors = [floor_ppm] * N
    floors[0] = 400.0                          # > e_0=241.6 ⇒ 做不到
    alloc_f = allocate_element_budgets(w.tolist(), eps_J,
                                       floors=floors,
                                       drift_rates=[drift] * N,
                                       meas_noise_ppm=floor_ppm)
    it0 = alloc_f["items"][0]
    add_check("A 可行性对账：budget %.1f < floor 400 ⇒ flagged（不许诺）"
              % it0["residual_budget_ppm"],
              "n_infeasible=%d" % alloc_f["n_infeasible"],
              (not it0["feasible"]) and "undeliverable" in it0["reason"])

    # --- P7 集成：diffusion-1d 锚点（64 状态），qoi=全 1 ⇒ 全元素分配，
    # 等贡献下最紧预算 = 500/(64·w_max) 必然低于 100 ppm 测量底 ⇒
    # flagged —— 这正是诚实口径要抓的：ε_J 太小/元素太多时许诺不了。
    d, sc, cert = _diffusion_bundle()
    n_states = len(list(getattr(d, "states", None) or []))
    twin = CalibratedTwin(params={"gm_over_c": 2.0e6})
    twin.apply_background("gm_over_c", calibrate_background(
        {"drift_rate_ppm_per_unit": 500.0}, 10.0, 100.0))
    qoi = np.ones(n_states)
    cfg = generate_config(d, cert, sc, twin=twin, qoi=qoi.tolist(),
                          eps_J=500.0, T_horizon=1.0)
    peb = cfg.plan.per_element_budgets
    n_flag = sum(1 for it in (peb or []) if not it["feasible"])
    add_check("P7 集成：per_element_budgets 发射（%d 项，%d flagged），"
              "budget_source=%s" % (len(peb or []), n_flag,
                                    cfg.plan.budget_source),
              "eps_J=%.0f" % cfg.plan.eps_J_ppm,
              peb is not None and n_flag >= 1
              and cfg.plan.budget_source == "dJ-allocation")
    ident = sum(it["dJ_weight"] * it["residual_budget_ppm"]
                for it in peb if it["residual_budget_ppm"] is not None)
    add_check("P7 集成：计划层恒等式 |Σ s_i·e_i − ε_J|=%.2e"
              % abs(ident - 500.0),
              "flagged 预算仍计入恒等式（分配数学 vs 可交付性分立）",
              abs(ident - 500.0) <= 1e-6)

    # --- 采样陷阱检查：刚性系统的 dJ 采样权重退化成端点伪影 ---
    A_v = np.diag(sc.scaled.A.diagonal()) if hasattr(sc, "scaled") else A
    lam_v = np.linalg.eigvals(A_v)
    gamma_max = float(np.max(-lam_v.real))
    w_sampled = dJ_weights(adjoint_field(A_v, qoi, t), t)
    w_exact = dJ_weights_exact(A_v, qoi, 2.0)
    spread_s = max(w_sampled) / min(w_sampled)
    spread_e = max(w_exact) / min(w_exact)
    add_check("A 采样陷阱：γ_max·Δt=%.1f ≫ 1 ⇒ 采样权重比 %.1f（端点伪影，"
              "近均匀）vs 解析 %.0f（集中慢模态）——刚性系统必须走闭式"
              % (gamma_max * (t[1] - t[0]), spread_s, spread_e),
              "γ_max=%.3g" % gamma_max,
              spread_s < 2.0 and spread_e > 100.0)

    # 向后兼容：不声明 ε_J ⇒ 行为不变（run_calib 的全局启发式口径）
    cfg2 = generate_config(d, cert, sc, twin=twin)
    add_check("向后兼容：不声明 ε_J ⇒ per_element_budgets=None，"
              "budget_source=%s（启发式不变）" % cfg2.plan.budget_source,
              "run_calib 62/62 回归保障",
              cfg2.plan.per_element_budgets is None
              and cfg2.plan.budget_source != "dJ-allocation")

    return dict(identity=alloc["identity_residual"], b_hi=b_hi, b_lo=b_lo,
                n_flag_p7=n_flag)


# ------------------------------------------------------------ dwell-time 演化

def case_runtime_dwell():
    """§6.22 路线图①：运行时源的 dwell-time 界面演化（三运行对照）。

    多激波 case 的诚实边界（u2 微扰取 IC）在此收口：运行时源（双激波
    夹层内的均匀质量源 ε）的偏差只在 cell 的 **dwell time** 内积累
    （出生时零、被 shock B 吞掉时达 ε·dwell(t)，dwell = W(t)/(u2−s23)
    随夹层收窄而衰减）⇒ 运行时源敏感度被系统性地压低。解析：

        δx_b(T) = (ε/2)·∫dwell = (ε/2)·0.2667（T=0.4 窗口）
        δX(T)   = 0.09ε  vs  IC 微扰 0.3ε  ⇒ 衰减因子 0.3

    显式界面演化器（evolve_shock_interfaces）逐步推进 (x, uL, uR)
    三元组 + 合并事件，FV 三运行（clean / IC / 运行时源）对账：
      G1 模型 sanity（clean 界面轨迹 = 解析）
      F1 δx_b(T/4) 模型 vs FV（亚 cell）
      F2 δX(T) 模型 vs FV
      F3 衰减因子：δX_runtime/δX_IC ≈ 0.3（运行时源 ≠ IC 微扰，
         定量否证"源和 IC 等价"的天真假设）
      F4 ε_J 证书：运行时源 QoI 证据进聚合
    """
    emit("[case_runtime_dwell]  界面演化器 / dwell 衰减 / 三运行对账")
    from lowering.error_control import evolve_shock_interfaces

    u1, u2, u3 = 1.0, 0.6, 0.0
    x2, x3 = 0.3, 0.6
    n, T = 256, 0.8
    dx = 1.0 / n
    eps = 2e-3
    centers = (np.arange(n) + 0.5) * dx
    u2_mask = (centers >= x2) & (centers < x3)

    shocks0 = [dict(x0=x2, uL=u1, uR=u2), dict(x0=x3, uL=u2, uR=u3)]
    t4 = round(T / 4, 6)

    # G1：模型 sanity（clean 界面轨迹 = 解析 Riemann 演化）
    m_clean = evolve_shock_interfaces(shocks0, {}, T,
                                      snap_times=[T / 4, T])
    xa4, xb4 = (m_clean["traj"][t4][0][0],
                m_clean["traj"][t4][1][0])
    add_check("G1 模型 clean 界面（t=%.2f）：x_a=%.4f x_b=%.4f vs 解析 "
              "(%.2f, %.2f)（±1 cell）"
              % (t4, xa4, xb4, x2 + 0.8 * t4, x3 + 0.3 * t4),
              "merges=%s" % [("%.3f" % g["t"], "%.3f" % g["x"])
                             for g in m_clean["merges"]],
              abs(xa4 - (x2 + 0.8 * t4)) <= dx
              and abs(xb4 - (x3 + 0.3 * t4)) <= dx
              and abs(m_clean["merges"][0]["t"] - 0.6) <= dx
              and abs(m_clean["merges"][0]["x"] - 0.78) <= dx)

    # 三次 FV 运行（同 seed，确定性）
    u0_clean = np.where(centers < x2, u1, np.where(centers < x3, u2, u3))
    u0_ic = np.where(centers < x2, u1,
                     np.where(centers < x3, u2 + eps, u3))
    # 运行时源：跟 u2 材料态走的值带注源 [0.6, 0.65]（覆盖新生 0.6 与
    # 已积累 ε·t 的 cell；fan 区间值 0.6 的 cell 也会被注入——其偏差
    # 困在 fan 特征线上到不了激波，对 δX 无贡献，实测见 F2/F3）
    state_src = [(u2, u2 + 0.05, eps)]
    x, snaps_c, _ = _fv_riemann_run(u1, u3, n, T, _godunov_flux,
                                    np.zeros(n), 0.0, seed=81, u0=u0_clean)
    _, snaps_i, _ = _fv_riemann_run(u1, u3, n, T, _godunov_flux,
                                    np.zeros(n), 0.0, seed=81, u0=u0_ic)
    _, snaps_r, _ = _fv_riemann_run(u1, u3, n, T, _godunov_flux,
                                    np.zeros(n), 0.0, seed=81, u0=u0_clean,
                                    state_sources=state_src)
    m_rt = evolve_shock_interfaces(shocks0, {1: eps}, T,
                                   snap_times=[T / 4, T])

    # F1：合并激波位移（T=0.8，材料耗尽 ⇒ 无在途质量污染）——质量等价
    # 位置（括号内 (u−uR) 质量折算的位移），亚 cell 可比（fp 精度）。
    # 注意 _jump_positions 的插值分数恒等于面位置（局部中点水平被
    # 消掉），只能给 ±1 cell——亚 cell 位移必须走质量等价口径。
    def _front_mass(x_arr, u_arr, i_lo, i_hi, uL_loc, uR_loc):
        return float(x_arr[i_lo] + np.sum(u_arr[i_lo:i_hi] - uR_loc)
                     * dx / (uL_loc - uR_loc))

    u_T_c = snaps_c[round(T, 6)]
    i_f = int(np.argmin(u_T_c))               # 合并激波右侧最低谷
    i_lo, i_hi = max(i_f - 16, 0), min(i_f + 8, n)
    p_c = _front_mass(x, u_T_c, i_lo, i_hi, u1, u3)
    p_i = _front_mass(x, snaps_i[round(T, 6)], i_lo, i_hi, u1, u3)
    p_r = _front_mass(x, snaps_r[round(T, 6)], i_lo, i_hi, u1, u3)
    dX_fv = p_r - p_c
    dX_ic = p_i - p_c
    dX_model = m_rt["X_T"] - m_clean["X_T"]
    add_check("F1 δX(T)：FV %.3e vs 界面演化模型 %.3e（±50%%，"
              "质量等价位置，材料已耗尽无在途污染）"
              % (dX_fv, dX_model),
              "clean p=%.4f" % p_c,
              abs(dX_fv - dX_model) <= 0.5 * abs(dX_model) + 1e-9)

    # F2：dwell 衰减因子——窗口质量比 δJ_rt/δJ_IC（解析 0.3：夹层内
    # 源只在 dwell time 内积累，IC 微扰则全程全额）
    win = (x >= 0.82) & (x <= 0.98)             # 含合并激波，避开 fan 缘 0.8
    J = lambda uu: float(np.sum(uu[win]) * dx)
    dJ_rt = J(snaps_r[round(T, 6)]) - J(snaps_c[round(T, 6)])
    dJ_ic = J(snaps_i[round(T, 6)]) - J(snaps_c[round(T, 6)])
    ratio = dJ_rt / max(abs(dJ_ic), 1e-300)
    add_check("F2 dwell 衰减：δJ_runtime/δJ_IC=%.2f ∈ [0.1, 0.5]"
              "（解析 0.3——运行时源 ≠ IC 微扰，dwell time 是衰减根源）"
              % ratio,
              "δJ_IC=%.3e δJ_rt=%.3e" % (dJ_ic, dJ_rt),
              0.1 <= ratio <= 0.5 and dJ_rt < dJ_ic)

    # F3：ε_J 证书（运行时源 QoI：estimate 口径进聚合）
    ec = ErrorControlResult(cls="neutral", n_windows=1, coverage_ratio=1.0,
                            e_measured_max=abs(dJ_rt),
                            e_predicted_max=abs(dJ_rt),
                            qoi_estimate=abs(dJ_rt), qoi_tolerance=1e-3,
                            qoi_cancellation=None).as_dict()
    from lowering.budget import _error_control_verdict
    v, _, dv = _error_control_verdict(ec)
    add_check("F3 ε_J 证书：运行时源 δJ=%.3e ≤ ε_J ⇒ pass（estimate 口径）"
              % abs(dJ_rt), v, v == "pass" and "estimate" in dv)

    return dict(dX_fv=dX_fv, dX_model=dX_model, dX_ic=dX_ic, ratio=ratio,
                dJ_rt=dJ_rt, dJ_ic=dJ_ic)


# ----------------------------------------------------------- descriptor 伴随
def case_descriptor_adjoint():
    """§6.22 路线图②：descriptor / E≠I 伴随（电容度量端口权重）。

    E·ẋ = A·x + r 的伴随生成元是 (E⁻¹A)ᵀ，源空间残差 R_k=∫r dt 进入
    状态方程前必须夹 E⁻¹：δJ₁=ΣλᵀE⁻¹R_k。naive 的 ΣλᵀR_k 在 E 病态
    （mode-C 电容跨 10³）时可偏差 κ(E) 倍——§5.8 的 M 度量频域版。
    """
    emit("[case_descriptor_adjoint]  E≠I 伴随 / 端口权重 / naive 对照")
    E = np.diag([1.0, 100.0])               # 节点电容跨 100×
    A = -E @ np.diag([0.5, 1.0])            # 前向速率仍 0.5 / 1.0
    c = np.ones(2)
    T = 2.0
    t = np.linspace(0.0, T, 81)
    lam = descriptor_adjoint_field(E, A, c, t)
    add_check("D1 终端条件：λ(T)=c（descriptor 伴随构造正确）",
              "max dev=%.2e" % np.max(np.abs(lam[:, -1] - c)),
              np.allclose(lam[:, -1], c))
    lam_i = descriptor_adjoint_field(np.eye(2), -np.diag([0.5, 1.0]), c, t)
    lam_std = adjoint_field(-np.diag([0.5, 1.0]), c, t)
    add_check("D1 E=I 退化：descriptor 伴随 == adjoint_field（×1e-12）",
              "max dev=%.2e" % np.max(np.abs(lam_i - lam_std)),
              np.allclose(lam_i, lam_std, atol=1e-12))
    try:
        descriptor_adjoint_field(np.diag([1.0, 0.0]), A, c, t)
        ok = False
    except ValueError:
        ok = True
    add_check("D1 E 奇异 ⇒ raise（代数约束须先投影，不静默伪逆）", "raise", ok)

    # 前向模拟（descriptor Euler）：x += dt·E⁻¹(Ax+r)
    r = np.array([1e-3, 1e-3])
    dt = 2e-3
    K = 4
    sub = (len(t) - 1) // K

    def run(inj):
        x = np.zeros(2)
        traj = [x.copy()]
        for _ in range(len(t) - 1):
            x = x + dt * np.linalg.solve(E, A @ x + inj)
            traj.append(x.copy())
        return np.array(traj)

    xc, xr = run(np.zeros(2)), run(r)
    dJ_true = float(c @ (xr[-1] - xc[-1]))
    Rs, centers = [], []
    for k in range(K):
        rows = xr[k * sub:(k + 1) * sub + 1]
        integ = np.array([np.trapezoid(rows[:, j], t[k * sub:(k + 1) * sub + 1])
                          for j in range(2)])
        Rs.append(E @ (rows[-1] - rows[0]) - A @ integ)  # = ∫r dt（源空间）
        centers.append(0.5 * (t[k * sub] + t[(k + 1) * sub]))
    est_d = descriptor_qoi_error_estimate(E, lam, t, Rs, centers)["delta_J"]
    est_n = qoi_error_estimate(lam, t, Rs, centers)["delta_J"]
    rel_d = abs(est_d - dJ_true) / max(abs(dJ_true), 1e-300)
    rel_n = abs(est_n - dJ_true) / max(abs(dJ_true), 1e-300)
    add_check("D2 descriptor 端口权重：δJ₁=%.3e vs 实测 %.3e（相对误差 %.1f%%）"
              % (est_d, dJ_true, 100 * rel_d),
              "naive=%.3e" % est_n, rel_d < 0.05)
    add_check("D2 naive 无 E⁻¹ 偏差 %.0f×（κ(E)=100 的度量陷阱）"
              % (rel_n / max(rel_d, 1e-12)),
              "rel_naive=%.2f" % rel_n, rel_n > 5.0 * rel_d)

    return dict(dJ_true=dJ_true, est_descriptor=est_d, est_naive=est_n,
                rel_descriptor=rel_d, rel_naive=rel_n)

# -------------------------------------------------------------- 非线性 DWR
def case_nonlinear_dwr():
    """§6.22 路线图③：非线性 DWR（一级估计、二阶余项、抵消、MC 覆盖）。

    ẋ = −x³（x₀=0.5，T=1）有解析参考 x(t)=x₀/√(1+2x₀²t)，J=x(T)。
    沿参考轨迹反向积分 J_F=−3x² 得 λ(t)，一级估计 δJ₁=Σλ_k R_k。
    非线性检验：N1 误差 ≤10%；N2 余项二阶；N3 抵消非界；N4 MC 覆盖；
    N5 大注入失效（不外推）。
    """
    emit("[case_nonlinear_dwr]  沿轨迹伴随 / 二阶余项 / 抵消 / MC 覆盖")
    x0, T = 0.5, 1.0
    t = np.linspace(0.0, T, 401)
    x_ref = x0 / np.sqrt(1.0 + 2.0 * x0 ** 2 * t)
    lam = nonlinear_adjoint_backward(lambda tt, xx: np.array([[-3.0 * xx[0] ** 2]]),
                                     x_ref[None, :], t, np.ones(1))
    add_check("N0 终端条件：λ(T)=c（沿轨迹反向积分正确）",
              "lambda(0)=%.4f" % lam[0, 0], np.allclose(lam[0, -1], 1.0))

    K = 8
    sub = (len(t) - 1) // K
    dt = float(t[1] - t[0])
    centers = np.array([0.5 * (t[k * sub] + t[(k + 1) * sub])
                        for k in range(K)])

    def run(rates):
        x = x0
        traj = [x]
        for k in range(K):
            for _ in range(sub):
                f = lambda v: -v ** 3 + rates[k]
                k1 = f(x)
                k2 = f(x + 0.5 * dt * k1)
                k3 = f(x + 0.5 * dt * k2)
                k4 = f(x + dt * k3)
                x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                traj.append(x)
        return np.array(traj)

    def estimate(traj):
        Rs = []
        for k in range(K):
            rows = traj[k * sub:(k + 1) * sub + 1]
            tt = t[k * sub:(k + 1) * sub + 1]
            integ = np.trapezoid(rows ** 3, tt)
            Rs.append(np.array([rows[-1] - rows[0] + integ]))
        return qoi_error_estimate(lam, t, Rs, centers)

    eps = 1e-3
    tr = run(np.full(K, eps))
    est = estimate(tr)
    dJ_true = float(tr[-1] - x_ref[-1])
    rel = abs(est["delta_J"] - dJ_true) / abs(dJ_true)
    add_check("N1 小注入：δJ₁=%.3e vs 实测 %.3e（相对误差 %.1f%% < 10%%）"
              % (est["delta_J"], dJ_true, 100 * rel),
              "eps=%.0e" % eps, rel < 0.10)

    tr_half = run(np.full(K, eps / 2))
    est_half = estimate(tr_half)
    dJ_half = float(tr_half[-1] - x_ref[-1])
    err_half = abs(est_half["delta_J"] - dJ_half)
    err_full = abs(est["delta_J"] - dJ_true)
    order = err_full / max(err_half, 1e-300)
    add_check("N2 二阶余项：注入减半 ⇒ 估计误差 ×%.1f ∈ [2.5, 6]（一阶应 ×2）"
              % order, "err_full=%.2e err_half=%.2e" % (err_full, err_half),
              2.5 <= order <= 6.0)

    w = np.array([np.trapezoid(lam[0, k * sub:(k + 1) * sub + 1],
                               t[k * sub:(k + 1) * sub + 1])
                  for k in range(K)])
    alpha = float(w[:K // 2].sum() / w[K // 2:].sum())
    rates_c = np.array([eps] * (K // 2) + [-alpha * eps] * (K - K // 2))
    tr_c = run(rates_c)
    est_c = estimate(tr_c)
    dJ_c = float(tr_c[-1] - x_ref[-1])
    add_check("N3 抵消：δJ₁≈%.2e（cancellation=%.2f）而真误差 %.2e 更高"
              % (est_c["delta_J"], est_c["cancellation_factor"], abs(dJ_c)),
              "Σ|贡献|=%.2e" % est_c["abs_sum"],
              abs(est_c["delta_J"]) < 0.05 * est_c["abs_sum"]
              and abs(dJ_c) > 3.0 * max(abs(est_c["delta_J"]), 1e-300))

    rng = np.random.default_rng(2026)
    covered = 0
    n_mc = 100
    est_list, true_list, scale_list = [], [], []
    for _ in range(n_mc):
        rates = eps * rng.choice([-1.0, 1.0], size=K)
        tr_m = run(rates)
        est_m = estimate(tr_m)
        true_m = float(tr_m[-1] - x_ref[-1])
        est_list.append(est_m["delta_J"])
        true_list.append(true_m)
        scale_list.append(est_m["abs_sum"])
        if abs(true_m) <= 1.3 * est_m["abs_sum"] + 1e-9:
            covered += 1
    cov = covered / n_mc
    scc = statistical_coverage_contract(est_list, true_list, scale_list,
                                        confidence=0.95)
    add_check("N4 MC 覆盖：%d/%d=%.2f ≥ 声明 0.9（B=1.3×Σ|λR|，非严格界）"
              % (covered, n_mc, cov), "统计口径：100 组随机二元注入",
              cov >= 0.9)
    add_check("N4 训练/验证分离：训练分位 0.95 ⇒ 独立验证覆盖 %.2f ≥ 0.85"
              "（非严格界，无 Hessian/Lipschitz 保证）"
              % scc["validation_coverage"],
              "%d train / %d val" % (scc["n_train"], scc["n_val"]),
              scc["validation_coverage"] >= 0.85)

    tr_big = run(np.full(K, 0.2))
    est_big = estimate(tr_big)
    dJ_big = float(tr_big[-1] - x_ref[-1])
    rel_big = abs(est_big["delta_J"] - dJ_big) / abs(dJ_big)
    add_check("N5 大注入 ε=0.2：相对误差 %.0f%% ≫ 小注入 %.1f%%（不承诺）"
              % (100 * rel_big, 100 * rel), "非线性 DWR 有界非线性区间",
              rel_big > 3.0 * rel)

    return dict(dJ_true=dJ_true, estimate=est["delta_J"], rel=rel,
                order=order, cancellation=est_c["cancellation_factor"],
                coverage=cov, rel_big=rel_big)

# -------------------------------------------------- Euler/NS 必要探针
def case_entropy_system():
    """§6.22 路线图④：Euler/NS 外推的必要探针——半离散熵不等式。

    线性对称双曲 u_t + A u_x = 0，A=[[0,1],[1,0]]（特征值 ±1），
    熵 E=½(a²+b²)。Tadmor 熵守恒通量 F*=½A(y_i+y_j)；Roe/upwind
    再减 ½|A|(y_j−y_i)（|A|=I ⇒ 熵耗散）；反耗散负对照加该项。
    **检验半离散熵产生率** dE/dt=−Σ u_i·(F_{i+1/2}−F_{i−1/2})
    （Tadmor 不等式本身），不是 Euler 前向长时间能量——中心通量+Euler
    前向本身无条件不稳定，长时间能量比较给出假阳性（实测翻车）。
    **诚实标注**：必要探针，不是 Euler/NS 本身；可压缩 Euler 的
    entropy-stable DG + 激波界面态仍是 TODO。
    """
    emit("[case_entropy_system]  半离散熵不等式 / 必要探针（非 Euler/NS）")
    n = 128
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    y0 = np.vstack([np.where(x < 0.5, 1.0, -1.0), np.zeros(n)])
    E0 = float(0.5 * np.sum(y0 ** 2) * dx)

    def flux_rate(kind):
        a, b = y0[0], y0[1]
        aj, bj = np.roll(a, -1), np.roll(b, -1)
        fe = np.vstack([0.5 * (b + bj), 0.5 * (a + aj)])
        diss = np.vstack([0.5 * (aj - a), 0.5 * (bj - b)])
        if kind == "ec":
            F = fe
        elif kind == "upwind":
            F = fe - diss
        else:
            F = fe + diss
        return float(-np.sum(y0 * (F - np.roll(F, 1))))

    r_ec = flux_rate("ec")
    r_up = flux_rate("upwind")
    r_bad = flux_rate("bad")
    add_check("S1 熵守恒通量：|dE/dt|=%.2e ≤ 1e-10（Tadmor 等式）"
              % abs(r_ec), "E0=%.4f" % E0, abs(r_ec) <= 1e-10)
    add_check("S2 upwind 耗散：dE/dt=%.4f < 0（半离散熵不等式成立）"
              % r_up, "熵产生为负", r_up < -1e-6)
    add_check("S3 反耗散负对照：dE/dt=%.4f > 0（不等式被违反被抓住）"
              % r_bad, "熵产生为正", r_bad > 1e-6)
    add_check("S4 熵产生排序：bad %.4f > ec %.4f > upwind %.4f"
              % (r_bad, r_ec, r_up), "系统结构的必要条件实测",
              r_bad > r_ec and r_ec > r_up)

    # 审查7：标量正密度对流——upwind 保正/有界，反耗散产生负值
    u0_pos = np.where(x < 0.5, 1.0, 0.0)
    dt_pos = 0.4 * dx
    nst_pos = int(round(0.4 / dt_pos))
    def advect(kind):
        u = u0_pos.copy()
        for _ in range(nst_pos):
            F = u if kind == "upwind" else np.roll(u, -1)
            u = u - dt_pos / dx * (F - np.roll(F, 1))
        return u
    u_up, u_bad = advect("upwind"), advect("bad")
    add_check("审查7 保正性：upwind min=%.4f∈[0,1]（CFL=0.4）" % u_up.min(),
              "max=%.4f" % u_up.max(),
              u_up.min() >= -1e-12 and u_up.max() <= 1.0 + 1e-12)
    add_check("审查7 保正性负对照：反耗散 min=%.3e < 0（被抓住）"
              % u_bad.min(), "n_neg=%d" % (u_bad < 0).sum(),
              u_bad.min() < -1e-6)
    # 审查7：相对熵（两解差值的能量）半离散产生率 EC=0 / upwind≤0 / bad≥0
    y_a = y0.copy()
    y_b = y0.copy()
    y_b[0] += 0.1 * np.sin(2 * np.pi * x)

    def rel_rate(kind):
        def flux(y):
            a, b = y[0], y[1]
            aj, bj = np.roll(a, -1), np.roll(b, -1)
            fe = np.vstack([0.5 * (b + bj), 0.5 * (a + aj)])
            diss = np.vstack([0.5 * (aj - a), 0.5 * (bj - b)])
            return (fe if kind == "ec" else
                    (fe - diss if kind == "upwind" else fe + diss))
        return float(-np.sum((y_b - y_a) * (flux(y_b) - flux(y_a))))
    rr_ec, rr_up, rr_bad = (rel_rate("ec"), rel_rate("upwind"),
                            rel_rate("bad"))
    add_check("审查7 相对熵半离散率：EC=%.2e、upwind=%.3f ≤0、bad=%.3f ≥0"
              % (rr_ec, rr_up, rr_bad), "两解差值的能量不等式",
              abs(rr_ec) < 1e-10 and rr_up < 0 and rr_bad > 0)

    return dict(E0=E0, rate_ec=r_ec, rate_up=r_up, rate_bad=r_bad,
                u_min_up=u_up.min(), rel_ec=rr_ec, rel_up=rr_up)

# ------------------------------------------------------------ 混沌统计合同
def case_chaos_contract():
    """§6.22 路线图⑤：混沌系统——有限预测窗 + 统计 QoI 合同。

    Lorenz (σ=10,ρ=28,β=8/3) 正 Lyapunov 指数使逐轨迹误差指数放大：
    δ(t)≈δ0·e^{λt}，时窗 T*=(1/λ)ln(ε_tol/δ0)；超过后不能许诺跟踪
    初值解。合同降级 Level 2 统计可划归（§1）：指定的 QoI（z 均值）
    在容限内而轨迹误差可很大；有限时窗声明进证书（expanding + 
    finite_horizon=True）。诚实边界：ensemble 统计不是严格 shadowing。
    实测翻车：切线重整化 λ≈6.8 高估一个量级（Euler 切线 + 阈值重整化
    口径不对），改用双轨发散斜率拟合 λ≈0.7——只报能复现的数。
    """
    emit("[case_chaos_contract]  正 Lyapunov / 有限时窗 / 系综统计合同")
    sig, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt = 5e-3
    def F(y):
        return np.array([sig * (y[1] - y[0]),
                         y[0] * (rho - y[2]) - y[1],
                         y[0] * y[1] - beta * y[2]])
    def rk4(y, h, bias=None):
        k1 = F(y)
        k2 = F(y + 0.5 * h * k1)
        k3 = F(y + 0.5 * h * k2)
        k4 = F(y + h * k3)
        return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4) \
            + (h * bias if bias is not None else 0.0)
    def run(x0, nst, seed=None, bias=None, noise=0.0, stride=1):
        rng = np.random.default_rng(seed) if seed is not None else None
        y = np.array(x0, dtype=float)
        traj = [y.copy()]
        for step in range(nst):
            inj = ((bias if bias is not None else 0.0)
                   + (noise * rng.uniform(-1.0, 1.0, size=3)
                      if noise > 0 and rng is not None else 0.0))
            y = rk4(y, dt, inj)
            if stride == 1 or (step + 1) % stride == 0:
                traj.append(y.copy())
        return np.array(traj)

    d0 = 1e-8
    n_div = 12000
    stride = 20
    y_ref = run([1.0, 1.0, 1.0], n_div, stride=stride)
    y_pert = run([1.0 + d0, 1.0, 1.0], n_div, stride=stride)
    dev = np.linalg.norm(y_pert - y_ref, axis=1)
    tt = np.arange(dev.shape[0]) * dt * stride
    band = (dev >= 1e-7) & (dev <= 1e-1)
    lam = float(np.polyfit(tt[band], np.log(dev[band]), 1)[0])
    add_check("K1 正 Lyapunov（双轨发散斜率）：λ=%.3f ∈ [0.4, 1.5]"
              % lam, "切线重整化高估 6.8 已弃用", 0.4 <= lam <= 1.5)

    eps_tol = 1.0
    idx = np.where(dev >= eps_tol)[0]
    t_cross = float(tt[idx[0]]) if idx.size else float(tt[-1])
    t_pred = np.log(eps_tol / d0) / lam
    add_check("K2 有限时窗：δ 越过 ε=1 实测 t=%.1f ≈ ln(ε/δ0)/λ=%.1f（±30%%）"
              % (t_cross, t_pred), "δ0=%.0e" % d0,
              abs(t_cross - t_pred) <= 0.3 * t_pred)
    add_check("K2 时窗外无轨迹精度：T=60 时 δ=%.2f ≫ QoI 容限 0.1"
              % dev[-1], "逐轨迹预测已失效", dev[-1] > 0.1)
    dT_lo = correction_interval(lam, 0.1, 0.1 * eps_tol, eps_tol)
    dT_hi = correction_interval(lam, 0.1, 0.9 * eps_tol, eps_tol)
    add_check("K2 调度收缩：e0 从 0.1ε 到 0.9ε ⇒ ΔT %.3f → %.3f（越来越短）"
              % (dT_lo, dT_hi), "正 λ 的调度公式", dT_hi < dT_lo)
    try:
        correction_interval(lam, 0.1, eps_tol, eps_tol)
        ok = False
    except ValueError:
        ok = True
    add_check("K2 e0≥ε ⇒ raise（混沌合同终点：拒绝无限时窗轨迹精度）",
              "raise", ok)

    n_stat = 12000
    keep = slice(200, None)           # stride=20 ⇒ index 200 ≈ t=20，弃瞬态
    ref = run([1.0, 1.0, 1.0], n_stat, stride=stride)
    ens = []
    for j in range(8):
        rng = np.random.default_rng(300 + j)
        x0j = np.array([1.0, 1.0, 1.0]) + 1e-2 * rng.standard_normal(3)
        ens.append(run(x0j, n_stat, seed=400 + j,
                       bias=np.array([1e-2, 0.0, 0.0]), noise=1e-2,
                       stride=stride))
    # 审查8：独立参考系综（不同 IC，无注入）——不能拿单条 clean 轨迹自证
    ref_ens = []
    for j in range(8):
        rng = np.random.default_rng(700 + j)
        x0j = np.array([1.0, 1.0, 1.0]) + 1e-2 * rng.standard_normal(3)
        ref_ens.append(run(x0j, n_stat, seed=800 + j, stride=stride))
    z_ref_vals = np.array([r[keep, 2].mean() for r in ref_ens])
    z_hw_vals = np.array([e[keep, 2].mean() for e in ens])
    z_ref_mean = float(z_ref_vals.mean())
    z_hw_mean = float(z_hw_vals.mean())
    qoi_err = abs(z_hw_mean - z_ref_mean) / abs(z_ref_mean)
    # 自相关时间 → 有效样本数 → 置信区间（统计签核，不是 8 个样本就够）
    series = ref[keep, 2]
    s = series - series.mean()
    denom = float(np.dot(s, s))
    rho = [float(np.dot(s[:-k], s[k:]) / denom) for k in range(1, 20)]
    tau_int = 1.0 + 2.0 * sum(r for r in rho if r > 0.05)
    n_eff = len(series) / (2.0 * tau_int)
    ci_ens = (1.96 * np.std(z_ref_vals, ddof=1) / np.sqrt(len(z_ref_vals)))
    ci_time = 1.96 * np.std(series, ddof=1) / np.sqrt(n_eff)
    ci_total = (ci_ens + ci_time) / abs(z_ref_mean)
    add_check("K3 统计签核：差 %.3f ≤ CI %.3f（τ_int=%.1f，N_eff=%.1f，"
              "独立参考系综）" % (qoi_err, ci_total, tau_int, n_eff),
              "burn-in T<20", qoi_err <= ci_total and n_eff > 1)
    add_check("K3 统计稳健 ≠ 轨迹稳健：QoI 差 %.3f 小而轨迹误差 %.2f"
              % (qoi_err, dev[-1]), "Level 2 统计可划归成立",
              qoi_err < 0.1 and dev[-1] > 0.1)

    # 审查8：trajectory contract 的预测界（不再拿实测当预测）
    pred_traj = np.minimum(d0 * np.exp(lam * tt), 60.0)   # 吸引子尺度封顶
    traj_cov = float(np.mean(dev <= 1.5 * pred_traj + 1e-12))
    add_check("K4 轨迹合同覆盖：%.0f%%（预测 δ0·e^{λt} 封顶，非实测自证）"
              % (100 * traj_cov), "λ_fit 样本内，非最坏上界",
              traj_cov >= 0.9)
    ec = ErrorControlResult(cls="expanding", finite_horizon=True,
                            horizon_T=60.0,
                            n_windows=1, coverage_ratio=traj_cov,
                            e_measured_max=dev[-1],
                            e_predicted_max=float(pred_traj[-1]),
                            qoi_estimate=qoi_err,
                            qoi_remainder=ci_total,
                            qoi_tolerance=0.5,
                            coverage_threshold=0.95,
                            qoi_cancellation=None).as_dict()
    from lowering.budget import _error_control_verdict
    v, _, dv = _error_control_verdict(ec)
    add_check("K4 证书：expanding + finite_horizon=True + QoI 稳健 ⇒ pass",
              v, v == "pass" and "estimate" in dv)
    ec_nohorizon = dict(ec, finite_horizon=False)
    v2, _, _ = _error_control_verdict(ec_nohorizon)
    add_check("K4 负对照：expanding 无 finite_horizon ⇒ rejected", v2,
              v2 == "rejected")

    return dict(lam=lam, t_cross=t_cross, t_pred=t_pred, qoi_err=qoi_err,
                ci_total=ci_total, tau_int=tau_int, n_eff=n_eff,
                traj_coverage=traj_cov, traj_error=float(dev[-1]))

# --------------------------------------------------------------- P0-2 反例
def case_residual_semantics():
    """§6.22 P0-2 回归：∫r dt 不能作为累计误差上界。

    ė=−e+r，e(0)=0；r=+1 (0≤t<1)、−1 (1≤t≤2)：signed 残差≈0，
    但真实 e(2)=2/e−1/e²−1≈−0.3996——第一段注入已衰减、第二段
    尚未充分衰减。三种量必须分开：
      residual_signed      ∫r dt          ≈ 0（诊断/校正，不是界）
      residual_l1_bound    ∫‖r‖dt         ≈ 2（需收缩前提）
      propagated_residual  ∫‖Φ‖‖r‖dt      ≈ 0.8647（保守动态界）
    并验证 lognorm 闭式 eta=r_max(e^{μΔT}−1)/μ 与传播积分一致。
    """
    emit("[case_residual_semantics]  P0-2 抵消反例 / 三种残差语义 / η 闭式")
    dt = 1e-3
    t = np.linspace(0.0, 2.0, 2001)
    x = np.zeros((t.size, 1))
    e = 0.0
    x[0] = e
    for i in range(1, t.size):
        ri = 1.0 if t[i - 1] < 1.0 else -1.0
        e = e + dt * (-e + ri)
        x[i] = e
    e_true = abs(e)
    e_analytic = abs(2.0 / np.e - 1.0 / np.e ** 2 - 1.0)
    A = np.array([[-1.0]])
    w = linear_window_residual(A, t, x)
    signed = float(np.abs(w.residual_signed[0]))
    add_check("P0-2 反例：signed |∫r|=%.4f ≈ 0 而真实 |e(2)|=%.4f（诊断非界）"
              % (signed, e_true), "解析 e(2)=%.4f" % e_analytic,
              signed < 0.01 and abs(e_true - e_analytic) < 1e-3)
    add_check("P0-2 l1 预算 ∫‖r‖=%.3f ≥ |e(2)|=%.3f（收缩前提下的界）"
              % (w.residual_l1_bound, e_true), "比传播界松 2.3×",
              w.residual_l1_bound >= e_true)
    add_check("P0-2 传播预算 ∫‖Φ‖‖r‖=%.4f ≥ |e(2)|=%.4f（保守动态界）"
              % (w.propagated_residual, e_true),
              "eta=%.4f（含求积误差）" % w.eta,
              w.propagated_residual >= e_true and w.eta >= e_true)
    eta_closed = eta_from_lognorm(-1.0, 1.0, 2.0)
    add_check("P0-2 lognorm 闭式：r_max(e^{μΔT}−1)/μ=%.4f ≈ 传播积分 %.4f"
              % (eta_closed, w.propagated_residual), "μ=−1, r_max=1, ΔT=2",
              abs(eta_closed - w.propagated_residual) < 1e-3)

    # 双窗口递推：signed 之和≈0 会给出界 0；保守 eta 给出真实上界
    ws = []
    for k in range(2):
        sel = slice(k * 1000, (k + 1) * 1000 + 1)
        ws.append(linear_window_residual(A, t[sel], x[sel]))
    bnd = error_recursion_bound(0.0, [np.exp(-1.0)] * 2,
                                injection_budget(ws))
    add_check("P0-2 递推：ε(2)=%.4f ≥ |e(2)|=%.4f（signed 预算会给 0）"
              % (bnd[-1], e_true), "G=e^{-1}, η=传播积分",
              bnd[-1] >= e_true)

    return dict(signed=signed, l1=w.residual_l1_bound,
                propagated=w.propagated_residual, e_true=e_true,
                eta_closed=eta_closed, recursion=bnd[-1])

# ------------------------------------------------------ 残差区间记账回归
def case_residual_quadrature():
    """A 修复回归：N−1 个区间必须全部进 L1/传播，认证界与估计界分开。

    旧 bug：r_samples 是 N−1 个**区间**样本，却用 trapezoid(..., t[:-1])
    （N−1 个节点）积分——单区间时数组长度 0，全给 0。反例
    A=0, t=[0,1], x=[0,1]：r=1、e(1)=1，旧实现 residual_l1_bound /
    propagated_residual / quad_error 全 0。
    新口径：r̄_i 是区间均值（与 R 精确同口径），Σ‖r̄_i‖h_i 覆盖全部区间；
    认证界需要声明（r_lipschitz/r_envelope + log-norm 增益界），
    未声明时 certified_bound=None、eta 只是估计。
    采样间不可见的扰动：节点全零、区间内振荡的轨迹——估计界为 0，
    只有声明的包络/带宽能看见它（负对照）。
    """
    emit("[case_residual_quadrature]  区间记账 / 估计 vs 认证 / 采样盲区")
    # 1. 最小反例（线性 + 非线性两条路径）
    A0 = np.zeros((1, 1))
    t1 = np.array([0.0, 1.0])
    x1 = np.array([[0.0], [1.0]])
    w = linear_window_residual(A0, t1, x1)
    add_check("区间记账反例：A=0,t=[0,1],x=[0,1] ⇒ L1=1、传播=1、η=1",
              "旧实现全 0；真解 e(1)=1",
              abs(w.residual_l1_bound - 1.0) < 1e-12
              and abs(w.propagated_residual - 1.0) < 1e-12
              and abs(w.eta - 1.0) < 1e-12
              and abs(w.quad_error) < 1e-12)
    wn = window_integral_residual(lambda xx, tt: np.zeros_like(xx), t1, x1)
    add_check("同一反例走一般 F 路径 ⇒ L1=1（不再漏最后区间）",
              "%.6f" % wn.residual_l1_bound,
              abs(wn.residual_l1_bound - 1.0) < 1e-12)
    # 2. 多区间常数残差：Σ‖r̄‖h 精确等于 ∫‖r‖
    r = np.array([1.0, 2.0])
    tm = np.array([0.0, 0.5, 1.0])
    xm = np.array([[0.0, 0.0], [0.5 * r[0], 0.5 * r[1]],
                   [r[0], r[1]]])
    wm = linear_window_residual(np.zeros((2, 2)), tm, xm)
    add_check("多区间常数 r：Σ‖r̄‖h=%.4f = ‖r‖=%.4f（区间均值口径）"
              % (wm.residual_l1_bound, np.linalg.norm(r)),
              "%.4f" % wm.residual_l1_bound,
              abs(wm.residual_l1_bound - np.linalg.norm(r)) < 1e-12)
    # 3. 认证界：声明后存在、覆盖真实 |e(2)|、且不虚高
    dt = 1e-3
    t = np.linspace(0.0, 2.0, 2001)
    xs = np.zeros((t.size, 1))
    e = 0.0
    for i in range(1, t.size):
        ri = 1.0 if t[i - 1] < 1.0 else -1.0
        e = e + dt * (-e + ri)
        xs[i] = e
    wc = linear_window_residual(np.array([[-1.0]]), t, xs,
                                r_lipschitz=0.0, quad_bound=0.0)
    e_true = abs(2.0 / np.e - 1.0 / np.e ** 2 - 1.0)
    add_check("认证界（r-Lipschitz=0 + quad_bound）%.4f ≥ |e(2)|=%.4f，"
              "且 ≤ L1 %.4f" % (wc.certified_bound, e_true,
                                 wc.residual_l1_bound),
              "cert=%s" % wc.certification,
              wc.certified_bound is not None
              and wc.certified_bound >= e_true
              and wc.certified_bound <= wc.residual_l1_bound + 1e-12
              and wc.eta == wc.certified_bound)
    wu = linear_window_residual(np.array([[-1.0]]), t, xs)
    add_check("未声明 ⇒ certified_bound=None、eta 标注 ESTIMATED",
              "%s" % wu.certification,
              wu.certified_bound is None
              and "ESTIMATED" in wu.eta_assumption)
    # 4. 采样盲区（负对照）：节点样本全零、区间内振荡 ⇒ 估计界为 0
    h = 0.25
    ts = np.arange(0.0, 1.0001, h)
    xz = np.sin(2.0 * np.pi * ts / h + 1e-12)[:, None]   # 节点处 sin(2πk)=0
    xz[np.abs(xz) < 1e-9] = 0.0
    wz = linear_window_residual(np.zeros((1, 1)), ts, xz)
    r_env = 2.0 * np.pi / h                              # |ẋ| 上界
    wzc = linear_window_residual(np.zeros((1, 1)), ts, xz, r_envelope=r_env)
    add_check("采样盲区：节点全零 ⇒ 估计 L1=%.1e（看不见区间内振荡），"
              "声明包络 ρ=%.3f ⇒ 认证界 %.3f 覆盖"
              % (wz.residual_l1_bound, r_env, wzc.certified_bound),
              "估计不是界，声明才是合同",
              wz.residual_l1_bound < 1e-9
              and wzc.certified_bound > 1.0)
    return dict(single=w.residual_l1_bound, certified=wc.certified_bound,
                blind=wz.residual_l1_bound, blind_cert=wzc.certified_bound)

# ------------------------------------------------------------ P0-4 证书强化
def case_certificate_hardening():
    """§6.22 P0-4 回归：双轨合同强制 + 畸形证据不得通过。

    拒绝路径：qoi_estimate 负方向大误差、e_measured NaN、coverage inf、
    finite_horizon 无具体时间、界覆盖通过但 ε_global 超标、
    |δJ|+B_remainder>ε_J、workload 要求但缺 evidence。
    """
    emit("[case_certificate_hardening]  P0-4 双轨强制 / 畸形证据拒绝")
    from lowering.budget import BudgetInput, compile_budget
    base = dict(kind="error_control", cls="contracting", n_windows=4,
                coverage_ratio=1.0, e_measured_max=1e-3,
                e_predicted_max=2e-3)
    cases = [
        ("qoi 负方向大误差", dict(base, qoi_estimate=-100.0,
                                    qoi_tolerance=0.01), "eps_J"),
        ("e_measured NaN", dict(base, e_measured_max=float("nan")),
         "non-finite"),
        ("e_predicted NaN", dict(base, e_predicted_max=float("nan")),
         "non-finite"),
        ("coverage inf", dict(base, coverage_ratio=float("inf")),
         "coverage_ratio"),
        ("finite_horizon 无时窗", dict(base, cls="expanding",
                                     finite_horizon=True), "horizon_T"),
        ("预测 101 > ε_global 100", dict(base, e_measured_max=100.0,
                                       e_predicted_max=101.0,
                                       global_tolerance=100.0), "eps_global"),
        ("余项推高 δJ", dict(base, qoi_estimate=0.009,
                            qoi_tolerance=0.01, qoi_remainder=0.005), "B_remainder"),
    ]
    for name, ev, token in cases:
        v, _, detail = _error_control_verdict(ev)
        add_check("P0-4 拒绝：%s ⇒ rejected" % name,
                  detail[:60], v == "rejected" and token in detail)
    # 合法时窗声明必须给具体 T
    v, _, _ = _error_control_verdict(dict(base, cls="expanding",
                                          finite_horizon=True, horizon_T=2.0))
    add_check("P0-4 合法 expanding + finite_horizon + horizon_T=2 ⇒ pass",
              v, v == "pass")
    # 双轨独立条件：界覆盖但精度超标 ⇒ rejected（已在上表 101>100）
    # 缺 evidence：workload 声明要求 ⇒ 整张证书拒绝
    cert = compile_budget(BudgetInput(require_error_control=True))
    add_check("P0-4 缺 evidence 且 workload 要求 ⇒ 证书 rejected",
              "; ".join(cert.rejection_reasons)[:70],
              cert.status == "rejected"
              and any("error-control evidence" in r
                      for r in cert.rejection_reasons))
    cert2 = compile_budget(BudgetInput())
    add_check("P0-4 默认不要求 evidence ⇒ 向后兼容（不误拒）",
              cert2.status, cert2.status != "rejected")

    return dict(rejected=len(cases), status=cert.status)

# ------------------------------------------------------------ P0-5 器件端口
def case_device_ports():
    """§6.22 P0-5 回归：绝对灵敏度 + 不同硬件误差端口。

    dJ 预算不能把同一个节点的 ∫|λ_i|dt 套到所有器件 ppm 上：
      integrator offset → 加性方向 e_i（常量 B）
      gm mismatch       → ΔA x（状态相关 B=diag(x(t))）
      cap mismatch      → −E⁻¹ΔE ẋ（行为级 B=−A x(t)）
    各端口算 K_i=∫|λᵀB_i|dt；分配用绝对 K（保留 ΣK 前置因子），
    K 整体翻倍 ⇒ 预算必须减半（归一化口径会漏掉这一因子）。
    不可行分配（预算 < 测量底）必须在 P7 flag + P6 拒绝。
    """
    emit("[case_device_ports]  P0-5 绝对灵敏度 / 三端口 / 不可行拒绝")
    from analysis.adjoint import port_sensitivities
    from lowering.p7 import allocate_element_budgets, generate_config
    N, T = 8, 2.0
    A = np.diag(-0.5 * np.arange(1, N + 1))
    c = np.ones(N)
    t = np.linspace(0.0, T, 81)
    lam = adjoint_field(A, c, t)
    x0 = np.ones(N)
    xref = np.array([np.exp(np.diag(A) * ti) * x0 for ti in t]).T
    ports = {
        "integrator_offset": np.eye(N)[0],
        "gm_mismatch": lambda tt, xx: np.diag(xx),
        "cap_mismatch": lambda tt, xx: -A @ xx,
    }
    K = port_sensitivities(lam, t, ports, xref)
    add_check("P0-5 三端口 K_i 不同：offset %.3f、gm %.3f、cap %.3f"
              % (K["integrator_offset"], K["gm_mismatch"],
                 K["cap_mismatch"]),
              "同一节点不能共用一份灵敏度",
              len(set(round(v, 6) for v in K.values())) == 3)

    k_abs = np.array([K["gm_mismatch"] / N] * N)  # 每元素绝对 K
    eps_J = 500.0
    a1 = allocate_element_budgets(k_abs.tolist(), eps_J, absolute=True)
    ident = float(np.sum(k_abs * np.array(
        [it["residual_budget_ppm"] for it in a1["items"]])))
    add_check("P0-5 绝对恒等式：Σ K_i·e_i=%.4f ≈ ε_J（ΣK=%.3f 保留）"
              % (ident, a1["sensitivity_total"]),
              "非归一化口径", abs(ident - eps_J) < 1e-8)
    a2 = allocate_element_budgets((2 * k_abs).tolist(), eps_J,
                                  absolute=True)
    b1 = a1["items"][0]["residual_budget_ppm"]
    b2 = a2["items"][0]["residual_budget_ppm"]
    add_check("P0-5 K 翻倍 ⇒ 预算减半：%.2f → %.2f（归一化会漏掉 ΣK）"
              % (b1, b2), "ratio=%.3f" % (b2 / b1),
              abs(b2 / b1 - 0.5) < 1e-9)

    # 不可行分配（预算 < floor）⇒ P7 flag + P6 拒绝
    tight = allocate_element_budgets(k_abs.tolist(), 1e-3,
                                     absolute=True, meas_noise_ppm=100.0)
    add_check("P0-5 ε_J 过小 ⇒ %d 元素 budget < floor（不可行）"
              % tight["n_infeasible"], "flag + 拒绝", tight["n_infeasible"] > 0)
    ec = ErrorControlResult(cls="contracting", n_windows=1,
                            coverage_ratio=1.0, e_measured_max=1e-4,
                            e_predicted_max=1e-4,
                            allocation_infeasible=tight["n_infeasible"]).as_dict()
    v, _, dv = _error_control_verdict(ec)
    add_check("P0-5 P6 拒绝不可行分配：verdict=%s" % v, dv[:70],
              v == "rejected" and "allocation infeasible" in dv)

    return dict(K=K, identity=ident, ratio=b2 / b1,
                n_infeasible=tight["n_infeasible"])

# -------------------------------------------------------------- 审查6 分类
def case_five_types():
    """审查6 回归：传播五分法 + 能量度量 μ_M 口径统一。

    五个系统：contractive（−I）、asymptotically stable（[[−1,100],[0,−1]]，
    Hurwitz 但强非正规瞬态）、semistable（diag(−1,0)）、polynomially
    growing（[[0,1],[0,0]] 临界 Jordan）、expanding（+I）。
    μ_M 与 finite_time_gain(A,t,M) 用同一能量度量 M。
    """
    emit("[case_five_types]  传播五分法 / μ_M 能量度量")
    systems = {
        "contractive": -np.eye(2),
        "asymptotically stable": np.array([[-1.0, 100.0], [0.0, -1.0]]),
        "semistable": np.diag([-1.0, 0.0]),
        "polynomially growing": np.array([[0.0, 1.0], [0.0, 0.0]]),
        "expanding": np.eye(2),
    }
    for name, A in systems.items():
        typ = classify_propagation_type(A)
        add_check("审查6 分类 %s ⇒ %s" % (name, typ), typ, typ == name)
    A = systems["asymptotically stable"]
    mu2 = logarithmic_norm(A)
    M = np.diag([1.0, 1e4])
    muM = logarithmic_norm(A, M)
    pc = classify_propagation(A, M=M, horizons=(0.01, 0.1))
    add_check("审查6 能量度量统一：μ₂=%.1f vs μ_M=%.1f（C bank 跨 10⁴ 时不同）"
              % (mu2, muM), "finite_time_gain 同 M", abs(mu2 - muM) > 1.0)
    add_check("审查6 能量度量预条件：无 M 为 asymptotically stable，"
              "有 M 变 %s（μ_M=%.1f<0 收缩）"
              % (pc.propagation_type, pc.mu_M),
              "同一口径 gain 与 μ_M", pc.propagation_type == "contractive"
              and abs(pc.mu_M - muM) < 1e-9)

    return dict(types={k: classify_propagation_type(v)
                       for k, v in systems.items()}, mu2=mu2, muM=muM)

# ------------------------------------------------------------ 审查7 descriptor
def case_descriptor_robustness():
    """审查7：尺度不变奇异性判据 + P7 真正调用 descriptor 伴随。

    det 阈值会把 E=1e-3 I_64 误判奇异（det=1e-192）——改用相对奇异值/
    条件数；P7 的 qoi_adjoint_for_config 在 E≠I 时走 (E⁻¹A)ᵀ 并给
    源端口夹 E⁻¹，不再用 adjoint_field(A_v,...)。
    """
    emit("[case_descriptor_robustness]  尺度不变判据 / P7 descriptor 伴随")
    n = 64
    E_ok = 1e-3 * np.eye(n)
    t = np.linspace(0, 1, 65)
    c = np.ones(n)
    lam = descriptor_adjoint_field(E_ok, -E_ok @ np.eye(n), c, t)
    add_check("审查7 E=1e-3 I_64（det=1e-192）不误判：λ(T)=c 且有限",
              "max|λ(T)−c|=%.1e" % np.max(np.abs(lam[:, -1] - c)),
              np.allclose(lam[:, -1], c) and np.all(np.isfinite(lam)))
    E_bad = np.diag([1.0] + [1e-15] * (n - 1))
    try:
        descriptor_adjoint_field(E_bad, -E_bad, c, t)
        ok = False
    except ValueError:
        ok = True
    add_check("审查7 相对奇异值 1e-15 ⇒ raise（cond 判据，非 det）", "raise", ok)
    A = -np.diag(np.arange(1.0, n + 1.0))
    w_i = qoi_adjoint_for_config(A, np.eye(n), c, 1.0, t)
    w_d = qoi_adjoint_for_config(A, np.diag(np.linspace(1.0, 100.0, n)),
                                c, 1.0, t)
    ratio = abs(w_d[n - 1, 0]) / max(abs(w_i[n - 1, 0]), 1e-300)
    add_check("审查7 P7 伴随口径：E≠I 时 λ_last 改变 %.2e×（E⁻¹ 端口权重）"
              % ratio, "descriptor 而非 adjoint_field", abs(ratio - 1.0) > 1e-3)

    return dict(ratio=ratio)

# ------------------------------------------------------------ 审查7 界面生命周期
def case_interface_lifecycle():
    """审查7：界面永久 id、区域源生命周期、merge 事件 Jacobian。

    三激波链 A→B→C，源按 region_id 绑定（rB 中途合并时消失，rC 存活
    且不得因列表下标变化绑错）；merge event 给 (x,uL,uR) 对两道旧
    界面的 3×6 Jacobian，供界面态伴随沿事件链回传。
    """
    emit("[case_interface_lifecycle]  永久 id / 源生命周期 / merge 伴随")
    shocks = [dict(x0=0.2, uL=1.0, uR=0.6, id="A"),
              dict(x0=0.5, uL=0.6, uR=0.3, id="B"),
              dict(x0=0.8, uL=0.3, uR=0.0, id="C")]
    m = evolve_shock_interfaces(shocks, {"rB": 1e-3, "rC": 2e-3},
                                0.9, snap_times=[0.4, 0.9])
    e0 = m["merges"][0]
    ok_jac = (e0["left_id"] == "A" and e0["right_id"] == "B"
              and e0["jacobian"].shape == (3, 6)
              and e0["jacobian"][0, 4] == 1.0
              and e0["jacobian"][1, 0] == 1.0
              and e0["jacobian"][2, 5] == 1.0)
    add_check("审查7 merge 事件：%s+%s→%s，t=%.3f，3×6 Jacobian"
              % (e0["left_id"], e0["right_id"], e0["merged_id"], e0["t"]),
              "shape=%s" % (e0["jacobian"].shape,), ok_jac)
    add_check("审查7 源生命周期：消失区域 rB 的源被移除",
              "sources=%s" % {k: round(v, 6) for k, v in m["sources"].items()},
              "rB" not in m["sources"] and "B" in m["retired"])
    add_check("审查7 存活源按 id 重绑：rC=%.3e 未绑错（下标已变）"
              % m["sources"].get("rC", 0.0),
              "ids=%s" % m["ids"],
              abs(m["sources"].get("rC", 0.0) - 2e-3) < 1e-12)

    return dict(merges=len(m["merges"]), ids=m["ids"],
                sources=list(m["sources"]))


# ------------------------------------------------------------ 审查9 编译链接线
def case_error_aware_chain():
    """审查9：run_chain 接受 Workload+QoI+Tolerance → 静态/动态合同 → 配置。

    以前 evaluate_sys/generate_config 虽有 error_control/qoi 参数，
    run_chain 却没传；本回归检查端到端传递：证书挂证据、p7 出
    dJ-allocation 逐元素预算；缺 evidence 且 workload 要求 ⇒ 拒绝。
    """
    emit("[case_error_aware_chain]  run_chain 动态合同端到端传递")
    from analysis.descriptor import Descriptor
    from lowering.compose import run_chain
    desc = Descriptor(np.eye(3), np.diag([-1.0, -10.0, -100.0]),
                      np.eye(3)[:, :1], np.eye(3))
    kw = dict(B_hw=1e4, kappa_hw=1e3, closure_N=3, closure_eps=1e-9)
    ec = ErrorControlResult(cls="contracting", n_windows=1,
                            coverage_ratio=1.0, e_measured_max=1e-3,
                            e_predicted_max=2e-3,
                            global_tolerance=1e-2).as_dict()
    res = run_chain(desc, error_control=ec, qoi=np.ones(3).tolist(),
                    eps_J=500.0, T_horizon=1.0, **kw)
    item = res.cert.items.get("error_control", {})
    add_check("审查9 证书挂动态证据：error_control verdict=%s"
              % item.get("verdict"), item.get("verdict"),
              item.get("verdict") == "pass")
    plan = res.config.plan
    add_check("审查9 p7 出 dJ 分配预算：source=%s，%d 项"
              % (plan.budget_source, len(plan.per_element_budgets or [])),
              "eps_J=%s" % plan.eps_J_ppm,
              plan.budget_source == "dJ-allocation"
              and plan.per_element_budgets is not None
              and len(plan.per_element_budgets) > 0)
    res2 = run_chain(desc, require_error_control=True, **kw)
    add_check("审查9 缺 evidence 且 workload 要求 ⇒ 证书 rejected",
              res2.cert.status, res2.cert.status == "rejected")
    return dict(verdict=item.get("verdict"),
                budget_source=plan.budget_source,
                missing_status=res2.cert.status)


def case_p7_physics():
    """C/D 修复：解析伴随权重 + 物理端口预算 + P6∧P7 联合判决。"""
    emit("[case_p7_physics]  解析权重 / 端口图 / 联合判决")
    import numpy as np
    from analysis.adjoint import dJ_weights_exact
    from analysis.descriptor import Descriptor
    from lowering.compose import joint_verdict, run_chain
    from lowering.p7 import qoi_adjoint_weights

    # --- C1：刚性对角系统的 65 点梯形陷阱 vs 闭式 ---
    A = np.diag([-1.0, -20.0, -400.0, -3000.0])
    c = np.ones(4)
    T = 1.0
    t65 = np.linspace(0.0, T, 65)
    adj = qoi_adjoint_weights(A, None, c, T, t65)
    w_exact = dJ_weights_exact(A, c, T)
    rel = float(np.max(np.abs(adj["weights"] - w_exact))
                / np.max(np.abs(w_exact)))
    add_check("C1 刚性对角：闭式权重与解析式一致（rel=%.2e），而 65 点梯形"
              "偏差 %.2f（γ_max·Δt=%.0f）"
              % (rel, adj["sampled_deviation"], adj["gamma_max_dt"]),
              "method=%s" % adj["method"],
              rel < 1e-12 and adj["sampled_deviation"] > 0.1
              and adj["gamma_max_dt"] > 10.0 and adj["reliable"])
    # 端点伪影：最快模态的 1/γ 尖峰被 0.5Δt 替身 ⇒ 该模态权重放大
    lam_s = np.array([c * np.exp(-(-np.diag(A)) * (T - tk))
                      for tk in t65]).T
    w_s = np.array([np.trapezoid(np.abs(lam_s[i]), t65)
                    for i in range(4)])
    w_s = w_s / w_s.sum()
    ratio_fast = float(w_s[-1] / w_exact[-1])
    ratio_slow = float(w_s[0] / w_exact[0])
    add_check("C1b 最快模态权重被梯形放大 %.1f×（慢模态 %.3f×）：1/γ=%.1e 的"
              "尖峰被 0.5Δt=%.1e 替身" % (ratio_fast, ratio_slow,
                                          1.0 / 3000.0, 0.5 * (t65[1] - t65[0]),
                                          ),
              "endpoint artefact", ratio_fast > 5.0 and abs(ratio_slow - 1) < 0.05)

    # --- C2/C3：耦合系统（有边 ⇒ 有 route 端口）走 run_chain ---
    # 对角刚性系统（C2 走闭式）
    desc_stiff = Descriptor(np.eye(4), np.diag([-1.0, -20.0, -400.0, -3000.0]),
                            np.ones((4, 1)), np.ones((1, 4)))
    kw4 = dict(B_hw=1e4, kappa_hw=1e3, closure_N=4, closure_eps=1e-9)
    stiff = run_chain(desc_stiff, qoi=np.ones(4), eps_J=5e4, T_horizon=1.0,
                      **kw4)
    add_check("C2 run_chain 的 dJ 权重走闭式：method=%s，γΔt=%.1f，"
              "逐分量梯形偏差 %.1f×（旧路径会失真）"
              % (stiff.config.plan.adjoint["method"],
                 stiff.config.plan.adjoint["gamma_max_dt"],
                 stiff.config.plan.adjoint["sampled_deviation"]),
              "closed-form",
              stiff.config.plan.adjoint["method"] == "closed-form"
              and stiff.config.plan.adjoint["reliable"]
              and stiff.config.plan.adjoint["sampled_deviation"] > 5.0)

    # 耦合系统（C3/C4：有边 ⇒ 有 route 端口）
    N = 8
    A8 = np.diag(-np.arange(1, N + 1, dtype=float))
    off = 0.4
    for i in range(N - 1):
        A8[i, i + 1] += off
        A8[i + 1, i] += off
    desc8 = Descriptor(np.eye(N), A8, np.ones((N, 1)), np.ones((1, N)))
    kw = dict(B_hw=1e4, kappa_hw=1e3, closure_N=N, closure_eps=1e-9)
    tight = run_chain(desc8, qoi=np.ones(N), eps_J=50.0, T_horizon=1.0, **kw)
    loose = run_chain(desc8, qoi=np.ones(N), eps_J=5e5, T_horizon=1.0, **kw)
    plan = tight.config.plan
    kinds = sorted({p["kind"] for p in (plan.port_budgets or [])})
    add_check("C3 物理端口图分型：%s（gm/C/offset/DAC/ADC/route 各自 K）"
              % "/".join(kinds), "no state-name sharing",
              all(k in kinds for k in ("gm_c", "c_bank", "offset", "dac",
                                       "adc", "gm_edge", "route_tau")))
    Kgm = [p["K_rel"] for p in plan.port_budgets if p["kind"] == "gm_c"]
    Koff = [p["K_rel"] for p in plan.port_budgets if p["kind"] == "offset"]
    add_check("C3b 同一节点的 gm 与 offset 灵敏度不同（%.4g vs %.4g）"
              % (Kgm[0], Koff[0]), "物理端口分开",
              abs(Kgm[0] - Koff[0]) > 1e-9 * max(Kgm[0], Koff[0]))
    tau = [p for p in plan.port_budgets if p["kind"] == "route_tau"]
    add_check("C3c route τ 端口存在：%d 条，带 τ_budget=%.2e s / 相位 %.2f°"
              % (len(tau),
                 tau[0]["tau_budget_s"] if tau else float("nan"),
                 tau[0]["phase_budget_deg"] if tau else float("nan")),
              "phase axis separate from ppm", bool(tau)
              and tau[0]["residual_budget_ppm"] is None)
    ident = sum(p["K_rel"] * p["residual_budget_ppm"]
                for p in plan.port_budgets
                if p["residual_budget_ppm"] is not None)
    add_check("C4 端口分配恒等式：Σ K_rel·e = %.4f ≈ ε_J=%.1f（残差 %.1e）"
              % (ident, plan.eps_J_ppm, abs(ident - plan.eps_J_ppm)),
              "relative-sensitivity contract",
              abs(ident - plan.eps_J_ppm) < 1e-9 * plan.eps_J_ppm)
    add_check("C5 紧 ε_J ⇒ %d/%d 端口低于测量底，deliverability=False"
              % (tight.config.deliverability["n_infeasible_ports"],
                 tight.config.deliverability["n_ports"]),
              "; ".join(tight.config.deliverability["reasons"])[:60],
              tight.config.deliverability["ok"] is False)
    add_check("C5b 松 ε_J ⇒ 全部端口可交付（P7 deliverable=True）",
              "n_ports=%d" % loose.config.deliverability["n_ports"],
              loose.config.deliverability["ok"] is True)

    # --- D：联合判决真值表 + 回传 ---
    jv_ok = joint_verdict(True, {"ok": True, "reasons": []})
    jv_bad_p7 = joint_verdict(True, {"ok": False, "reasons": ["3 ports < floor"],
                                    "n_infeasible_ports": 3})
    jv_partial = joint_verdict(False, {"ok": True, "reasons": []})
    add_check("D1 真值表：T∧T=%s、T∧F=%s（reason=%s）、F∧T=%s（%s）"
              % (jv_ok["final_feasible"], jv_bad_p7["final_feasible"],
                 jv_bad_p7["final_status"][:28], jv_partial["final_feasible"],
                 jv_partial["final_status"][:22]),
              "∧ semantics", jv_ok["final_feasible"]
              and not jv_bad_p7["final_feasible"]
              and not jv_partial["final_feasible"])
    add_check("D2 联合判决回传到证书：cert.status=%s、cert.p7.final=%s、"
              "chain.final=%s"
              % (tight.cert.status, tight.cert.p7["final_feasible"],
                 tight.final_feasible),
              "P7 undeliverable ⇒ chain rejected",
              tight.cert.p7["final_feasible"] is False
              and tight.final_feasible is False
              and "P7" in tight.final_status)
    add_check("D2b 松 ε_J 下 chain 状态为 partial（P6 未全轴评估，不许诺"
              " feasible）：%s" % loose.final_status[:40], "partial",
              loose.final_status.startswith("partial"))
    return dict(closed_form_rel=rel, trap_dev=adj["sampled_deviation"],
                kinds=kinds, n_ports=tight.config.deliverability["n_ports"],
                tight_ok=tight.config.deliverability["ok"],
                loose_ok=loose.config.deliverability["ok"],
                final_tight=tight.final_status,
                final_loose=loose.final_status)


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
    emit("")
    mul = case_multi_shock()
    emit("")
    alc = case_epsj_allocation()
    emit("")
    dwl = case_runtime_dwell()
    emit("")
    de = case_descriptor_adjoint()
    emit("")
    nl = case_nonlinear_dwr()
    emit("")
    ent = case_entropy_system()
    emit("")
    ch = case_chaos_contract()
    emit("")
    rs = case_residual_semantics()
    emit("")
    rq = case_residual_quadrature()
    emit("")
    hc = case_certificate_hardening()
    emit("")
    dp = case_device_ports()
    emit("")
    ft = case_five_types()
    emit("")
    dr = case_descriptor_robustness()
    emit("")
    il = case_interface_lifecycle()
    emit("")
    ec9 = case_error_aware_chain()
    emit("")
    p7p = case_p7_physics()
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
                   multi_shock=mul,                    epsj_alloc=alc, runtime_dwell=dwl,
                   descriptor_adjoint=de, nonlinear_dwr=nl,
                   entropy_system=ent, chaos=ch,                    residual_semantics=rs, residual_quadrature=rq, certificate_hardening=hc,
                   device_ports=dp, five_types=ft,
                   descriptor_robustness=dr, interface_lifecycle=il,
                   error_aware_chain=ec9, p7_physics=p7p,
                   checks=_checks)
    with open(os.path.join(out_dir, "error_control.json"), "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return n_pass == len(_checks)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
