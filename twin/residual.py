"""分段积分残差与误差源归因（DESIGN.md §6.22 / §6.22.1）。

数字监督器对模拟轨迹能做的最便宜、最通用的健康检查：**硬件输出是否
自洽于目标方程**。真实系统 ẋ=F(x)，硬件实现了 ẋ=F(x̂)+r(t)，则窗口
[t_k, t_{k+1}] 上的积分残差

    R_k = x̂(t_{k+1}) − x̂(t_k) − ∫ F(x̂(t), t) dt = ∫ r(t) dt

恰好等于该窗口内注入的扰动积累 —— 所以残差不仅判"是否守恒于方程"，
还直接给出误差传播递推 ε_{k+1} ≤ G_k ε_k + η_k 里的**新增误差预算
η_k**。这是 residual.py 与 error_propagation.py 的分工边界：
这里测量，那边传播。

**区间记账（2026-11-27 修复）**：N 个采样点给出 N−1 个区间，每个区间
的残差样本是该区间的**均值** r̄_i（前向差分的点样本会被求积权重重
复计数，且最后一个区间根本进不了旧实现的 trapezoid(t[:-1])）。本模块
统一用

    r̄_i = (x_{i+1}−x_i)/h_i − (F_i + F_{i+1})/2
    ∫_window r dt  = Σ_i r̄_i h_i          （与 R 精确同口径）
    L1_est         = Σ_i ‖r̄_i‖ h_i        （覆盖包括最后区间在内的全部）

P0-2 反例回归（A=0, t=[0,1], x=[0,1] ⇒ r=1, e(1)=1）：旧实现的
trapezoid 在单区间上取空数组，l1/propagated/quad_error 全给 0；新实现
三者都给 1。

**估计界 vs 认证界（同一字段的两种强度）**：

  estimated_bound   采样区间和。它假设区间内 r 的变化不可见——采样间
                    的扰动可以整段躲过（区间平均恰好抵消），因此**不是**
                    误差上界，只是"看得见的部分"。
  certified_bound   需要外部声明之一：
                      (a) r_lipschitz L_r：‖r(t)−r̄_i‖ ≤ L_r·h_i/2，
                      (b) r_envelope ρ：‖r(t)‖ ≤ ρ 全窗口，
                      (c) 求积误差硬界 quad_bound（否则梯形积分误差
                          只进 quad_error 估计）。
                    线性 A 上再叠区间的增益硬界
                    ‖Φ(T,t_{i+1})‖·e^{μ h_i}（μ ≥ μ₂(A)，用
                    ‖expm(Aτ)‖ ≤ e^{μτ}, τ≥0），非线性则要求调用方
                    声明局部 log-norm 上界 μ（沿轨迹的 Jacobian 界）。
                    η 取 certified_bound 当且仅当它存在；否则 η 是
                    estimated + 求积估计，字段如实标注。

三条诚实边界（都写进了 run_error_control 的负对照）：

1. **残差小 ≠ 误差小。** 线性齐次系统里，任何初始偏移 c 的轨迹
   x̂+c 与 x̂ 满足同一个方程，残差为 ~0 但解误差 = ‖Φ(t)c‖。残差
   监视对"一致偏移"不可见，只能靠独立参考（高精度参考解/可观测量/
   校准激励）发现 —— 见 detect_offset 的合同注释。
2. **归因是投影不是证明。** attribute_sources 把 R 投影到候选源方向
   上，系数大只说明"与该源相容"，不能排除未建模源的巧合相容。
3. **残差自己也是测出来的。** 读出精度有限时 R 的可信下限是测量底
   （calib.py 的口径），低于底的改善不可声称。

依赖方向：只做窗口测量与归因，不含求解器与误差传播（那是
analysis/error_propagation.py 的职责）；不 import analysis/。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

__all__ = [
    "WindowResidual",
    "window_integral_residual",
    "linear_window_residual",
    "interval_residual_samples",
    "injection_budget",
    "conservation_residual",
    "attribute_sources",
    "detect_offset",
]


@dataclass
class WindowResidual:
    """一个检查窗口的残差记录（§6.22 P0-2 语义拆分 + 估计/认证双界）。

    三种量必须分开，不能互相替代：
      residual_signed      ∫r dt —— 诊断/校正估计（可正负抵消）
      residual_l1_bound    Σ‖r̄_i‖h_i —— **估计**的保守预算（采样可见部分）
      propagated_residual Σ G_i‖r̄_i‖h_i —— **估计**的动态误差预算
    eta 优先取 certified_bound（存在时）；否则取 propagated/l1 估计 +
    求积估计，并在 eta_assumption 里写明"未认证"。R 保留为 signed 的
    别名（向后兼容），但**不是**累计误差界。
    """
    t0: float
    t1: float
    R: np.ndarray                            # residual_signed 的别名
    eta: float                               # η：certified 优先，否则 estimated
    residual_signed: np.ndarray | None = None
    residual_l1_bound: float | None = None   # estimated Σ‖r̄‖h（全区间）
    propagated_residual: float | None = None # estimated Σ G_i‖r̄‖h
    quad_error: float = 0.0                  # 梯形求积误差的**估计**
    quad_bound: float | None = None          # 声明的求积硬界（认证用）
    estimated_bound: float | None = None     # max(propagated, l1) 估计
    certified_bound: float | None = None     # 认证上界（需声明）
    certification: str = "none"              # 认证依据的短标签
    r_envelope: float | None = None
    r_lipschitz: float | None = None
    mu: float | None = None
    meas_floor: float = 0.0
    eta_assumption: str = ""
    quad_points: int = 0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(t0=self.t0, t1=self.t1,
                    residual_signed_norm=float(np.linalg.norm(self.R)),
                    residual_l1_bound=self.residual_l1_bound,
                    propagated_residual=self.propagated_residual,
                    estimated_bound=self.estimated_bound,
                    certified_bound=self.certified_bound,
                    certification=self.certification,
                    eta=self.eta, eta_assumption=self.eta_assumption,
                    quad_error=self.quad_error, quad_bound=self.quad_bound,
                    meas_floor=self.meas_floor,
                    r_envelope=self.r_envelope,
                    r_lipschitz=self.r_lipschitz, mu=self.mu,
                    quad_points=self.quad_points, meta=dict(self.meta))


def interval_residual_samples(x_traj, t_grid, F) -> tuple:
    """N−1 个区间均值残差 r̄_i（与窗口 R 精确同口径）。

    返回 (r_bar, h)：r_bar[i] = (x_{i+1}−x_i)/h_i − (F_i+F_{i+1})/2，
    Σ_i r_bar[i]·h_i = x_end−x_start−trapezoid(F)。**这是唯一合法的
    残差采样口径**：前向差分的点样本会把每个区间算两次（旧 bug 的
    另一半）。
    """
    t = np.asarray(t_grid, dtype=float)
    x = np.asarray(x_traj, dtype=float)
    F = np.asarray(F, dtype=float)
    h = np.diff(t)
    if np.any(h <= 0):
        raise ValueError("t_grid 必须严格递增")
    r_bar = np.diff(x, axis=0) / h[:, None] - 0.5 * (F[:-1] + F[1:])
    return r_bar, h


def _phi_norms(A, t_grid, T) -> np.ndarray:
    """‖Φ(T,t_i)‖₂ 采样（对角 A 走逐模态 exp 的向量化 fast path）。

    对角系统 ‖diag(e^{λ_i(T−t)})‖₂ = max_i |e^{λ_i(T−t)}|——避免每次
    采样一次 expm（500 点 × 24 态窗口会慢两个量级）。
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    diag = np.diag(np.diag(A))
    if np.abs(A - diag).max() <= 1e-12 * max(1.0, float(np.abs(A).max())):
        lam = np.diag(A)
        E = np.exp(np.outer(lam, T - t))
        return np.max(np.abs(E), axis=0)
    return np.array([np.linalg.norm(expm(A * (T - ti)), 2) for ti in t])


def _gain_bound_boundary(A, t_grid, T, mu):
    """每区间**认证**的伴随增益硬界 G_i = ‖Φ(T,t_{i+1})‖·e^{μ h_i}。

    依据：对 τ≥0，‖expm(Aτ)‖₂ ≤ e^{μ₂τ}（μ₂ 为对数范数上界）；
    s∈[t_i,t_{i+1}] ⇒ Φ(T,s)=Φ(T,t_{i+1})·expm(A(t_{i+1}−s))，
    τ=t_{i+1}−s∈[0,h] ⇒ ‖Φ(T,s)‖ ≤ ‖Φ(T,t_{i+1})‖·e^{max(μ,0)·h}。
    注意 μ<0 时 e^{μτ}≤1（不能乘 e^{μh}——那会把右端点 I 的 1 压小，
    得到假下界）。
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    Phi_end = _phi_norms(A, t[1:], T)
    h = np.diff(t)
    return Phi_end * np.exp(max(float(mu), 0.0) * h)


def window_integral_residual(rhs, t_grid, x_traj, *,
                             r_envelope=None, r_lipschitz=None, mu=None,
                             quad_bound=None) -> WindowResidual:
    """一般 F 的窗口积分残差：R = x_end − x_start − ∫F(x̂(t),t)dt。

    区间记账见模块 docstring：l1/prop 都覆盖**全部 N−1 个区间**。
    认证界需要 r_lipschitz（区间内 r 的 Lipschitz 界）或 r_envelope
    （全窗口 ‖r‖ 包络）；非线性系统还要 mu（沿轨迹的 Jacobian
    log-norm 上界）才能给传播加权认证界。缺声明 ⇒ certified_bound=None，
    eta 只是估计（字段写明），绝不把估计冒充认证。

    求积：梯形。quad_error 是**估计**（‖F'‖ 用 ΔF 近似），quad_bound
    才是可认证的硬界——两者都进记录，不混用。
    """
    t = np.asarray(t_grid, dtype=float)
    x = np.asarray(x_traj, dtype=float)
    if t.shape[0] != x.shape[0]:
        raise ValueError("t_grid 与 x_traj 行数必须一致")
    if t.shape[0] < 2:
        raise ValueError("窗口至少两个采样点")
    F = np.array([np.asarray(rhs(xi, ti), dtype=float) for xi, ti in zip(x, t)])
    integral = np.array([np.trapezoid(F[:, j], t) for j in range(x.shape[1])])
    R = x[-1] - x[0] - integral
    r_bar, h = interval_residual_samples(x, t, F)
    l1 = float(np.sum(np.linalg.norm(r_bar, axis=1) * h))
    # 求积误差估计：‖F''‖·T·h²/12 量级，用 ‖ΔF‖/h 近似 ‖F'‖。
    dF = float(np.max(np.linalg.norm(np.diff(F, axis=0), axis=1))) if F.shape[0] > 1 else 0.0
    h_max = float(np.max(h))
    q_err = dF * (t[-1] - t[0]) * h_max / 8.0

    est_prop = None                      # 非线性：无 A/Φ，不给传播估计
    cert = None
    cert_tag = "none"
    rho_i = None
    if r_lipschitz is not None:
        L = float(r_lipschitz)
        if L < 0:
            raise ValueError("r_lipschitz 必须 >= 0")
        rho_i = np.linalg.norm(r_bar, axis=1) + L * h / 2.0
    if r_envelope is not None:
        rho = float(r_envelope)
        if rho < 0:
            raise ValueError("r_envelope 必须 >= 0")
        rho_i = np.full(h.shape, rho)
    if rho_i is not None and mu is not None:
        mu_f = float(mu)
        r_env_tot = float(np.max(rho_i))
        if abs(mu_f) <= 1e-15:
            cert = r_env_tot * (t[-1] - t[0])
        elif mu_f > 0:
            cert = r_env_tot * math.expm1(mu_f * (t[-1] - t[0])) / mu_f
        else:
            cert = r_env_tot * math.expm1(mu_f * (t[-1] - t[0])) / mu_f
        if cert < 0:
            raise ValueError("mu>0 时指数界无效：请缩短窗口")
        cert_tag = ("lognorm-envelope"
                    if r_envelope is not None else "lognorm-lipschitz")
        if quad_bound is not None:
            cert += float(quad_bound)
            cert_tag += "+quad-bound"
    est = max(l1, est_prop if est_prop is not None else 0.0)
    if cert is not None:
        eta = float(cert)
        eta_src = ("CERTIFIED: mu=%.4g, %s" % (float(mu), cert_tag))
    else:
        eta = float(est + q_err)
        eta_src = ("ESTIMATED (not certified): interval sums from sampled "
                   "residuals; sub-sample variation unaccounted — declare "
                   "r_lipschitz/r_envelope (+mu for nonlinear) to certify")
    return WindowResidual(
        t0=float(t[0]), t1=float(t[-1]), R=R,
        residual_signed=R.copy(), residual_l1_bound=l1,
        propagated_residual=est_prop,
        quad_error=q_err, quad_bound=(float(quad_bound)
                                      if quad_bound is not None else None),
        estimated_bound=float(est), certified_bound=(float(cert)
                                                     if cert is not None else None),
        certification=cert_tag, r_envelope=r_envelope,
        r_lipschitz=r_lipschitz, mu=mu,
        eta=eta, eta_assumption=eta_src,
        quad_points=int(t.shape[0]),
        meta=dict(quad="trapezoid", intervals=int(h.size),
                  trapz_err_scale=q_err, r_bar_norm_max=float(
                      np.max(np.linalg.norm(r_bar, axis=1))) if r_bar.size else 0.0))


def linear_window_residual(A, t_grid, x_traj, *,
                           r_envelope=None, r_lipschitz=None, mu=None,
                           quad_bound=None) -> WindowResidual:
    """线性 ẋ=Ax+r 的窗口残差 + 估计/认证双界。

    R=∫r dt 只用于诊断/校正；η 的认证路径（存在声明时）：
      G_i^cert = ‖Φ(T,t_{i+1})‖·e^{μ h_i}（μ 缺省取 μ₂(A)）
      ρ_i      = ‖r̄_i‖ + L_r·h_i/2   （或 r_envelope）
      certified = Σ G_i^cert·ρ_i·h_i (+ quad_bound)
    这是 ‖∫Φ(T,s)r(s)ds‖ 的硬上界（区间三角不等式 + 半群增益界），
    覆盖采样间不可见的扰动（L_r 声明的那部分）。
    无声明时 propagated_residual 仍是**估计**：G_i 取区间两端 ‖Φ‖ 的
    较大者，只说明"看得见的"残差样本。
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    x = np.asarray(x_traj, dtype=float)
    integral = np.array([np.trapezoid(x[:, j], t) for j in range(x.shape[1])])
    R = x[-1] - x[0] - A @ integral
    r_bar, h = interval_residual_samples(x, t, x @ A.T)
    l1 = float(np.sum(np.linalg.norm(r_bar, axis=1) * h))
    T = float(t[-1])

    # 估计传播：每个区间的增益取两端样本的较大者（不再把左端点当上界）
    Phi = _phi_norms(A, t, T)
    G_est = np.maximum(Phi[:-1], Phi[1:])
    prop = float(np.sum(G_est * np.linalg.norm(r_bar, axis=1) * h))

    # 梯形求积误差估计（F=A x̂）
    dF = float(np.max(np.linalg.norm(np.diff(x @ A.T, axis=0), axis=1)))
    h_max = float(np.max(h))
    q_err = dF * (T - t[0]) * h_max / 8.0

    # ---- 认证路径 -----------------------------------------------------------
    mu_used = None
    if mu is not None:
        mu_used = float(mu)
    else:
        # μ₂(A) = λ_max((A+Aᵀ)/2)：‖expm(Aτ)‖₂ ≤ e^{μ₂τ}
        mu_used = float(np.max(np.linalg.eigvalsh((A + A.T) / 2.0)))
    if r_lipschitz is not None:
        L = float(r_lipschitz)
        if L < 0:
            raise ValueError("r_lipschitz 必须 >= 0")
        rho_i = np.linalg.norm(r_bar, axis=1) + L * h / 2.0
    elif r_envelope is not None:
        rho_i = np.full(h.shape, float(r_envelope))
    else:
        rho_i = None
    cert = None
    cert_tag = "none"
    if rho_i is not None:
        G_cert = _gain_bound_boundary(A, t, T, mu_used)
        cert = float(np.sum(G_cert * rho_i * h))
        if quad_bound is not None:
            cert += float(quad_bound)
            cert_tag = ("interval-gain+%s+quad-bound"
                        % ("r-envelope" if r_lipschitz is None
                           else "r-lipschitz"))
        else:
            cert_tag = ("interval-gain+%s"
                        % ("r-envelope" if r_lipschitz is None
                           else "r-lipschitz"))
    est = max(l1, prop)
    if cert is not None:
        eta = float(cert)
        eta_src = ("CERTIFIED: interval gain ||Phi(T,t_i+1)||*exp(mu*h) "
                   "with mu=%.4g, %s" % (mu_used, cert_tag))
    else:
        eta = float(est + q_err)
        eta_src = ("ESTIMATED (not certified): interval gain = max endpoint "
                   "||Phi||, sampled r only; sub-sample variation and "
                   "quadrature error are NOT bounded — declare r_lipschitz/"
                   "r_envelope (+quad_bound) to certify")
    return WindowResidual(
        t0=float(t[0]), t1=T, R=R,
        residual_signed=R.copy(), residual_l1_bound=l1,
        propagated_residual=prop,
        quad_error=q_err, quad_bound=(float(quad_bound)
                                      if quad_bound is not None else None),
        estimated_bound=float(est), certified_bound=(float(cert)
                                                     if cert is not None else None),
        certification=cert_tag, r_envelope=r_envelope,
        r_lipschitz=r_lipschitz, mu=(mu_used if rho_i is not None else None),
        eta=eta, eta_assumption=eta_src,
        quad_points=int(t.shape[0]),
        meta=dict(quad="interval-sum", form="linear",
                  intervals=int(h.size), mu2=float(mu_used),
                  trapz_err_scale=q_err,
                  r_bar_norm_max=float(np.max(np.linalg.norm(r_bar, axis=1))) if r_bar.size else 0.0))


def injection_budget(residuals) -> list:
    """窗口残差序列 → **保守**新增误差预算序列 [η_0, η_1, …]。

    取 WindowResidual.eta：有认证界（declared envelope/Lipschitz + log-norm
    增益界）时是 certified_bound，否则是 estimated + 求积估计（调用方
    读 eta_assumption 决定能不能引用为"界"）。**不是** ‖∫r dt‖——后者
    可因注入正负抵消而归零（P0-2 反例：ė=−e+r，+1 后 −1，signed R=0
    而 e(2)=0.3996）。
    """
    return [float(w.eta) for w in residuals]


def conservation_residual(M_t0: float, M_t1: float) -> float:
    """守恒量漂移：M(t1) − M(t0)。线性版本请配合
    analysis.error_propagation.conservation_drift_rate（率）与本函数
    （实测累计）对账：率 × 时间 × ‖x‖ 应覆盖实测漂移。
    """
    return float(M_t1 - M_t0)


def attribute_sources(R, source_dirs: dict) -> dict:
    """把残差 R 投影到候选源方向上做归因（§6.22 边界 2：相容 ≠ 证明）。

    source_dirs: {name: 单位方向向量}。做非负最小二乘没必要——源可以
    反号；用普通 lstsq，返回 {name: 系数}，并给 residual-after-fit
    （归因后剩余），剩余大 ⇒ 有未建模源，如实报告。
    """
    names = list(source_dirs.keys())
    S = np.column_stack([np.asarray(source_dirs[k], dtype=float) for k in names])
    R = np.asarray(R, dtype=float)
    coeff, res, _, _ = np.linalg.lstsq(S, R, rcond=None)
    leftover = float(np.linalg.norm(R - S @ coeff))
    out = {k: float(c) for k, c in zip(names, coeff)}
    out["residual_after_fit"] = leftover
    return out


def detect_offset(x_hat, x_ref, tol: float) -> dict:
    """一致偏移检测：**必须**有独立参考，残差通路对此不可见（边界 1）。

    返回 {offset_norm, detectable}。x_ref 来自高精度参考解 / 校准激励
    / 可观测量，不能是硬件自己的输出。没有参考时调用方不得伪造
    detectable=True —— 这是 §6.22 拒绝过的第 4 类合同违规。
    """
    if x_ref is None:
        return dict(offset_norm=None, detectable=False,
                    reason="无独立参考：残差通路对一致偏移不可见")
    off = float(np.linalg.norm(np.asarray(x_hat, dtype=float)
                               - np.asarray(x_ref, dtype=float)))
    return dict(offset_norm=off, detectable=bool(off > tol),
                tol=float(tol))
