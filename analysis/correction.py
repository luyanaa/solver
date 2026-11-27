"""可部署校正算子：q/ζ 的离线 characterization 与 held-out 验证
（DESIGN.md §6.22 P0-3 续，2026-11-27）。

**上一轮的合同违规**：diffusion 窗口实验把硬件状态直接回注到解析真解
（`xh = x_true_end + injection_noise`），再用真解反推 q_k=‖e⁺‖/‖e⁻‖。
那只能做离线 characterization，不能当运行时证据——运行时没有真解。

本模块提供三种**真正可部署**的校正器（都只消费硬件自己的状态、模型
和可信外部信号，不读真解）：

  1. DigitalRecompute      数字参考短窗口重算（一般 ODE）
       从窗口起点状态用高精度模型重放到窗口末：x⁺=Φ_dig(h)x⁻。
       q=‖Φ(h)‖（线性解析；非线性由离线 characterization 测），
       ζ=数字积分器模型误差 + DAC 回写量化地板。
  2. ResidualPropagation   高精度残差 + correction solve（Poisson/隐式 PDE 的 ODE 版）
       用窗口残差样本估计注入贡献 δ=∫Φ(t₁,s)r(s)ds，再扣除：
       x⁺=x_meas−δ。这是"传播加权扣除"，不是 naive x−R（后者会把
       注入的**非传播**部分也扣掉，q>1 的负对照）。
       q≈‖Φ(h)‖，ζ=残差求积误差 + 回写地板。
  3. ObserverCorrection    稀疏可信观测 + observer（可观测线性系统）
       y=C x_true + noise 来自**独立可信传感器**（数字监督器持有的
       边界/守恒量/复本，不是硬件自己的输出——硬件自观测 y=Cx_meas
       只剩噪声，对误差零信息）。x⁺=x_meas+L(y−Cx_meas)，
       q=‖(I−LC)Φ(h)‖，ζ=‖L‖σ_y + 回写地板。

  负对照：SignedSubtraction  x⁺=x_meas−R（naive，不认证）。

characterization 协议（与 calib.py 的 VERIFY 同构）：
  * 在**训练集**（不同种子/激励幅度 ≥ 运行时包络）上测 q_max；
  * ζ 是**声明**的硬件地板（DAC LSB、传感器噪声增益），不是拟合参数；
  * 在**未参与拟合**的 held-out workload 上跑递推
        ε⁺_{k+1} ≤ q_max·G_k·ε⁺_k + q_max·η_k + ζ_max
    并报覆盖率。覆盖率 < 目标 ⇒ 合同不成立（不许拿训练集分位数冒充）。

依赖方向：只做校正算子与统计协议；不 import twin/，不含求解器。
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
from scipy.linalg import expm

from analysis.error_propagation import corrected_recursion

__all__ = [
    "CorrectionWindow",
    "residual_correction_eta_factor",
    "DigitalRecompute",
    "ResidualPropagation",
    "ObserverCorrection",
    "SignedSubtraction",
    "characterize_q",
    "heldout_coverage",
    "recursion_with_correction",
]


@dataclass
class CorrectionWindow:
    """一个校正窗口的硬件采样（局部时间从 0 到 h，行 = 时刻）。"""

    t: np.ndarray
    xs: np.ndarray

    @property
    def h(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def x_end(self) -> np.ndarray:
        return np.asarray(self.xs[-1], dtype=float)


def _window_residual_interval(A, w: CorrectionWindow) -> np.ndarray:
    """窗口积分残差 R=x_end−x_start−A∫x dt（梯形，与 residual.py 同口径）。"""
    A = np.asarray(A, dtype=float)
    integral = np.array([np.trapezoid(w.xs[:, j], w.t)
                         for j in range(w.xs.shape[1])])
    return w.x_end - np.asarray(w.xs[0], dtype=float) - A @ integral


class DigitalRecompute:
    """数字参考短窗口重算：x⁺ = Φ_dig(h) x⁻（不读真解）。

    A 为线性模型时用 expm（解析）；给 rhs 时用 RK4（n_sub 子步）。
    zeta 由调用方声明（数字积分器模型误差上界 + DAC 回写 LSB）。
    """

    def __init__(self, A=None, rhs=None, n_sub: int = 8, name="digital-recompute"):
        self.A = None if A is None else np.asarray(A, dtype=float)
        self.rhs = rhs
        self.n_sub = int(n_sub)
        self.name = name

    def apply(self, w: CorrectionWindow, y=None) -> np.ndarray:
        x0 = np.asarray(w.xs[0], dtype=float)
        h = w.h
        if self.A is not None:
            return expm(self.A * h) @ x0
        if self.rhs is None:
            raise ValueError("DigitalRecompute 需要 A 或 rhs")
        x = x0.copy()
        dt = h / self.n_sub
        for _ in range(self.n_sub):
            k1 = np.asarray(self.rhs(x, 0.0), dtype=float)
            k2 = np.asarray(self.rhs(x + 0.5 * dt * k1, 0.0), dtype=float)
            k3 = np.asarray(self.rhs(x + 0.5 * dt * k2, 0.0), dtype=float)
            k4 = np.asarray(self.rhs(x + dt * k3, 0.0), dtype=float)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return x


class ResidualPropagation:
    """高精度残差 + correction solve：x⁺ = x_meas − Σ Φ(t₁,t_mid) r̄_i h_i。

    只消费窗口采样与模型；q ≈ ‖Φ(h)‖（估计误差进 ζ）。
    与 naive x−R 的差别就是"传播加权"：R 是 ∫r，扣除它等于假设
    Φ≡I，在没有强收缩时会过校正（负对照 q>1）。
    """

    def __init__(self, A, name="residual-propagation"):
        self.A = np.asarray(A, dtype=float)
        self.name = name

    def injection_estimate(self, w: CorrectionWindow) -> np.ndarray:
        t = np.asarray(w.t, dtype=float)
        xs = np.asarray(w.xs, dtype=float)
        h = np.diff(t)
        F = xs @ self.A.T
        r_bar = np.diff(xs, axis=0) / h[:, None] - 0.5 * (F[:-1] + F[1:])
        t_mid = 0.5 * (t[:-1] + t[1:])
        T = float(t[-1])
        delta = np.zeros(xs.shape[1])
        diag = np.diag(np.diag(self.A))
        off = np.abs(self.A - diag).max()
        if off <= 1e-12 * max(1.0, float(np.abs(self.A).max())):
            # 对角 fast path：Φ(T,t_mid)=diag(exp(λ(T−t_mid)))——避免 O(N) 次 expm
            lam = np.diag(self.A)
            for i in range(h.size):
                delta += np.exp(lam * (T - t_mid[i])) * r_bar[i] * h[i]
            return delta
        for i in range(h.size):
            delta += expm(self.A * (T - t_mid[i])) @ r_bar[i] * h[i]
        return delta

    def apply(self, w: CorrectionWindow, y=None) -> np.ndarray:
        return w.x_end - self.injection_estimate(w)


class ObserverCorrection:
    """稀疏可信观测 + Luenberger/最小二乘 observer。

    C: m×n 观测矩阵；y 是**独立可信**观测（C x_true + noise），不是
    硬件自己的输出。L 缺省为最小二乘投影 Cᵀ(CCᵀ)⁻¹（m≤n 且行满秩）。
    """

    def __init__(self, C, L=None, name="observer"):
        self.C = np.asarray(C, dtype=float)
        if self.C.ndim == 1:
            self.C = self.C[None, :]
        if L is None:
            G = self.C @ self.C.T
            self.L = self.C.T @ np.linalg.inv(G)
        else:
            self.L = np.asarray(L, dtype=float)
            if self.L.ndim == 1:
                self.L = self.L[:, None]
        self.name = name

    def apply(self, w: CorrectionWindow, y=None) -> np.ndarray:
        if y is None:
            raise ValueError("ObserverCorrection 需要可信观测 y")
        y = np.asarray(y, dtype=float).ravel()
        x = w.x_end
        return x + self.L @ (y - self.C @ x)


class SignedSubtraction:
    """负对照：x⁺ = x_meas − R（naive，不认证；用于证明它不满足收缩）。"""

    def __init__(self, A, name="signed-subtraction"):
        self.A = np.asarray(A, dtype=float)
        self.name = name

    def apply(self, w: CorrectionWindow, y=None) -> np.ndarray:
        return w.x_end - _window_residual_interval(self.A, w)


def residual_correction_eta_factor(A, h: float) -> float:
    """ResidualPropagation 的求积因子上界 ρ_q（无量纲，乘 η）。

    δ_error_i = ∫_0^h [e^{λ_i(h−s)} − e^{λ_i h/2}] r_i(s) ds。
    对角实负时逐模态：|∫(Φ−Φ_mid)r| ≤ e^{λ_i h/2}·h·g(a_i)·|r̄|，
    a_i=|λ_i|h/2，g(a)=(e^a−1−a)/a（g(0)=0）；取 max_i。
    非对角退回保守式 ‖Φ‖·g(‖A‖h/2)（可能偏松，如实标注）。
    """
    A = np.asarray(A, dtype=float)
    h = float(h)
    if h <= 0:
        raise ValueError("h 必须 > 0")
    diag = np.diag(np.diag(A))
    off = np.abs(A - diag).max()
    if off <= 1e-12 * max(1.0, float(np.abs(A).max())):
        lam = np.diag(A)
        if np.any(lam > 0):
            return 1.0                      # 非收缩：不给小因子，退回 η
        a = np.abs(lam) * h / 2.0
        g = np.where(a > 1e-12, (np.expm1(a) - a) / np.maximum(a, 1e-300), 0.0)
        return float(np.max(np.exp(-a) * g))
    a = float(np.linalg.norm(A, 2)) * h / 2.0
    g = (math.expm1(a) - a) / a if a > 1e-12 else 0.0
    return float(min(1.0, math.exp(-a) * g))


def characterize_q(pairs, zeta: float = 0.0, q_floor: float = 1e-15,
                   floor_multiple: float = 5.0) -> dict:
    """训练集的 (‖e⁻‖, ‖e⁺‖) → q_max（扣除声明 ζ 后的最坏比值）。

    合同是 ‖e⁺‖ ≤ q‖e⁻‖ + ζ，ζ 是**声明**的硬件地板（DAC 回写 LSB、
    传感器噪声增益），不是拟合参数。因此 q 的离线识别必须先把 ζ 的
    贡献从 e⁺ 里扣掉：否则零注入角落 e⁺≈ζ 会把比值抬到 2–3，使一个
    实际工作的校正器被 q>1 误判死。扣 ζ 后 q_max 是观测上确界，不做
    平滑/回归；held-out 验证负责抓住过度乐观（覆盖失败即合同不成立）。

    真解只在**离线** characterization 里出现——运行时合同只引用
    q_max/ζ，不引用任何 ‖e‖。
    """
    em = np.asarray([p[0] for p in pairs], dtype=float)
    ep = np.asarray([p[1] for p in pairs], dtype=float)
    if em.size == 0:
        raise ValueError("q 样本为空")
    if np.any(em < 0) or np.any(ep < 0):
        raise ValueError("误差范数必须 >= 0")
    raw = ep / np.maximum(em, q_floor)
    # 分辨率地板：e⁻ 与 ζ 同量级时"扣 ζ"的减法噪声主导，比率不可辨识。
    # 这些窗口只用来验证覆盖（递推本身含 ζ），不参与 q 的识别。
    resolvable = em >= float(floor_multiple) * max(float(zeta), q_floor)
    if not np.any(resolvable):
        resolvable = em > q_floor
    q = np.maximum(ep[resolvable] - float(zeta), 0.0) / em[resolvable]
    return dict(q_max=float(np.max(q)),
                q_median=float(np.median(q)),
                q_quantile_95=float(np.quantile(q, 0.95)),
                raw_ratio_max=float(np.max(raw)),
                n=int(q.size), n_resolvable=int(np.sum(resolvable)),
                zeta=float(zeta), q_floor=float(q_floor),
                floor_multiple=float(floor_multiple))


def heldout_coverage(predicted, measured, tol: float = 1.2) -> dict:
    """held-out 覆盖：measured ≤ tol×predicted 的比例 + 最坏比值。"""
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    if p.shape != m.shape or p.size == 0:
        raise ValueError("predicted/measured 同形且非空")
    ok = m <= tol * np.maximum(p, 1e-300)
    ratio = m / np.maximum(p, 1e-300)
    return dict(coverage=float(np.mean(ok)), n=int(p.size),
                worst_ratio=float(np.max(ratio)), tol=float(tol),
                n_covered=int(np.sum(ok)))


def recursion_with_correction(contract, gains, etas, e0=0.0):
    """把 (q_max, ζ_max) 代进 corrected_recursion 的薄封装。

    ε⁺_{k+1} ≤ q·G_k·ε⁺_k + q·η_k + ζ。返回 corrected_recursion 的
    dict（含 rho、eta_bar、steady_bound、eps_plus）。
    """
    q = float(contract["q_max"])
    zeta = float(contract.get("zeta", 0.0))
    n = len(gains)
    return corrected_recursion(e0, gains, etas, [q] * n, [zeta] * n)
