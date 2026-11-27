"""分段积分残差与误差源归因（DESIGN.md §6.22）。

数字监督器对模拟轨迹能做的最便宜、最通用的健康检查：**硬件输出是否
自洽于目标方程**。真实系统 ẋ=F(x)，硬件实现了 ẋ=F(x̂)+r(t)，则窗口
[t_k, t_{k+1}] 上的积分残差

    R_k = x̂(t_{k+1}) − x̂(t_k) − ∫ F(x̂(t), t) dt = ∫ r(t) dt

恰好等于该窗口内注入的扰动积累 —— 所以残差不仅判"是否守恒于方程"，
还直接给出误差传播递推 ε_{k+1} ≤ G_k ε_k + η_k 里的**新增误差预算
η_k = ‖R_k‖**。这是 residual.py 与 error_propagation.py 的分工边界：
这里测量，那边传播。

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

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

__all__ = [
    "WindowResidual",
    "window_integral_residual",
    "linear_window_residual",
    "injection_budget",
    "conservation_residual",
    "attribute_sources",
    "detect_offset",
]


@dataclass
class WindowResidual:
    """一个检查窗口的残差记录（§6.22 P0-2 语义拆分）。

    三种量必须分开，不能互相替代：
      residual_signed      ∫r dt —— 诊断/校正估计（可正负抵消）
      residual_l1_bound    ∫‖r‖ dt —— 保守预算（Φ 未知时的上界前提）
      propagated_residual  ∫‖Φ(t1,s)‖‖r(s)‖ ds —— 动态误差预算（Φ 已知）
    eta 取 propagated_residual（线性/已知 A）或 residual_l1_bound
    （仅当已知收缩 ‖Φ‖≤1）+ 求积误差 + 测量底。
    R 保留为 signed 的别名（向后兼容），但**不是**累计误差界。
    """
    t0: float
    t1: float
    R: np.ndarray                            # residual_signed 的别名
    eta: float                               # 保守新增误差预算（非 ‖R‖）
    residual_signed: np.ndarray | None = None
    residual_l1_bound: float | None = None
    propagated_residual: float | None = None
    quad_error: float = 0.0
    meas_floor: float = 0.0
    eta_assumption: str = ""
    quad_points: int = 0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(t0=self.t0, t1=self.t1,
                    residual_signed_norm=float(np.linalg.norm(self.R)),
                    residual_l1_bound=self.residual_l1_bound,
                    propagated_residual=self.propagated_residual,
                    eta=self.eta, eta_assumption=self.eta_assumption,
                    quad_error=self.quad_error, meas_floor=self.meas_floor,
                    quad_points=self.quad_points, meta=dict(self.meta))


def window_integral_residual(rhs, t_grid, x_traj) -> WindowResidual:
    """一般 F 的窗口积分残差：R = x_end − x_start − ∫F(x̂(t),t)dt。

    rhs(x, t) -> ẋ（**目标**方程的 RHS，不含硬件扰动）。x_traj 是
    硬件轨迹的采样（行 = 时刻），t_grid 等长。积分用梯形法则 ——
    采样足够密时梯形误差 O(h²) 进 meta（调用方保证它 ≪ η 才有意义；
    这是测量底的一部分，不隐藏）。
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
    # 源样本 r_i ≈ (x_{i+1}−x_i)/dt − F(x_i,t_i)（前向差分；求积误差
    # 诚实记入 quad_error）。signed 与 l1 由同一批样本积分，避免
    # 两次独立抽样造成口径不一致。
    dt_loc = np.diff(t)[:, None]
    r_samples = np.diff(x, axis=0) / dt_loc - F[:-1]
    l1 = float(np.trapezoid(np.linalg.norm(r_samples, axis=1), t[:-1]))
    h = float(np.max(np.diff(t)))
    # 梯形误差量级：‖ẍ‖·T·h²/12 的上界估计用 ‖ΔF‖ 近似
    dF = float(np.max(np.linalg.norm(np.diff(F, axis=0), axis=1))) if F.shape[0] > 1 else 0.0
    q_err = dF * (t[-1] - t[0]) * h / 8.0
    return WindowResidual(t0=float(t[0]), t1=float(t[-1]), R=R,
                          residual_signed=R.copy(),
                          residual_l1_bound=l1,
                          propagated_residual=None,       # 非线性：Φ 未知，不外推
                          eta=l1 + q_err, quad_error=q_err,
                          eta_assumption="l1 bound valid if ||Phi||<=1 "
                                         "(contraction must be certified)",
                          quad_points=int(t.shape[0]),
                          meta=dict(quad="trapezoid",
                                    trapz_err_scale=q_err))


def linear_window_residual(A, t_grid, x_traj) -> WindowResidual:
    """线性 ẋ=Ax+r 的窗口残差 + **传播加权的保守新增误差界**。

    R=∫r dt 仍只用于诊断/校正；eta=∫‖Φ(t1,s)‖‖r(s)‖ds + 求积误差，
    满足 ‖∫Φr‖ ≤ eta（线性 LTI，源样本由轨迹前向差分得到）。
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    x = np.asarray(x_traj, dtype=float)
    integral = np.array([np.trapezoid(x[:, j], t) for j in range(x.shape[1])])
    R = x[-1] - x[0] - A @ integral
    dt_loc = np.diff(t)[:, None]
    r_samples = np.diff(x, axis=0) / dt_loc - x[:-1] @ A.T
    l1 = float(np.trapezoid(np.linalg.norm(r_samples, axis=1), t[:-1]))
    T = float(t[-1])
    g = np.array([np.linalg.norm(expm(A * (T - ti)), 2)
                  * np.linalg.norm(r_samples[i])
                  for i, ti in enumerate(t[:-1])])
    prop = float(np.trapezoid(g, t[:-1]))
    dF = float(np.max(np.linalg.norm(np.diff(x @ A.T, axis=0), axis=1)))
    h = float(np.max(np.diff(t)))
    q_err = dF * (T - t[0]) * h / 8.0
    return WindowResidual(t0=float(t[0]), t1=T, R=R,
                          residual_signed=R.copy(), residual_l1_bound=l1,
                          propagated_residual=prop, eta=prop + q_err,
                          quad_error=q_err,
                          eta_assumption="propagated LTI integral with "
                                         "sampled r (trapezoid), no cancellation",
                          quad_points=int(t.shape[0]),
                          meta=dict(quad="trapezoid", form="linear"))


def injection_budget(residuals) -> list:
    """窗口残差序列 → **保守**新增误差预算序列 [η_0, η_1, …]。

    取 WindowResidual.eta（传播加权积分或经认证收缩下的 l1 界），
    **不是** ‖∫r dt‖——后者可因注入正负抵消而低估甚至归零
    （P0-2 反例：ė=−e+r，+1 后 −1，signed R=0 而 e(2)=0.3996）。
    供 error_propagation.error_recursion_bound 的 etas 参数使用。
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
