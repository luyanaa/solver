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
    """一个检查窗口的残差记录（§6.22 evidence 对象）。"""
    t0: float
    t1: float
    R: np.ndarray                 # 残差向量 = ∫r dt（状态空间的注入积累）
    eta: float                    # ‖R‖：误差递推的新增预算
    quad_points: int = 0          # 积分用的采样点数（诚实记录网格）
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(t0=self.t0, t1=self.t1,
                    R_norm=float(np.linalg.norm(self.R)),
                    eta=self.eta, quad_points=self.quad_points,
                    meta=dict(self.meta))


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
    h = float(np.max(np.diff(t)))
    # 梯形误差量级：‖ẍ‖·T·h²/12 的上界估计用 ‖ΔF‖ 近似
    dF = float(np.max(np.linalg.norm(np.diff(F, axis=0), axis=1))) if F.shape[0] > 1 else 0.0
    return WindowResidual(t0=float(t[0]), t1=float(t[-1]), R=R,
                          eta=float(np.linalg.norm(R)),
                          quad_points=int(t.shape[0]),
                          meta=dict(quad="trapezoid",
                                    trapz_err_scale=dF * (t[-1] - t[0]) * h / 8.0))


def linear_window_residual(A, t_grid, x_traj) -> WindowResidual:
    """线性 ẋ=Ax 的窗口残差：∫Ax dt = A·∫x dt，一次矩阵乘完成。"""
    A = np.asarray(A, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    x = np.asarray(x_traj, dtype=float)
    integral = np.array([np.trapezoid(x[:, j], t) for j in range(x.shape[1])])
    R = x[-1] - x[0] - A @ integral
    return WindowResidual(t0=float(t[0]), t1=float(t[-1]), R=R,
                          eta=float(np.linalg.norm(R)),
                          quad_points=int(t.shape[0]),
                          meta=dict(quad="trapezoid", form="linear"))


def injection_budget(residuals) -> list:
    """窗口残差序列 → 新增误差预算序列 [η_0, η_1, …]。

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
