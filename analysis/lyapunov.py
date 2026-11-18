"""广义 Lyapunov → 输出方差（DESIGN.md §0(B)、§1.2）。

    E P Aᵀ + A P Eᵀ + B_p Q B_pᵀ = 0 ,   σ_y² = tr(C P Cᵀ)

E 奇异时 P 在代数子空间上不唯一，噪声注入代数方程也会产生脉冲。做法是
QZ 排序取出稳定有限谱的 deflating subspace，只在那个子空间上解方程；
如果 B 在其余子空间上有分量，H2 直接判为无穷（这是物理事实，不是数值问题）。

返回的 residual 是稳定子空间上的相对残差，作用与 coupling.py 的质量账本相同：
它必须 ~1e-12，否则说明求解器本身有隐藏错误。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .descriptor import Descriptor

__all__ = ["LyapunovResult", "solve_generalized_lyapunov"]


@dataclass
class LyapunovResult:
    P: np.ndarray
    sigma_y2: float
    residual: float
    rel_residual: float
    h2_finite: bool
    n_stable: int
    n_total: int
    driven_other: float
    method: str

    @property
    def sigma_y(self) -> float:
        return float(np.sqrt(max(self.sigma_y2, 0.0)))


def _stable_finite(alpha, beta, zero_tol: float, scale: float):
    finite = np.abs(beta) > 1e-12 * (np.abs(alpha) + np.abs(beta) + 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        re = np.real(alpha / np.where(finite, beta, 1.0))
    return finite & (re < -zero_tol * scale)


def solve_generalized_lyapunov(
    sys: Descriptor, Q: np.ndarray | None = None, zero_tol: float = 1e-12
) -> LyapunovResult:
    """解 E P Aᵀ + A P Eᵀ + B Q Bᵀ = 0。"""
    if Q is None:
        Q = np.eye(sys.m)
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (sys.m, sys.m):
        raise ValueError("Q 形状应为 (m, m)")
    W = sys.B @ Q @ sys.B.T
    n = sys.n

    # 严格稳定且 E 良态：直接降为 (E⁻¹A, E⁻¹B)，精度最好。
    use_direct = sys.is_stable() and np.linalg.cond(sys.E) < 1e10
    if use_direct:
        As = linalg.solve(sys.E, sys.A)
        Bs = linalg.solve(sys.E, sys.B)
        P = linalg.solve_lyapunov(As, -(Bs @ Q @ Bs.T))
        P = 0.5 * (P + P.T)
        R = sys.E @ P @ sys.A.T + sys.A @ P @ sys.E.T + W
        num = np.linalg.norm(R)
        den = np.linalg.norm(sys.E @ P @ sys.A.T) + np.linalg.norm(sys.A @ P @ sys.E.T) + np.linalg.norm(W)
        sigma_y2 = float(np.real(np.trace(sys.C @ P @ sys.C.T)))
        return LyapunovResult(P, sigma_y2, float(num), float(num / max(den, 1e-300)),
                              True, n, n, 0.0, "direct")

    # 奇异/临界：QZ 排序，取稳定有限谱的 deflating subspace。
    w = sys.finite_eigs()
    scale = max(1.0, float(np.max(np.abs(w))) if w.size else 1.0)
    sort = lambda a, b: _stable_finite(a, b, zero_tol, scale)
    AA, BB, alpha, beta, Qz, Z = linalg.ordqz(sys.A, sys.E, sort=sort, output="real")
    k = int(np.count_nonzero(sort(alpha, beta)))
    Z1 = Z[:, :k]
    Z2 = Z[:, k:]

    if k == 0:
        P = np.zeros((n, n))
        B2 = Z2.T @ sys.B
        driven = float(np.linalg.norm(B2 @ Q @ B2.T))
        return LyapunovResult(P, float("inf") if driven > 1e-12 else 0.0,
                              0.0, 0.0, driven <= 1e-12, 0, n, driven, "qz-empty")

    Er = Z1.T @ sys.E @ Z1
    Ar = Z1.T @ sys.A @ Z1
    Br = Z1.T @ sys.B
    As = linalg.solve(Er, Ar)
    Bs = linalg.solve(Er, Br)
    Wr = Bs @ Q @ Bs.T
    Pr = linalg.solve_lyapunov(As, -Wr)
    Pr = 0.5 * (Pr + Pr.T)
    P = Z1 @ Pr @ Z1.T
    P = 0.5 * (P + P.T)

    # 稳定块上的残差（与 coupling.py 的 mass budget 同一种自检）
    Rr = Er @ Pr @ Ar.T + Ar @ Pr @ Er.T + Wr
    num = np.linalg.norm(Rr)
    den = np.linalg.norm(Er @ Pr @ Ar.T) + np.linalg.norm(Ar @ Pr @ Er.T) + np.linalg.norm(Wr)
    # B 在非稳定子空间上的投影：不为零 ⇒ 噪声驱动了零/不稳定模态，H2 = ∞
    B2 = Z2.T @ sys.B
    driven = float(np.linalg.norm(B2 @ Q @ B2.T))
    h2_finite = driven <= 1e-12 * max(1.0, float(np.linalg.norm(W)))
    sigma_y2 = float(np.real(np.trace(sys.C @ P @ sys.C.T)))
    return LyapunovResult(P, sigma_y2 if h2_finite else float("inf"),
                          float(num), float(num / max(den, 1e-300)),
                          h2_finite, k, n, driven, "qz")
