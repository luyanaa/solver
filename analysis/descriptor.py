"""Descriptor 容器与谱诊断（DESIGN.md §0、§1）。

一个对象只回答四个问题：

    κ_t = max|Re λ| / min|Re λ|     频率窗宽度（去掉守恒零模态）
    B_req = max|λ| / 2π             最快模态对应的带宽
    H(jω) = C(jωE − A)⁻¹B + D       所有扰动的传递函数
    σ_min(jωE − A)                  最坏情况放大（Γ 的倒数）

不在这里做降阶、不做硬件映射；那些属于 lowering/。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

__all__ = ["Descriptor", "stable_finite_mask"]

_TINY = 1e-300


def stable_finite_mask(alpha, beta, zero_tol: float, scale: float):
    """QZ 的排序谓词：有限且 Re(λ) < −zero_tol·scale。"""
    finite = np.abs(beta) > 1e-12 * (np.abs(alpha) + np.abs(beta) + _TINY)
    with np.errstate(divide="ignore", invalid="ignore"):
        re = np.real(alpha / np.where(finite, beta, 1.0))
    return finite & (re < -zero_tol * scale)


@dataclass
class Descriptor:
    """E·ẋ = A·x + B·u, y = C·x + D·u。

    E 可以奇异（QSSA/虚地/KCL 节点）。B/C 允许 1-D，自动补成列/行。
    """

    E: np.ndarray
    A: np.ndarray
    B: np.ndarray | None = None
    C: np.ndarray | None = None
    D: np.ndarray | None = None

    def __post_init__(self):
        self.E = np.asarray(self.E, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        if self.E.ndim != 2 or self.E.shape[0] != self.E.shape[1]:
            raise ValueError("E 必须是方阵")
        if self.A.shape != self.E.shape:
            raise ValueError("A 必须与 E 同形")
        n = self.E.shape[0]
        if self.B is None:
            self.B = np.zeros((n, 1))
        if self.C is None:
            self.C = np.eye(n)
        self.B = np.asarray(self.B, dtype=float)
        if self.B.ndim == 1:
            self.B = self.B[:, None]
        self.C = np.asarray(self.C, dtype=float)
        if self.C.ndim == 1:
            self.C = self.C[None, :]
        if self.D is None:
            self.D = np.zeros((self.C.shape[0], self.B.shape[1]))
        self.D = np.asarray(self.D, dtype=float)
        if self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError("B 形状应为 (n, m)")
        if self.C.ndim != 2 or self.C.shape[1] != n:
            raise ValueError("C 形状应为 (p, n)")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D 形状应为 (p, m)")
        for name in ("E", "A", "B", "C", "D"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError("%s 含非有限值" % name)
        # 频响的 O(n) 快路径：E=I 且 A 对角。1024 模态的算例靠它才跑得动。
        self._diag = None
        if np.allclose(self.E, np.eye(n)) and np.allclose(self.A, np.diag(np.diag(self.A))):
            self._diag = np.diag(self.A).copy()
        # 奇异 E 的广义 Lyapunov/H∞ 需要知道 pencil 是否正则。
        self._regular = None
        # 广义特征值在 κ/B/has_imaginary 里被多次用到；n=1024 时重算是分钟级。
        self._eig_cache = None

    # ---- 基本属性 ----
    @property
    def n(self) -> int:
        return self.E.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    @property
    def p(self) -> int:
        return self.C.shape[0]

    def eigvals(self) -> np.ndarray:
        """广义特征值；无穷特征值以 inf/nan 返回，由调用者过滤。"""
        if self._eig_cache is None:
            self._eig_cache = linalg.eigvals(self.A, self.E)
        return self._eig_cache

    def finite_eigs(self) -> np.ndarray:
        w = self.eigvals()
        return w[np.isfinite(w)]

    def is_regular(self, tol: float = 1e-10) -> bool:
        """pencil 正则：不存在同时为零的 (alpha, beta) 对。"""
        if self._regular is None:
            try:
                _, _, alpha, beta, _, _ = linalg.ordqz(self.A, self.E, output="complex")
                finite = np.isfinite(alpha) & np.isfinite(beta)
                scale = max(1.0, float(np.max(np.abs(alpha[finite])) if np.any(finite) else 1.0))
                bad = finite & (np.abs(alpha) < tol * scale) & (np.abs(beta) < tol * scale)
                self._regular = not bool(np.any(bad))
            except linalg.LinAlgError:
                self._regular = False
        return self._regular

    # ---- 谱指标 ----
    def kappa(self, zero_tol: float = 1e-12) -> float:
        """κ_t = max|Re λ|/min|Re λ|，去掉守恒零模态。

        规则（DESIGN.md §0）：只去掉 |λ| < zero_tol·max|λ| 的数值零模态。
        物理慢模态（哪怕 τ > 观测窗）保留——它正是硬件窗口装不下的原因。
        有正实部模态时返回 inf。
        """
        w = self.finite_eigs()
        if w.size == 0:
            return float("nan")
        scale = max(1.0, float(np.max(np.abs(w))))
        if np.any(np.real(w) > 1e-10 * scale):
            return float("inf")
        r = -np.real(w)
        r = r[r > zero_tol * scale]
        if r.size < 2:
            return 1.0
        return float(r.max() / r.min())

    def has_imaginary(self, zero_tol: float = 1e-9) -> bool:
        """是否存在落在虚轴上的有限模态（含守恒零模态）。"""
        w = self.finite_eigs()
        if w.size == 0:
            return False
        scale = max(1.0, float(np.max(np.abs(w))))
        return bool(np.any(np.abs(np.real(w)) <= zero_tol * scale))

    def is_stable(self) -> bool:
        """所有有限模态严格在左半平面。零模态返回 False。"""
        w = self.finite_eigs()
        if w.size == 0:
            return True
        scale = max(1.0, float(np.max(np.abs(w))))
        return bool(np.all(np.real(w) < -1e-10 * scale))

    def bandwidth(self) -> float:
        """B_req = max|λ| / 2π（Hz）。"""
        w = self.finite_eigs()
        if w.size == 0:
            return float("nan")
        return float(np.max(np.abs(w)) / (2.0 * np.pi))

    def spectral_radius(self) -> float:
        w = self.finite_eigs()
        return float(np.max(np.abs(w))) if w.size else 0.0

    def stable_projection(self, zero_tol: float = 1e-12):
        """QZ 排序取稳定有限谱的 deflating subspace 上的降阶系统。

        左投影必须用 Q（不是 Z）：ordqz 返回 Qᵀ A Z = AA、Qᵀ E Z = BB（块上三角），
        变换后的方程是 BB·ż = AA·z + QᵀB·u，状态坐标在 Z 里但方程行空间在 Q 里。
        用 Z1ᵀ 当左投影只有 Q=Z（对称 pencil / E=I 且 A 对称）时才等价；
        非对称奇异 DAE（如 P0-E 反例的推广）下会留下交叉项、给出错误的降阶系统。

        Q1 = Q[:, :k] 张成稳定有限谱的左 deflating subspace：
        E_r = Q1ᵀ E Z1、A_r = Q1ᵀ A Z1、B_r = Q1ᵀ B、C_r = C Z1。
        P（状态协方差）回到全空间用 P = Z1 Pr Z1ᵀ（右子空间）。
        守恒零模态（以及不稳定/无限模态）被投影掉，降阶系统正则且严格稳定。
        返回 (Descriptor, k)；k=0 时返回 (None, 0)。
        """
        red, k, _ = self._stable_projection_z1(zero_tol)
        return red, k

    def _stable_projection_z1(self, zero_tol: float = 1e-12):
        """stable_projection 的私有变体：额外返回右 deflating subspace Z1。

        Z1 的列是稳定有限谱子空间的坐标基（x = Z1·z）。resolvent 的度量
        投影需要它：M_red = Z1ᵀ M Z1 才能把能量度量带到降阶系统上
        （peak_with_projection 用）。公开签名保持 (red, k) 不变。
        """
        w = self.finite_eigs()
        scale = max(1.0, float(np.max(np.abs(w))) if w.size else 1.0)
        sort = lambda a, b: stable_finite_mask(a, b, zero_tol, scale)
        _, _, alpha, beta, Q, Z = linalg.ordqz(self.A, self.E, sort=sort, output="real")
        k = int(np.count_nonzero(sort(alpha, beta)))
        if k == 0:
            return None, 0, None
        Q1, Z1 = Q[:, :k], Z[:, :k]
        red = Descriptor(Q1.T @ self.E @ Z1, Q1.T @ self.A @ Z1,
                         Q1.T @ self.B, self.C @ Z1, self.D)
        return red, k, Z1

    # ---- 频响 ----
    def freqresp(self, omega) -> np.ndarray:
        """H(jω)，返回 (nω, p, m)；标量 ω 返回 (p, m)。"""
        scalar = np.isscalar(omega)
        w = np.atleast_1d(np.asarray(omega, dtype=complex))
        if self._diag is not None:
            inv = 1.0 / (1j * w[:, None] - self._diag[None, :])
            H = np.einsum("pi,wi,im->wpm", self.C, inv, self.B) + self.D
        else:
            H = np.empty((w.size, self.p, self.m), dtype=complex)
            for i, wi in enumerate(w):
                M = 1j * wi * self.E - self.A
                try:
                    X = linalg.solve(M, self.B)
                except linalg.LinAlgError:
                    X = linalg.lstsq(M, self.B)[0]
                H[i] = self.C @ X + self.D
        return H[0] if scalar else H

    def sigma_min(self, omega) -> np.ndarray:
        """σ_min(jωE − A) 的网格；对角系统用 O(n) 公式。"""
        w = np.atleast_1d(np.asarray(omega, dtype=float))
        if self._diag is not None:
            return np.min(np.abs(1j * w[:, None] - self._diag[None, :]), axis=1).real
        out = np.empty(w.size)
        for i, wi in enumerate(w):
            M = 1j * wi * self.E - self.A
            # svdvals 在本机 OpenBLAS 上对 128×128 复矩阵要 6 s；Gram + eigvalsh 快 40×。
            e = np.linalg.eigvalsh(M.conj().T @ M)
            out[i] = np.sqrt(max(float(e[0]), 0.0))
        return out

    def check(self) -> dict:
        """一次性自检，返回可写进 JSON 的字典。"""
        w = self.finite_eigs()
        return dict(
            n=self.n, m=self.m, p=self.p,
            regular=bool(self.is_regular()) if self.n <= 256 else None,
            n_finite=int(w.size),
            n_infinite=int(self.n - w.size),
            kappa=self.kappa(),
            bandwidth_hz=self.bandwidth(),
            spectral_radius=self.spectral_radius(),
            stable=bool(self.is_stable()),
        )
