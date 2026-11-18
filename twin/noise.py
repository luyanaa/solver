"""噪声源模型(twin 层):有色噪声、白极限、相关矩阵 ρ、replay(DESIGN.md §1.3 噪声合同)。

这是噪声**源**模型,不是 SDE 积分器——积分器属于 bench/(那里是唯一允许放
solver 的地方,见 AGENTS.md 分层规则)。

PSD 约定(写死,因为差 2× 就是 3 dB,§1.3):

    双边谱 S(f) 定义在 f ∈ (−∞,+∞),且 ∫ S df = E[ξ²](方差守恒,自洽标定)。
    单边谱 = 2 × 双边谱,只定义在 f ≥ 0。
    本模块的一切文档与 `psd()` 输出一律用**双边谱**;要用单边谱的场合必须
    显式写出 "one-sided = 2 × two-sided"。

ColorNoise 的平稳统计(由下面的精确更新直接推出):

    E[ξ²] = D/τ_c          (D = intensity 是**白等效**强度,见下)
    R(s)   = (D/τ_c)·e^{−|s|/τ_c}
    S_ξ(f) = 2D / (1 + (2π f τ_c)²)      [双边]
    单边   = 4D / (1 + (2π f τ_c)²)

    平坦段(双边)= 2D,3 dB 点在 f = 1/(2πτ_c) =: B_noise。
    τ_c → 0 时,该源驱动任何积分器都趋于白噪声 √(2D)·η(双边平坦谱 2D)
    ——这就是 D 叫"白等效强度"的原因:它直接对齐 OU 方程
    dx = −λx dt + √(2D) dW 的 D,白极限下方差 D/λ,无重标定。

精确更新(推导,不是猜的):把 OU 源写成 AR(1) 并要求平稳矩匹配。连续
    τ_c ξ̇ = −ξ + √(2D/τ_c)·τ_c dW/dt ⇒ dξ = −ξ/τ_c dt + √(2D/τ_c²) dW
的精确转移是

    ξ_{n+1} = ξ_n·e^{−dt/τ_c} + √((D/τ_c)·(1 − e^{−2 dt/τ_c}))·N

验证:平稳方差 v = s²/(1−e^{−2dt/τ_c}) = D/τ_c ✓;自相关衰减率 e^{−dt/τ_c} ✓。

用法(驱动 ẋ = f(x) + ξ):

    dt ≪ τ_c 时(工程上 dt ≲ 0.1 τ_c)ξ_n·dt ≈ ∫_0^dt ξ ds,直接用。
    步内积分的精确方差是(对 R(s) 积分可得,用于 dt 不 ≪ τ_c 的场合)

        Var(∫_0^dt ξ ds) = 2D·[dt − τ_c(1 − e^{−dt/τ_c})]   → 白极限 2D·dt ✓

    `increment_variance(dt)` 返回这个数;需要精确时把每步驱动缩放到它。

replay 模式(§6.12 CRN 反问题):`start_capture()` 记录每一步实际消费的
噪声向量,`stop_capture()` 取走;`replay(draws)` 之后 `sample()` 按原顺序
重放同一串噪声 → 两次参数评估见到逐点相同的一条实现。同 seed 重实例化
等价,但 capture/replay 不依赖全局 seed 顺序,更稳。
"""

from __future__ import annotations

import numpy as np

__all__ = ["ColorNoise", "white_matrix", "bandwidth_drop_factor"]


def _psi(dt: float, tau_c: float) -> float:
    """Var(∫_0^dt ξ ds)/2D = dt − τ_c(1−e^{−dt/τ_c}),白极限 → dt。"""
    if tau_c <= 0.0:
        return dt
    return dt - tau_c * (1.0 - np.exp(-dt / tau_c))


class ColorNoise:
    """OU 滤波有色噪声源 ξ̇ = −ξ/τ_c + √(2D)/τ_c · η(见模块 docstring 约定)。

    tau_c <= 0 即白极限:sample(dt) 返回 √(2D/dt)·L·N,使 ξ·dt 的方差
    = 2D·dt,与 √(2D)·ΔW 逐点一致。
    dim>1 时用相关矩阵 rho(缺省单位阵);S_ij(f) = 2D·ρ_ij/(1+(2πfτ_c)²)。
    """

    def __init__(self, tau_c: float, intensity: float, dim: int = 1,
                 rho=None, rng: np.random.Generator | None = None):
        if intensity <= 0:
            raise ValueError("intensity D must be > 0")
        self.tau_c = float(tau_c)
        self.intensity = float(intensity)
        self.dim = int(dim)
        self.rng = rng if rng is not None else np.random.default_rng()
        if rho is None:
            self.L = np.eye(self.dim)
        else:
            rho = np.asarray(rho, dtype=float)
            if rho.shape != (self.dim, self.dim):
                raise ValueError("rho must be dim×dim")
            if not np.allclose(rho, rho.T):
                raise ValueError("rho must be symmetric")
            if not np.allclose(np.diag(rho), 1.0):
                raise ValueError("rho must have unit diagonal")
            evals = np.linalg.eigvalsh(rho)
            if evals.min() < -1e-10:
                raise ValueError("rho must be PSD")
            self.L = np.linalg.cholesky(rho + 1e-12 * np.eye(self.dim))
        self.xi = np.zeros(self.dim)
        self._cap: list[np.ndarray] | None = None
        self._rep: np.ndarray | None = None
        self._rep_i = 0

    # -- 噪声合同字段 -----------------------------------------------------
    @property
    def B_noise(self) -> float:
        """PSD 平坦带宽(Hz):双边谱的 3 dB 点 1/(2πτ_c);白极限 → ∞。"""
        return np.inf if self.tau_c <= 0 else 1.0 / (2.0 * np.pi * self.tau_c)

    def psd(self, f):
        """双边 PSD S(f) = 2D/(1+(2πfτ_c)²);单边请自乘 2(见模块 docstring)。"""
        f = np.asarray(f, dtype=float)
        if self.tau_c <= 0:
            return np.full_like(f, 2.0 * self.intensity)
        return 2.0 * self.intensity / (1.0 + (2.0 * np.pi * f * self.tau_c) ** 2)

    def increment_variance(self, dt: float) -> float:
        """Var(∫_0^dt ξ ds)(dim=1、ρ=I 时;相关时按 ρ 缩放)。"""
        return 2.0 * self.intensity * _psi(dt, self.tau_c)

    def describe(self) -> dict:
        """§1.3 噪声合同五字段,缺一项就不能编译。"""
        return dict(
            S_xi="two-sided 2D/(1+(2πfτc)²), flat = 2D (one-sided = 2×)",
            tau_c=self.tau_c,
            rho="I" if self.dim == 1 else "user ρ_ij (Cholesky L)",
            B_noise=("inf (white limit)" if self.tau_c <= 0
                     else "%.6g Hz = 1/(2πτc)" % self.B_noise),
            mode="physical",
        )

    # -- 采样 --------------------------------------------------------------
    def _draw(self) -> np.ndarray:
        """一步的标准正态(已过 L);capture / replay 都作用在这一层。"""
        if self._rep is not None:
            if self._rep_i >= self._rep.shape[0]:
                raise RuntimeError("replay buffer exhausted")
            z = self._rep[self._rep_i]
            self._rep_i += 1
            return z
        z = self.L @ self.rng.standard_normal(self.dim)
        if self._cap is not None:
            self._cap.append(z)
        return z

    def sample(self, dt: float) -> np.ndarray:
        """推进 dt 并返回该格点的 ξ(形状 (dim,));驱动时用 ξ·dt。"""
        if dt <= 0:
            raise ValueError("dt must be > 0")
        z = self._draw()
        if self.tau_c <= 0.0:
            # 白极限:ξ·dt ≡ √(2D)·ΔW
            self.xi = np.sqrt(2.0 * self.intensity / dt) * z
        else:
            e = np.exp(-dt / self.tau_c)
            s = np.sqrt((self.intensity / self.tau_c) * (1.0 - e * e))
            self.xi = self.xi * e + s * z
        return self.xi

    # -- replay(CRN 用) ---------------------------------------------------
    def start_capture(self) -> None:
        self._cap = []

    def stop_capture(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("start_capture() first")
        draws = np.asarray(self._cap)
        self._cap = None
        return draws

    def replay(self, draws: np.ndarray) -> None:
        """之后 sample() 逐条重放 draws((n_steps, dim))。"""
        draws = np.asarray(draws, dtype=float)
        if draws.ndim != 2 or draws.shape[1] != self.dim:
            raise ValueError("draws must be (n_steps, dim)")
        self._rep = draws
        self._rep_i = 0
        self.xi = np.zeros(self.dim)


def white_matrix(rng: np.random.Generator, intensity: float, dt: float,
                 shape) -> np.ndarray:
    """ColorNoise 白极限(τ_c→0)的向量化版本。

    返回形状 shape 的 ξ 样本,ξ·dt ≡ √(2D)·ΔW(每样本方差 2D/dt)。
    这是 sample() 在 τ_c≤0 时的批量等价,供 bench 大 ensemble 用。
    """
    return np.sqrt(2.0 * intensity / dt) * rng.standard_normal(shape)


def bandwidth_drop_factor(fc_over_b: float) -> float:
    """白噪声经理想低通(截止 f_c)驱动一阶系统的方差比(DESIGN.md §2)。

    Var/Var_white = (2/π)·arctan(f_c/B_req),B_req = λ/2π。
    f_c/B = 10 → 0.93655(降 6.4%);f_c/B = 100 → 0.99363(降 0.64%)。
    这是 §6.11 噪声带宽验收的解析预测。
    """
    return 2.0 / np.pi * np.arctan(float(fc_over_b))
