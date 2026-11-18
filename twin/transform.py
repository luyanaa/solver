"""表示层原语：smoothness sensor、WENO 三算子、lifting wavelet、transform contract（DESIGN.md §5.7–§5.9）。

    S:   u → σ(x)          singularity / smoothness detector
    T_σ: u → z             state-dependent representation / scale transform
    F_σ: z → F_{i+1/2}     conservative flux operator

三条规则写死在这里，bench 只负责验收：

1. σ 是连续坐标，不是 if/else；`sigma_from_indicator` 是 sigmoid，
   `SigmaTracker` 加迟滞和速率限制，避免阈值附近的极限环。
2. 表示混合必须发生在**同一个 face flux** 上；先混状态再各自算通量，
   守恒误差会变成 O(σ) 且不可观测。
3. forward / inverse / adjoint 是三个对象。biorthogonal 时
   `W_inv ≠ W_star`，用 synthesis 当 adjoint 会让 forward 解看起来正确、
   梯度慢慢偏。

wavelet 用 lifting 实现，因为它的 predict/update 都是 local 且可转置：
`(AB)^* = B^* A^*`，所以 adjoint 只是反向顺序 + 转置/共轭。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "smoothness_indicator", "sigma_from_indicator", "SigmaTracker",
    "weno5_face_flux", "central_face_flux", "rusanov_face_flux",
    "muscl_face_flux", "blend_face_flux", "rhs_from_face_flux",
    "conservation_drift", "count_threshold_crossings",
    "tracker_rate_bounds",
    "LiftingWavelet", "stationary_haar", "stationary_haar_inverse",
    "gram_calibrate", "isometry_error", "pr_error",
    "TransposeableEdge", "edge_matrix", "transpose_apply",
    "adjoint_asym_bound",
    "quadratic_gradient", "finite_difference_gradient",
    "adjoint_ode_gradient", "rk4_matrix", "scale_poles", "scale_time_error",
]

_TINY = 1e-300


# ---------------------------------------------------------------------------
# S: singularity / smoothness detector
# ---------------------------------------------------------------------------
def smoothness_indicator(u: np.ndarray, eps: float = 1e-12,
                         rel_eps: float = 1e-2) -> np.ndarray:
    """归一化二阶差分，周期边界。

        S_i = |u_{i+1} - 2u_i + u_{i-1}| / (|u_{i+1}-u_i| + |u_i-u_{i-1}| + eps)

    smooth → 0，shock → 1。它只需要 T2 的差分/绝对值/归一化，不需要除法器阵列。

    `eps` 必须是**相对场幅的 floor**，不能是 1e-12 的绝对数：光滑极值处
    一阶差分为零，绝对 eps 会让 0/0 饱和成 shock。默认
    `eps = max(eps, rel_eps * (max(u)-min(u)))`，rel_eps≈1% 是硬件偏置的
    自然尺度（相当于 ADC/DAC 满量程的一个小分数）。
    """
    u = np.asarray(u, dtype=float)
    span = float(np.max(u) - np.min(u))
    floor = max(float(eps), float(rel_eps) * span)
    up = np.roll(u, -1)
    um = np.roll(u, 1)
    num = np.abs(up - 2.0 * u + um)
    den = np.abs(up - u) + np.abs(u - um) + floor
    return num / den


def sigma_from_indicator(s: np.ndarray, beta0: float = 0.5,
                         k: float = 12.0) -> np.ndarray:
    """把 sensor 映射成 [0,1] 的 representation coordinate。"""
    s = np.asarray(s, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (s - beta0)))


@dataclass
class SigmaTracker:
    """带迟滞与速率限制的 σ。硬件上就是 supervisor 的判决更新律。

    attack/release 必须不对称：σ 上升要快（抓住激波），下降要慢（释放）。
    对称速率会和 representation 反馈形成极限环：σ 降 → 中心格式 → 振荡 →
    S 升 → σ 升 → MUSCL → S 降，如此循环。`release_rate` 就是硬件 hold time。
    """

    beta0: float = 0.5
    k: float = 12.0
    hysteresis: float = 0.01
    max_rate: float = 500.0
    release_rate: float | None = None
    sigma: np.ndarray | None = None

    # jacobian_coupling(dF) — 表示反馈的线性化说明（仅文档级，不新增机制）：
    #
    # 当 σ(u) 调制 face flux 时，F_σ = (1−σ_f)·F_smooth + σ_f·F_shock
    # （σ_f = 两侧 cell 平均）。对状态求导，effective Jacobian 多出
    #
    #     ∂F_σ/∂σ · ∂σ/∂u = (F_shock − F_smooth) · ∂σ_f/∂u
    #
    # 这一项就是 representation-feedback：σ 降 → 中心格式 → 振荡 → S 升 →
    # σ 升 → MUSCL → S 降 的闭环。它使极限环不再是纯经验观察，而是一个
    # **线性稳定性问题**：环增益 ∝ ‖(F_shock−F_smooth)·∂σ/∂u‖ 随
    # attack/release 速率与 hysteresis 改变（见 tracker_rate_bounds 的推导）。
    # tracker.update 的速率限制把这个反馈限制在有限带宽内；若把该耦合项
    # 显式线性化，chatter 判据等价于闭环特征值的实部符号。

    def reset(self):
        self.sigma = None

    def update(self, s: np.ndarray, dt: float) -> np.ndarray:
        raw = sigma_from_indicator(s, self.beta0, self.k)
        if self.sigma is None:
            self.sigma = raw.copy()
            return self.sigma
        delta = raw - self.sigma
        # deadband：小于 hysteresis 的差不动，避免在阈值附近来回抖
        small = np.abs(delta) < self.hysteresis
        delta = np.where(small, 0.0, delta - np.sign(delta) * self.hysteresis)
        rise = max(self.max_rate * dt, 0.0)
        fall = max((self.release_rate if self.release_rate is not None
                    else self.max_rate) * dt, 0.0)
        step = np.where(delta > 0.0, rise, fall)
        self.sigma = np.clip(self.sigma + np.clip(delta, -step, step), 0.0, 1.0)
        return self.sigma


# ---------------------------------------------------------------------------
# F_σ: face fluxes（conservative by topology）
# ---------------------------------------------------------------------------
def _weno5_reconstruct(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """WENO5-JS 对 cell averages 的 i+1/2 面重构（周期边界）。"""
    vm2 = np.roll(v, 2)
    vm1 = np.roll(v, 1)
    vp1 = np.roll(v, -1)
    vp2 = np.roll(v, -2)

    v0 = (2.0 * vm2 - 7.0 * vm1 + 11.0 * v) / 6.0
    v1 = (-vm1 + 5.0 * v + 2.0 * vp1) / 6.0
    v2 = (2.0 * v + 5.0 * vp1 - vp2) / 6.0

    b0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v) ** 2 \
        + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v) ** 2
    b1 = (13.0 / 12.0) * (vm1 - 2.0 * v + vp1) ** 2 \
        + 0.25 * (vm1 - vp1) ** 2
    b2 = (13.0 / 12.0) * (v - 2.0 * vp1 + vp2) ** 2 \
        + 0.25 * (3.0 * v - 4.0 * vp1 + vp2) ** 2

    d0, d1, d2 = 0.1, 0.6, 0.3
    a0 = d0 / (eps + b0) ** 2
    a1 = d1 / (eps + b1) ** 2
    a2 = d2 / (eps + b2) ** 2
    inv = 1.0 / (a0 + a1 + a2)
    return (a0 * v0 + a1 * v1 + a2 * v2) * inv


def weno5_face_flux(u: np.ndarray, flux, alpha: float,
                    eps: float = 1e-6) -> np.ndarray:
    """WENO5 + global Lax–Friedrichs splitting；返回 F_{i+1/2}。"""
    u = np.asarray(u, dtype=float)
    f = flux(u)
    alpha = max(float(alpha), 1e-12)
    fp = 0.5 * (f + alpha * u)
    fm = 0.5 * (f - alpha * u)
    f_plus = _weno5_reconstruct(fp, eps)
    f_minus = np.roll(_weno5_reconstruct(fm[::-1], eps)[::-1], -1)
    return f_plus + f_minus


def central_face_flux(u: np.ndarray, flux) -> np.ndarray:
    """4 阶中心面插值；smooth 区用它，shock 区会被 σ 混掉。"""
    u = np.asarray(u, dtype=float)
    uf = (-np.roll(u, 1) + 7.0 * u + 7.0 * np.roll(u, -1)
          - np.roll(u, -2)) / 12.0
    return flux(uf)


def rusanov_face_flux(u: np.ndarray, flux, alpha: float) -> np.ndarray:
    """一阶 robust FV，作为最保守的 shock 候选。"""
    u = np.asarray(u, dtype=float)
    f = flux(u)
    alpha = max(float(alpha), 1e-12)
    return 0.5 * (f + np.roll(f, -1)) - 0.5 * alpha * (np.roll(u, -1) - u)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def muscl_face_flux(u: np.ndarray, flux, alpha: float) -> np.ndarray:
    """二阶 MUSCL + minmod + Rusanov；比一阶 Rusanov 尖锐，比 WENO5 少振荡。

    这是 F_σ 在 shock 区的默认候选：TVD，face flux 单值，守恒由拓扑保证。
    """
    u = np.asarray(u, dtype=float)
    alpha = max(float(alpha), 1e-12)
    slope = _minmod(u - np.roll(u, 1), np.roll(u, -1) - u)
    uL = u + 0.5 * slope
    uR = np.roll(u, -1) - 0.5 * np.roll(slope, -1)
    return 0.5 * (flux(uL) + flux(uR)) - 0.5 * alpha * (uR - uL)


def blend_face_flux(F_smooth: np.ndarray, F_shock: np.ndarray,
                    sigma: np.ndarray) -> np.ndarray:
    """在**同一个 face** 上混合；σ 是 cell-centered，face 取两侧平均。"""
    F_smooth = np.asarray(F_smooth, dtype=float)
    F_shock = np.asarray(F_shock, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sf = 0.5 * (sigma + np.roll(sigma, -1))
    return (1.0 - sf) * F_smooth + sf * F_shock


def rhs_from_face_flux(F: np.ndarray, dx: float) -> np.ndarray:
    """C_i dV_i/dt = F_{i-1/2} - F_{i+1/2}；F[i] 是 i+1/2。"""
    F = np.asarray(F, dtype=float)
    return -(F - np.roll(F, 1)) / dx


def conservation_drift(u: np.ndarray, u0: np.ndarray,
                       dx: float) -> float:
    """相对质量漂移；用于检查 telescoping 是否被表示切换破坏。"""
    u = np.asarray(u, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    ref = max(abs(np.sum(u0) * dx), _TINY)
    return abs((np.sum(u) - np.sum(u0)) * dx) / ref


def count_threshold_crossings(history: np.ndarray,
                              threshold: float = 0.5) -> np.ndarray:
    """每个 cell 的 σ 穿越次数。移动激波正常是 2 次（进/出），chatter 会远大于 2。"""
    above = np.asarray(history) > threshold
    return np.sum(np.abs(np.diff(above.astype(np.int8), axis=0)), axis=0)


def tracker_rate_bounds(dx: float, nu_num: float, s_shock: float,
                        dsigma_hyst: float) -> tuple[float, float]:
    """SigmaTracker attack/release 速率的解析可行域（外部 review 的可证伪预测）。

    返回 (r_attack_min, r_release_max)。推导（可证伪：数字错了就推翻）：

    1. **advection 下界（attack）**。激波每扫过一个 cell，表示必须在该
       cell 内完成 σ 的上升，否则新到达的激波先被中心格式处理一拍，产生
       振荡 → S 再升 → 二次切换（chatter）。一个 cell 的 advection 时间
           τ_adv = Δx / |s_shock|，
       所以 attack 速率必须满足
           r_attack ≫ 1/τ_adv = |s_shock| / Δx。
       “≫”按工程取 ~3× 余量解读；本函数只返回硬下界本身。

    2. **耗散上界（release）**。σ 一旦升高，表示反馈闭环（见
       SigmaTracker 的 jacobian_coupling 注释）靠格式自身的数值黏性把
       振荡压下去，之后才允许 σ 下降。激波区 Rusanov/MUSCL 的数值黏性
           ν_num ≈ ½·alpha·Δx
       在网格尺度 k_max = π/Δx 上的衰减时间为
           τ_dissip = Δx² / (π²·ν_num)，
       （扩散 e^{−ν k² t} 在 k=π/Δx 处衰减 e^{−1} 的时间）。若 release
       快于该时间，格式还没把振荡耗散掉 σ 就回退，闭环重振 → 极限环。
       一次完整迟滞回环要求 σ 变化 ≥ Δσ_hyst，所以
           r_release ≤ Δσ_hyst / τ_dissip = π²·ν_num·Δσ_hyst / Δx²。

    注意 Δσ_hyst 是 **σ 坐标**里的迟滞（SigmaTracker.hysteresis），
    不是 S 的；用它时要用 tracker 的实际配置，不是默认值。

    对 bench 的 Burgers（N=256, Δx=1/256, ν_num≈3e-3, |s|≈0.5），
    外部 review 用 Δσ_hyst≈0.008 预测 r_attack_min≈128, r_release_max≈15；
    用 bench 实际的 hysteresis=0.08 时 r_release_max 放大 10×。两种取法
    （400, 10）都在可行域内 — 这是本函数的可检验预测。
    """
    dx = float(dx)
    nu_num = float(nu_num)
    s_shock = float(s_shock)
    dsigma_hyst = float(dsigma_hyst)
    if dx <= 0 or nu_num <= 0 or dsigma_hyst <= 0:
        raise ValueError("dx, nu_num, dsigma_hyst 必须为正")
    tau_adv = dx / abs(s_shock) if s_shock != 0 else float("inf")
    tau_dissip = dx * dx / (np.pi ** 2 * nu_num)
    r_attack_min = 1.0 / tau_adv if tau_adv > 0 else 0.0
    r_release_max = dsigma_hyst / tau_dissip
    return (r_attack_min, r_release_max)


# ---------------------------------------------------------------------------
# T: lifting wavelet（forward / inverse / adjoint）
# ---------------------------------------------------------------------------
def _haar_fwd(x: np.ndarray):
    even = x[0::2]
    odd = x[1::2]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    s = (even + odd) * inv_sqrt2
    d = (odd - even) * inv_sqrt2
    return s, d


def _haar_inv(s: np.ndarray, d: np.ndarray):
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    even = (s - d) * inv_sqrt2
    odd = (s + d) * inv_sqrt2
    out = np.empty(even.size + odd.size, dtype=float)
    out[0::2] = even
    out[1::2] = odd
    return out


def _cdf53_fwd(x: np.ndarray):
    """CDF 5/3 lifting（周期边界）。biorthogonal，W_inv ≠ W_star。"""
    even = x[0::2]
    odd = x[1::2]
    d = odd - 0.5 * (even + np.roll(even, -1))
    s = even + 0.25 * (np.roll(d, 1) + d)
    return s, d


def _cdf53_inv(s: np.ndarray, d: np.ndarray):
    even = s - 0.25 * (np.roll(d, 1) + d)
    odd = d + 0.5 * (even + np.roll(even, -1))
    out = np.empty(even.size + odd.size, dtype=float)
    out[0::2] = even
    out[1::2] = odd
    return out


class LiftingWavelet:
    """1D 多级 lifting wavelet，显式矩阵只用于合同验收（n 小）。"""

    def __init__(self, n: int, kind: str = "haar", levels: int = 3):
        if n <= 0 or n % (1 << levels):
            raise ValueError("n 必须能被 2**levels 整除")
        if kind not in ("haar", "cdf53"):
            raise ValueError("kind 必须是 haar 或 cdf53")
        self.n = int(n)
        self.kind = kind
        self.levels = int(levels)
        self.W = np.column_stack([self.forward(e) for e in np.eye(self.n)])
        self.W_inv = np.linalg.inv(self.W)
        self.W_star = self.W.T

    # -- level ------------------------------------------------------------
    def _level_forward(self, x: np.ndarray):
        return _haar_fwd(x) if self.kind == "haar" else _cdf53_fwd(x)

    def _level_inverse(self, s: np.ndarray, d: np.ndarray):
        return _haar_inv(s, d) if self.kind == "haar" else _cdf53_inv(s, d)

    # -- multilevel -------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        details = []
        for _ in range(self.levels):
            a, d = self._level_forward(a)
            details.append(d)
        return np.concatenate([a] + details[::-1])

    def inverse(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        approx_size = self.n >> self.levels
        a = c[:approx_size].copy()
        idx = approx_size
        details = []
        for l in range(self.levels, 0, -1):
            size = self.n >> l
            details.append(c[idx:idx + size].copy())
            idx += size
        for d in details:
            a = self._level_inverse(a, d)
        return a

    def adjoint(self, c: np.ndarray) -> np.ndarray:
        return self.W_star @ np.asarray(c, dtype=float)

    # -- contract metrics -------------------------------------------------
    def isometry_error(self, metric: np.ndarray | None = None) -> float:
        target = np.eye(self.n) if metric is None else np.asarray(metric)
        g = self.W.T @ self.W - target
        return float(np.linalg.norm(g, "fro") / np.sqrt(self.n))

    def pr_error(self) -> float:
        e = self.W_inv @ self.W - np.eye(self.n)
        return float(np.linalg.norm(e, "fro") / np.sqrt(self.n))

    def adjoint_error(self) -> float:
        den = max(np.linalg.norm(self.W_inv, "fro"), _TINY)
        return float(np.linalg.norm(self.W_inv - self.W_star, "fro") / den)

    def dot_error(self, x: np.ndarray, y: np.ndarray) -> float:
        lhs = np.vdot(self.W @ x, y)
        rhs = np.vdot(x, self.W_star @ y)
        den = max(np.linalg.norm(x) * np.linalg.norm(y), _TINY)
        return float(abs(lhs - rhs) / den)

    @property
    def noise_gain(self) -> float:
        return float(np.linalg.norm(self.W, 2) *
                     np.linalg.norm(self.W_inv, 2))

    @property
    def condition(self) -> float:
        return float(np.linalg.cond(self.W))


def stationary_haar(x: np.ndarray, levels: int):
    """Undecimated/stationary Haar：保留所有 location，只改变 scale。

    返回 (approx, [d_1, ..., d_L])，每个数组都是 length N；redundancy = L+1。
    不做 ↓2，所以不会引入隐藏的 S/H 和 aliasing。
    """
    a = np.asarray(x, dtype=float).copy()
    details = []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for j in range(levels):
        shift = 1 << j
        d = (a - np.roll(a, shift)) * inv_sqrt2
        a = (a + np.roll(a, shift)) * inv_sqrt2
        details.append(d)
    return a, details


def stationary_haar_inverse(a: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(a, dtype=float).copy()
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for d in reversed(details):
        out = (out + d) * inv_sqrt2
    return out


# ---------------------------------------------------------------------------
# TransformContract：Gram 标定
# ---------------------------------------------------------------------------
def isometry_error(T: np.ndarray, metric: np.ndarray | None = None) -> float:
    T = np.asarray(T, dtype=float)
    target = np.eye(T.shape[1]) if metric is None else np.asarray(metric)
    g = T.T @ T - target
    return float(np.linalg.norm(g, "fro") / np.sqrt(T.shape[1]))


def pr_error(T: np.ndarray, T_inv: np.ndarray) -> float:
    T = np.asarray(T, dtype=float)
    T_inv = np.asarray(T_inv, dtype=float)
    e = T_inv @ T - np.eye(T.shape[1])
    return float(np.linalg.norm(e, "fro") / np.sqrt(T.shape[1]))


def _inv_sqrt_psd(G: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (G + G.T))
    w = np.maximum(w, 1e-15)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def gram_calibrate(W_hw: np.ndarray, side: str = "right") -> dict:
    """测 G_T = T̃^* T̃，做 whitening，使 T_cal^* T_cal ≈ I。

    数学上要区分左/右：
      right: T_cal = T̃ G_T^{-1/2},  G_T = T̃^*T̃   → T_cal^*T_cal = I
      left : T_cal = (T̃T̃^*)^{-1/2} T̃              → T_cal^*T_cal = I
    两者一般不同，只有 T̃ 可交换时才一样。DESIGN §5.9 的 G_T 是右 Gram，
    所以默认 right；left 保留给“只能在后级做增益修正”的硬件。

    Δ_asym 预算（transpose-reuse 合法性合同）：`adjoint_before` 里的
    A↔Aᵀ 一致性（~2.7e-16，§6.16）是**理想开关假设**下的数——硬件里
    forward i→j 与 reverse j→i 的晶体管级传输不相等，Ã_adj = Aᵀ+Δ_asym。
    相对梯度误差界：err ≤ Γ_proj(A)·‖Δ_asym‖·T_obs（resolvent 在观测窗
    内放大不对称）。预算的实测（case G, bench/run_sde.py）：误差随
    ‖Δ_asym‖ 线性（log-log 斜率≈1）、落在界的 ~0.05–0.5 倍带内。
    用法：‖Δ_asym‖·Γ_proj·T_obs ≪ 目标梯度精度时才能免费复用 forward
    阵列做 adjoint；否则必须先跑本函数的 Gram whitening 把 T̃ 校成
    等距，再按 `adjoint_after` 判定。
    """
    W = np.asarray(W_hw, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W_hw 必须是方阵")
    if side == "right":
        G = W.T @ W
        K = _inv_sqrt_psd(G)
        T = W @ K
    elif side == "left":
        G = W @ W.T
        K = _inv_sqrt_psd(G)
        T = K @ W
    else:
        raise ValueError("side 必须是 right 或 left")
    T_inv = np.linalg.inv(T)
    return dict(
        G=G,
        K=K,
        T_cal=T,
        T_inv=T_inv,
        side=side,
        isometry_before=isometry_error(W),
        isometry_after=isometry_error(T),
        pr_after=pr_error(T, T_inv),
        adjoint_before=float(np.linalg.norm(np.linalg.inv(W) - W.T)
                             / max(np.linalg.norm(np.linalg.inv(W)), _TINY)),
        adjoint_after=float(np.linalg.norm(T_inv - T.T)
                            / max(np.linalg.norm(T_inv), _TINY)),
        cond=float(np.linalg.cond(G)),
        noise_gain=float(np.linalg.norm(K, 2)),
    )


# ---------------------------------------------------------------------------
# Transposeable edges / adjoint
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransposeableEdge:
    """forward: I_j += g V_i；adjoint: I_i += g V_j。

    transpose-reuse 合同：这里假设同一条 gm 边正反向增益严格相等
    （理想开关）。真实器件 forward/reverse 传输差 Δ_asym，其梯度预算
    是 err ≤ Γ_proj(A)·‖Δ_asym‖·T_obs（推导与实测见 gram_calibrate
    docstring 与 bench/run_sde.py case G）——不对称预算超差时，
    adjoint 必须走独立校准边或先 Gram whitening，不能白拿 forward。
    """

    i: int
    j: int
    gain: float
    transposeable: bool = True

    def apply(self, x: np.ndarray, out: np.ndarray,
              transpose: bool = False) -> None:
        if transpose:
            if not self.transposeable:
                raise ValueError("edge 不可转置")
            out[self.i] += self.gain * x[self.j]
        else:
            out[self.j] += self.gain * x[self.i]


def edge_matrix(edges, n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for e in edges:
        A[e.j, e.i] += e.gain
    return A


def transpose_apply(edges, x: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.zeros(n, dtype=float)
    for e in edges:
        e.apply(x, out, transpose=True)
    return out


def quadratic_gradient(A: np.ndarray, b: np.ndarray,
                       x: np.ndarray) -> np.ndarray:
    """J(x) = 0.5‖A x − b‖² 的解析梯度 = Aᵀ(Ax−b)。"""
    A = np.asarray(A, dtype=float)
    return A.T @ (A @ x - b)


def finite_difference_gradient(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (fun(xp) - fun(xm)) / (2.0 * eps)
    return g


def rk4_matrix(A: np.ndarray, dt: float) -> np.ndarray:
    """线性 ODE dx/dt = A x 的 RK4 一步矩阵。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n)
    return (I + dt * A + 0.5 * dt ** 2 * A @ A
            + (1.0 / 6.0) * dt ** 3 * A @ A @ A
            + (1.0 / 24.0) * dt ** 4 * A @ A @ A @ A)


def adjoint_asym_bound(gamma_proj: float, delta_norm: float,
                       T_obs: float) -> float:
    """Ã_adj = Aᵀ + Δ_asym 时 adjoint 梯度的相对误差上界。

    err ≤ Γ_proj(A)·‖Δ_asym‖·T_obs：resolvent 峰值把单步不对称放大，
    观测窗 T_obs 内累加。这是必要方向的一阶界（报告里同时给实测比值
    err/bound，case G）——bound 不是紧的，但不空转：case G 断言
    bound/100 ≤ err ≤ bound。
    """
    if gamma_proj <= 0 or delta_norm < 0 or T_obs < 0:
        raise ValueError("gamma_proj>0, delta_norm/T_obs≥0")
    return gamma_proj * delta_norm * T_obs


def adjoint_ode_gradient(A: np.ndarray, x0: np.ndarray, T: float,
                         nsteps: int, C: np.ndarray, y: np.ndarray) -> dict:
    """线性 ODE 的离散 adjoint。

    forward:  x_{k+1} = M x_k,  M = RK4(A, dt)
    cost:     J = 0.5‖C x_N − y‖²
    adjoint:  λ_k = Mᵀ λ_{k+1}, λ_N = Cᵀ(Cx_N − y)
    grad:     λ_0

    这里用的是离散 RK4 map 的转置，所以与 finite difference 应当到机器精度，
    不是“连续 adjoint 的近似”。checkpoint 量按保存全部状态计。
    """
    A = np.asarray(A, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    C = np.asarray(C, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = T / nsteps
    M = rk4_matrix(A, dt)
    xs = np.empty((nsteps + 1, x0.size), dtype=float)
    xs[0] = x0
    for k in range(nsteps):
        xs[k + 1] = M @ xs[k]
    lam = C.T @ (C @ xs[-1] - y)
    for k in range(nsteps, 0, -1):
        lam = M.T @ lam
    return dict(
        grad=lam,
        xT=xs[-1],
        checkpoint_bytes=int(xs.nbytes),
        checkpoint_states=int(xs.size),
    )


# ---------------------------------------------------------------------------
# scale-dependent poles：mode C vs mode g_m
# ---------------------------------------------------------------------------
def scale_poles(levels: int, C0: float = 1e-12, gm0: float = 1e-12,
                base: float = 2.0, mode: str = "C") -> dict:
    """Wavelet scale j 的 pole 阶梯。

    model:  λ_j = ν (2^j k0)² ∝ 4^j
    mode C:  C_j ∝ 4^{-j}, g_m 固定  → g_m 范围小，噪声 ∝ 4^j
    mode g_m:g_m,j ∝ 4^j, C 固定     → 噪声平，g_m 范围 4^L
    mode none: 两者都固定            → 高 j 模态慢 4^j，model time 错
    """
    j = np.arange(levels, dtype=float)
    if mode == "C":
        C = C0 * base ** (-2.0 * j)
        gm = np.full(levels, gm0)
    elif mode == "gm":
        C = np.full(levels, C0)
        gm = gm0 * base ** (2.0 * j)
    elif mode == "none":
        C = np.full(levels, C0)
        gm = np.full(levels, gm0)
    else:
        raise ValueError("mode 必须是 C、gm 或 none")
    omega = gm / C
    noise = 1.0 / C
    return dict(
        mode=mode, C=C, gm=gm, omega=omega, noise=noise,
        pole_spread=float(omega[-1] / omega[0]),
        gm_spread=float(gm[-1] / gm[0]),
        noise_spread=float(noise[-1] / noise[0]),
    )


def scale_time_error(levels: int, nu: float = 1.0, t: float = 1.0,
                     base: float = 2.0, mode: str = "C") -> dict:
    """比较 physical pole 与 model λ_j 在 t 时刻的衰减误差。"""
    C0 = 1e-12
    gm0 = nu * C0
    p = scale_poles(levels, C0=C0, gm0=gm0, base=base, mode=mode)
    j = np.arange(levels, dtype=float)
    lam = nu * base ** (2.0 * j)
    exact = np.exp(-lam * t)
    phys = np.exp(-p["omega"] * t)
    den = np.maximum.reduce([np.abs(exact), np.abs(phys),
                             np.full_like(exact, 1e-3)])
    rel = np.abs(phys - exact) / den
    p.update(dict(lambda_model=lam, exact=exact, physical=phys,
                  max_rel_error=float(np.max(rel))))
    return p
