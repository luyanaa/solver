"""Route 不是 wire 长度，是一条 transfer function（DESIGN.md §3.2、§6.15）。

    Y_c(s) = G_c + sC_c          （lumped π 模型；distributed 模型是 1/cosh√(sRC)）
    H_ij(s) = H_wire(s) · ∏_k H_buf,k(s) · e^{−sτ_ij}

每条 edge 的 closure 是四条，不是一条：

    gain       sup_ω | |H(jω)|/g_ij − 1 | < ε_g
    phase      sup_ω | ∠H(jω) + ωτ_ij |   < ε_φ
    noise      Σ_b N_b σ_b²              < S_n,budget
    absolute   τ_loop < φ_reserve/ω_c     （只有 feedback_critical 需要）

前三条可以靠 pad/deskew 修，第四条不能：把两条路径 pad 成同样的延迟，
skew 是零，但闭环仍然是 ẋ=Ax(t−τ0)（§2.8、§6.14）。所以 “matched” 与
“instantaneous” 是两个判据，不是同一个。

block 只暴露三个量：H(s)、DC latency、input-referred noise variance。
`S_τ,ij(ω) = ω|J_ij|·‖(jωI−J)⁻¹‖` 从动力学本身给出每条 edge 的延迟预算，
而不是让 designer 手写 “这条线很重要”。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.skew import phase_budget

__all__ = [
    "Wire", "Buffer", "AllPass", "PureDelay", "RouteEdge",
    "rc_wire", "buffer_chain",
    "group_delay", "edge_metrics", "eta_route", "j_h", "j_tau", "j_tau_group",
    "latency_class", "CoupledNets", "coupled_nets",
    "edge_sensitivity", "sensitivity_summary",
    "RouteSpec", "route_closure", "deskew",
]

_TINY = 1e-300


# ---------------------------------------------------------------------------
# 基本 block
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PureDelay:
    """理想延迟 e^{−sτ}。它保存的是 waveform history，不是静态状态。"""

    tau: float

    def __post_init__(self):
        if self.tau < 0:
            raise ValueError("tau 必须非负")

    @property
    def latency(self) -> float:
        return float(self.tau)

    @property
    def noise_var(self) -> float:
        return 0.0

    def dc_gain(self) -> float:
        return 1.0

    def H(self, s):
        return np.exp(-np.asarray(s, dtype=complex) * self.tau)


@dataclass(frozen=True)
class AllPass:
    """一阶 all-pass (1−sτ/2)/(1+sτ/2)：|H|=1，DC group delay=τ。

    这是 Padé 延迟近似的电路版本；用于 latency matching，不像 RC 那样
    用幅度衰减换相位。τ 很小时它就是一个 tunable delay trim。
    """

    tau: float

    def __post_init__(self):
        if self.tau < 0:
            raise ValueError("tau 必须非负")

    @property
    def latency(self) -> float:
        return float(self.tau)

    @property
    def noise_var(self) -> float:
        return 0.0

    def dc_gain(self) -> float:
        return 1.0

    def H(self, s):
        s = np.asarray(s, dtype=complex)
        return (1.0 - s * self.tau / 2.0) / (1.0 + s * self.tau / 2.0)


@dataclass(frozen=True)
class Buffer:
    """单极点 buffer：g/(1+s/ω_p)·e^{−sτ_b} + 输入折算噪声 σ²。

    buffer 不是 memory，是阻抗隔离 + 能量补充。代价是每级一个极点、
    一份 offset/THD/噪声；fanout tree 的深度受 N_b σ_b² 与相位预算限制。
    """

    gain: float = 1.0
    f_p: float = 5e9
    tau: float = 0.0
    sigma: float = 0.0

    def __post_init__(self):
        if self.f_p <= 0:
            raise ValueError("f_p 必须为正")
        if self.tau < 0:
            raise ValueError("tau 必须非负")

    @property
    def latency(self) -> float:
        return float(self.tau + 1.0 / (2.0 * np.pi * self.f_p))

    @property
    def noise_var(self) -> float:
        return float(self.sigma ** 2)

    def dc_gain(self) -> float:
        return float(self.gain)

    def H(self, s):
        s = np.asarray(s, dtype=complex)
        wp = 2.0 * np.pi * self.f_p
        return self.gain / (1.0 + s / wp) * np.exp(-s * self.tau)


@dataclass(frozen=True)
class Wire:
    """分布式/lumped RC 线。

    length [µm]，r [Ω/µm]，c [fF/µm]，于是 R=length·r、C=length·c、
    τ_rc=R·C（Ω·fF = fs）。distributed 模型是 1/cosh√(sτ_rc)，它的
    DC group delay 是 τ_rc/2，阶跃 50% 点是 0.3795τ_rc；lumped π 模型是
    1/(1+sτ_rc/2)，同样 DC 斜率，但阶跃 50% 点是 ln2·τ_rc/2。
    """

    length: float
    r: float = 0.2
    c: float = 0.2
    model: str = "distributed"

    def __post_init__(self):
        if self.length <= 0 or self.r <= 0 or self.c <= 0:
            raise ValueError("length/r/c 必须为正")
        if self.model not in ("distributed", "lumped"):
            raise ValueError("model 必须是 distributed 或 lumped")

    @property
    def R(self) -> float:
        return float(self.r * self.length)

    @property
    def C(self) -> float:
        return float(self.c * self.length)

    @property
    def tau_rc(self) -> float:
        """R·C；Ω·fF = fs，所以乘 1e-15 得秒。"""
        return float(self.R * self.C * 1e-15)

    @property
    def latency(self) -> float:
        """DC group delay = τ_rc/2（两种模型相同）。"""
        return self.tau_rc / 2.0

    @property
    def delay50(self) -> float:
        k = 0.3795 if self.model == "distributed" else np.log(2.0) / 2.0
        return float(k * self.tau_rc)

    @property
    def noise_var(self) -> float:
        return 0.0

    def dc_gain(self) -> float:
        return 1.0

    def H(self, s):
        s = np.asarray(s, dtype=complex)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if self.model == "distributed":
                return 1.0 / np.cosh(np.sqrt(s * self.tau_rc))
            return 1.0 / (1.0 + s * self.tau_rc / 2.0)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------
@dataclass
class RouteEdge:
    """一串 block 的级联；H 是乘积，latency 是 DC 延迟之和。"""

    blocks: tuple = field(default_factory=tuple)
    name: str = ""

    def __post_init__(self):
        self.blocks = tuple(self.blocks)

    def H(self, s):
        s = np.asarray(s, dtype=complex)
        out = np.ones_like(s, dtype=complex)
        for b in self.blocks:
            out = out * b.H(s)
        return out

    def freqresp(self, omega):
        scalar = np.ndim(omega) == 0
        out = self.H(1j * np.asarray(omega, dtype=float))
        return complex(out) if scalar else out

    @property
    def latency(self) -> float:
        return float(sum(b.latency for b in self.blocks))

    @property
    def dc_gain(self) -> float:
        return float(np.real(self.H(0.0)))

    @property
    def noise_var(self) -> float:
        """把各级输入折算噪声搬到 edge 输出；g≈1 时退化为 N_b σ_b²。"""
        v = 0.0
        g2 = 1.0
        for b in reversed(self.blocks):
            v += g2 * b.noise_var
            g2 *= b.dc_gain() ** 2
        return float(v)

    def with_block(self, block) -> "RouteEdge":
        return RouteEdge(self.blocks + (block,), self.name)


def rc_wire(length: float, r: float = 0.2, c: float = 0.2,
            model: str = "distributed", name: str = "") -> RouteEdge:
    return RouteEdge((Wire(length, r, c, model),), name)


def buffer_chain(depth: int, gain: float = 1.0, f_p: float = 5e9,
                 tau: float = 0.0, sigma: float = 0.0,
                 name: str = "") -> RouteEdge:
    if depth < 0:
        raise ValueError("depth 必须非负")
    return RouteEdge(tuple(Buffer(gain, f_p, tau, sigma) for _ in range(depth)),
                     name)


# ---------------------------------------------------------------------------
# 频域指标
# ---------------------------------------------------------------------------
def group_delay(edge: RouteEdge, omega, rel: float = 1e-3):
    """τ_g = −d∠H/dω，中心差分；三个点一起 unwrap 防止跨 π。"""
    scalar = np.ndim(omega) == 0
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    if np.any(w <= 0):
        raise ValueError("omega 必须 > 0")
    out = np.empty(w.size)
    for i, wi in enumerate(w):
        h = rel * wi
        ws = np.array([wi - h, wi, wi + h])
        p = np.unwrap(np.angle(edge.freqresp(ws)))
        out[i] = -(p[2] - p[0]) / (2.0 * h)
    return float(out[0]) if scalar else out


def _f3db(w: np.ndarray, mag: np.ndarray) -> float:
    """|H| 首次落到 DC/√2 的频率；全通返回 inf。"""
    target = mag[0] / np.sqrt(2.0)
    idx = np.where(mag <= target)[0]
    if idx.size == 0:
        return float("inf")
    i = int(idx[0])
    if i == 0:
        return float(w[0])
    if mag[i] <= 0 or mag[i - 1] <= 0:
        return float(w[i])
    x0, x1 = np.log(w[i - 1]), np.log(w[i])
    y0, y1 = np.log(mag[i - 1]), np.log(mag[i])
    return float(np.exp(x0 + (np.log(target) - y0) * (x1 - x0) / (y1 - y0)))


def edge_metrics(edge: RouteEdge, omega, g_ideal: float = 1.0,
                 tau_ideal: float = 0.0) -> dict:
    """per-edge 的 gain/phase/group-delay/η；ideal 写成 g·e^{−sτ}。"""
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    if np.any(w <= 0):
        raise ValueError("omega 必须 > 0")
    H = edge.freqresp(w)
    mag = np.abs(H)
    phase = np.unwrap(np.angle(H))
    tau_g = group_delay(edge, w)
    Href = g_ideal * np.exp(-1j * w * tau_ideal)
    rel = np.abs(H / Href - 1.0)
    gain_err = np.abs(mag / g_ideal - 1.0)
    phase_err = np.abs(np.angle(np.exp(1j * (phase + w * tau_ideal))))
    return dict(
        omega=w, H=H, mag=mag, phase=phase, tau_g=tau_g,
        gain_err=gain_err, phase_err_deg=np.degrees(phase_err),
        group_err=np.abs(tau_g - tau_ideal),
        eta=float(np.max(rel)), f_3db=_f3db(w, mag),
        g_dc=edge.dc_gain, latency=edge.latency,
    )


def eta_route(edge: RouteEdge, omega, g_ideal: float = 1.0,
              tau_ideal: float = 0.0) -> float:
    """η_route = sup_ω |H/H_ideal − 1|（标量 edge 的 matrix 范数特例）。"""
    return edge_metrics(edge, omega, g_ideal, tau_ideal)["eta"]


def _weighted_mean(w: np.ndarray, y: np.ndarray, weight) -> float:
    W = np.ones_like(w) if weight is None else np.asarray(weight, dtype=float)
    if W.shape != w.shape:
        raise ValueError("weight 必须与 omega 同形")
    dw = np.diff(w)
    num = np.sum(0.5 * (y[1:] + y[:-1]) * dw)
    den = np.sum(0.5 * (W[1:] + W[:-1]) * dw)
    return float(num / den) if den > 0 else float("nan")


def j_h(edge: RouteEdge, omega, g_ideal: float = 1.0,
        tau_ideal: float = 0.0, weight=None) -> float:
    """J_H = ∫W(ω)|H/H_ref−1|²dω / ∫W dω。"""
    m = edge_metrics(edge, omega, g_ideal, tau_ideal)
    return _weighted_mean(m["omega"], np.abs(m["H"] / (g_ideal * np.exp(
        -1j * m["omega"] * tau_ideal)) - 1.0) ** 2, weight)


def j_tau(edge: RouteEdge, omega, tau_ideal: float = 0.0, weight=None) -> float:
    """J_τ = ∫W(ω)|τ_g−τ_ideal|²dω / ∫W dω，单位为 s²。"""
    m = edge_metrics(edge, omega, tau_ideal=tau_ideal)
    return _weighted_mean(m["omega"], m["group_err"] ** 2, weight)


def j_tau_group(edges, omega, tau_ref=None, weight=None) -> float:
    """一组 edge 的 group-delay spread：∫⟨(τ_g,i−τ_ref)²⟩dω / ∫dω。

    deskew 降低的是这个量；绝对延迟是另一条判据。τ_ref 缺省取每个频点的
    组内均值，所以它衡量的是“相互匹配”，不是“接近零”。
    """
    edges = list(edges)
    if not edges:
        raise ValueError("edges 不能为空")
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    gd = np.array([group_delay(e, w) for e in edges], dtype=float)
    if tau_ref is None:
        ref = gd.mean(axis=0)
    elif np.ndim(tau_ref) == 0:
        ref = np.full(w.size, float(tau_ref))
    else:
        ref = np.asarray(tau_ref, dtype=float)
        if ref.shape != w.shape:
            raise ValueError("tau_ref 必须是标量或与 omega 同形")
    return _weighted_mean(w, np.mean((gd - ref) ** 2, axis=0), weight)


# ---------------------------------------------------------------------------
# Latency class
# ---------------------------------------------------------------------------
def latency_class(tau: float, sampled: bool = False) -> str:
    """L0 <10 ps；L1 10–50；L2 50–200；L3 0.2–1 ns；L4 ≥1 ns 或 sampled。

    L4 不代表“不能用”，而代表“compiler 必须显式知道 τ”，不能再把它
    当成 instantaneous wire（§2.8 的时间域 hierarchy）。
    """
    if sampled:
        return "L4"
    tau = float(tau)
    if tau < 10e-12:
        return "L0"
    if tau < 50e-12:
        return "L1"
    if tau < 200e-12:
        return "L2"
    if tau < 1e-9:
        return "L3"
    return "L4"


# ---------------------------------------------------------------------------
# Crosstalk：两条耦合线
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupledNets:
    """两条并行 RC 线，耦合电容 C_m（fF），输出节点各带 C_i 到地。

    节点方程 (Vin_i−Vo_i)/R_i = sC_iVo_i + sC_m(Vo_i−Vo_j) 给出 2×2 H(s)。
    高频极限 H_12/H_11 → C_m/(C_2+C_m)：这就是串扰地板，不能靠 buffer 消掉。
    """

    R1: float
    C1: float
    R2: float
    C2: float
    Cm: float

    def __post_init__(self):
        if min(self.R1, self.R2, self.C1, self.C2) <= 0 or self.Cm < 0:
            raise ValueError("R/C 必须为正，Cm 非负")

    def H(self, omega):
        scalar = np.ndim(omega) == 0
        w = np.atleast_1d(np.asarray(omega, dtype=float))
        s = 1j * w
        m11 = 1.0 / self.R1 + s * (self.C1 + self.Cm) * 1e-15
        m22 = 1.0 / self.R2 + s * (self.C2 + self.Cm) * 1e-15
        m12 = -s * self.Cm * 1e-15
        det = m11 * m22 - m12 ** 2
        out = np.empty((w.size, 2, 2), dtype=complex)
        out[:, 0, 0] = m22 / (det * self.R1)
        out[:, 0, 1] = -m12 / (det * self.R1)
        out[:, 1, 0] = -m12 / (det * self.R2)
        out[:, 1, 1] = m11 / (det * self.R2)
        return out[0] if scalar else out

    def crosstalk(self, omega):
        """|H_12/H_11|：aggressor 2 → victim 1。"""
        scalar = np.ndim(omega) == 0
        H = self.H(omega)
        H = H[None, :, :] if scalar else H
        ratio = np.abs(H[:, 0, 1]) / np.abs(H[:, 0, 0])
        return float(ratio[0]) if scalar else ratio

    @property
    def crosstalk_hf(self) -> float:
        return float(self.Cm / (self.C2 + self.Cm)) if self.C2 + self.Cm else 0.0


def coupled_nets(R: float = 400.0, C: float = 200.0, Cm: float = 20.0) -> CoupledNets:
    return CoupledNets(R, C, R, C, Cm)


# ---------------------------------------------------------------------------
# 动力学给出的延迟敏感度
# ---------------------------------------------------------------------------
def edge_sensitivity(J, omega, eps_phase_deg: float = 1.0):
    """S_τ,ij(ω) = ω|J_ij|·‖(jωI−J)⁻¹‖ 与 τ_tol = ε_φ/S。

    延迟对 edge j→i 的一阶扰动是 ΔJ_ij ≈ −sτJ_ij；resolvent 把这个误差
    放大 ‖(jωI−J)⁻¹‖。所以强耦合、高频、接近 resonance 的 edge 自动拿到
    更紧的延迟预算——不需要手写 “这条线很重要”。
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J 必须是方阵")
    scalar = np.ndim(omega) == 0
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    n = J.shape[0]
    eye = np.eye(n)
    rn = np.empty(w.size)
    for i, wi in enumerate(w):
        M = 1j * wi * eye - J
        e = np.linalg.eigvalsh(M.conj().T @ M)
        smin = np.sqrt(max(float(e[0]), 0.0))
        rn[i] = np.inf if smin <= 0.0 else 1.0 / smin
    S = w[:, None, None] * np.abs(J)[None, :, :] * rn[:, None, None]
    eps = np.deg2rad(float(eps_phase_deg))
    tau_tol = np.where(S > 0.0, eps / np.maximum(S, _TINY), np.inf)
    return (S[0], tau_tol[0]) if scalar else (S, tau_tol)


def sensitivity_summary(J, omega, eps_phase_deg: float = 1.0) -> dict:
    """带上/逐 edge 的 S 上确界、最坏频率、τ_tol 与高频极限 ε/|J_ij|。"""
    J = np.asarray(J, dtype=float)
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    S, _ = edge_sensitivity(J, w, eps_phase_deg)
    Smax = S.max(axis=0)
    iw = S.argmax(axis=0)
    eps = np.deg2rad(float(eps_phase_deg))
    tau_tol = np.where(Smax > 0.0, eps / np.maximum(Smax, _TINY), np.inf)
    return dict(
        omega=w, S=S, S_max=Smax, tau_tol=tau_tol, worst_omega=w[iw],
        hf_tau_tol=np.where(np.abs(J) > 0.0, eps / np.maximum(np.abs(J), _TINY),
                            np.inf),
    )


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------
@dataclass
class RouteSpec:
    omega_max: float
    eps_gain: float = 2e-3
    eps_phase_deg: float = 3.0
    noise_budget: float = 80e-6
    g_ideal: float = 1.0
    tau_ideal: float = 0.0
    feedback_critical: bool = False
    omega_c: float | None = None
    phi_reserve_deg: float = 5.0
    n_grid: int = 400


def _band(omega_max: float, n: int = 400) -> np.ndarray:
    return np.logspace(np.log10(omega_max) - 5.0, np.log10(omega_max), n)


def route_closure(edge: RouteEdge, spec: RouteSpec) -> dict:
    """四条 closure 一次判完；返回每个条件的 pass 与绝对延迟预算。"""
    w = _band(spec.omega_max, spec.n_grid)
    m = edge_metrics(edge, w, spec.g_ideal, spec.tau_ideal)
    noise_rms = float(np.sqrt(edge.noise_var))
    gain_ok = bool(np.max(m["gain_err"]) <= spec.eps_gain)
    phase_ok = bool(np.max(m["phase_err_deg"]) <= spec.eps_phase_deg)
    noise_ok = bool(noise_rms <= spec.noise_budget)
    if spec.feedback_critical:
        if spec.omega_c is None:
            raise ValueError("feedback_critical 需要 omega_c")
        tau_max = phase_budget(spec.omega_c, spec.phi_reserve_deg)
        absolute_ok = bool(edge.latency <= tau_max)
    else:
        tau_max = float("inf")
        absolute_ok = True
    return dict(
        name=edge.name, latency=edge.latency,
        latency_class=latency_class(edge.latency),
        noise_rms=noise_rms, tau_max_abs=tau_max,
        eta=m["eta"], f_3db=m["f_3db"],
        max_gain_err=float(np.max(m["gain_err"])),
        max_phase_err_deg=float(np.max(m["phase_err_deg"])),
        max_group_err=float(np.max(m["group_err"])),
        j_h=j_h(edge, w, spec.g_ideal, spec.tau_ideal),
        j_tau=j_tau(edge, w, spec.tau_ideal),
        pass_gain=gain_ok, pass_phase=phase_ok,
        pass_noise=noise_ok, pass_absolute=absolute_ok,
        verdict=bool(gain_ok and phase_ok and noise_ok and absolute_ok),
        pad_can_fix=not spec.feedback_critical,
    )


def deskew(edges, omega0: float, max_iter: int = 8) -> dict:
    """用 all-pass 把每条 edge 在 ω0 的 group delay 对齐到最慢的一条。

    对齐后 skew→0，但每条的绝对延迟仍然是 target；闭环相位预算不会因此变好。
    """
    edges = list(edges)
    if not edges:
        raise ValueError("edges 不能为空")
    d0 = np.array([group_delay(e, omega0) for e in edges], dtype=float)
    target = float(np.max(d0))
    out = []
    for e in edges:
        pad = max(target - group_delay(e, omega0), 0.0)
        for _ in range(max_iter):
            cand = e.with_block(AllPass(pad))
            err = target - group_delay(cand, omega0)
            if abs(err) < 1e-18:
                break
            pad = max(pad + err, 0.0)
        out.append(e.with_block(AllPass(pad)))
    d1 = np.array([group_delay(e, omega0) for e in out], dtype=float)
    return dict(
        edges=out, omega0=omega0, target=target,
        before=d0, after=d1, pad=np.maximum(target - d0, 0.0),
        skew_before=float(d0.max() - d0.min()),
        skew_after=float(d1.max() - d1.min()),
    )
