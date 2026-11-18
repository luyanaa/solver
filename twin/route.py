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
    "Wire", "Buffer", "AllPass", "PureDelay", "ZOH", "RouteEdge",
    "rc_wire", "buffer_chain",
    "group_delay", "edge_metrics", "eta_route", "j_h", "j_tau", "j_tau_group",
    "latency_class", "CoupledNets", "coupled_nets",
    "edge_sensitivity", "sensitivity_summary",
    "RouteSpec", "route_closure", "deskew",
    "phase_delay_at", "loop_gain_matrix", "loop_margins", "PhysicalScaling",
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
class ZOH:
    """零阶保持的真实等效传递 H_ZOH(s) = (1−e^{−sT_s})/(sT_s)。

    = e^{−sT_s/2}·sinc(sT_s/2)：相位与 PureDelay(T_s/2) **完全一致**，
    但幅度带 sinc droop——jω 处 |H| = |sin(ωT_s/2)/(ωT_s/2)|，
    DC=1，Nyquist(ω=π/T_s) 处只有 2/π≈0.637（−3.92 dB）。PureDelay
    只在 ωT_s≪1（droop<ε）时是合法近似；带宽逼近 Nyquist 时必须
    用本块，把 droop 记进 gain 轴。dc_gain=1，noise_var=0（采样折叠
    噪声是单独的 B_p 端口，不在这里）。
    """

    T_s: float

    def __post_init__(self):
        if self.T_s <= 0:
            raise ValueError("T_s 必须为正")

    @property
    def latency(self) -> float:
        return float(self.T_s / 2.0)

    @property
    def noise_var(self) -> float:
        return 0.0

    def dc_gain(self) -> float:
        return 1.0

    def droop(self, omega):
        """jω 处的幅度 |sinc(ωT_s/2)|（相对 DC 的幅度损失因子）。"""
        x = np.asarray(omega, dtype=float) * self.T_s / 2.0
        return np.sinc(x / np.pi)          # np.sinc(x)=sin(πx)/(πx)

    def H(self, s):
        s = np.asarray(s, dtype=complex)
        scalar = s.shape == ()
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (1.0 - np.exp(-s * self.T_s)) / (s * self.T_s)
        # s=0 极限是 1；exp 饱和引起的 0/0 同样按极限 1 处理
        bad = ~np.isfinite(v)
        v = np.where(bad, 1.0 + 0j, v)
        return complex(v) if scalar else v


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

    # 明确的两个简化（P2-route-1 / P2-route-2），读数时不要超出：
    # 1. H 是各 block **开路** transfer function 的乘积。两个无 buffer 隔离
    #    的 Wire 段级联时，真实电路有段间阻抗加载（第二段的输入阻抗
    #    分压第一段的输出），乘积模型高估带宽。正确的做法是 ABCD 矩阵
    #    级联；本文件故意不做——本仓库的立场是每段关键路径由 buffer
    #    隔离，加载效应是 PEX/SPICE 阶段的职责。
    # 2. noise_var 把各级输入折算噪声按 **DC 增益²** 搬到输出，忽略了
    #    后级极点对噪声的等效噪声带宽（ENBW）压缩：单极点链的输出噪声
    #    实际上比 Σg²σ² 略小。这也是保守方向的近似，不是精确值。


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


def phase_delay_at(edge: RouteEdge, omega: float, n_decades: float = 12.0,
                   step: float = 1.02) -> float:
    """pd(ω) = −∠H(jω)/ω：闭环在 crossover 真正感到的 phase delay。

    ∠H 沿频率连续 unwrap 到目标 ω 再除——AllPass 级联的总相位会越过
    −π，np.angle 的单点卷绕值会低估甚至变号（4×AllPass(50ps) 在 10 GHz
    处真值 −8.08 rad 卷绕成 −1.80 rad）。实现：从 ω/10^n_decades 起
    以固定步长比 march 到 ω，逐点 unwrap；要求每步相位增量 < π，对本
    文的 block（pure delay/单极点/all-pass/分布式 RC）在 ωτ≲1e4 内成立。

    独立参照：−φ(ωc) = ∫₀^{ωc} τ_g dω（group delay 的积分），bench 用它
    交叉核对 unwrap 的正确性。

    与 edge.latency（DC group delay 之和）的关系：对上述 block，每段的
    phase delay ≤ 自身 DC group delay，所以 pd(ω) ≤ latency 恒成立——
    latency 判据是保守侧。两者只在 pad 的极点落进带内时才在数值上
    分开（pad 极点在带外时 pd ≈ latency 到 1e-4）。
    """
    w_target = float(omega)
    if w_target <= 0:
        raise ValueError("omega 必须 > 0")
    n = int(np.ceil(n_decades * np.log(10.0) / np.log(step)))
    w = w_target * np.power(step, np.arange(-n, 1))
    p = np.unwrap(np.angle(edge.H(1j * w)))
    # 减去起始相位：反相 buffer 的 DC 相位是 ±π，unwrap 从 p[0]=π 出发，
    # 不减会把 DC 反相（符号翻转，不是延迟）计成 −p[-1]/ω 的巨大伪延迟
    # （实测 −4934 ps vs 真值 ~66 ps）。相位增量沿路径累加，与常数无关。
    return float(-(p[-1] - p[0]) / w_target)


def _weighted_mean(w: np.ndarray, y: np.ndarray, weight) -> float:
    W = np.ones_like(w) if weight is None else np.asarray(weight, dtype=float)
    if W.shape != w.shape:
        raise ValueError("weight 必须与 omega 同形")
    dw = np.diff(w)
    num = np.sum(0.5 * (W[1:] * y[1:] + W[:-1] * y[:-1]) * dw)
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
    高频地板有两个不相等的版本（非对称网）：|H_01/H_11| → Cm/(C1+Cm)
    （crosstalk_hf_victim(1)，纯电容分压）与 |H_01/H_00| →
    (Cm/(C2+Cm))·(R1/R2)（crosstalk()，带电阻比）。对称网两者重合，
    crosstalk_hf = Cm/(C2+Cm)。这就是串扰地板，不能靠 buffer 消掉。
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
        out[:, 0, 1] = -m12 / (det * self.R2)
        out[:, 1, 0] = -m12 / (det * self.R1)
        out[:, 1, 1] = m11 / (det * self.R2)
        return out[0] if scalar else out

    def crosstalk(self, omega):
        """|H_12/H_11|：aggressor 2 → victim 1。

        注意（非对称网）：这个比值的分母是 **victim 自己的对角项**
        H_00（含 1/R1），所以高频地板带 R1/R2 因子：
        |H_01/H_00| → (Cm/(C2+Cm))·(R1/R2)。纯电容分压（V1/V2）是
        |H_01/H_11|，见 crosstalk_hf_victim。
        """
        scalar = np.ndim(omega) == 0
        H = self.H(omega)
        H = H[None, :, :] if scalar else H
        ratio = np.abs(H[:, 0, 1]) / np.abs(H[:, 0, 0])
        return float(ratio[0]) if scalar else ratio

    @property
    def crosstalk_hf(self) -> float:
        """高频串扰地板 Cm/(C2+Cm)。

        这是对称网（R1=R2）的 crosstalk() 极限，也是 |H_01/H_00| 对
        R1=R2 的值。数值上等于 Cm/(C1+Cm) 仅当 C1=C2；非对称网请用
        crosstalk_hf_victim 取 victim 自己的对角比值。数值不改动——
        bench/run_route.py 的 crosstalk 算例依赖它。
        """
        return float(self.Cm / (self.C2 + self.Cm)) if self.C2 + self.Cm else 0.0

    def crosstalk_hf_victim(self, net: int = 1) -> float:
        """victim 自对角比值的高频电容分压地板 |H_01/H_11|。

        H_01 = −m12/(det·R2)、H_11 = m11/(det·R2) 共享同一个 R2 因子，
        所以 |H_01/H_11| → ωCm/(ω(C1+Cm)) = Cm/(C1+Cm)：这是"victim 1
        节点上，每单位 aggressor 在 victim 2 节点上的电压"的地板，
        与电阻比无关。net=2 给对称的 Cm/(C2+Cm)（H_10/H_22）。
        与 crosstalk()（|H_01/H_00|，带 R1/R2 因子）不同——两个地板
        在非对称网上不相等，混用会把 R 比当成电容比。
        """
        if net == 1:
            return float(self.Cm / (self.C1 + self.Cm)) if self.C1 + self.Cm else 0.0
        if net == 2:
            return float(self.Cm / (self.C2 + self.Cm)) if self.C2 + self.Cm else 0.0
        raise ValueError("net 必须是 1 或 2")


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
    """四条 closure 一次判完；返回每个条件的 pass 与绝对延迟预算。

    absolute 判据用的是 **ωc 处的 phase delay** −∠H(jωc)/ωc
    （phase_delay_at），不是 DC latency：闭环感到的是 crossover 处的
    相位，deskew 插入的 AllPass pad 在 DC 贡献 τ 的 group delay，但其
    极点落进带内时在 ωc 处贡献的 phase delay 小于 τ（pd ≤ latency，
    latency 是保守侧；pad 极点在带外时两者到 1e-4 一致）。两个数都
    报告：`latency`（DC）与 `phase_delay_omega_c`（判定用量）。
    """
    w = _band(spec.omega_max, spec.n_grid)
    m = edge_metrics(edge, w, spec.g_ideal, spec.tau_ideal)
    noise_rms = float(np.sqrt(edge.noise_var))
    gain_ok = bool(np.max(m["gain_err"]) <= spec.eps_gain)
    phase_ok = bool(np.max(m["phase_err_deg"]) <= spec.eps_phase_deg)
    noise_ok = bool(noise_rms <= spec.noise_budget)
    pd_omega = (spec.omega_c if (spec.feedback_critical and spec.omega_c is not None)
                else spec.omega_max)
    phase_delay = phase_delay_at(edge, pd_omega)
    if spec.feedback_critical:
        if spec.omega_c is None:
            raise ValueError("feedback_critical 需要 omega_c")
        tau_max = phase_budget(spec.omega_c, spec.phi_reserve_deg)
        absolute_ok = bool(phase_delay <= tau_max)
    else:
        tau_max = float("inf")
        absolute_ok = True
    return dict(
        name=edge.name, latency=edge.latency,
        phase_delay_omega_c=phase_delay,
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


# ---------------------------------------------------------------------------
# 闭环 closure：per-edge H_ij → whole-loop（Review item B，§2.8/§6.14）
# ---------------------------------------------------------------------------
def loop_gain_matrix(J, edge_map, s):
    """延迟耦合矩阵 J∘H(s)：edge j→i 把 J_ij 换成 J_ij·H_ij(s)。

    edge_map 是 {(i, j): RouteEdge}；没被覆盖的非零 J_ij 视为瞬时（H=1），
    零元素不需要 edge。这是 §6.14 的 A∘H(s) 的逐 edge 版本：H_ij 不是
    一个标量 e^{−sτ}，而是整条 RouteEdge 的 transfer function。
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J 必须是方阵")
    s_arr = np.atleast_1d(np.asarray(s, dtype=complex))
    n = J.shape[0]
    out = np.zeros((s_arr.size, n, n), dtype=complex)
    covered = set()
    for (i, j), edge in edge_map.items():
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("edge_map 下标越界: (%d,%d)" % (i, j))
        out[:, i, j] = J[i, j] * np.asarray(edge.H(s_arr), dtype=complex)
        covered.add((i, j))
    for i in range(n):
        for j in range(n):
            if (i, j) not in covered and J[i, j] != 0.0:
                out[:, i, j] = J[i, j]
    return out


def _sigma_min(M: np.ndarray) -> float:
    e = np.linalg.eigvalsh(M.conj().T @ M)
    return float(np.sqrt(max(e[0], 0.0)))


def loop_margins(J, edge_map, omega_grid, with_first_order: bool = True) -> dict:
    """闭环 stability margin：det(jωI − J∘H(jω)) 沿 jω 轴的最近距离。

    返回三件事：

    (a) sigma_min(ω) = σ_min(jωI − J∘H(jω)) 的带内最小值与所在频率。
        σ_min→0 的邻接近似特征根穿过 jω 轴的位置——这是必要条件探针，
        不是 DDE 精确根：对 DDE 用 argument principle 找全部根超出范围；
        标量 1-DOF 的精确参照是 analysis.skew.scalar_dde_root（§6.14）。

    (b) with_first_order=True 时同时给出 E=I+A∘T 版本
        σ_min(jωE − J)（复用 analysis.skew.delay_mass_matrix，T 取每条
        edge 的 DC latency）。这是 §6.14 的乐观性检查在 loop 层的重复：
        一阶模型只在 ωτ≪1 可信。

    (c) S(jω) = [I + L(jω)]⁻¹，L(jω) = J∘H(jω)/(jω) 是把状态注入点
        割开后的 loop transfer（ẋ = J∘H(D)x 是正反馈互连：延迟后的状态
        被重新注入，符号折进 L 的定义）。‖S‖∞ 是 1/σ_min(I−L) 的带内
        峰值；靠近失稳时它发散，比 σ_min 更接近 classical margins。
    """
    J = np.asarray(J, dtype=float)
    w = np.atleast_1d(np.asarray(omega_grid, dtype=float))
    if np.any(w <= 0):
        raise ValueError("omega 必须 > 0")
    n = J.shape[0]
    eye = np.eye(n)
    A = loop_gain_matrix(J, edge_map, 1j * w)
    smin = np.empty(w.size)
    speak = np.empty(w.size)
    for k in range(w.size):
        smin[k] = _sigma_min(1j * w[k] * eye - A[k])
        speak[k] = 1.0 / _sigma_min(eye - A[k] / (1j * w[k]))
    i0 = int(np.argmin(smin))
    i1 = int(np.argmax(speak))
    out = dict(
        omega=w, sigma_min=smin, sensitivity_peak=speak,
        sigma_min_min=float(smin[i0]), omega_at_min=float(w[i0]),
        S_peak=float(speak[i1]), S_peak_omega=float(w[i1]),
    )
    if with_first_order:
        tau_mat = np.zeros((n, n))
        for (i, j), edge in edge_map.items():
            tau_mat[i, j] = edge.latency
        from analysis.skew import delay_mass_matrix
        E = delay_mass_matrix(J, tau_mat)
        smin_fo = np.array([_sigma_min(1j * wk * E - J) for wk in w])
        j0 = int(np.argmin(smin_fo))
        out["sigma_min_first_order"] = smin_fo
        out["sigma_min_min_first_order"] = float(smin_fo[j0])
        out["omega_at_min_first_order"] = float(w[j0])
    return out


# ---------------------------------------------------------------------------
# 物理时间/电压 scaling（Review item B 第二半）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicalScaling:
    """模型时间/状态单位 → 硬件物理量的显式换算字段。

    方向（唯一的、显式的）：

        t_phys = alpha · t_model      alpha [s / model-time-unit]

    alpha 是“一个 model-time 单位在硬件上花多少秒”。alpha=1e-6 意味着
    硬件比模型快 1e6×，硬件看到的频率/特征值是

        omega_phys = omega_model / alpha        lambda_hw = lambda_model / alpha

    所以任何 B_hw 比较**必须**经过 to_physical_band：模型坐标里 8 MHz
    的模态在 alpha=1e-6 的硬件上是 8 THz 的需求，跳过这一步的带宽
    verdict 是错的。电压侧同理：x = V / S_x（S_x [V / model-unit]），
    噪声预算在 V 与 model 单位之间也要过 S_x。
    """

    S_x: float
    alpha: float
    v_swing: float

    def __post_init__(self):
        if self.S_x <= 0 or self.alpha <= 0 or self.v_swing <= 0:
            raise ValueError("S_x/alpha/v_swing 必须为正")

    def to_physical_band(self, omega_model):
        """rad/model-time → rad/s 硬件需求。"""
        return np.asarray(omega_model, dtype=float) / self.alpha

    def to_model_band(self, omega_phys):
        """rad/s → rad/model-time。与 to_physical_band 互逆。"""
        return np.asarray(omega_phys, dtype=float) * self.alpha

    def state_from_voltage(self, V):
        """x = V / S_x：硬件电压 → 模型状态单位。"""
        return np.asarray(V, dtype=float) / self.S_x

    def voltage_from_state(self, x):
        """V = x · S_x。"""
        return np.asarray(x, dtype=float) * self.S_x

    def noise_budget_model(self, sigma_v: float) -> float:
        """V 的噪声预算折算到 model 单位（σ_model = σ_V / S_x）。"""
        return float(sigma_v) / self.S_x

    def bandwidth_pass(self, omega_model, B_hw_hz: float) -> bool:
        """带宽 verdict：物理需求 |lambda_hw|/(2π) ≤ B_hw [Hz]。"""
        return bool(np.max(np.abs(self.to_physical_band(omega_model)))
                    / (2.0 * np.pi) <= float(B_hw_hz))
