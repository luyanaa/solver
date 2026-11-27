"""有限时间误差传播与校正调度（DESIGN.md §6.22）。

模拟 fabric 的局部误差（噪声/失配/量化/带宽）不可消除，能做的是控制
误差的**传播、累积与对结果的影响**。本模块把 §1.2 的扰动端口视角换成
误差状态视角：真实系统 ẋ=F(x)，硬件 ẋ=F(x̂)+r(t)，线性化

    ė ≈ J(t) e + r(t)  ⟹  e(T) ≈ Φ(T,0) e(0) + ∫Φ(T,t) r(t) dt

三种情形给出完全不同的控制策略（§6.22 的分类合同）：

  contracting  谱横坐标 < 0 且有限时间增益不爆炸 —— 误差被自然压制在
               r_max/γ 量级；只需按校正调度公式选检查间隔。
  neutral      谱横坐标 ≈ 0（纯积分器/守恒零模态）—— 恒定偏置线性累积
               e ∝ t；必须监视漂移率并周期校正。
  expanding    谱横坐标 > 0 —— 误差指数放大；有限时间失去逐轨迹预测，
               只能缩时窗或降级到 QoI/统计量（§1 的 Level 2 可划归）。
  leak         泄漏型扰动（ΔA 列和≠0，如电流镜失配打到单条通量边）把
               守恒零模态激发成慢漂移。Γ_proj 看不见它（slack.py 的 P0-D
               合同已声明），必须另算 dM/dt 漂移率 —— 这里给
               conservation_drift_rate 做这件事的线性化版本。

非正规系统不能只看特征值：所有特征值稳定仍可能有很大的瞬态放大
（shear 流动 Γ=5.12 的已知案例，stagea.json）。所以分类同时报告
谱横坐标和**有限时间增益** ‖expm(A·T)‖（能量度量可选），两者分开。

校正调度（§6.22）：一次校正后残余误差 e0，区间内扰动上界 r_max，
不稳定率 λ>0，要求区间末误差 ≤ ε，则最大检查间隔

    ΔT ≤ (1/λ) · ln( (ε + r_max/λ) / (e0 + r_max/λ) )    （右侧>0 时）

λ→0（中性）退化为 ΔT ≤ (ε−e0)/r_max：线性累积的预算式。若每次校正
满足递推 ε_{k+1} ≤ ρ ε_k + η（ρ<1），长期界为 η/(1−ρ) —— 这就是
"低精度单元 + 校正闭环给出有限长期误差"的形式化依据。

依赖方向：纯线性代数，不 import twin/，不含求解器。resolvent.py 的
Γ 是频域最坏增益（∞时间），这里是有限时间窗口的对应物，两者互补。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

__all__ = [
    "PropagationClass",
    "finite_time_gain",
    "logarithmic_norm",
    "classify_propagation",
    "conservation_drift_rate",
    "correction_interval",
    "steady_bound",
    "error_recursion_bound",
    "eta_from_lognorm",
    "classify_propagation_type",
    "CorrectionOperator",
    "corrected_recursion",
]


# dataclass 容器 -----------------------------------------------------------

@dataclass
class PropagationClass:
    """误差传播分类结果（§6.22 合同的 evidence 对象）。

    class 的判定规则：谱横坐标 α_spec > tol ⇒ expanding；|α_spec| ≤ tol
    ⇒ neutral；α_spec < −tol ⇒ contracting（但若有限时间增益在任何
    检查点超过 transient_threshold，降级标注 transient_amplified——
    非正规瞬态不能被谱判据覆盖，必须单独看见）。
    """
    spectral_abscissa: float          # max Re(λ)
    transient_gains: dict             # {horizon: ‖Φ(h,0)‖_M}
    lognorm: float                    # μ₂(A)：2-范数对数范数（能量上界速率）
    cls: str                          # legacy：contracting/neutral/expanding
    transient_amplified: bool         # 谱稳定但瞬态增益超阈值
    mu_M: float | None = None         # 能量度量 M 下的 μ_M（审查6）
    propagation_type: str | None = None  # 五分法类型（审查6）
    horizons: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(cls=self.cls, propagation_type=self.propagation_type,
                    mu_M=self.mu_M,
                    spectral_abscissa=self.spectral_abscissa,
                    transient_gains=dict(self.transient_gains),
                    lognorm=self.lognorm,
                    transient_amplified=self.transient_amplified,
                    notes=list(self.notes))


# 核心 API -----------------------------------------------------------------

def finite_time_gain(A, t, M=None) -> float:
    """‖expm(A·t)‖（M 度量可选：‖M^{1/2} Φ M^{−1/2}‖₂）。

    直接 expm 的代价 O(n³) 级，n≤256 的 tile 规模内可接受；更大规模
    应走特征分解（正规系统）或 Krylov，本模块不做那个优化——这是
    判定器不是求解器。
    """
    A = np.asarray(A, dtype=float)
    Phi = expm(A * t)
    if M is None:
        return float(np.linalg.norm(Phi, 2))
    w, Q = np.linalg.eigh(np.asarray(M, dtype=float))
    if np.any(w <= 0):
        raise ValueError("metric M 必须正定")
    Mh = Q @ np.diag(np.sqrt(w)) @ Q.T
    Mhi = Q @ np.diag(1.0 / np.sqrt(w)) @ Q.T
    return float(np.linalg.norm(Mh @ Phi @ Mhi, 2))


def logarithmic_norm(A, M=None) -> float:
    """μ_M(A) = λ_max[½ M^{-1/2}(AᵀM+MA)M^{-1/2}]（能量度量对数范数）。

    M=None ⇒ 普通二范数 μ₂(A)=λ_max((A+Aᵀ)/2)。M=diag(C) 时给出
    电容/状态尺度正确的收缩速率——C bank 跨数量级时 μ₂ 可能严重误导
    （审查6）。μ<0 ⇒ 该范数下 ‖e(t)‖_M 单调收缩。
    """
    A = np.asarray(A, dtype=float)
    if M is None:
        return float(np.max(np.linalg.eigvalsh((A + A.T) / 2.0)))
    M = np.asarray(M, dtype=float)
    w, Q = np.linalg.eigh(M)
    if np.any(w <= 0):
        raise ValueError("度量 M 必须正定")
    Mh = Q @ np.diag(np.sqrt(w)) @ Q.T
    Mhi = Q @ np.diag(1.0 / np.sqrt(w)) @ Q.T
    B = 0.5 * Mhi @ (A.T @ M + M @ A) @ Mhi
    return float(np.max(np.linalg.eigvalsh((B + B.T) / 2.0)))


def classify_propagation_type(A, M=None, tol=1e-9) -> str:
    """审查6 五分法（返回五类之一）：

      contractive           μ_M(A)<0 —— 直接收缩界
      asymptotically stable Hurwitz 但 μ_M≥0 —— 非正规瞬态放大
      semistable            零/纯虚半单模态，其余 Re<0 —— 有界中性
      polynomially growing  临界 Jordan block —— 多项式增长
      expanding             max Re(λ)>0 —— 指数放大（有限时窗/统计合同）

    M 是能量度量（如节点电容矩阵）；μ_M 与 finite_time_gain(A,t,M)
    使用同一度量，口径统一。半单判定用零（虚轴）特征值的几何重数
    == 代数重数（数值容差 tol）。
    """
    A = np.asarray(A, dtype=float)
    w = np.linalg.eigvals(A)
    alpha = float(np.max(w.real))
    if alpha > tol:
        return "expanding"
    mu = logarithmic_norm(A, M)
    if mu < -tol:
        return "contractive"
    # 临界模态（Re≈0）：Jordan 块 ⇒ 多项式增长；半单 ⇒ 有界中性
    crit = w[np.abs(w.real) <= tol]
    if crit.size:
        for lam in crit:
            # rank(A−λI) 的零空间维数 vs 重数（几何 vs 代数）
            mult = int(np.sum(np.abs(w - lam) <= 1e-7 * max(1.0, abs(lam))))
            rank = np.linalg.matrix_rank(A - lam * np.eye(A.shape[0]),
                                         tol=1e-8 * max(1.0, float(np.abs(A).max())))
            if A.shape[0] - rank < mult:
                return "polynomially growing"
        return "semistable"
    return "asymptotically stable"


def classify_propagation(A, horizons=(1e-2, 1e-1, 1.0),
                         neutral_tol=1e-9, transient_threshold=10.0,
                         M=None) -> PropagationClass:
    """按谱横坐标 + 有限时间增益分类（§6.22 的三分类 + 瞬态标注）。"""
    A = np.asarray(A, dtype=float)
    w = np.linalg.eigvals(A)
    alpha = float(np.max(w.real)) if w.size else 0.0
    gains = {float(h): finite_time_gain(A, h, M=M) for h in horizons}
    mu = logarithmic_norm(A, M)
    ptype = classify_propagation_type(A, M, tol=neutral_tol)

    if alpha > neutral_tol:
        cls = "expanding"
    elif alpha < -neutral_tol:
        cls = "contracting"
    else:
        cls = "neutral"

    amplified = cls == "contracting" and max(gains.values()) > transient_threshold
    notes = []
    if amplified:
        notes.append(
            "谱稳定但有限时间增益 %.3g > %.3g：非正规瞬态，"
            "误差合同必须用 Φ(T,0) 而非谱横坐标"
            % (max(gains.values()), transient_threshold))
    if cls == "neutral":
        notes.append("中性模态：恒定偏置线性累积 e∝t，需周期校正或漂移预算")
    if cls == "expanding":
        notes.append("扩张模态：有限时间失去逐轨迹预测，缩时窗或降级 QoI")

    return PropagationClass(spectral_abscissa=alpha,
                            transient_gains=gains,
                            lognorm=mu,
                            mu_M=mu, propagation_type=ptype,
                            cls=cls,
                            transient_amplified=amplified,
                            horizons=[float(h) for h in horizons],
                            notes=notes)


def conservation_drift_rate(A, dA) -> float:
    """泄漏型扰动的守恒漂移率（线性化 dM/dt，§1.2 泄漏合同）。

    守恒量 M = 1ᵀx（行和为零的 A 保 M）。泄漏型 dA（列和≠0）使
    dM/dt = 1ᵀ·dA·x，单位状态范数的漂移率 = ‖1ᵀ·dA‖₂ / ‖1‖₂。
    保守型 dA（列和=0，slack.py P0-D 合同）返回 ~0。
    注意：这是**率**，累计漂移 = 率 × 时间 × ‖x‖ —— 调用方自己乘。
    """
    A = np.asarray(A, dtype=float)
    dA = np.asarray(dA, dtype=float)
    if dA.shape != A.shape:
        raise ValueError("dA 形状必须与 A 相同")
    ones = np.ones(A.shape[0])
    col_sums = dA.sum(axis=0)           # 保守型 ⇒ 每列和 = 0
    return float(np.linalg.norm(col_sums) / np.linalg.norm(ones))


def correction_interval(lam: float, r_max: float, e0: float,
                        eps: float) -> float:
    """校正调度公式（§6.22）：不稳定率 λ ≥ 0 时下一次检查的最大间隔。

        λ > 0:  ΔT = (1/λ)·ln((ε + r_max/λ)/(e0 + r_max/λ))
        λ = 0:  ΔT = (ε − e0)/r_max（线性累积预算式）

    前提：e0 < ε（校正后的误差必须已进入容限内），r_max ≥ 0，λ ≥ 0。
    e0 ≥ ε ⇒ raise：调度器不得静默给出∞间隔（先修校正机制，
    不是调间隔）。
    """
    if e0 >= eps:
        raise ValueError("e0=%.3g ≥ eps=%.3g：校正本身未把误差压进容限，"
                         "无法调度（先修校正机制，不是调间隔）" % (e0, eps))
    if lam < 0:
        raise ValueError("lam<0 是收缩情形：误差自衰减，无需调度公式，"
                         "传 lam=0 用预算式即可")
    if lam == 0.0:
        if r_max <= 0:
            return math.inf            # 无扰动无增长：无需再检查
        return (eps - e0) / r_max
    num = lam * eps + r_max
    den = lam * e0 + r_max
    return math.log(num / den) / lam


def steady_bound(rho: float, eta: float) -> float:
    """校正闭环的长期界：limsup ε_k ≤ η/(1−ρ)（ρ<1）。

    递推 ε_{k+1} ≤ ρ ε_k + η 的经典不动点。ρ≥1 ⇒ inf（闭环不收敛，
    诚实返回，不抛异常——调用方证书层决定怎么拒绝）。
    """
    if rho >= 1.0:
        return math.inf
    return eta / (1.0 - rho)


@dataclass
class CorrectionOperator:
    """校正算子合同（§6.22 P0-3）：‖e⁺‖ ≤ q‖e⁻‖ + ζ。

    q: 校正后的残余收缩系数（0 表示理想复位，但真实 ADC/DAC/
       参考模型/状态回注做不到）；ζ: 校正自身引入的新误差上界。
    合同必须**显式声明** q 与 ζ，不能默认 q=0、ζ=0。
    """
    q: float
    zeta: float

    def residual_bound(self, e_minus: float) -> float:
        if self.q < 0 or self.zeta < 0:
            raise ValueError("q/zeta 必须 >= 0")
        return self.q * e_minus + self.zeta


def corrected_recursion(e0_plus: float, gains, etas, qs, zetas) -> dict:
    """校正闭环递推（§6.22 P0-3）：

        ε_{k+1}⁺ ≤ q_{k+1}·G_k·ε_k⁺ + q_{k+1}·η_k + ζ_{k+1}

    ρ=sup(q_{k+1}G_k)<1 且新增项有统一上界 η̄=sup(qη+ζ) 时，
    limsup ε⁺ ≤ η̄/(1−ρ)。**不声明 q/ζ 就不能声称长期有界**。
    返回 dict(eps_plus, rho, eta_bar, steady_bound)。
    """
    n = len(gains)
    if not (len(etas) == len(qs) == len(zetas) == n):
        raise ValueError("gains/etas/qs/zetas 必须等长")
    eps = [e0_plus]
    rho = 0.0
    eta_bar = 0.0
    for k in range(n):
        q, G, eta, zeta = qs[k], gains[k], etas[k], zetas[k]
        if q < 0 or zeta < 0 or G < 0:
            raise ValueError("q/zeta/G 必须 >= 0")
        rho = max(rho, q * G)
        eta_bar = max(eta_bar, q * eta + zeta)
        eps.append(q * G * eps[-1] + q * eta + zeta)
    return dict(eps_plus=eps, rho=rho, eta_bar=eta_bar,
                steady_bound=(eta_bar / (1.0 - rho) if rho < 1.0
                              else math.inf))


def eta_from_lognorm(mu: float, r_max: float, dT: float) -> float:
    """对数范数闭式新增误差界（§6.22 P0-2）：

        eta ≤ r_max·(e^{mu·dT} − 1)/mu   (mu != 0)
        eta ≤ r_max·dT                   (mu = 0)

    前提：mu 是**度量 M 下**的对数范数上界，‖r(t)‖ ≤ r_max。
    这是 ∫‖Φ(t1,s)‖‖r(s)‖ds 的上界（不必计算逐点 Φ）。
    """
    if r_max < 0 or dT < 0:
        raise ValueError("r_max/dT 必须 >= 0")
    if mu == 0.0:
        return r_max * dT
    v = r_max * math.expm1(mu * dT) / mu
    if v < 0:
        raise ValueError("mu>0 时指数界溢出/无效：请缩短时窗")
    return float(v)


def error_recursion_bound(e0: float, gains, etas) -> list:
    """逐窗口误差上界传播 ε_{k+1} = G_k ε_k + η_k（§6.22 的在线估计器）。

    **P0-2**：etas 必须是**保守**新增预算 ∫‖Φ‖‖r‖（或经认证的
    l1 界），绝不能用 ‖∫r dt‖——注入正负抵消会让后者归零而真实
    误差非零。

    gains/etas 等长。返回每窗口末的界序列（含 k=0 的 e0）。残差小
    不代表界小——初始误差经 G_k 传播的那一项一直在，这正是"必须同时
    传播过去累计的误差界"的形式化。
    """
    if len(gains) != len(etas):
        raise ValueError("gains 与 etas 必须等长")
    out = [e0]
    e = e0
    for G, eta in zip(gains, etas):
        e = G * e + eta
        out.append(e)
    return out
