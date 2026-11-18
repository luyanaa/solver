"""状态/时间 scaling（DESIGN.md §4.0 P3、§10 repo skeleton "A_t 公式 + 频窗检查"）。

给定（通常由归一化系统线性化得到的）Descriptor (E,A,B,C)，计算：

- **状态缩放 S_x**（每状态满摆映射）：V = S·x，S_i = v_swing/(√P_ii·k_head)，
  P 是可控性 Gramian（单位白输入下状态的协方差，x̂ 单位——**必须在
  Normalization 之后的描述符上调用**），k_head=3 表示 3σ 占满量程。
  V=Sx 不改变特征值/κ（scaling 是预条件，不是单位换算），但改变 Γ、
  瞬态增长和硬件扰动范数——这正是 S 要同时满足 |V_i| ≤ V_max 与
  Γ(T0·S A S⁻¹·(E S⁻¹)⁻¹?) 尽量小的原因；两种口径（rms3sigma 与
  Moore 平衡 d_i=(P_ii·Wo_ii)^{1/4}）都报告，由证书 lane 选。
- **时间缩放 alpha = T0**：λ_hw = λ_model/alpha，取 alpha 使最快特征值
  （max|λ|，实部虚部同时映射）落进频窗：alpha = max|λ|/(2π·B_hw·margin)。
  verdict 用 twin.route.PhysicalScaling.bandwidth_pass 复核。

物理缩放后的描述符（ODE 形式推导，保留 descriptor 形式）：

    x = S⁻¹V,  ẋ = S⁻¹V̇  ⇒  (E S⁻¹) V̇ = (A S⁻¹) V
    τ = t/T0             ⇒  A_v = T0·A·S⁻¹,  E_v = E·S⁻¹

    ⇒ eig(A_v, E_v) = T0 · eig(A, E)      （selftest 里断言）

已知限制：PhysicalScaling.S_x 是标量（twin/route.py:745），本模块只存
S_x_vec 的几何均值作为代表值；完整逐状态向量在 ScalingResult.S_x_vec。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from analysis.descriptor import Descriptor
from analysis.lyapunov import solve_generalized_lyapunov
from analysis.resolvent import resolvent_peak
from twin.route import PhysicalScaling

__all__ = [
    "ScalingResult", "gramian_scaling", "alpha_for_bandwidth",
    "state_scale_from_ranges",
]

_TINY = 1e-300


# ---------------------------------------------------------------------------
# 可观测性 Gramian
# ---------------------------------------------------------------------------
def _observability_gramian(sys: Descriptor) -> np.ndarray | None:
    """直接路径：EᵀWoA + AᵀWoE + CᵀC = 0（stable、cond(E)<1e10）。

    与 solve_generalized_lyapunov 转置调用的方程一致：
    Descriptor(Eᵀ, Aᵀ, Cᵀ, Bᵀ) 解出 EᵀWoA + AᵀWoE + CᵀC = 0，恰好是对偶。
    但转置调用会经过 D≠0 守卫 / 边缘模态路径，可能给出零矩阵或 inf；
    这里只镜像 direct 路径，条件不满足时返回 None（由调用方降级）。
    """
    if not sys.is_stable() or np.linalg.cond(sys.E) >= 1e10:
        return None
    As = linalg.solve(sys.E, sys.A)
    # A_s = E⁻¹A；对偶系统 (A_sᵀ, C_s=E S⁻¹?) —— 保持 E 形式更稳妥：
    # Wo 解 EᵀWoA + AᵀWoE + CᵀC = 0 ⇔ (E⁻¹A)ᵀ Wo (E⁻¹A) 关于 (E⁻¹E⁻ᵀ)…直接用
    # 广义 Lyapunov 的 direct 变换：乘 E⁻ᵀ E⁻¹ 得
    # Wo (E⁻¹A) + (Aᵀ E⁻ᵀ) Wo = −E⁻ᵀ Cᵀ C E⁻¹。
    W_rhs = -(linalg.solve(sys.E.T, sys.C.T) @ (sys.C @ linalg.solve(sys.E, np.eye(sys.n))))
    Wo = linalg.solve_lyapunov(As.T, W_rhs)
    return 0.5 * (Wo + Wo.T)


def _Wo_via_transposed(sys: Descriptor) -> tuple[np.ndarray | None, str]:
    """先试转置调用 solve_generalized_lyapunov；失败则走 _observability_gramian。"""
    try:
        dual = Descriptor(sys.E.T, sys.A.T, sys.C.T, sys.B.T, sys.D.T)
        ly = solve_generalized_lyapunov(dual)
        if ly.h2_finite and np.linalg.norm(ly.P) > 0:
            Wo = 0.5 * (ly.P + ly.P.T)
            return Wo, "transposed-generalized-lyapunov"
    except (ValueError, np.linalg.LinAlgError):
        pass
    Wo = _observability_gramian(sys)
    if Wo is not None:
        return Wo, "direct-dual-lyapunov"
    return None, "unavailable"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
@dataclass
class ScalingResult:
    """P3 的输出证书：S_x 向量、alpha、Γ 前后对照、两条口径。"""

    S_x_vec: np.ndarray              # 每状态满摆尺度（V per x̂ 单位）
    alpha: float                     # t_phys = alpha · t_model
    T0: float                        # = alpha（别名，§4.0 记号）
    physical_scaling: PhysicalScaling  # 标量摘要（S_x = 几何均值）
    method: str                      # "rms3sigma" | fallback 路径名
    d_balanced: np.ndarray | None    # Moore 口径 d_i=(P_ii·Wo_ii)^{1/4}
    Wo: np.ndarray | None
    gamma_before: float
    gamma_after: float
    k_head: float
    bandwidth_pass: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(
            S_x_vec=[float(v) for v in self.S_x_vec],
            alpha=float(self.alpha), T0=float(self.T0),
            S_x_scalar=float(self.physical_scaling.S_x),
            method=self.method,
            d_balanced=([float(v) for v in self.d_balanced]
                        if self.d_balanced is not None else None),
            gamma_before=float(self.gamma_before),
            gamma_after=float(self.gamma_after),
            k_head=float(self.k_head),
            bandwidth_pass=bool(self.bandwidth_pass),
            notes=list(self.notes),
        )


def alpha_for_bandwidth(sys: Descriptor, B_hw: float, margin: float = 0.8) -> float:
    """alpha = max|λ|/(2π·B_hw·margin)（有限特征值；实虚部同时映射）。"""
    lam = sys.finite_eigs()
    if lam.size == 0:
        raise ValueError("没有有限特征值，无法定时间尺度")
    lam_max = float(np.max(np.abs(lam)))
    if lam_max <= 0:
        raise ValueError("max|λ| = 0：纯零谱系统不约束时间尺度")
    return lam_max / (2.0 * np.pi * float(B_hw) * float(margin))


def state_scale_from_ranges(ranges, v_swing: float) -> np.ndarray:
    """范围兜底口径：S_i = v_swing / range_i（range_i = 状态的物理跨度）。"""
    r = np.asarray(ranges, dtype=float).ravel()
    if np.any(r <= 0):
        raise ValueError("ranges 必须为正")
    return float(v_swing) / r


def gramian_scaling(sys: Descriptor, v_swing: float, B_hw: float,
                    margin: float = 0.8, k_head: float = 3.0,
                    ranges=None) -> ScalingResult:
    """计算 S_x（rms3sigma 主口径 + Moore 平衡对照）与 alpha，装配 PhysicalScaling。

    - P 来自 solve_generalized_lyapunov(sys).P（x̂ 单位，须在归一化之后调用）；
      !h2_finite 或 P≈0（不受驱动）时降级：ranges 给了就用
      state_scale_from_ranges，否则用 ‖B‖/max|λ| 的均方启发式——路径记在 method。
    - Wo：先试转置调用（EᵀWoA + AᵀWoE + CᵀC = 0 恰为对偶方程），失败走
      _observability_gramian（stable、cond(E)<1e10），再失败 Wo=None。
    """
    if v_swing <= 0 or B_hw <= 0:
        raise ValueError("v_swing 与 B_hw 必须为正")
    n = sys.n
    notes: list[str] = []

    ly = solve_generalized_lyapunov(sys)
    sqrt_p = np.sqrt(np.clip(np.diag(ly.P), 0.0, None))
    S_x: np.ndarray
    d_balanced: np.ndarray | None
    Wo: np.ndarray | None = None
    if not ly.h2_finite or float(np.max(sqrt_p)) <= 1e-14 * max(1.0, float(np.linalg.norm(sys.A))):
        notes.append("lyapunov: h2_finite=%s, P 不可用 → 降级" % ly.h2_finite)
        if ranges is not None:
            S_x = state_scale_from_ranges(ranges, v_swing)
            method = "ranges-fallback"
        else:
            lam = sys.finite_eigs()
            lam_max = float(np.max(np.abs(lam))) if lam.size else 1.0
            exc = float(np.linalg.norm(sys.B)) / max(lam_max, _TINY)
            sqrt_p = np.full(n, max(exc, 1e-12))
            method = "heuristic-B-over-lambda"
        d_balanced = None
    else:
        method = "rms3sigma"
        Wo, wo_path = _Wo_via_transposed(sys)
        if Wo is None:
            notes.append("Wo 不可用（%s）：Moore 口径缺省" % wo_path)
            d_balanced = None
        else:
            d = (np.clip(np.diag(ly.P), 0.0, None)
                 * np.clip(np.diag(Wo), 0.0, None)) ** 0.25
            d_balanced = d
    if method != "ranges-fallback":
        S_x = float(v_swing) / np.maximum(sqrt_p * float(k_head), _TINY)

    # 时间缩放
    alpha = alpha_for_bandwidth(sys, B_hw, margin)
    S_scalar = float(np.exp(np.mean(np.log(np.clip(S_x, _TINY, None)))))
    ps = PhysicalScaling(S_x=S_scalar, alpha=alpha, v_swing=float(v_swing))

    # 缩放后的描述符：E_v = E·S⁻¹, A_v = alpha·A·S⁻¹ ⇒ eig = alpha·eig(orig)
    Sinv = np.diag(1.0 / S_x)
    scaled = Descriptor(sys.E @ Sinv, alpha * sys.A @ Sinv,
                        sys.B, sys.C @ Sinv, sys.D)

    g_before = float(resolvent_peak(sys)["Gamma"])
    g_after = float(resolvent_peak(scaled)["Gamma"])

    lam = np.abs(sys.finite_eigs())
    bw_ok = ps.bandwidth_pass(lam, B_hw) if lam.size else True

    return ScalingResult(
        S_x_vec=S_x.copy(), alpha=alpha, T0=alpha, physical_scaling=ps,
        method=method, d_balanced=d_balanced, Wo=Wo,
        gamma_before=g_before, gamma_after=g_after,
        k_head=float(k_head), bandwidth_pass=bool(bw_ok), notes=notes,
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def selftest() -> dict:
    """快速验证：特征值不变性、bandwidth_pass、两种 S 口径分离、Wo 对拍 scipy。"""
    rng = np.random.default_rng(7)
    n = 3
    # 稳定非对称系统：A = QΛQ⁻¹, λ = −(0.5, 1, 4)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag([-0.5, -1.0, -4.0]) @ Q.T
    E = np.eye(n)
    B = rng.standard_normal((n, 1))
    C = rng.standard_normal((1, n))
    sys = Descriptor(E, A, B, C)

    res = gramian_scaling(sys, v_swing=1.0, B_hw=1.0)

    # 1) 特征值不变性：eig(A_v, E_v) = alpha·eig(A, E)
    lam0 = np.sort_complex(sys.finite_eigs())
    Sinv = np.diag(1.0 / res.S_x_vec)
    scaled = Descriptor(sys.E @ Sinv, res.alpha * sys.A @ Sinv,
                        sys.B, sys.C @ Sinv)
    lam1 = np.sort_complex(scaled.finite_eigs())
    assert np.allclose(lam1, res.alpha * lam0, rtol=1e-9, atol=1e-9), (lam0, lam1)

    # 2) bandwidth_pass 由构造为真
    assert res.bandwidth_pass, res.as_dict()

    # 3) alpha 公式
    lam_max = float(np.max(np.abs(sys.finite_eigs())))
    assert abs(res.alpha - lam_max / (2 * np.pi * 1.0 * 0.8)) < 1e-12

    # 4) fallback：不受驱动（B=0）→ P=0，给 ranges 就走 ranges-fallback，
    #    且与 Gramian 口径不同（机器有意义）
    sys_u = Descriptor(E, A, np.zeros((n, 1)), C)
    res_r = gramian_scaling(sys_u, v_swing=1.0, B_hw=1.0,
                            ranges=np.full(n, 1.0))
    assert res_r.method == "ranges-fallback", res_r.method
    assert np.allclose(res_r.S_x_vec, 1.0)
    assert not np.allclose(res.S_x_vec, res_r.S_x_vec, rtol=1e-3)
    # 无 ranges 时的启发式兜底
    res_h = gramian_scaling(sys_u, v_swing=1.0, B_hw=1.0)
    assert res_h.method == "heuristic-B-over-lambda", res_h.method
    assert np.all(res_h.S_x_vec > 0)

    # 5) Wo 对拍 scipy（E=I）：AᵀWo + WoA + CᵀC = 0
    Wo = res.Wo
    assert Wo is not None
    Wo_scipy = linalg.solve_lyapunov(A.T, -(C.T @ C))
    assert np.allclose(Wo, Wo_scipy, rtol=1e-8, atol=1e-10), \
        np.max(np.abs(Wo - Wo_scipy))
    # 对偶方程残差
    R = E.T @ Wo @ A + A.T @ Wo @ E + C.T @ C
    assert np.max(np.abs(R)) < 1e-9

    # 6) Moore 平衡向量存在且为正
    assert res.d_balanced is not None and np.all(res.d_balanced > 0)

    # 7) state_scale_from_ranges
    s = state_scale_from_ranges([0.5, 2.0, 4.0], 1.0)
    assert np.allclose(s, [2.0, 0.5, 0.25])

    # 8) as_dict 可 JSON 化
    d = res.as_dict()
    import json
    json.dumps(d)

    return dict(ok=True, method=res.method, gamma_before=res.gamma_before,
                gamma_after=res.gamma_after)
