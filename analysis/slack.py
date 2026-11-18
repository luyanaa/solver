"""六轴 slack 向量（DESIGN.md §1.2、§7 Phase 0）。

    slack = (σ_y/σ_req, B_req/B_hw, κ_t/κ_hw, Γ_proj‖ΔA_cons‖, ‖H‖∞‖Δu‖,
             η_route, E_tail/tol_L2²)

鲁棒轴拆成两个对象：Γ_proj‖ΔA_cons‖ 是"离不稳定还有多远"（器件失配），
‖H‖∞‖Δu‖ 是"接口扰动被放大多少"（small-gain）。全 Γ 含守恒零模态的
s=0 极点，只作诊断，不进 slack——否则 Burgers 这类守恒系统会被误判为
"无限脆弱"。

Γ_proj 的使用有一个硬前提（P0-D 合同）：ΔA 必须是**保守型**扰动，
即保持守恒不变子空间（列和/行和结构）不变。验证方法：
ΔA_conservative ⇒ A+dA 的零模态仍在机器精度内（列和仍为零）。
一旦 ΔA 是**泄漏型**（两侧注入电流不相等，列和≠0，如电流镜失配
映射到单条通量边），原来的零模态被激发成有限时间增益或慢漂移，
Γ_proj 会把真正危险的误差排除。泄漏型扰动必须单独计算质量/电荷
漂移率（如 coupling.py 的 dM/dt = (F_A_fv−F_A_spec)−(F_B_fv−F_B_spec)）
或有限时间增益，不得用 Γ_proj‖ΔA‖ 判据。HardwareSpec.delta_A_norm
的调用方负责声明扰动类型；本模块只按声明计算，不做守恒性检查。

任一已评估轴 > 1 即不可行；inf 是"确定违规"，NaN 表示该轴没有输入
（例如没有 PEX 就没有 η_route）。不把 NaN 当 0，也不把"未评估"当"可行"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .descriptor import Descriptor

__all__ = ["HardwareSpec", "Slack", "evaluate"]

_AXES = ["noise", "bandwidth", "stiffness", "robustness_A", "robustness_io",
         "routing", "representation"]

_REASONS = {
    "noise": "add C only on nodes with large ||H_i||₂ (σ² = αkT/C)",
    "bandwidth": "shift A_t or widen the hardware window",
    "stiffness": "delete modes (QSSA/CSP) or use multirate",
    "robustness_A": "conservative ΔA only: recondition slow modes, calibrate, or change basis; leakage ΔA needs drift budget, not Γ_proj",
    "robustness_io": "lower the interface disturbance or the closed-loop gain",
    "routing": "reduce parasitic or reconfigure the switch matrix",
    "representation": "switch basis or move the singular region to flux tiles",
}


@dataclass
class HardwareSpec:
    kappa_hw: float = 1e6          # ω_max/ω_min
    B_hw: float = 10e6             # Hz
    sigma_req: float | None = None  # 输出 RMS 要求；None ⇒ noise 轴不评估
    delta_A_norm: float | None = None      # ‖ΔA‖（器件失配界，**相对原始 A**：
                                           # lowering.budget 在 sc.scaled 上评时
                                           # 按 δ_rel=‖ΔA‖/‖A‖ → ‖ΔA_v‖=δ_rel‖A_v‖
                                           # 缩放，保持 Γ·‖ΔA‖ 量纲一致）
    delta_input_norm: float | None = None  # ‖Δu‖（接口扰动界）
    eta_route: float | None = None   # 已算好的路由误差
    tol_l2: float = 1e-2             # 允许的 L2 表示误差（默认 1%）

    @property
    def etail_req(self) -> float:
        """谱尾能量阈值 = tol_L2²（1% L2 ⇒ 1e-4 能量）。"""
        return self.tol_l2 ** 2


@dataclass
class Slack:
    values: dict[str, float]
    violated: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.violated

    @property
    def status(self) -> str:
        if self.violated:
            return "infeasible"
        if self.unknown:
            return "no violation on evaluated axes"
        return "feasible"

    def as_dict(self) -> dict:
        return dict(values=self.values, violated=self.violated,
                    unknown=self.unknown, reasons=self.reasons,
                    status=self.status)


def _representation(reg: dict, tol_l2: float) -> float:
    """max(E_tail/tol_L2², p_thresh/p)；阈值只在这里定义一次。"""
    p = float(reg.get("p", float("nan")))
    e = float(reg.get("E_tail", float("nan")))
    p_thresh = float(reg.get("p_thresh", 1.5))
    rep_e = e / (tol_l2 ** 2) if np.isfinite(e) else float("nan")
    rep_p = p_thresh / p if np.isfinite(p) and p > 0 else 0.0
    return float(max(rep_e, rep_p)) if np.isfinite(rep_e) else float(rep_p)


def evaluate(sys: Descriptor, spec: HardwareSpec, *,
             lyap=None, reg=None,
             gamma_proj: float | None = None,
             hinf: float | None = None) -> Slack:
    vals: dict[str, float] = {}
    vals["noise"] = (float("nan") if spec.sigma_req is None or lyap is None
                     else (float(lyap.sigma_y / spec.sigma_req) if lyap.h2_finite
                           else float("inf")))
    vals["bandwidth"] = float(sys.bandwidth() / spec.B_hw)
    vals["stiffness"] = float(sys.kappa() / spec.kappa_hw)
    vals["robustness_A"] = (float("nan") if spec.delta_A_norm is None or gamma_proj is None
                            else float(gamma_proj * spec.delta_A_norm))
    vals["robustness_io"] = (float("nan") if spec.delta_input_norm is None or hinf is None
                             else float(hinf * spec.delta_input_norm))
    vals["routing"] = (float("nan") if spec.eta_route is None
                       else float(spec.eta_route))
    vals["representation"] = (float("nan") if reg is None
                              else _representation(reg, spec.tol_l2))
    vals = {k: float(v) for k, v in vals.items()}
    # inf 是确定违规；只有 NaN 才是"没有输入"。
    violated = [k for k in _AXES
                if np.isinf(vals[k]) or (np.isfinite(vals[k]) and vals[k] > 1.0)]
    unknown = [k for k in _AXES if np.isnan(vals[k])]
    reasons = {k: _REASONS[k] for k in violated}
    return Slack(values=vals, violated=violated, unknown=unknown, reasons=reasons)
