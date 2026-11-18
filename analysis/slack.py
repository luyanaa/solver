"""六轴 slack 向量（DESIGN.md §1.2、§7 Phase 0）。

    slack = (σ_y/σ_req, B_req/B_hw, κ_t/κ_hw, Γ‖Δ‖∞, η_route, E_tail/E_tail_req)

任一已评估轴 >1 即不可行；NaN 表示该轴没有输入（例如没有 PEX 就没有 η_route）。
不把 NaN 当 0，也不把"未评估"当"可行"——这是 §1.2 的诚实版本。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .descriptor import Descriptor

__all__ = ["HardwareSpec", "Slack", "evaluate"]

_AXES = ["noise", "bandwidth", "stiffness", "robustness", "routing", "representation"]

_REASONS = {
    "noise": "add C only on nodes with large ||H_i||₂ (σ² = αkT/C)",
    "bandwidth": "shift A_t or widen the hardware window",
    "stiffness": "delete modes (QSSA/CSP) or use multirate",
    "robustness": "lower Q / calibrate / change basis",
    "routing": "reduce parasitic or reconfigure the switch matrix",
    "representation": "switch basis or move the singular region to flux tiles",
}


@dataclass
class HardwareSpec:
    kappa_hw: float = 1e6          # ω_max/ω_min
    B_hw: float = 10e6             # Hz
    sigma_req: float | None = None  # 输出 RMS 要求；None ⇒ noise 轴不评估
    delta_norm: float | None = None  # ‖Δ‖∞（结构化界）
    eta_route: float | None = None   # 已算好的路由误差
    etail_req: float = 1e-4          # 允许的谱尾能量占比


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


def evaluate(sys: Descriptor, spec: HardwareSpec, *,
             lyap=None, reg=None, gamma: float | None = None) -> Slack:
    vals: dict[str, float] = {}
    vals["noise"] = (float("nan") if spec.sigma_req is None or lyap is None
                     else (float(lyap.sigma_y / spec.sigma_req) if lyap.h2_finite
                           else float("inf")))
    vals["bandwidth"] = float(sys.bandwidth() / spec.B_hw)
    vals["stiffness"] = float(sys.kappa() / spec.kappa_hw)
    vals["robustness"] = (float("nan") if spec.delta_norm is None or gamma is None
                          else float(gamma * spec.delta_norm))
    vals["routing"] = (float("nan") if spec.eta_route is None
                       else float(spec.eta_route))
    vals["representation"] = (float("nan") if reg is None
                              else float(reg["representation"]))
    vals = {k: float(v) for k, v in vals.items()}
    violated = [k for k in _AXES if np.isfinite(vals[k]) and vals[k] > 1.0]
    unknown = [k for k in _AXES if not np.isfinite(vals[k])]
    reasons = {k: _REASONS[k] for k in violated}
    return Slack(values=vals, violated=violated, unknown=unknown, reasons=reasons)
