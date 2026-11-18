"""可行性判定器（DESIGN.md §7 Phase 0）。

纯线性代数，不含求解器与硬件模型：输入 (E,A,B,C)，输出 κ_t、Γ、σ_y、B
与六轴 slack。所有结论都能回到 §0 的三个泛函。
"""

from .descriptor import Descriptor
from .hinf import hinf_norm
from .lyapunov import LyapunovResult, solve_generalized_lyapunov
from .regularity import energy_tail, fourier_metrics, shock_sensor, tail_exponent
from .resolvent import peak_with_projection, resolvent_peak
from .skew import (dde_characteristic_roots, delay_mass_matrix, phase_budget,
                   scalar_dde_root, scalar_tau_crit, skew_descriptor,
                   skew_report, uniform_delay_poles, uniform_tau_crit)
from .slack import HardwareSpec, Slack, evaluate

__all__ = [
    "Descriptor", "LyapunovResult", "solve_generalized_lyapunov",
    "hinf_norm", "resolvent_peak", "peak_with_projection",
    "tail_exponent", "energy_tail", "fourier_metrics", "shock_sensor",
    "delay_mass_matrix", "skew_descriptor", "skew_report",
    "uniform_delay_poles", "dde_characteristic_roots",
    "uniform_tau_crit", "scalar_tau_crit", "scalar_dde_root", "phase_budget",
    "HardwareSpec", "Slack", "evaluate",
]
