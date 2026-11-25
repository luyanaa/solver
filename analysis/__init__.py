"""可行性判定器（DESIGN.md §7 Phase 0）。

纯线性代数，不含求解器与硬件模型：输入 (E,A,B,C)，输出 κ_t、Γ、σ_y、B
与六轴 slack。所有结论都能回到 §0 的三个泛函。
"""

from .descriptor import Descriptor
from .error_propagation import (PropagationClass, classify_propagation,
                                 conservation_drift_rate,
                                 correction_interval, error_recursion_bound,
                                 finite_time_gain, logarithmic_norm,
                                 steady_bound)
from .hinf import hinf_norm
from .lyapunov import LyapunovResult, solve_generalized_lyapunov
from .regularity import energy_tail, fourier_metrics, shock_sensor, tail_exponent
from .resolvent import peak_with_projection, resolvent_peak
from .skew import (dde_characteristic_roots, delay_mass_matrix, phase_budget,
                   scalar_dde_root, scalar_tau_crit, skew_descriptor,
                   skew_report, uniform_delay_poles, uniform_tau_crit)
from .sde import (NoiseRealizability, NoiseSpec, SdeResult,
                  hardware_drift, ito_to_stratonovich, linear_correction,
                  linear_sde_covariance, multiplicative_correction,
                  noise_realizability, ou_variance_deficit, sylvester_operator,
                  stratonovich_to_ito)
from .slack import HardwareSpec, Slack, evaluate

__all__ = [
    "Descriptor", "LyapunovResult", "solve_generalized_lyapunov",
    "PropagationClass", "classify_propagation", "finite_time_gain",
    "logarithmic_norm", "conservation_drift_rate", "correction_interval",
    "steady_bound", "error_recursion_bound",
    "hinf_norm", "resolvent_peak", "peak_with_projection",
    "tail_exponent", "energy_tail", "fourier_metrics", "shock_sensor",
    "delay_mass_matrix", "skew_descriptor", "skew_report",
    "uniform_delay_poles", "dde_characteristic_roots",
    "uniform_tau_crit", "scalar_tau_crit", "scalar_dde_root", "phase_budget",
    "HardwareSpec", "Slack", "evaluate",
    "SdeResult", "linear_sde_covariance", "sylvester_operator",
    "multiplicative_correction", "ito_to_stratonovich",
    "stratonovich_to_ito", "linear_correction", "hardware_drift",
    "NoiseSpec", "NoiseRealizability", "noise_realizability",
    "ou_variance_deficit",
]
