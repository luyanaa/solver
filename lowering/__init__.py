"""前端 lowering（DESIGN.md §1.5、§4.0）。

reduce：方程 → 稀疏多项式/有理 IR（P1）。
normalize：无量纲化 + 坐标中心化（P0）。
scaling：状态/时间缩放，A_t 公式 + 频窗检查（P3）。
"""

from .discretize import Discretized, PDESpec, discretize, linearize_rhs
from .normalize import (Normalization, center_system, fold,
                        nondimensionalize, pipeline_normalize)
from .reduce import (Canonical, Expr, Reduced, System, add, besselj, C,
                     complete_env, cos, div, eval_expr, eval_system, exp,
                     first_order, gamma, mul, pow, reduce_system, sin, sub,
                     tanh, V)
from .scaling import (ScalingResult, alpha_for_bandwidth, gramian_scaling,
                      state_scale_from_ranges)
from .p7 import CalibPlan, Config, ElementConfig, generate_config

__all__ = [
    "Expr", "C", "V", "add", "mul", "pow", "sub", "div",
    "sin", "cos", "exp", "tanh", "besselj", "gamma",
    "System", "Canonical", "Reduced", "first_order", "reduce_system",
    "eval_expr", "eval_system", "complete_env",
    "Normalization", "center_system", "nondimensionalize",
    "pipeline_normalize", "fold",
    "PDESpec", "Discretized", "discretize", "linearize_rhs",
    "ScalingResult", "gramian_scaling", "alpha_for_bandwidth",
    "state_scale_from_ranges",
    "ElementConfig", "CalibPlan", "Config", "generate_config",
]
