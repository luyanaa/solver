"""精度作为编译优化变量（§6.24）：边界位宽按灵敏度分配。

原来的 P7 只回答"给定 ε_J，每个器件要多准"。这里把方向反过来：

    min_θ  E_solve(θ)   s.t.  ε_global ≤ ε_x, ε_J ≤ ε_J,req,
                             T_solve ≤ T_deadline, R_link ≤ R_capacity

θ 至少包含边界位宽 b_i、采样率 f_s、物理时间缩放 α（已由 P3 定）与
校正周期 ΔT_cal（已由 calib 反解）。

量化误差的确定性上界（每条边界 i，相对满量程）：

    |δJ| ≤ Σ_i K_i · (V_FS,i / 2^{b_i+1})

其中 K_i 是 **相对** 灵敏度 |∂J/∂lnθ_i|（`analysis.portgraph` 的 K_rel）。
等贡献分配 K_i·e_i = ε_J/n_alloc 给出连续位宽：

    b_i = log2(n_alloc·K_i / ε_J) − 1

随后取整/夹到 [min_bits, max_bits]。整数化有两种锚：
  * anchor="max"  最高灵敏度边界取 max_bits，其余按 round(log2(K_i/K_max))
                偏移（这就是 §6.24 示例 16/15/13/11/10 的产生规则）；
  * anchor="budget" 直接由 ε_J 解 b_i（可能低于 max_bits）。

能耗诚实口径：ADC 每次转换的能量按 Walden 型 E0·2^b 计。**只有转换器
真正支持可变精度（可变比较次数/采样配置）时，降低位宽才省能**；否则
converter 仍按 max_bits 工作，省下来的只是链路带宽（`variable_resolution`
=False 时如实返回同样的转换能耗，不假装省电）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BitPlan", "allocate_bits", "uniform_bits", "energy_model",
           "precision_plan", "optimize_precision"]


@dataclass
class BitPlan:
    bits: list
    K_rel: list
    eps_rel: float
    achieved: float
    feasible: bool
    total_bits: int
    anchor: str
    note: str = ""

    def as_dict(self) -> dict:
        return dict(bits=[int(b) for b in self.bits],
                    K_rel=[float(k) for k in self.K_rel],
                    eps_rel=float(self.eps_rel),
                    achieved=float(self.achieved),
                    feasible=bool(self.feasible),
                    total_bits=int(self.total_bits),
                    anchor=self.anchor, note=self.note)


def allocate_bits(K_rel, eps_rel: float, max_bits: int = 16,
                  min_bits: int = 10, anchor: str = "max") -> BitPlan:
    """按相对灵敏度分配边界位宽。

    K_rel 是 |∂J/∂lnθ|（无量纲 QoI / 相对参数误差）；eps_rel 是 QoI 合同
    （无量纲，ppm/1e6）。量化项 K_i/2^{b_i+1} 的和必须 ≤ eps_rel。
    """
    K = np.asarray(K_rel, dtype=float).ravel()
    if np.any(K < 0):
        raise ValueError("K_rel 必须非负")
    if eps_rel <= 0:
        raise ValueError("eps_rel 必须 > 0")
    n = K.size
    kmax = float(K.max())
    if kmax <= 0:
        raise ValueError("K_rel 全零：无分配对象")

    if anchor == "max":
        # 最高灵敏度边界取满位宽；其余按相对灵敏度偏移（示例表的规则）
        offsets = np.round(np.log2(np.maximum(K, 1e-300) / kmax))
        bits = np.clip(max_bits + offsets, min_bits, max_bits).astype(int)
        note = ("anchor=max_bits: b_i = %d + round(log2 K_i/K_max), "
                "clipped to [%d,%d]" % (max_bits, min_bits, max_bits))
    elif anchor == "budget":
        # 等贡献连续解取整（每项 K_i/2^{b+1} = eps/n）
        b_cont = np.log2(n * np.maximum(K, 1e-300) / eps_rel) - 1.0
        bits = np.clip(np.ceil(b_cont), min_bits, max_bits).astype(int)
        note = ("anchor=budget: ceil(log2(n K_i/eps) - 1), clipped to "
                "[%d,%d]" % (min_bits, max_bits))
    else:
        raise ValueError("anchor ∈ {'max','budget'}")

    terms = K / np.power(2.0, bits + 1)
    achieved = float(np.sum(terms))
    feasible = bool(achieved <= eps_rel)
    if not feasible:
        note += ("; achieved %.4g > eps %.4g: infeasible at these bit "
                 "limits (raise max_bits, relax eps, or drop a boundary)"
                 % (achieved, eps_rel))
    return BitPlan(bits=bits.tolist(), K_rel=K.tolist(), eps_rel=float(eps_rel),
                   achieved=achieved, feasible=feasible,
                   total_bits=int(np.sum(bits)), anchor=anchor, note=note)


def uniform_bits(K_rel, eps_rel: float, max_bits: int = 16) -> dict:
    """两个均匀基线：满足 ε 的最小均匀位宽；与差异化方案等界时的位宽。"""
    K = np.asarray(K_rel, dtype=float).ravel()
    n = K.size
    summed = float(K.sum())
    # 满足 eps 的最小 b：ΣK/2^{b+1} ≤ eps
    b_min = int(np.ceil(np.log2(summed / eps_rel) - 1.0))
    b_eq = int(np.ceil(np.log2(summed / max(float(np.sum(K / 2.0 ** (
        np.asarray(allocate_bits(K, eps_rel, max_bits=max_bits).bits) + 1))),
        1e-300)) - 1.0))
    return dict(uniform_for_eps=b_min, total_for_eps=n * b_min,
                uniform_at_equal_bound=b_eq, total_at_equal_bound=n * b_eq,
                total_at_max=n * max_bits)


def energy_model(bits, fs, max_bits: int = 16, E0: float = 1.0,
                 E_link_per_bit: float = 0.0,
                 variable_resolution: bool = True) -> dict:
    """ADC + 链路能耗（相对单位）。

    variable_resolution=True：E_conv ∝ E0·2^{b}（Walden 型），降位宽省能；
    False：转换器固定按 max_bits 工作，只有链路项随 b 变化——不许诺
    "降位宽一定省电"。
    """
    bits = np.asarray(bits, dtype=float).ravel()
    n = bits.size
    if variable_resolution:
        e_conv = E0 * np.power(2.0, bits)
    else:
        e_conv = E0 * np.power(2.0, max_bits) * np.ones(n)
    e_link = E_link_per_bit * bits
    e_tot = (e_conv + e_link) * float(fs)
    return dict(e_conv=float(np.sum(e_conv)),
                e_link=float(np.sum(e_link)),
                e_total=float(np.sum(e_tot)),
                e_conv_per_sample=float(np.sum(e_conv)),
                variable_resolution=bool(variable_resolution))


def precision_plan(K_rel, eps_rel: float, fs: float, max_bits: int = 16,
                   min_bits: int = 10, anchor: str = "max",
                   E0: float = 1.0, E_link_per_bit: float = 0.0,
                   variable_resolution: bool = True,
                   link_capacity_bps: float | None = None) -> dict:
    """单点精度计划：位宽 + 误差上界 + 能耗 + 链路负载。"""
    bp = allocate_bits(K_rel, eps_rel, max_bits=max_bits, min_bits=min_bits,
                       anchor=anchor)
    en = energy_model(bp.bits, fs, max_bits=max_bits, E0=E0,
                      E_link_per_bit=E_link_per_bit,
                      variable_resolution=variable_resolution)
    load = float(np.sum(np.asarray(bp.bits) * float(fs)))
    feasible = bp.feasible
    reasons = []
    if not bp.feasible:
        reasons.append("quantization bound %.3g > eps %.3g"
                       % (bp.achieved, eps_rel))
    if link_capacity_bps is not None and load > link_capacity_bps:
        feasible = False
        reasons.append("link load %.1f bps > capacity %.1f bps"
                       % (load, link_capacity_bps))
    return dict(bits=bp.bits, total_bits=bp.total_bits,
                achieved=bp.achieved, eps_rel=float(eps_rel),
                feasible=bool(feasible), reasons=reasons,
                energy=en, link_load_bps=load, fs=float(fs),
                anchor=anchor, variable_resolution=bool(variable_resolution))


def optimize_precision(K_rel, eps_rel: float, T_deadline: float,
                       fs_grid=None, max_bits: int = 16, min_bits: int = 10,
                       anchor: str = "max", E0: float = 1.0,
                       E_link_per_bit: float = 0.0,
                       variable_resolution: bool = True,
                       link_capacity_bps: float | None = None) -> dict:
    """在 (fs, 位宽) 上做最小能耗搜索，约束 ε_J / deadline / link。

    采样率下界由 deadline 推出：fs ≥ 1/T_deadline（窗口至少一个采样）。
    能耗随 fs 单调增，故最优在网格最小可行点；链路容量随 fs 增而收紧，
    两者可能冲突——冲突时如实返回 infeasible，不给"看起来可行"的配置。
    """
    fs_min = 1.0 / float(T_deadline)
    grid = (np.array([fs_min, 2 * fs_min, 4 * fs_min])
            if fs_grid is None else np.asarray(fs_grid, dtype=float))
    grid = grid[grid >= fs_min - 1e-15]
    best, tried = None, []
    for fs in grid:
        p = precision_plan(K_rel, eps_rel, fs, max_bits=max_bits,
                           min_bits=min_bits, anchor=anchor, E0=E0,
                           E_link_per_bit=E_link_per_bit,
                           variable_resolution=variable_resolution,
                           link_capacity_bps=link_capacity_bps)
        tried.append(dict(fs=float(fs), energy=p["energy"]["e_total"],
                          feasible=p["feasible"], reasons=p["reasons"]))
        if p["feasible"] and (best is None or
                              p["energy"]["e_total"] < best["energy"]["e_total"]):
            best = p
    if best is None:
        return dict(feasible=False, plan=None, tried=tried,
                    reasons=["no (fs, bits) point satisfies eps/link/deadline "
                             "simultaneously"])
    return dict(feasible=True, plan=best, tried=tried, reasons=[])
