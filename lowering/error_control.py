"""§6.22 与约束 fabric 的综合桥：误差控制合同在 T10 上的实例化。

依赖方向：消费 twin.constraint 的 mesh/ConstraintResult/mismatch_floor 与
analysis.error_propagation 的分类/调度原语，产出可被 lowering.budget 聚合
的 evidence 对象（ErrorControlResult.as_dict()）。本模块只做实例化与组装，
不重算前级验收——聚合语义在 budget.py。

三条合同：

1.  **约束网分类**：KCL 松弛 C_y·V̇ = −G·V + I 的 A = −G/C_y 是收缩系统
    （SPD G ⇒ μ₂ < 0），classify_constraint 自动出分类 + 有限时间增益。
    τ_relax 时标以内取检查点——非正规瞬态对纯电导网不会出现（对称
    M-矩阵），但非对称 gm 网络沿用 solve_constraint 的 λ_min((A+Aᵀ)/2)
    守卫与 spectral_abscissa，分类结果必须一致才放行。

2.  **约束网上的 iterative refinement**：模拟稳态解 V_h = G⁻¹(I+δI) 的
    mismatch 偏置会被 KCL 残差 r = I − G·V_m **直接暴露**——I 是数字
    配置（精确已知），残差里的 −δI 项不依赖任何对 δI 的先验估计。
    修正解把 δI 扣掉后，attainable 逼近修正解的量化地板，**低于
    mismatch_floor**。这是 §3.6 "物理残差 R(x)=0（ΣI、ΣV）做自监督"
    的定量版：约束 fabric 在数字残差通路上是自校准的，mismatch 地板
    不是不可逾越的精度极限。可达精度的真正瓶颈是 (a) V 的读出精度
    u_meas（一阶相消：测量与残差看到同一个 δV_m），(b) 修正解本身的
    量化地板（κ·u_a 级）。residual 粗量化（粗 ADC）拖垮 attainable 的
    结论与 §6.22 N1 同构。

3.  **校正调度**：约束网的背景重校间隔不再是配置项——漂移是中性
    累积（λ=0），按调度公式的预算式 ΔT = (ε−e₀)/r_max 反解，并与
    twin/calib.py 的 residual = drift_rate·interval/(1+loop_gain) + floor
    语义对账：两条公式在 loop_gain=0 时重合，残差预测必须一致。

诚实边界：IR 的模拟核是量化 Richardson 行为级（与 run_error_control
同口径），mismatch 注入是 Pelgrom 线性模型（delta_v=None，系统项为零）；
SPICE 级验证是 B2 联合闭环的职责。本模块不宣称修正能消除随机噪声——
噪声零均值部分进方差，不进残差扣除的合同。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.error_propagation import (PropagationClass,
                                        classify_propagation,
                                        correction_interval)
from twin.constraint import ConstraintMesh, mismatch_floor, relaxation_modes

__all__ = [
    "ErrorControlResult",
    "classify_constraint",
    "constraint_refine",
    "constraint_schedule",
]


@dataclass
class ErrorControlResult:
    """运行时误差控制合同的 evidence 对象（供 budget.py 聚合）。

    聚合规则（_error_control_verdict）只读这些字段，不重算：
      * kind/cls 缺失 ⇒ rejected（malformed）；
      * cls == "expanding" 且 finite_horizon=False ⇒ rejected
        （不许诺无限时窗轨迹精度）；
      * n_windows > 0 且 coverage_ratio < 1.0 ⇒ rejected（界未覆盖）；
      * e_measured_max > e_predicted_max ⇒ rejected；
      * 其余 ⇒ pass（IR/调度数字进 detail，不是可行性判决）。
    """
    kind: str = "error_control"
    cls: str = ""
    n_windows: int = 0
    coverage_ratio: float = 0.0
    e_measured_max: float = 0.0
    e_predicted_max: float = 0.0
    eta_mean: float | None = None
    steady_bound: float | None = None
    ir_improve: float | None = None
    schedule_dT: float | None = None
    finite_horizon: bool = False
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(kind=self.kind, cls=self.cls,
                    n_windows=int(self.n_windows),
                    coverage_ratio=float(self.coverage_ratio),
                    e_measured_max=float(self.e_measured_max),
                    e_predicted_max=float(self.e_predicted_max),
                    eta_mean=self.eta_mean, steady_bound=self.steady_bound,
                    ir_improve=self.ir_improve, schedule_dT=self.schedule_dT,
                    finite_horizon=bool(self.finite_horizon),
                    notes=list(self.notes))


def _quantized_richardson(G: np.ndarray, b: np.ndarray, u_a: float,
                          scale: float, iters: int = 400) -> np.ndarray:
    """量化 Richardson（T10 松弛网的行为级求解核）。

    ω = 1/λ_max(G) 保证收缩；量化噪声每步注入，收敛停在
    u_a·scale·√n/λ_min 量级的地板。与 bench/run_error_control 的同名
    模型同口径（那里是独立副本，bench 不得 import lowering——方向
    反了；这里是 lowering 侧的正式实现）。
    """
    lam_max = float(np.linalg.eigvalsh(0.5 * (G + G.T)).max())
    if lam_max <= 0:
        raise ValueError("G 的对称部分不满足 λ_max > 0：不是可松弛网")
    omega = 1.0 / lam_max
    x = np.zeros_like(b)
    rng = np.random.default_rng(0)
    for _ in range(iters):
        x = x + omega * (b - G @ x)
        x = x + u_a * scale * rng.uniform(-1.0, 1.0, size=x.shape)
    return x


def classify_constraint(mesh: ConstraintMesh, c_node: float,
                        n_tau=(0.2, 1.0, 5.0)) -> PropagationClass:
    """约束网的误差传播分类（合同 1）。

    A = −G/C_y，检查点取 τ_relax 的分数倍（τ_relax = C_y/λ_min_sym）。
    纯电导网必然 contracting——若不是，说明 G 不 SPD（有源 gm 网络），
    solve_constraint 的 monotone_spd 守卫本来就该已拦截；这里再次如实
    报告，不静默改判。
    """
    modes = relaxation_modes(mesh)
    if modes["singular"]:
        raise ValueError("mesh 奇异（纯 Neumann）：先由 solve_constraint "
                         "判 SINGULAR，误差合同不评奇异网")
    G = mesh.G
    A = -G / float(c_node)
    tau_relax = float(c_node) / modes["lam_min_sym"]
    horizons = tuple(float(n) * tau_relax for n in n_tau)
    pc = classify_propagation(A, horizons=horizons)
    if pc.cls != "contracting":
        pc.notes.append(
            "约束网分类=%s（期望 contracting）：G 非 SPD 或有源网络，"
            "须与 solve_constraint 的 monotone_spd/spectral_abscissa 对账"
            % pc.cls)
    return pc


def constraint_refine(mesh: ConstraintMesh, i_rhs: np.ndarray, c_node: float,
                      u_a: float = 2.0 ** -10, u_meas: float = 2.0 ** -12,
                      lam: float = 0.0, eps_dev: float = 1e-3,
                      seed: int = 2026, iters: int = 400) -> dict:
    """约束网 iterative refinement（合同 2）：mismatch 偏置被 KCL 残差暴露。

    模型口径：
      * 模拟核：量化 Richardson 松弛到 G·V = I+δI 的稳态（δI 为 Pelgrom
        随机项，I·eps_dev·ξ，种子固定；与 mismatch_floor 同模型同种子
        ⇒ 同一个 δI 实现）；
      * 读出：V_m = V_h + u_meas·‖V‖ 的 ADC 量化；
      * 数字残差：r = I − G·V_m（float64，I 精确已知）；
      * 修正解：量化 Richardson 对 r 松弛，修正量经 DAC 注入（注入
        精度 = u_meas，与读出同级）；
      * 每轮重新读出，残差与量化都如实体现在轨迹里。
    返回实测改进链 + mismatch_floor 预测的对账字段。
    """
    G = mesh.G
    n = mesh.n
    I = np.asarray(i_rhs, dtype=float)
    if I.shape != (n,):
        raise ValueError("i_rhs 形状必须与 mesh 一致")
    V_star = np.linalg.solve(G, I)

    rng = np.random.default_rng(seed)
    dI = I * eps_dev * rng.standard_normal(n)
    scale = max(float(np.max(np.abs(V_star))), 1e-30)

    # 模拟核：对 I+δI 松弛（稳态等价于 V̇ = −G/C·V + (I+δI)/C 的平衡点）
    V_h = _quantized_richardson(G, I + dI, u_a, scale, iters=iters)
    e_analog = float(np.linalg.norm(V_h - V_star))

    # mismatch_floor 预测（同种子 ⇒ 同一 δI 实现）：bound 是保守上界
    mf = mismatch_floor(mesh, I, lam, eps_dev, seed=seed, n_draws=1)

    # IR 主线：数字残差暴露 δI，修正解扣除
    rng_m = np.random.default_rng(seed + 2)     # ADC 读出量化
    rng_i = np.random.default_rng(seed + 3)     # DAC 注入量化
    v_acc = V_h.copy()
    e_meas_hist = []
    for _ in range(4):
        v_meas = v_acc + u_meas * scale * rng_m.uniform(-1.0, 1.0, size=n)
        r = I - G @ v_meas                      # KCL 残差（数字，float64）
        dV = _quantized_richardson(G, r, u_a, scale, iters=iters)
        v_acc = v_acc + dV
        # DAC 注入精度与读出同级
        v_true = v_acc + u_meas * scale * rng_i.uniform(-1.0, 1.0, size=n)
        e_meas_hist.append(float(np.linalg.norm(v_true - V_star)))
    e_final = e_meas_hist[-1]
    kcl_res_analog = float(np.linalg.norm(I - G @ V_h))   # 修正前的 KCL 失衡
    kcl_res_final = float(np.linalg.norm(I - G @ v_acc))
    improve = e_analog / max(e_final, 1e-300)

    # mismatch 分量的解析误差（同一 δI 实现）：bound_inf 是它的保守上界。
    # e_analog 里还叠加量化地板——那是独立预算行，不进这个 bound。
    mismatch_err = float(np.linalg.norm(np.linalg.solve(G, dI)))

    return dict(
        e_analog=e_analog,
        e_final=e_final,
        ir_improve=improve,
        e_history=[e_analog] + e_meas_hist,
        kcl_residual_analog=kcl_res_analog,
        kcl_residual_final=kcl_res_final,
        kcl_residual_reduction=kcl_res_analog / max(kcl_res_final, 1e-300),
        mismatch_floor_inf=mf["floor_inf"],
        mismatch_bound_inf=mf["bound_inf"],
        mismatch_err=mismatch_err,
        mismatch_within_bound=bool(mismatch_err <= mf["bound_inf"]),
        final_below_floor=bool(e_final < mf["floor_inf"]),
        dI_norm=float(np.linalg.norm(dI)),
        u_a=u_a, u_meas=u_meas, eps_dev=eps_dev,
    )


def constraint_schedule(mesh: ConstraintMesh, c_node: float,
                        drift_rate: float, calib_floor: float,
                        eps: float, e0: float = 0.0,
                        loop_gain: float = 0.0) -> dict:
    """约束网校正调度（合同 3）：漂移是中性累积，用预算式反解间隔。

    ΔT = (ε−e₀)/r_max，r_max 取漂移率（有界注入上界）。与 calib.py 的
    background 语义对账：loop_gain=0 时 residual(interval) =
    drift_rate·ΔT + floor 与预算式重合——两条公式必须一致，不一致
    说明其中一条的假设（有界性/线性累积）被违反。
    calib_floor ≥ ε ⇒ raise：地板已在容限之上，任何间隔都救不了。
    """
    if calib_floor >= eps:
        raise ValueError("calib_floor=%.3g ≥ eps=%.3g：校准地板已在容限之上，"
                         "调度无解（先修硬件，不是调间隔）" % (calib_floor, eps))
    # 预算式的容限要扣掉地板：drift·ΔT/(1+T) + floor ≤ eps
    #   ⇒ ΔT ≤ (eps−floor)·(1+T)/drift（e0=0 时与 calib 语义严格重合）
    eps_head = (eps - calib_floor) * (1.0 + loop_gain)
    dT = correction_interval(0.0, drift_rate, e0, eps_head)
    calib_residual = drift_rate * dT / (1.0 + loop_gain) + calib_floor
    return dict(
        dT=dT,
        predicted_residual=calib_residual,
        calib_consistent=bool(calib_residual <= eps),
        loop_gain=loop_gain,
        drift_rate=drift_rate,
        calib_floor=calib_floor,
    )
