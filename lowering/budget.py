"""P6 可行性证书聚合（DESIGN.md §4 P6 / §10 budget.py 骨架）。

P0–P7 流水线里，前面各阶段分别产出**已算好的证据对象**：

    P0  (E,A,B,C) Descriptor + 谱指标      → analysis.descriptor
    P0  六轴 slack 判据                    → analysis.slack（§1.2）
    P0  σ_y²（广义 Lyapunov）              → analysis.lyapunov
    P0  Γ / Γ_proj（resolvent 峰值）        → analysis.resolvent
    P0  ‖H‖∞                               → analysis.hinf
    P1  manifold-γ 可行窗 [γ_min, γ_max]   → lowering.reduce / bench.run_reduce
    P2  loop 闭环敏感度峰 ‖S‖∞             → twin.route.loop_margins（§6.15）
    P3  multirate 合法性（ρ_sup, k_max）   → twin.multirate（§4.3）
    P4  随机证书（noise_realizability）    → analysis.sde（§1.3/§4.4）

本模块只做**聚合**：把各证据拼成结构化 Certificate 或 Rejection，不改任何
一项的判定语义。规则与 §1.2 的"不把未评估当可行"一致：

  * 结构化拒绝（manifold 空窗 / ‖S‖∞ 超阈 / ρ_sup≥1 / k_max<1 / sde 违规 /
    constraint fabric ε_c 违约或 deadline 未收敛）
    ⇒ status="rejected"，拒绝原因逐条进 rejection_reasons；
  * 只有 slack 轴违规 ⇒ status="infeasible"，binding_axis 取被违反的轴；
  * 有 NaN 轴（未评估）⇒ status="no violation on evaluated axes"，
    **不写 "feasible"**——证书只覆盖已评估的轴。

assumptions 列出会让证书失效的假设合同（§1.2/§4.3/§4.4/§6），聚合层
永远假设前级产物是**已验证**的；本模块不重跑前级的验收。

依赖方向：budget.py 不 import lowering.normalize / lowering.scaling
（并行 lane，证书只消费预算对象）。bench 侧同理自建测试系统。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from analysis.descriptor import Descriptor
from analysis.hinf import hinf_norm
from analysis.lyapunov import LyapunovResult, solve_generalized_lyapunov
from analysis.resolvent import peak_with_projection
from analysis.slack import HardwareSpec, Slack, evaluate as slack_evaluate

__all__ = ["BudgetInput", "Certificate", "compile_budget", "evaluate_sys"]


@dataclass
class BudgetInput:
    """P6 的输入账本：每个字段对应一条已算出的证据；None ⇒ 该项不评估。"""

    sys: Descriptor | None = None          # (E,A,B,C) — slack/Γ/σ_y 的载体
    spec: HardwareSpec | None = None       # 六轴判据
    lyap: LyapunovResult | None = None     # σ_y² 证据（None ⇒ 自算）
    reg: dict | None = None                # regularity/shock_sensor 输出
    gamma_M: float | None = None           # 度量 Γ（peak_with_projection(metric=...)）
    hinf: float | None = None
    manifold: dict | None = None           # {"gamma_min","gamma_max","chosen"} 来自 reduce/bench
    loop: dict | None = None               # loop_margins 输出（S_peak 等）
    multirate: dict | None = None          # {"rho_sup","k_max","eps_replay",...} 来自 twin.multirate
    sde_cert: dict | None = None           # noise_realizability 输出（随机证书，§4.4）
    metric: object | None = None           # Γ 的度量矩阵 / "energy"（§5.8）
    constraint: dict | None = None         # twin.constraint.ConstraintResult.as_dict()
                                           # T10 约束 fabric 时序证据：eps_c（对
                                           # 显式声明的 external_state_timescale）
                                           # 或纯稳态 deadline/residual 事实
    error_control: dict | None = None      # lowering.error_control.ErrorControlResult.as_dict()
                                           # §6.22 运行时误差控制合同：分类/覆盖/
                                           # IR/调度。字段缺失或界未覆盖 ⇒ rejected
    require_error_control: bool = False    # workload 声明动态误差合同 ⇒ 没有
                                           # evidence 也必须拒绝（不许静默继续编译）
    scaling: object | None = None          # ScalingResult：给定时 slack/Γ/σ_y/‖H‖∞
                                           # 在其 .scaled（物理时间）descriptor 上评；
                                           # delta_A_norm 按相对失配自动缩放
                                           # （δ_rel·‖A_v‖），防"缩时间白嫖裕度"


# ‖S‖∞ 的阈值（loop 轴）：§6.15 的必要条件探针。靠近失稳时 S_peak 发散，
# 有限带宽网格上峰值冲破该值就拒绝（阈值是合同参数，不是物理常数）。
S_PEAK_THRESHOLD = 50.0


@dataclass
class Certificate:
    feasible: bool
    status: str                  # feasible | no violation on evaluated axes | infeasible | rejected
    slack: dict | None           # Slack.as_dict()
    items: dict                  # 每项证据的 {name: {value, verdict, detail}}
    binding_axis: str | None     # slack 里 >1 或最接近 1 的轴名
    constraint: dict | None = None   # T10 约束时序 verdict（constraint 证据）
    error_control: dict | None = None   # §6.22 误差控制合同 verdict
    rejection_reasons: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(
            feasible=bool(self.feasible), status=self.status,
            slack=self.slack, items=self.items,
            binding_axis=self.binding_axis,
            constraint=(dict(self.constraint) if self.constraint is not None else None),
            error_control=(dict(self.error_control) if self.error_control is not None else None),
            rejection_reasons=list(self.rejection_reasons),
            assumptions=list(self.assumptions),
        )


def _fmt(v) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "inf" if fv > 0 else ("-inf" if fv < 0 else "nan")
    return "%.4g" % fv


def _manifold_verdict(manifold: dict):
    """γ 可行窗检查：窗空（γ_min≥γ_max）或 chosen 不在开区间内都拒绝。

    §4.3/P1：Baumgarte γ 只在 [γ_min, γ_max] 内既镇定流形又不把刚体
    模态推得太快；窗口为空 ⇒ 该前端在这一档下无法稳定，必须换基。
    """
    g_min = manifold.get("gamma_min")
    g_max = manifold.get("gamma_max")
    chosen = manifold.get("chosen")
    try:
        g_min_f, g_max_f = float(g_min), float(g_max)
    except (TypeError, ValueError):
        return "rejected", float("nan"), "manifold window malformed"
    empty = not (g_min_f < g_max_f)
    inside = (chosen is not None and np.isfinite(float(chosen))
              and g_min_f < float(chosen) < g_max_f)
    if empty:
        return "rejected", float("nan"), (
            "manifold stabilization infeasible: no γ in [γ_min,γ_max]")
    if not inside:
        c = "None" if chosen is None else _fmt(chosen)
        return "rejected", (float(chosen) if chosen is not None else float("nan")), (
            "manifold stabilization infeasible: chosen γ=%s outside "
            "(γ_min=%s, γ_max=%s)" % (c, _fmt(g_min_f), _fmt(g_max_f)))
    return "pass", float(chosen), "chosen γ=%s ∈ (%s, %s)" % (
        _fmt(chosen), _fmt(g_min_f), _fmt(g_max_f))


def _loop_verdict(loop: dict, threshold: float):
    """‖S‖∞ 检查：loop_margins 的 S_peak < threshold（§6.15）。

    loop_margins 是必要条件探针（σ_min(jωI−J∘H) 最近距离 + S=(I+L)⁻¹ 峰），
    不是 DDE 精确根——超阈值拒绝，未超也只说明"该网格上未见失稳迹象"。
    """
    s_peak = loop.get("S_peak")
    try:
        s_f = float(s_peak)
    except (TypeError, ValueError):
        return "rejected", float("nan"), "loop S_peak missing/malformed"
    if not np.isfinite(s_f):
        return "rejected", s_f, "loop S_peak non-finite (grid near instability)"
    if s_f >= threshold:
        return "rejected", s_f, (
            "loop sensitivity S_peak=%s exceeds threshold %s"
            % (_fmt(s_f), _fmt(threshold)))
    return "pass", s_f, "S_peak=%s < threshold %s" % (_fmt(s_f), _fmt(threshold))


def _multirate_verdict(mr: dict):
    """WR 合法性：ρ_sup<1（收敛）且 k_max≥1（迭代预算非空）。§4.3。"""
    rho = mr.get("rho_sup")
    kmax = mr.get("k_max")
    reasons = []
    try:
        rho_f = float(rho)
    except (TypeError, ValueError):
        rho_f = float("nan")
    try:
        k_f = float(kmax)
    except (TypeError, ValueError):
        k_f = float("nan")
    if not np.isfinite(rho_f) or rho_f >= 1.0:
        reasons.append("multirate split divergent: rho_sup=%s ≥ 1" % _fmt(rho_f))
    if (not np.isfinite(k_f)) or k_f < 1.0:
        reasons.append("multirate split illegal: k_max=%s < 1" % _fmt(kmax))
    if reasons:
        return "rejected", dict(rho_sup=rho_f, k_max=kmax), "; ".join(reasons)
    return "pass", dict(rho_sup=rho_f, k_max=kmax), (
        "rho_sup=%s < 1, k_max=%d" % (_fmt(rho_f), int(k_f)))


def _sde_verdict(cert: dict):
    """随机证书检查（§1.3 噪声合同 + §4.4）。

    noise_realizability 的硬判据是 B_noise/B_req ≥ 100，10–100 是警告带；
    聚合层的合同：**< 10 拒绝**（白噪声近似失效），10–100 记为带警告的
    pass（assumptions 里显式留痕）。tau_c·λ > 0.01 时 Itô 语义有 O(τ_c)
    偏差，同样进 warnings；聚合层不重判，只读它的字段。
    """
    ratio = cert.get("ratio")
    passes_10 = bool(cert.get("passes_10x", False))
    warnings = list(cert.get("warnings", []))
    detail = "B_noise/B_req=%s, passes_10x=%s, passes_100x=%s" % (
        _fmt(ratio), cert.get("passes_10x"), cert.get("passes_100x"))
    if warnings:
        detail += "; warnings: " + " | ".join(warnings)
    if not passes_10:
        return "rejected", {"ratio": ratio, "warnings": warnings}, (
            "sde_cert noise contract violated: B_noise/B_req=%s < 10 "
            "(white-noise approximation fails)" % _fmt(ratio))
    return "pass", {"ratio": ratio, "warnings": warnings}, detail


def _constraint_verdict(cons: dict):
    """T10 约束 fabric 时序检查（twin.constraint.ConstraintResult 字段）。

    语义（ε_c 修复后的合同）：
      * FAIL / SINGULAR            → rejected（ε_c 违约或网格奇异）；
      * STEADY（纯稳态事实，无 ε_c）→ pass，但 detail 明确"这不是可行性判决，
        是 deadline/residual 事实"——纯 Poisson 不被伪装成 infeasible；
      * PASS                       → pass（ε_c 已对声明的 external_state_
        timescale 判定 < 0.01，或 deadline 内收敛）。
    聚合层不重算 ε_c，只读前级字段；eps_c ≥ 0.01 的 FAIL 必须让证书不再
    "no violation"（约束失败不许埋在 side-record 里）。
    """
    verdict = cons.get("verdict")
    eps = cons.get("eps_c")
    detail = ("verdict=%s, eps_c=%s, steady_state=%s, converged=%s, "
              "residual_at_deadline=%s"
              % (verdict, _fmt(eps), cons.get("steady_state"),
                 cons.get("converged"), _fmt(cons.get("residual_at_deadline"))))
    if verdict in ("FAIL", "SINGULAR"):
        return "rejected", dict(cons), ("constraint fabric %s: %s"
                                        % (verdict, detail))
    if verdict == "STEADY":
        return "pass", dict(cons), (
            "constraint steady-state fact (no ε_c defined, no external "
            "dynamic state declared): tau_relax=%s; not a feasibility verdict"
            % _fmt(cons.get("tau_relax")))
    if verdict == "PASS":
        return "pass", dict(cons), detail
    return "rejected", dict(cons), "constraint evidence malformed (no verdict)"


def _error_control_verdict(ec: dict):
    """§6.22 运行时误差控制合同检查（ErrorControlResult 字段）。

    聚合语义（与 _constraint_verdict 同风格——只读字段，不重算）：
      * 缺 kind/cls                     ⇒ rejected（malformed 证据）；
      * cls == "expanding" 且无
        finite_horizon 声明             ⇒ rejected（不许诺无限时窗
        轨迹精度——这是合同的终点，不是告警）；
      * n_windows > 0 且 coverage < 1.0 ⇒ rejected（界未覆盖：预测界
        没罩住实测误差，合同失效）；
      * e_measured_max > e_predicted_max ⇒ rejected（同上，逐窗覆盖
        之外的总账复核）；
      * 其余                            ⇒ pass，IR/调度数字进 detail。
        n_windows == 0（纯 IR 静态事实）也 pass，但 detail 明确
        "这不是动态可行性判决"——与 constraint 的 STEADY 同口径。
    """
    kind = ec.get("kind")
    cls = ec.get("cls")
    if kind != "error_control" or cls not in ("contracting", "neutral",
                                              "expanding"):
        return "rejected", dict(ec), "error_control evidence malformed "\
            "(kind=%s, cls=%s)" % (_fmt(kind), _fmt(cls))
    finite = bool(ec.get("finite_horizon", False))
    if cls == "expanding" and not finite:
        return "rejected", dict(ec), (
            "error_control expanding without finite_horizon declaration: "
            "no infinite-horizon trajectory promise")
    if finite:
        hT = ec.get("horizon_T")
        if (not isinstance(hT, (int, float)) or not math.isfinite(hT)
                or hT <= 0):
            return "rejected", dict(ec), (
                "error_control finite_horizon=True without a concrete "
                "positive horizon_T: cannot verify the runtime window")
    n_win = ec.get("n_windows", 0)
    cov = ec.get("coverage_ratio", 0.0)
    e_meas = ec.get("e_measured_max")
    e_pred = ec.get("e_predicted_max")
    if not isinstance(cov, (int, float)) or not math.isfinite(cov) \
            or cov < 0.0 or cov > 1.0:
        return "rejected", dict(ec), (
            "error_control coverage_ratio %s not finite in [0,1]"
            % _fmt(cov))
    if isinstance(e_meas, (int, float)) and not math.isfinite(e_meas):
        return "rejected", dict(ec), (
            "error_control e_measured_max non-finite: not a pass")
    if isinstance(e_pred, (int, float)) and not math.isfinite(e_pred):
        return "rejected", dict(ec), (
            "error_control e_predicted_max non-finite: not a pass")
    detail = ("cls=%s, n_windows=%s, coverage=%s, e_measured_max=%s, "
              "e_predicted_max=%s, ir_improve=%s, schedule_dT=%s"
              % (cls, _fmt(n_win), _fmt(cov), _fmt(e_meas), _fmt(e_pred),
                 _fmt(ec.get("ir_improve")), _fmt(ec.get("schedule_dT"))))
    if ec.get("qoi_estimate") is not None:
        detail += (", qoi_estimate=%s, qoi_tolerance=%s, "
                   "qoi_cancellation=%s (estimate, not a bound)"
                   % (_fmt(ec.get("qoi_estimate")), _fmt(ec.get("qoi_tolerance")),
                      _fmt(ec.get("qoi_cancellation"))))
    cov_thr = ec.get("coverage_threshold", 1.0)
    if isinstance(n_win, (int, float)) and n_win > 0 and not (
            isinstance(cov, (int, float)) and cov >= cov_thr):
        return "rejected", dict(ec), (
            "error_control bound coverage %s < threshold %s over %s windows: %s"
            % (_fmt(cov), _fmt(cov_thr), _fmt(n_win), detail))
    if not isinstance(e_meas, (int, float)) or not isinstance(e_pred,
                                                              (int, float)):
        return "rejected", dict(ec), (
            "error_control e_measured_max/e_predicted_max missing: %s" % detail)
    # ε_J 双轨：qoi_tolerance 声明才判（§6.22 续）。δJ₁ 是估计不是上界——
    # cancellation_factor 进 detail，抵消严重时靠 ε_global 兜底。
    qoi_tol = ec.get("qoi_tolerance")
    qoi_est = ec.get("qoi_estimate")
    if qoi_tol is not None:
        if not isinstance(qoi_est, (int, float)):
            return "rejected", dict(ec), (
                "error_control qoi_tolerance declared but qoi_estimate "
                "missing: %s" % detail)
        if not math.isfinite(qoi_est):
            return "rejected", dict(ec), (
                "error_control qoi_estimate non-finite (NaN/inf): "
                "not a pass — %s" % detail)
        rem = ec.get("qoi_remainder", 0.0)
        if not isinstance(rem, (int, float)) or not math.isfinite(rem) \
                or rem < 0:
            return "rejected", dict(ec), (
                "error_control qoi_remainder invalid: %s" % _fmt(rem))
        if abs(qoi_est) + rem > qoi_tol:
            return "rejected", dict(ec), (
                "error_control eps_J violated: |qoi_estimate| %s + "
                "B_remainder %s > tolerance %s (cancellation=%s): %s"
                % (_fmt(abs(qoi_est)), _fmt(rem), _fmt(qoi_tol),
                   _fmt(ec.get("qoi_cancellation")), detail))
    if int(ec.get("allocation_infeasible", 0)) > 0:
        return "rejected", dict(ec), (
            "error_control P7 allocation infeasible for %s elements "
            "(budget < measurement floor): redesign devices or relax tol"
            % ec.get("allocation_infeasible"))
    # ε_global 是独立条件：界覆盖 ≠ 精度达标（pred=101, measured=100 仍超标）
    g_tol = ec.get("global_tolerance")
    if g_tol is not None:
        if (not isinstance(g_tol, (int, float)) or not math.isfinite(g_tol)
                or g_tol <= 0 or e_pred > g_tol):
            return "rejected", dict(ec), (
                "error_control eps_global accuracy violated: predicted %s > "
                "tolerance %s (coverage alone is not accuracy): %s"
                % (_fmt(e_pred), _fmt(g_tol), detail))
    if e_meas > e_pred:
        return "rejected", dict(ec), (
            "error_control e_measured_max %s > e_predicted_max %s: %s"
            % (_fmt(e_meas), _fmt(e_pred), detail))
    if not n_win:
        return "pass", dict(ec), (
            "error_control static IR facts (no dynamic windows; not a "
            "dynamic feasibility verdict): %s" % detail)
    return "pass", dict(ec), detail


def compile_budget(inp: BudgetInput, s_max_threshold: float = S_PEAK_THRESHOLD) -> Certificate:
    """聚合 P0–P5 的证据 → P6 证书。只判聚合语义，不重算前级验收。"""
    items: dict[str, dict] = {}
    reasons: list[str] = []
    assumptions: list[str] = []
    slack_block: dict | None = None
    binding_axis: str | None = None
    constraint_val: dict | None = None
    error_control_val: dict | None = None

    sys = inp.sys
    spec = inp.spec
    if sys is not None and spec is not None:
        # P3 衔接：给了 ScalingResult 就把 slack/Γ/σ_y/‖H‖∞ 全部评在物理时间
        # descriptor（sc.scaled：eig=λ/α、双边预条件）上 —— model-time 的
        # bandwidth/κ 值不是物理量；没有 scaling 就回退原始 sys（§6.17 待办：
        # slack 轴目前不接受单独 α，scaled-descriptor 是唯一一致入口）。
        eval_sys = sys
        spec_eval = spec
        if inp.scaling is not None and getattr(inp.scaling, "scaled", None) is not None:
            eval_sys = inp.scaling.scaled
            # ΔA 量纲一致性陷阱：物理器件失配是**相对**的（δ_rel·A），在
            # scaled 系统上 ‖ΔA_v‖ = δ_rel·‖A_v‖ 会随 ‖A_v‖ 同步放大
            # （≈α⁻¹，叠加 S 的范数变化）。若直接把 model-time 的绝对
            # delta_A_norm 传给 scaled slack，等于"缩快时间白嫖失配裕度"。
            # 口径：delta_A_norm 始终按**原始**系统的相对失配解释
            #   δ_rel = delta_A_norm/‖A‖  ⇒  ‖ΔA_v‖ = δ_rel·‖A_v‖，
            # 使 Γ·‖ΔA‖ 对 scaling 保持量纲一致（Γ 的真实变化来自 S 的
            # 预条件，不是单位换算）。delta_A_norm=None ⇒ 轴不评估，跳过。
            if spec.delta_A_norm is not None:
                nA = float(np.linalg.norm(sys.A, 2))
                nAv = float(np.linalg.norm(eval_sys.A, 2))
                if nA > 0 and np.isfinite(nAv):
                    d_rel = float(spec.delta_A_norm) / nA
                    spec_eval = dataclasses.replace(
                        spec, delta_A_norm=d_rel * nAv)
        lyap = inp.lyap
        if lyap is None:
            lyap = solve_generalized_lyapunov(eval_sys)
        if inp.gamma_M is not None:
            gamma_proj = float(inp.gamma_M)
            proj_meta = dict(Gamma_proj=gamma_proj, metric="caller-supplied")
        else:
            res = peak_with_projection(eval_sys, metric=inp.metric)
            gamma_proj = res["Gamma_proj"]
            proj_meta = dict(Gamma=float(res["Gamma"]), Gamma_proj=gamma_proj,
                             method=res["method"], projected=bool(res["projected"]))
        hi = inp.hinf
        if hi is None:
            hi = hinf_norm(eval_sys)["value"]
        sl = slack_evaluate(eval_sys, spec_eval, lyap=lyap, reg=inp.reg,
                            gamma_proj=gamma_proj, hinf=hi)
        slack_block = sl.as_dict()
        items["slack"] = dict(
            value={k: sl.values[k] for k in sl.values},
            verdict=sl.status,
            detail=("violated=%s, unknown=%s" % (sl.violated or "-", sl.unknown or "-")))
        items["resolvent"] = dict(value=proj_meta, verdict="pass",
                                  detail="Γ_proj=%s" % _fmt(gamma_proj))
        items["hinf"] = dict(value=float(hi), verdict="pass", detail="‖H‖∞")
        items["lyapunov"] = dict(
            value=dict(sigma_y=float(lyap.sigma_y), sigma_y2=float(lyap.sigma_y2),
                       h2_finite=bool(lyap.h2_finite), method=lyap.method),
            verdict="pass" if lyap.h2_finite else "violation",
            detail="σ_y=%s (%s)" % (_fmt(lyap.sigma_y), lyap.method))
        if lyap.h2_finite:
            assumptions.append("τ_c/τ_min≪1 white-noise approx (§1.4: finite τ_c "
                               "gives O(τ_c) Itô bias; σ_y² from generalized Lyapunov "
                               "assumes white injection)")
        assumptions.append(
            "ΔA declared conservative — leakage ΔA needs drift budget not Γ_proj "
            "(§1.2/P0-D: robustness_A axis only valid for column-sum-preserving ΔA)")

        violated = sl.violated
        if violated:
            binding_axis = violated[0]
        else:
            finite = {k: v for k, v in sl.values.items() if np.isfinite(v)}
            if finite:
                binding_axis = max(finite, key=lambda k: finite[k])
            else:
                binding_axis = None

    if inp.manifold is not None:
        verdict, val, detail = _manifold_verdict(inp.manifold)
        items["manifold"] = dict(value=val, verdict=verdict, detail=detail)
        if verdict == "rejected":
            reasons.append(detail)

    if inp.loop is not None:
        verdict, val, detail = _loop_verdict(inp.loop, s_max_threshold)
        items["loop"] = dict(value=val, verdict=verdict, detail=detail)
        if verdict == "rejected":
            reasons.append(detail)
        assumptions.append(
            "loop_margins is a necessary-condition probe, not exact DDE roots "
            "(§6.15: σ_min dip on a finite ω grid; scalar_dde_root is the exact "
            "1-DOF analog)")

    if inp.multirate is not None:
        verdict, val, detail = _multirate_verdict(inp.multirate)
        items["multirate"] = dict(value=val, verdict=verdict, detail=detail)
        if verdict == "rejected":
            reasons.extend(detail.split("; "))
        assumptions.append(
            "σ_max WR bound sufficient-not-necessary (§4.3: non-normal splits can "
            "converge below the σ_max certificate; contract rejects only on ρ_sup≥1)")

    if inp.sde_cert is not None:
        verdict, val, detail = _sde_verdict(inp.sde_cert)
        items["sde_cert"] = dict(value=val, verdict=verdict, detail=detail)
        if verdict == "rejected":
            reasons.append(detail)

    if inp.constraint is not None:
        verdict, val, detail = _constraint_verdict(inp.constraint)
        items["constraint"] = dict(value=val, verdict=verdict, detail=detail)
        constraint_val = val
        if verdict == "rejected":
            reasons.append(detail)
        assumptions.append(
            "constraint ε_c judged against a caller-declared external_state_"
            "timescale (§3.4); pure-steady-state problems report τ_relax/"
            "deadline+residual facts instead of an ε_c verdict — they are "
            "not silently marked infeasible or feasible on timing")

    if inp.error_control is None and inp.require_error_control:
        reasons.append("error-control evidence required by workload but "
                       "not supplied: dynamic contract cannot be certified "
                       "(no silent compile)")
    if inp.error_control is not None:
        verdict, val, detail = _error_control_verdict(inp.error_control)
        items["error_control"] = dict(value=val, verdict=verdict, detail=detail)
        error_control_val = val
        if verdict == "rejected":
            reasons.append(detail)
        assumptions.append(
            "error_control bound coverage is checked against the DECLARED "
            "error model (§6.22 behavioral: quantize + bounded noise + bias); "
            "a coverage pass on this model is not a SPICE-level claim")

    # 状态聚合：结构化拒绝优先于 slack 违规（rejected > infeasible）。
    if reasons:
        status = "rejected"
        feasible = False
    elif slack_block is not None:
        status = slack_block["status"]
        feasible = status == "feasible"
    else:
        status = "no evidence evaluated"
        feasible = False

    return Certificate(feasible=feasible, status=status, slack=slack_block,
                       items=items, binding_axis=binding_axis,
                       constraint=constraint_val,
                       error_control=error_control_val,
                       rejection_reasons=reasons, assumptions=assumptions)


def evaluate_sys(sys: Descriptor, spec: HardwareSpec, metric=None, loop=None,
                 multirate=None, manifold=None, sde_cert=None, reg=None,
                 scaling=None, gamma_M=None, hinf=None, lyap=None,
                 constraint=None, error_control=None,
                 require_error_control=False) -> Certificate:
    """便捷入口：由 (sys, spec) + 可选证据直接建 BudgetInput 聚合。

    scaling: ScalingResult（lowering.scaling）—— 给定时证书评在物理时间
    descriptor 上；不传则评 model-time 原始 sys。
    constraint: twin.constraint.ConstraintResult.as_dict() —— T10 约束时序
    证据（ε_c 或纯稳态 deadline/residual 事实），FAIL ⇒ 证书 rejected。
    gamma_M/hinf/lyap: 直接供给（跳过 resolvent sweep / hinf bisect / lyap
    求解）—— delta_A_norm=None 或调用方已有证据时避免无谓的昂贵重算。"""
    return compile_budget(BudgetInput(sys=sys, spec=spec, metric=metric,
                                      loop=loop, multirate=multirate,
                                      manifold=manifold, sde_cert=sde_cert,
                                      reg=reg, scaling=scaling,
                                      gamma_M=gamma_M, hinf=hinf, lyap=lyap,
                                      constraint=constraint,
                                      error_control=error_control,
                                      require_error_control=require_error_control))
