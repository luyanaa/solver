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

  * 结构化拒绝（manifold 空窗 / ‖S‖∞ 超阈 / ρ_sup≥1 / k_max<1 / sde 违规）
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
    rejection_reasons: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(
            feasible=bool(self.feasible), status=self.status,
            slack=self.slack, items=self.items,
            binding_axis=self.binding_axis,
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


def compile_budget(inp: BudgetInput, s_max_threshold: float = S_PEAK_THRESHOLD) -> Certificate:
    """聚合 P0–P5 的证据 → P6 证书。只判聚合语义，不重算前级验收。"""
    items: dict[str, dict] = {}
    reasons: list[str] = []
    assumptions: list[str] = []
    slack_block: dict | None = None
    binding_axis: str | None = None

    sys = inp.sys
    spec = inp.spec
    if sys is not None and spec is not None:
        lyap = inp.lyap
        if lyap is None:
            lyap = solve_generalized_lyapunov(sys)
        if inp.gamma_M is not None:
            gamma_proj = float(inp.gamma_M)
            proj_meta = dict(Gamma_proj=gamma_proj, metric="caller-supplied")
        else:
            res = peak_with_projection(sys, metric=inp.metric)
            gamma_proj = res["Gamma_proj"]
            proj_meta = dict(Gamma=float(res["Gamma"]), Gamma_proj=gamma_proj,
                             method=res["method"], projected=bool(res["projected"]))
        hi = inp.hinf
        if hi is None:
            hi = hinf_norm(sys)["value"]
        sl = slack_evaluate(sys, spec, lyap=lyap, reg=inp.reg,
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
                       rejection_reasons=reasons, assumptions=assumptions)


def evaluate_sys(sys: Descriptor, spec: HardwareSpec, metric=None, loop=None,
                 multirate=None, manifold=None, sde_cert=None, reg=None) -> Certificate:
    """便捷入口：由 (sys, spec) + 可选证据直接建 BudgetInput 聚合。"""
    return compile_budget(BudgetInput(sys=sys, spec=spec, metric=metric,
                                      loop=loop, multirate=multirate,
                                      manifold=manifold, sde_cert=sde_cert,
                                      reg=reg))
