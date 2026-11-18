"""P7 配置 + 校准计划生成（DESIGN.md §4 P7 / Stage B1 最后一个编译 pass）。

输入都是**已算好的证据对象**：

    P0/P0.5  Discretized / Descriptor
    P6       Certificate（lowering.budget，含六轴 slack）
    P3       ScalingResult（S_x 向量 + alpha + PhysicalScaling）
    校准     CalibratedTwin（twin.calib：这颗芯片的实测参数）

输出是一个**计划**，不是网表：

    Config
      elements[]   每状态/每边一条：primitive（integrator-gm_c /
                   mvm-tile / flux-edge）、target_gm_over_c、C_bank、
                   s_x、alpha、noise_setting、cal 引用
      plan         foreground[]（对哪些量、用什么激励）、
                   background{rate, reference, update_interval}、
                   residual_budget
      flags[]      证书判 infeasible 的量被**标记**而不是悄悄计划
                   （§1.2 "不把未评估当可行" 的 P7 版）

残差预算的诚实合同：plan 的 residual_budget 不能比证书的 noise 轴更紧——
标定残差最终落进 σ²=αkT/C 噪声预算里；若 noise 轴已经 >1（违约），计划
只能把预算定在该轴允许的范围之外并 flag，而不是许诺一个做不到的数字。

依赖方向：p7.py 只消费证据对象，不重算任何判定（与 budget.py 同口径）。
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field

__all__ = ["ElementConfig", "CalibPlan", "Config", "generate_config",
           "C_BANK_CODES"]

# 集成器 tile 的名义电容（计划层面的占位；PDK 数字来自 twin.spice.extract）
C_BANK_F = 1e-12

# C_bank 离散档位（计划层的 code 空间；真实 code 表是 B3 表征Target，
# 这里只保证"档位被显式选中并记录"，不冒充已量化的 PDK 值。）
C_BANK_CODES = (0.5e-12, 1.0e-12, 2.0e-12, 5.0e-12)


# ---------------------------------------------------------------------------
# 输出对象
# ---------------------------------------------------------------------------
@dataclass
class ElementConfig:
    """一个 tile/一条边的可编程配置（plan，不是 netlist）。"""

    name: str
    primitive: str               # "integrator-gm_c" | "mvm-tile" | "flux-edge"
    target_gm_over_c: float      # gm/C 目标（rad/s，物理时间轴）
    C_bank: float = C_BANK_F
    s_x: float | None = None     # 该状态的满摆尺度（V per x̂ 单位）
    alpha: float | None = None   # t_phys = alpha · t_model
    # --- 系数图字段（P0-3 扩展：config 必须编码动力学，不只是速度spec）---
    a_ii: float | None = None    # 该积分器**带符号**有效自系数 A_v[i,i]（rad/s）
    C_code: int = 0              # C_BANK_CODES 的档位序号（最近的档）
    gm_code: int = 0             # gm 码位占位（连续可编程；码表是 B3 Target）
    noise_setting: dict = field(default_factory=dict)
    cal_method: str = "nominal"  # "foreground" | "background" | "replica" | "nominal"
    flagged: bool = False        # 证书判不可行 ⇒ 标记，不悄悄计划
    flag_reason: str = ""

    def as_dict(self) -> dict:
        return dict(name=self.name, primitive=self.primitive,
                    target_gm_over_c=float(self.target_gm_over_c),
                    C_bank=float(self.C_bank), s_x=self.s_x, alpha=self.alpha,
                    a_ii=(float(self.a_ii) if self.a_ii is not None else None),
                    C_code=int(self.C_code), gm_code=int(self.gm_code),
                    noise_setting=dict(self.noise_setting),
                    cal_method=self.cal_method,
                    flagged=bool(self.flagged), flag_reason=self.flag_reason)


@dataclass
class CalibPlan:
    """校准计划：前台量清单 + 后台调度 + 残差预算。"""

    foreground: list              # [{quantity, method, stimulus, ...}]
    background: dict              # {"rate_ppm_per_unit", "reference", "update_interval"}
    residual_budget_ppm: float | None   # None ⇒ 噪声轴未评估，不承诺
    budget_source: str = "none"   # "noise-axis" | "floored" | "none"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(foreground=[dict(f) for f in self.foreground],
                    background=dict(self.background),
                    residual_budget_ppm=self.residual_budget_ppm,
                    budget_source=self.budget_source,
                    notes=list(self.notes))


@dataclass
class Config:
    elements: list
    plan: CalibPlan
    flags: list = field(default_factory=list)
    source: dict = field(default_factory=dict)   # 证据对象的溯源摘要
    route_contracts: list = field(default_factory=list)  # per-edge 路由契约
    # （route_contract.RouteContractReport；as_dict 已展开。聚合的
    #  "route-edges" ElementConfig 保留为向后兼容的摘要——真实记录在这里。）
    # --- 系数图（P0-3）：config 编码**整个** A,B,C 图，不只是 max|λ|/alpha ---
    coeff_edges: list = field(default_factory=list)
    # 每条非对角 A_v[i,j] 一条：{i, j, src, dst, gain(物理 rad/s),
    # tile, route_contract(契约序号或 None)}
    b_ports: list = field(default_factory=list)
    # 每条非零 B_v[i,p] 一条：{input=p, i, dst, gain(物理 V/s per V_in)}
    c_taps: list = field(default_factory=list)
    # 每条非零 C_v[q,i] 一条：{tap=q, i, src, gain}
    cal_bindings: list = field(default_factory=list)
    # 每元素一条：{element, quantity, method, reference}——哪个元素锁哪个
    # 前台量/哪个后台参考。

    def as_dict(self) -> dict:
        return dict(elements=[e.as_dict() for e in self.elements],
                    plan=self.plan.as_dict(), flags=list(self.flags),
                    source=dict(self.source),
                    route_contracts=[c.as_dict() for c in self.route_contracts],
                    coeff_edges=[dict(e) for e in self.coeff_edges],
                    b_ports=[dict(p) for p in self.b_ports],
                    c_taps=[dict(t) for t in self.c_taps],
                    cal_bindings=[dict(b) for b in self.cal_bindings])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def generate_config(discretized, cert, scaling,
                    twin=None, C_bank: float = C_BANK_F,
                    meas_noise_ppm: float = 100.0) -> Config:
    """把 (Discretized, Certificate, ScalingResult, CalibratedTwin) 编成配置。

    逻辑（每条都是消费既有证据，不重算）：
      * alpha / s_x 直接取自 ScalingResult（P3 已做频窗检查）；
      * 系数图（P0-3）：scaling.scaled 的 A_v = α⁻¹ S A S⁻¹ / B_v / C_v
        被完整展开——对角元成为每积分器**各自**的有效自系数 a_ii
        （target_gm_over_c = |a_ii|；diag(-1,-100) 出两个不同的系数，
        不再收敛到全局 max|λ|/alpha），非对角非零元成为带物理增益的
        coeff_edges（引用 route_contract 的契约序号），B/C 非零元成为
        b_ports/c_taps。纯积分器（a_ii≈0）退回 tile 速度 spec
        max|λ|/alpha 并记录 tile_speed_spec；descriptor 缺失（burgers）
        时如实标注 coeff_graph_encoded=False，不编造系数图；
      * C/gm 码位：C_code 选 C_BANK_CODES 中最接近请求 C_bank 的档，
        gm_code 是连续可编程占位（真实码表是 B3 表征 Target）；
      * cal_bindings：每元素锁哪个前台量/后台参考；
      * noise_setting 的 σ_scale 取证书 slack 的 noise 轴（>1 说明噪声
        预算已经越界，配置如实放大 σ_scale=axis 值并 flag，不自造裕度）；
      * residual_budget 取 max(标定噪声底, noise 轴隐含的容许分数误差)，
        且 plan.residual_budget ≥ meas_noise_ppm（承诺不能低于测量底）。
        floor/axis 是启发式：跨轴的 dJ 敏感度分配是 B2 refinement；
      * 证书不可行/rejected ⇒ 对应元素 flagged=True，plan 只标不计划。
    """
    desc = getattr(discretized, "descriptor", discretized)
    lam = desc.finite_eigs() if desc is not None else None
    lam_max = float(max(abs(lam))) if lam is not None and lam.size else float("nan")

    alpha = float(scaling.alpha)
    s_vec = list(scaling.S_x_vec)
    states = list(getattr(discretized, "states", None) or
                  [f"x{i}" for i in range(len(s_vec))])

    slack = (cert.slack or {}) if cert is not None else {}
    values = slack.get("values", {}) if isinstance(slack, dict) else {}
    noise_axis = values.get("noise", float("nan"))
    status = getattr(cert, "status", "unknown") if cert is not None else "unknown"
    # 证书语义（budget.py）：status "infeasible"/"rejected" ⇒ 不可计划；
    # "no violation on evaluated axes" 是部分评估——继续出计划但记 note。
    infeasible = status in ("infeasible", "rejected")
    binding = getattr(cert, "binding_axis", None) if cert is not None else None
    flags: list[str] = []
    notes: list[str] = []
    if cert is not None and not infeasible and not getattr(cert, "feasible", True):
        notes.append("certificate %s (partial evaluation): plan emitted, "
                     "unevaluated axes unclaimed" % status)
    # ---- 残差预算：不能比 noise 轴更紧，且不低于测量底 ----
    if _finite(noise_axis) and noise_axis > 0:
        # 诚实口径：noise 轴 = σ_y/σ_req，轴 ≤1 说明证书还有 (1-轴) 的
        # 噪声预算余量；允许的标定残差按 floor/axis 量化（轴 → 1 ⇒ 预算
        # 收缩到测量底；轴 → 0 ⇒ 余量大），且永不低于测量底。
        # 轴 >1（已违约）⇒ 预算置 None：计划不许诺做不到的数字。
        if noise_axis > 1.0:
            budget = None
            flags.append("noise axis %.3g > 1: residual budget not promised"
                         % noise_axis)
        else:
            budget = max(float(meas_noise_ppm),
                         float(meas_noise_ppm) / noise_axis)
            notes.append("residual budget %.4g ppm = floor/axis "
                         "(noise axis %.3g <= 1) — HEURISTIC: per-axis dJ "
                         "sensitivity allocation is a B2 refinement, not "
                         "implemented here" % (budget, noise_axis))
        budget_source = "noise-axis" if budget is not None else "none"
    else:
        budget = None
        budget_source = "none"
        notes.append("noise axis not evaluated (sigma_req None?): "
                     "no residual budget promised")

    # ---- per-edge route contracts（§3.2/§6.15 四判据进编译输出）----
    # 先于系数图发出：coeff_edges 直接引用契约序号（消费既有记录，
    # 不重建）。惰性导入：保持 lowering → twin 只出现在调用期。聚合的
    # "route-edges" ElementConfig 保留（向后兼容），真实记录在这里。
    from lowering.route_contract import emit_route_contracts
    report = emit_route_contracts(discretized, scaling=scaling)

    # ---- per-element 配置 ----
    # P0-3：系数图取自**物理缩放后的描述符**（scaling.scaled：
    # E_v = S E S⁻¹, A_v = α⁻¹ S A S⁻¹, B_v = S B, C_v = C S⁻¹）。
    # A_v 的对角元是该积分器的有效自系数（rad/s，物理时间轴），非对角
    # 非零元是 MVM/flux 边的物理增益。没有 scaled/descriptor（burgers
    # 非线性）时退回全局速度 spec——如实记录，不编造系数图。
    scaled = getattr(scaling, "scaled", None)
    A_v = getattr(scaled, "A", None) if scaled is not None else None
    B_v = getattr(scaled, "B", None) if scaled is not None else None
    C_v = getattr(scaled, "C", None) if scaled is not None else None
    if A_v is None and desc is not None and alpha > 0 and len(s_vec) == desc.n:
        S = np.diag(s_vec)
        A_v = (1.0 / alpha) * (S @ np.asarray(desc.A, dtype=float) @ np.diag(1.0 / np.asarray(s_vec)))
        B_v = S @ np.asarray(desc.B, dtype=float)
        C_v = np.asarray(desc.C, dtype=float) @ np.diag(1.0 / np.asarray(s_vec))
    A_v = np.asarray(A_v, dtype=float) if A_v is not None else None
    graph_ok = A_v is not None and A_v.shape == (len(states), len(states))
    a_scale = float(np.max(np.abs(A_v))) if graph_ok else 0.0
    a_tol = 1e-12 * a_scale if a_scale > 0 else 0.0
    # tile 速度 spec 仍是 max|λ|/alpha（tile 必须能 slew 到最快模态速率）
    tile_speed = (lam_max / alpha if _finite(lam_max) and alpha > 0
                  else float("nan"))

    elements: list[ElementConfig] = []
    for i, name in enumerate(states):
        s_x = float(s_vec[i]) if i < len(s_vec) and s_vec[i] else None
        flagged = infeasible
        reason = ""
        if flagged:
            reason = ("certificate %s%s" %
                      (status, ("; binding axis %s" % binding) if binding else ""))
        # 逐元素有效自系数：a_ii 带符号（dot V_i = a_ii V_i + Σ_j a_ij V_j + …）
        a_ii = float(A_v[i, i]) if graph_ok else None
        if a_ii is not None and abs(a_ii) > a_tol:
            gm_c_target = abs(a_ii)     # 集成器极点放在**自己的**模态速率处
        else:
            gm_c_target = tile_speed    # 纯积分器（a_ii=0）：cell 仍须跟最快速率
        # C 档位：选最接近请求 C_bank 的档（显式选中并记录，不冒充 PDK 值）
        c_code = int(min(range(len(C_BANK_CODES)),
                         key=lambda k: abs(C_BANK_CODES[k] - C_bank)))
        elements.append(ElementConfig(
            name=name,
            primitive="integrator-gm_c",
            target_gm_over_c=gm_c_target,
            C_bank=C_bank,
            s_x=s_x,
            alpha=alpha,
            a_ii=a_ii,
            C_code=c_code,
            noise_setting={
                "type": "gm-tile white",
                "sigma_scale": (float(noise_axis) if _finite(noise_axis)
                                and noise_axis > 1.0 else 1.0),
            },
            cal_method=("background" if not flagged else "nominal"),
            flagged=flagged,
            flag_reason=reason,
        ))
    c_code_used = elements[0].C_code if elements else 0

    # ---- 系数图：A 边 / B 输入 / C 抽头 ----
    coeff_edges: list[dict] = []
    b_ports: list[dict] = []
    c_taps: list[dict] = []
    if graph_ok:
        contract_of = {}
        for k, c in enumerate(report.contracts):
            contract_of[(int(c.i), int(c.j))] = k
        ii, jj = np.nonzero(A_v - np.diag(np.diag(A_v)))
        for i, j in zip(ii, jj):
            if abs(A_v[i, j]) <= a_tol:
                continue
            coeff_edges.append(dict(
                i=int(i), j=int(j), src=states[j], dst=states[i],
                gain=float(A_v[i, j]), tile="mvm-tile",
                route_contract=contract_of.get((int(i), int(j)))))
        if B_v is not None:
            bi, bp = np.nonzero(np.asarray(B_v, dtype=float))
            for i, p in zip(bi, bp):
                b_ports.append(dict(input=int(p), i=int(i), dst=states[i],
                                    gain=float(B_v[i, p])))
        if C_v is not None:
            q_, ci_ = np.nonzero(np.asarray(C_v, dtype=float))
            for q, i in zip(q_, ci_):
                c_taps.append(dict(tap=int(q), i=int(i), src=states[i],
                                   gain=float(C_v[q, i])))

    # ---- 校准绑定：哪个元素锁哪个参考/前台量 ----
    cal_bindings = []
    for el in elements:
        if el.primitive == "integrator-gm_c" and not el.flagged:
            cal_bindings.append(dict(
                element=el.name, quantity="integrator offset/gain",
                method="foreground",
                reference="known step (ratiometric against reference)"))
            cal_bindings.append(dict(
                element=el.name, quantity="gm/C pole drift",
                method="background",
                reference="on-chip replica cell (or MEMS omega_ref)"))
        elif el.primitive == "integrator-gm_c":
            cal_bindings.append(dict(element=el.name, quantity="none (flagged)",
                                     method="nominal", reference=None))

    # flux-edge（耦合边，Q10/T7 route）：一条记录代表边集合的口径
    edges = ElementConfig(
        name="route-edges", primitive="flux-edge",
        target_gm_over_c=float("nan"), s_x=None, alpha=alpha,
        noise_setting={"type": "interface", "sigma_scale": 1.0},
        cal_method="foreground" if not infeasible else "nominal",
        flagged=infeasible,
        flag_reason=("certificate %s" % status if infeasible else ""),
    )
    elements.append(edges)

    # ---- 校准计划 ----
    foreground = []
    if not infeasible:
        foreground.append({
            "quantity": "integrator offset/gain (per element)",
            "method": "foreground",
            "stimulus": "known step (ratiometric against reference)",
            "meas_noise_ppm": float(meas_noise_ppm),
        })
        foreground.append({
            "quantity": "transform Gram G_T (T~^* T~)",
            "method": "foreground",
            "stimulus": "known excitation, right-Gram calibration",
            "meas_noise_ppm": float(meas_noise_ppm),
        })
    bg_rate = 0.0
    if twin is not None:
        # 漂移速率来自这颗芯片的孪生记账（apply_background 存下的
        # rate_ppm_per_unit）；孪生未注入速率时不排程，不编造数字。
        twin_rate = twin.rates.get("gm_over_c")
        if twin_rate is not None and _finite(twin_rate):
            bg_rate = float(twin_rate)
    background = {
        "quantity": "gm/C pole drift vs temperature (replica-locked)",
        "reference": "on-chip replica cell (or MEMS omega_ref when present)",
        "rate_ppm_per_unit": bg_rate,
        "update_interval": None,      # 速率未注入前不排程
        "meas_noise_ppm": float(meas_noise_ppm),
    }
    if bg_rate > 0.0 and budget is not None:
        # 惰性导入：保持 lowering → twin 只出现在调用期，不在模块加载期
        from twin.calib import update_interval_for_budget
        background["update_interval"] = update_interval_for_budget(
            bg_rate, budget, meas_noise_ppm)
    plan = CalibPlan(foreground=foreground, background=background,
                     residual_budget_ppm=budget, budget_source=budget_source,
                     notes=notes)

    src = dict(alpha=alpha, lam_max=lam_max,
               cert_status=status, binding_axis=binding,
               n_states=len(states), C_bank=float(C_bank),
               tile_speed_spec=tile_speed, C_code=c_code_used,
               n_coeff_edges=len(coeff_edges), n_b_ports=len(b_ports),
               n_c_taps=len(c_taps),
               coeff_graph_encoded=bool(graph_ok))

    src["n_route_contracts"] = report.n_edges
    src["n_route_feedback"] = report.n_feedback
    # 系数谱摘要：对角有效自系数的离散度（diag(-1,-100) 类系统的判据）
    if graph_ok:
        diag_abs = np.abs(np.diag(A_v))
        nz = diag_abs[diag_abs > a_tol]
        src["a_ii_spread"] = (float(nz.max() / nz.min())
                              if nz.size and nz.min() > 0 else None)

    return Config(elements=elements, plan=plan, flags=flags, source=src,
                  route_contracts=report.contracts, coeff_edges=coeff_edges,
                  b_ports=b_ports, c_taps=c_taps, cal_bindings=cal_bindings)


def _finite(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")
