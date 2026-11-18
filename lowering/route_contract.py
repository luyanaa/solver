"""Per-edge route contracts（B1 最后一个编译 pass；DESIGN.md §3.2/§6.15）。

P7 的 generate_config 原来只发一条聚合的 "route-edges" ElementConfig——
边的四判据 closure（twin.route.route_closure）没有进编译输出。本模块把
每条耦合边变成一条**契约**，不是布线结果：

  耦合图 = 描述符 A 的非对角非零元。A[i,j] ≠ 0（i≠j）就是一条 edge
  j → i：状态 j 的输出进入节点 i 的求和结点。与 loop_gain_matrix 的
  edge_map {(i,j): RouteEdge} 同一约定。

每条 EdgeRouteContract 三层内容：

  1. 需求（REQUIREMENT，P7 已知）：
       omega_max      缩放后带宽 max|λ|/alpha（rad/s 物理轴）
       omega_c        feedback 边的 crossover（=omega_max，最保守）
       tau_max_abs    feedback 边的绝对延迟预算 phase_budget(ωc, φ_reserve)
                      ——匹配 deskew 负控制：把 skew 修到零也不改这个数
       eps_gain/eps_phase_deg/noise_budget
  2. 代表性 candidate edge 的 feasibility verdict（四判据 + η_route +
     J_H/J_τ + latency_class）。P7 不知道物理长度/fanout，所以 candidate
     是一条满足预算的短分布式 RC 线——verdict=True 的意思是 "这样的
     边存在且满足契约"，不是 "已布好"。candidate 可被 candidate_length
     强制成长线做负控制（verdict 翻 False，pad_can_fix 诚实：feedback
     边 pad 修不了绝对延迟）。
  3. 动力学敏感度（edge_sensitivity）：S_τ,ij 与 τ_tol；高频极限
     hf_tau_tol = ε/|A_ij| 给出 1/|A_ij| 的延迟预算排序。

诚实边界：per-edge 契约是 REQUIREMENT + 代表性可行性，不是 P5/B2 的
放置布线结果；physical lengths/fanout 到 B2/PEX 才知道。

依赖方向：route_contract.py 消费 Discretized/Descriptor + ScalingResult，
调用 twin.route（lowering → twin 方向，与 p7.py 的惰性导入同先例）；
不重算任何 P0–P6 判定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twin.route import (RouteEdge, RouteSpec, Wire, edge_sensitivity,
                        latency_class, phase_delay_at, route_closure)
from analysis.skew import phase_budget

__all__ = ["EdgeRouteContract", "RouteContractReport",
           "derive_coupling_edges", "emit_route_contracts"]

# 名义 candidate wire 参数（Ω/µm、fF/µm —— twin.route.Wire 缺省值）
_WIRE_R = 0.2
_WIRE_C = 0.2
# 缺省 candidate：100 µm 分布式 RC 线（latency = τ_rc/2 = 2e-13 s）
_NOMINAL_LENGTH_UM = 100.0
_LENGTH_MIN_UM = 10.0
_LENGTH_MAX_UM = 1e6


# ---------------------------------------------------------------------------
# 输出对象
# ---------------------------------------------------------------------------
@dataclass
class EdgeRouteContract:
    """一条耦合边 (j → i) 的路由契约：需求 + 代表性 verdict + 敏感度。"""

    i: int                       # 汇节点（求和结点）
    j: int                       # 源节点（状态 j 输出）
    src: str                     # 状态 j 名
    dst: str                     # 状态 i 名
    weight: float                # |A_ij|
    feedback_critical: bool
    omega_max: float             # rad/s 物理轴（=max|λ|/alpha）
    omega_c: float | None        # feedback 边 crossover；否则 None
    tau_max_abs: float           # feedback: φ_reserve/ωc；否则 inf
    eps_gain: float
    eps_phase_deg: float
    noise_budget: float
    phi_reserve_deg: float
    candidate_length_um: float
    closure: dict                # twin.route.route_closure 输出（含四判据）
    S_tau: float                 # ω|A_ij|·‖(jωI−J)⁻¹‖ 在 omega_max
    tau_tol: float               # ε_φ/S_tau（resolvent 口径）
    hf_tau_tol: float            # ε_φ/|A_ij|（高频极限 ⇒ 1/|A_ij| 排序）
    flagged: bool = False        # 契约本身有缺陷（奇异/defective spec）⇒ 标记
    flag_reason: str = ""

    @property
    def verdict(self) -> bool:
        """契约可行性：candidate 过四判据且契约本身未 flagged。"""
        return bool(self.closure.get("verdict", False)) and not self.flagged

    def as_dict(self) -> dict:
        return dict(
            i=int(self.i), j=int(self.j), src=self.src, dst=self.dst,
            weight=float(self.weight),
            feedback_critical=bool(self.feedback_critical),
            omega_max=float(self.omega_max), omega_c=self.omega_c,
            tau_max_abs=float(self.tau_max_abs),
            eps_gain=float(self.eps_gain),
            eps_phase_deg=float(self.eps_phase_deg),
            noise_budget=float(self.noise_budget),
            phi_reserve_deg=float(self.phi_reserve_deg),
            candidate_length_um=float(self.candidate_length_um),
            closure=dict(self.closure),
            S_tau=float(self.S_tau), tau_tol=float(self.tau_tol),
            hf_tau_tol=float(self.hf_tau_tol),
            flagged=bool(self.flagged), flag_reason=self.flag_reason,
            verdict=bool(self.verdict),
        )


@dataclass
class RouteContractReport:
    """一次 emit_route_contracts 的完整记录（进 Config.route_contracts）。"""

    contracts: list              # [EdgeRouteContract]
    n_edges: int
    n_feedback: int
    omega_max: float
    J: object | None             # 用作敏感度的动力学矩阵（模型坐标 A）
    source: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(
            contracts=[c.as_dict() for c in self.contracts],
            n_edges=int(self.n_edges), n_feedback=int(self.n_feedback),
            omega_max=(float(self.omega_max)
                       if self.omega_max is not None else None),
            source=dict(self.source),
        )


# ---------------------------------------------------------------------------
# 耦合图
# ---------------------------------------------------------------------------
def derive_coupling_edges(discretized) -> tuple[list, np.ndarray | None]:
    """耦合边列表 [(i, j, A_ij)] + 动力学矩阵 J。

    优先级：Discretized 显式携带的边/图字段（meta["edges"] /
    coupling_graph 属性）> 从 A 的非对角非零元推导。J 用于
    edge_sensitivity；burgers（descriptor=None、非线性）时 J=None，
    调用方只能拿到显式边或空表。
    """
    explicit = getattr(discretized, "coupling_graph", None)
    meta = getattr(discretized, "meta", None) or {}
    if explicit is None and isinstance(meta, dict):
        explicit = meta.get("edges")
    desc = getattr(discretized, "descriptor", discretized)
    A = getattr(desc, "A", None) if desc is not None else None
    J = np.asarray(A, dtype=float) if A is not None else None
    if explicit is not None:
        edges = [(int(i), int(j), float(w)) for (i, j, w) in explicit]
        return edges, J
    if J is None or J.ndim != 2 or J.shape[0] != J.shape[1]:
        return [], None
    off = J - np.diag(np.diag(J))
    ii, jj = np.nonzero(off)
    edges = [(int(i), int(j), float(J[i, j])) for i, j in zip(ii, jj)]
    return edges, J


def _feedback_edges(edges: list) -> set:
    """edge (i,j) 是 feedback_critical ⇔ 去掉它之后仍有路径 i → … → j。

    即该边参与一个有向环（把状态 j 的输出经一串 tile 送回 i，而 i 的
    结果又能回到 j）——它闭合一个 loop，绝对延迟判据才适用。BFS 逐边
    判定；边数 ≤ n² 时成本可忽略。
    """
    adj: dict[int, set] = {}
    for i, j, _ in edges:
        adj.setdefault(j, set()).add(i)      # 边 j → i
    fb: set = set()
    for idx, (i, j, _) in enumerate(edges):
        # 从 i 出发（不经本边）能否到 j
        seen = set()
        stack = [i]
        while stack:
            u = stack.pop()
            if u == j and seen:
                fb.add(idx)
                break
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj.get(u, ()))
        # 修正：u==j 且 seen 判定要求路径 ≥1 步；起点 i 直达 j 意味着
        # 存在另一条边 i→j（并行边），也算闭环。
    return fb


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def emit_route_contracts(discretized, scaling=None, spec_template=None,
                         candidate_length: float | None = None,
                         states: list | None = None) -> RouteContractReport:
    """P7 per-edge 路由契约：每条非对角耦合边一条 EdgeRouteContract。

    spec_template：RouteSpec 实例或其 kwargs dict（eps_gain /
    eps_phase_deg / noise_budget / phi_reserve_deg / n_grid）——
    omega_max/omega_c/feedback_critical 由本模块按边推导，不从模板抄。
    candidate_length：强制所有 candidate 线长（µm）；None ⇒ 按预算
    自动选一条满足契约的短线（feedback 边取 tau_max_abs 的 1/4 作
    latency 目标，非 feedback 边取名义 100 µm）。
    """
    edges, J = derive_coupling_edges(discretized)
    states = list(states if states is not None else
                  getattr(discretized, "states", None) or [])

    # 物理带宽：alpha 来自 ScalingResult（t_phys = alpha·t_model）
    alpha = float(getattr(scaling, "alpha", 1.0)) if scaling is not None else 1.0
    lam_max = float("nan")
    if J is not None and J.size:
        try:
            if np.array_equal(J, np.diag(np.diag(J))):
                lam = np.diag(J).astype(complex)   # 对角谱：O(n)，不跑 QR
            else:
                lam = np.asarray(getattr(
                    getattr(discretized, "descriptor", discretized),
                    "finite_eigs", lambda: None)(), dtype=complex)
        except (np.linalg.LinAlgError, TypeError, ValueError):
            lam = None
        if lam is not None:
            lam = lam[np.isfinite(lam)]
            lam_max = float(np.max(np.abs(lam))) if lam.size else float("nan")
    omega_max = lam_max / alpha if np.isfinite(lam_max) and lam_max > 0 \
        else float("nan")

    # RouteSpec 模板：只取预算字段
    base = dict(eps_gain=2e-3, eps_phase_deg=3.0, noise_budget=80e-6,
                phi_reserve_deg=5.0, n_grid=400)
    if spec_template is not None:
        if isinstance(spec_template, RouteSpec):
            spec_template = dict(
                eps_gain=spec_template.eps_gain,
                eps_phase_deg=spec_template.eps_phase_deg,
                noise_budget=spec_template.noise_budget,
                phi_reserve_deg=spec_template.phi_reserve_deg,
                n_grid=spec_template.n_grid)
        for k in base:
            if k in spec_template and spec_template[k] is not None:
                base[k] = spec_template[k]

    fb_idx = _feedback_edges(edges)
    contracts: list[EdgeRouteContract] = []
    for idx, (i, j, a_ij) in enumerate(edges):
        src = states[j] if j < len(states) else "x%d" % j
        dst = states[i] if i < len(states) else "x%d" % i
        feedback = idx in fb_idx
        flag = ""
        flagged = False
        # ---- 契约量（defective spec ⇒ flag，不静默）----
        omega_c = omega_max if feedback else None
        weight = abs(float(a_ij))
        if not np.isfinite(omega_max) or omega_max <= 0:
            flagged = True
            flag = ("defective spec: omega_max not finite/positive "
                    "(lam_max=%r, alpha=%r)" % (lam_max, alpha))
            omega_c = omega_c if (feedback and np.isfinite(omega_max)
                                  and omega_max > 0) else None
        if not np.isfinite(weight):
            flagged = True
            flag = (flag + "; " if flag else "") + \
                "defective edge: |A_ij| not finite"
        tau_max_abs = float("inf")
        if feedback and omega_c is not None and not flagged:
            tau_max_abs = phase_budget(omega_c, base["phi_reserve_deg"])

        # ---- candidate：一条满足预算的短分布式 RC 线 ----
        length = _candidate_length(tau_max_abs if feedback else None,
                                   omega_max, candidate_length)
        edge = RouteEdge((Wire(length, _WIRE_R, _WIRE_C),),
                         name="route_%s->%s" % (src, dst))
        if flagged:
            closure = dict(name=edge.name, verdict=False, latency=edge.latency,
                           latency_class=latency_class(edge.latency),
                           pad_can_fix=not feedback)
            S_tau = tau_tol = hf_tau_tol = float("inf") if weight == 0 else float("nan")
            if np.isfinite(weight) and weight > 0:
                hf_tau_tol = base["eps_phase_deg"] * np.pi / 180.0 / weight
                S_tau = float("nan")
                tau_tol = float("nan")
        else:
            spec = RouteSpec(
                omega_max=omega_max,
                eps_gain=base["eps_gain"],
                eps_phase_deg=base["eps_phase_deg"],
                noise_budget=base["noise_budget"],
                feedback_critical=feedback,
                omega_c=omega_c,
                phi_reserve_deg=base["phi_reserve_deg"],
                n_grid=base["n_grid"],
            )
            closure = route_closure(edge, spec)
            # 敏感度：J 在 omega_max 的 resolvent + 高频极限
            S_tau = tau_tol = hf_tau_tol = float("nan")
            if J is not None and np.isfinite(weight) and weight > 0:
                try:
                    S, tol = edge_sensitivity(J, float(omega_max),
                                              eps_phase_deg=base["eps_phase_deg"])
                    S_tau = float(np.asarray(S)[i, j])
                    tau_tol = float(np.asarray(tol)[i, j])
                except (np.linalg.LinAlgError, ValueError):
                    pass
                hf_tau_tol = (base["eps_phase_deg"] * np.pi / 180.0) / weight
        contracts.append(EdgeRouteContract(
            i=i, j=j, src=src, dst=dst, weight=weight,
            feedback_critical=feedback,
            omega_max=omega_max, omega_c=omega_c,
            tau_max_abs=tau_max_abs,
            eps_gain=base["eps_gain"], eps_phase_deg=base["eps_phase_deg"],
            noise_budget=base["noise_budget"],
            phi_reserve_deg=base["phi_reserve_deg"],
            candidate_length_um=length, closure=closure,
            S_tau=S_tau, tau_tol=tau_tol, hf_tau_tol=hf_tau_tol,
            flagged=flagged, flag_reason=flag,
        ))

    n_fb = sum(1 for c in contracts if c.feedback_critical)
    return RouteContractReport(
        contracts=contracts, n_edges=len(contracts), n_feedback=n_fb,
        omega_max=omega_max, J=J,
        source=dict(
            derivation=("explicit" if getattr(discretized, "coupling_graph",
                                              None) is not None
                        or (getattr(discretized, "meta", None) or {}).get("edges")
                        else "A off-diagonal nnz"),
            alpha=alpha, lam_max=lam_max,
            candidate="distributed RC wire (r=%.3g Ω/µm, c=%.3g fF/µm)"
                      % (_WIRE_R, _WIRE_C),
            honesty="requirement + representative-candidate feasibility, "
                    "NOT a placed/routed result (P5/B2)"),
    )


def _candidate_length(tau_max: float | None, omega_max: float,
                      forced: float | None) -> float:
    """选 candidate 线长：forced 优先；feedback 边把 latency 目标定为
    tau_max/4（分布式 RC latency = τ_rc/2）；非 feedback/无预算 ⇒ 名义
    100 µm。夹在 [_LENGTH_MIN_UM, _LENGTH_MAX_UM]。"""
    if forced is not None:
        return float(min(max(forced, _LENGTH_MIN_UM), 1e12))
    if tau_max is None or not np.isfinite(tau_max) or tau_max <= 0:
        return _NOMINAL_LENGTH_UM
    target_latency = tau_max / 4.0
    # latency = τ_rc/2 = L²·(r·c·1e-15)/2  ⇒  L = sqrt(2·latency/(r·c·1e-15))
    denom = _WIRE_R * _WIRE_C * 1e-15
    length = float(np.sqrt(2.0 * target_latency / denom))
    return float(min(max(length, _LENGTH_MIN_UM), _LENGTH_MAX_UM))
