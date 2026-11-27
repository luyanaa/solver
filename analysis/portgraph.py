"""物理参数端口图（§6.22 P0-5 的落地版）：把 dJ 灵敏度分到**器件参数**。

P0-5 的绝对灵敏度接口已经存在（`analysis.adjoint.port_sensitivities`，
K_i = ∫|λᵀ∂F/∂θ_i|dt），但 P7 的 `device_sensitivities` 一直按 **state
name** 取值——同一个节点上的 gm、C、offset、DAC、ADC、route delay 共用
一份 ∫|λ_i|dt，这在物理上不成立：

    offset   → 常量注入 e_i          ⇒ K = ∫|λ_i|dt
    gm (a_ii)→ Δa·x_i                ⇒ K = ∫|λ_i x_i|dt
    C bank   → Δa = −a/C·δC          ⇒ K = |a|/C·∫|λ_i x_i|dt
    gm edge  → Δa_ij·x_j (i≠j)       ⇒ K = ∫|λ_i x_j|dt
    DAC      → Δb·u_p                ⇒ K = ∫|λ_i u_p|dt
    ADC      → Δc·x_i（QoI 读数）    ⇒ K = ∫|x_i|dt
    route τ  → a_ij·ω·δτ（一阶）     ⇒ K = K_edge·ω（近似，phases 线性化）

本模块只做一件事：给定 (A,B,C,qoi,T) 与输入/初值，为**每个物理端口**
生成一条 `PortSense`，并给出绝对 K 与单位。它不含分配逻辑，也不读真解；
分配在 `lowering.p7.allocate_port_budgets`，物理码位/位宽在
`analysis.precision`。

诚实的边界：
  * K 是**线性化**灵敏度（沿标称轨迹），大扰动的非线性项不在内；
  * route τ 的 K 是 ω 处的一阶相位近似（`S_τ` 的完整 resolvent 口径
    在 `twin.route.edge_sensitivity`，这里只用它给出的带宽）；
  * E≠I（descriptor）时轨迹用 E⁻¹A 推进；E 奇异用 pinv 并在 note 里
    标注（守恒零模态本身不产生 QoI 敏感度）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PortSense", "PortGraph", "build_port_graph"]


@dataclass
class PortSense:
    """一条物理端口的绝对 dJ 灵敏度（|δJ| ≤ K·δθ + h.o.t.）。"""

    kind: str                 # gm_c | c_bank | gm_edge | offset | dac | adc | route_tau
    name: str
    K: float                  # 绝对灵敏度（|δJ| per 参数单位）
    K_rel: float              # 相对灵敏度 |∂J/∂lnθ| = K·θ_ref（ppm 预算口径）
    theta_ref: float | None   # 归一化参考（a_ii / C_bank / |a_ij| / v_swing）
    unit: str
    element: int | None = None      # 所属积分器下标（若适用）
    edge: tuple | None = None       # (i, j)（若适用）
    floor_ppm: float | None = None  # 该端口的测量/量化底（未声明 ⇒ None）
    note: str = ""

    def as_dict(self) -> dict:
        return dict(kind=self.kind, name=self.name, K=float(self.K),
                    K_rel=float(self.K_rel), theta_ref=self.theta_ref,
                    unit=self.unit, element=self.element,
                    edge=(list(self.edge) if self.edge is not None else None),
                    floor_ppm=(float(self.floor_ppm)
                               if self.floor_ppm is not None else None),
                    note=self.note)


@dataclass
class PortGraph:
    """一次标称运行下的全端口灵敏度快照（物理参数图）。"""

    ports: list[PortSense]
    t: np.ndarray
    x_traj: np.ndarray
    input_kind: str
    notes: list = field(default_factory=list)

    def by_kind(self, kind: str) -> list[PortSense]:
        return [p for p in self.ports if p.kind == kind]

    def element_totals(self, n_states: int) -> list[float]:
        """逐积分器聚合的绝对灵敏度 ΣK（per-element 预算视图）。"""
        tot = np.zeros(n_states, dtype=float)
        for p in self.ports:
            if p.element is not None and 0 <= p.element < n_states:
                tot[p.element] += p.K_rel
        return [float(v) for v in tot]

    def as_dict(self) -> dict:
        return dict(ports=[p.as_dict() for p in self.ports],
                    input_kind=self.input_kind,
                    n_ports=len(self.ports),
                    kinds={k: len(self.by_kind(k))
                           for k in sorted({p.kind for p in self.ports})},
                    notes=list(self.notes))


def _trajectory(A, B, x0, t, E=None):
    """线性标称轨迹（单位阶跃输入），与伴随同一网格。

    x(t) = expm(A t)x0 + Σ_p ∫_0^t e^{A(t−s)}B_p u_p ds，u_p = 1(t≥0)；
    用增广矩阵指数一次算出（A 奇异/守恒零模态也成立）。
    """
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if E is not None and np.any(np.abs(np.asarray(E) - np.eye(n)) > 1e-12):
        Ep = np.asarray(E, dtype=float)
        try:
            A = np.linalg.solve(Ep, A)
        except np.linalg.LinAlgError:
            A = np.linalg.pinv(Ep) @ A
    x0 = (np.zeros(n) if x0 is None else np.asarray(x0, dtype=float))
    m = 0 if B is None else np.asarray(B, dtype=float).shape[1]
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = A
    if m:
        aug[:n, n:] = np.asarray(B, dtype=float)
    x = np.empty((n, t.size))
    for k, tk in enumerate(t):
        M = expm(aug * float(tk))
        x[:, k] = M[:n, :n] @ x0 + M[:n, n:] @ np.ones(m)
    return x


def build_port_graph(A, B=None, C=None, qoi=None, T=None, E=None,
                     x0=None, route_contracts=None, C_bank: float = 1e-12,
                     t_grid=None, input_kind: str = "unit step",
                     v_swing: float = 1.0) -> PortGraph:
    """为 (A,B,C,qoi,T) 生成逐物理端口的绝对灵敏度图。

    A/B/C 用**物理时间缩放后**的 A_v/B_v/C_v（P7 系数图口径）；qoi 是
    C_v 的行（或任意 QoI 向量）；T 是物理时窗。route_contracts 提供每边
    ω（feedback 边用 ω_c，否则 ω_max），用于 τ 端口的一阶灵敏度。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    T = float(T) if T is not None else 1.0
    t = (np.asarray(t_grid, dtype=float) if t_grid is not None
         else np.linspace(0.0, T, 81))
    if t[0] != 0.0 or abs(t[-1] - T) > 1e-12 * max(1.0, T):
        t = np.linspace(0.0, T, 81)
    x = _trajectory(A, B, x0, t, E=E)
    notes = []
    if E is not None and np.any(np.abs(np.asarray(E) - np.eye(n)) > 1e-12):
        notes.append("descriptor E≠I: trajectory advanced on E⁻¹A "
                     "(pinv fallback if singular); conservation null modes "
                     "carry no QoI sensitivity by construction")

    # ---- 伴随场：与 p7 同一口径（对角闭式 / 一般采样）----
    c = np.ones(n) if qoi is None else np.asarray(qoi, dtype=float).ravel()
    if c.size != n:
        raise ValueError("qoi 维度 %d != 状态数 %d" % (c.size, n))
    lam = np.empty((n, t.size))
    diag_only = np.abs(A - np.diag(np.diag(A))).max() <= 1e-9 * max(
        1.0, float(np.abs(np.diag(A)).max()))
    if diag_only and np.all(-np.diag(A) > 0) and np.all(c >= 0):
        gamma = -np.diag(A)
        for k, tk in enumerate(t):
            lam[:, k] = c * np.exp(-gamma * (float(T) - tk))
        notes.append("adjoint: closed form λ_i(t)=c_i e^{−γ_i(T−t)} "
                     "(diagonal, real-negative)")
    else:
        from analysis.adjoint import adjoint_field, descriptor_adjoint_field
        if E is not None and np.any(np.abs(np.asarray(E) - np.eye(n)) > 1e-12):
            lam = descriptor_adjoint_field(E, A, c, t)
        else:
            lam = adjoint_field(A, c, t)
        notes.append("adjoint: sampled field (non-diagonal or sign-indefinite)")

    ports: list[PortSense] = []

    def K_of(colfn):
        """∫|λᵀ b(t)|dt，b(t) 是 n×1（时间相关时传 callable）。"""
        vals = np.empty(t.size)
        for k, tk in enumerate(t):
            b = colfn(tk, x[:, k]) if callable(colfn) else colfn
            vals[k] = abs(float(lam[:, k] @ np.asarray(b, dtype=float).ravel()))
        return float(np.trapezoid(vals, t))

    def e(i):
        v = np.zeros(n)
        v[i] = 1.0
        return v

    diag = np.diag(A)
    # ---- 逐积分器：gm / C bank / offset ----
    for i in range(n):
        name = "x%d" % i
        ki = K_of(lambda tk, xx, i=i: e(i) * xx[i])       # Δa_ii·x_i
        a_ii = float(diag[i])
        ports.append(PortSense(
            kind="gm_c", name="gm_c[%s]" % name, K=ki,
            K_rel=ki * abs(a_ii), theta_ref=abs(a_ii),
            unit="ppm of gm (δa_ii = a_ii·δgm/gm)", element=i,
            note="δa_ii = δgm/C ⇒ 与 C 端口的 K_rel 相等（相对误差口径）"))
        if C_bank > 0 and abs(a_ii) > 0:
            ports.append(PortSense(
                kind="c_bank", name="c_bank[%s]" % name,
                K=ki * abs(a_ii) / float(C_bank), K_rel=ki * abs(a_ii),
                theta_ref=float(C_bank),
                unit="ppm of C (δa_ii = −a_ii·δC/C)", element=i))
        ports.append(PortSense(
            kind="offset", name="offset[%s]" % name,
            K=K_of(e(i)), K_rel=K_of(e(i)) * float(v_swing),
            theta_ref=float(v_swing),
            unit="ppm of full-scale offset", element=i,
            note="常量电流/失调注入；θ_ref = v_swing"))

    # ---- 耦合边：gm edge (+ route τ 一阶) ----
    omega_of = {}
    if route_contracts is not None:
        for cc in getattr(route_contracts, "contracts", route_contracts):
            try:
                i, j = int(cc.i), int(cc.j)
            except AttributeError:
                continue
            om = (cc.omega_c if getattr(cc, "feedback_critical", False)
                  and cc.omega_c is not None else cc.omega_max)
            omega_of[(i, j)] = float(om) if om is not None else np.nan
    off = A - np.diag(diag)
    for i, j in zip(*np.nonzero(off)):
        if abs(off[i, j]) <= 1e-12 * max(1.0, float(np.abs(A).max())):
            continue
        kij = K_of(lambda tk, xx, i=i, j=j: e(i) * xx[j])
        ports.append(PortSense(kind="gm_edge", name="gm_edge[%d<-%d]" % (i, j),
                              K=kij, K_rel=kij * abs(off[i, j]),
                              theta_ref=abs(off[i, j]),
                              unit="ppm of edge gain a_ij",
                              element=i, edge=(i, j)))
        om = omega_of.get((int(i), int(j), ))
        if om is not None and np.isfinite(om) and om > 0:
            ports.append(PortSense(
                kind="route_tau", name="route_tau[%d<-%d]" % (i, j),
                K=kij * om, K_rel=float("nan"), theta_ref=1.0 / om,
                unit="per second of delay (first order ω·δτ)",
                element=i, edge=(i, j),
                note="linearized: δa_ij ≈ a_ij·ω·δτ at ω=%.3g rad/s; "
                     "phase budget, not a ppm budget" % om))

    # ---- DAC（输入端口）/ ADC（QoI 抽头）----
    if B is not None:
        Bm = np.asarray(B, dtype=float)
        for i, p in zip(*np.nonzero(Bm)):
            ports.append(PortSense(
                kind="dac", name="dac[p%d->x%d]" % (p, i),
                K=K_of(lambda tk, xx, i=i: e(i) * 1.0),
                K_rel=K_of(lambda tk, xx, i=i: e(i) * 1.0),
                theta_ref=1.0,
                unit="ppm of input gain (unit-step drive)", element=i,
                note="与 offset 同 ∫|λ_i|dt 形态，但作用在输入增益而非失调；"
                     "输入波形改变 K（这里按 unit step 标定）"))
    if C is not None:
        Cm = np.asarray(C, dtype=float).reshape(-1, n)
        for q_, i in zip(*np.nonzero(Cm)):
            ports.append(PortSense(
                kind="adc", name="adc[q%d<-x%d]" % (q_, i),
                K=float(np.trapezoid(np.abs(x[i]), t)),
                K_rel=float(np.trapezoid(np.abs(x[i]), t)), theta_ref=1.0,
                unit="ppm of tap gain (QoI readout)", element=i,
                note="QoI 读数增益；ADC 量化误差按 LSB → δc/F.S. 映射"))

    return PortGraph(ports=ports, t=t, x_traj=x, input_kind=input_kind,
                     notes=notes)
