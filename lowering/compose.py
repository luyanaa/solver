"""编译 pass 链的组合驱动（B1 closeout：run_compiler_e2e 的 findings F1/F2/F3/F5）。

`bench/run_compiler_e2e.py` 原来手工串各 pass，串法本身有四个组合缺口
（见 results/compiler_e2e.json 的 findings）。本模块把"串"变成**一等
公民**：`run_chain` 是唯一合法的链驱动，每个 stage 声明自己的输入/输出
坐标系，坐标不匹配是显式错误而不是静默错答案。

坐标系（declared domain，全链可追踪）：

  "source"        输入对象（ProblemBundle/Discretized/Descriptor）
  "model"         模型时间 descriptor 坐标（x，t = t_model）
  "physical-time" P3 缩放后 descriptor（V = S x，t_phys = α·t_model）
  "basis-T"       基赢家坐标（x = T z ⇒ z = T⁻¹x；model 或 physical-time
                  皆可，stage 记录里用 "basis-T(model)" / "basis-T(physical)"）
  "field"         物理场（周期网格 u，interface 的输入域）

四条 finding 的关闭机制：

  F1（typed boundary）run_chain 只接受 descriptor 产线对象
     （ProblemBundle / Discretized / Descriptor）。reduce System IR /
     NormalizedSystem 进来 ⇒ TypedBoundaryError，错误文本点名正确的
     入口（normalize 消费 System IR；descriptor lane 从 discretize/scaling
     进链）。边界是可执行断言，不是文档。

  F2（projected 改 max|λ|）closure 的 P_N 投影会改谱（diffusion N=32:
     4.096 → 1.024）。carry="projected" 时链把 projected descriptor 作为
     **唯一下游载体**，gramian_scaling（alpha + S_x）在它上面**重推**，
     全链（cert / contracts / p7）吃同一份 scaling；两次 scaling 都如实
     记录（alpha_full vs alpha_carried）。绝不把 projected 谱配全谱 alpha。

  F3（interface 输入域）interface 消费**物理场**，不消费 modal 坐标。
     链里这是一个显式声明的域边界：workload 带场（field= 参数或
     spec_meta 的网格场）⇒ interface 作为 EVIDENCE stage 运行，输入域
     标 "field"；workload 不带场 ⇒ 记 "interface not applicable"（不是
     静默跳过）。modal↔field 的表示变换（§5.7 的 T）不在本链内接线，
     边界文本如实写明。

  F5（basis 复合进 scaling）赢家 ≠ identity 时，T 已经换了 tile 的坐标
     系；下游两个消费者（p7 系数图、route 契约）必须吃**同一个**
     transformed descriptor。链的做法：
       desc_final = 赢家 transformed（先做共轭卫生清理：eigenvector 基
         的 T⁻¹AT 带机器精度尘埃 ~1e-14，按相对容差 1e-12·max|A| 清零——
         精确共轭处是精确零）；
       sc_final  = gramian_scaling(desc_final)（F5 的复合：scaling 在
         T 坐标上重推，B_v = α⁻¹·S_z·B_z 语义按 scaling.py 的同一合同
         在 T 坐标上成立；外部传递函数不变，数值验证）；
       边准入 mask：p7 的 a_tol = 1e-12·max|A_v| 语义（缩放坐标下可忽略
         的边不进系数图）换算回 desc_final 的 A 上，把**同一个**
         masked descriptor 发给 emit_route_contracts 和 generate_config
         ⇒ 两个消费者的边集由构造相等；assert 契约边集 == coeff 边集
         （同一批 (i,j) + 契约序号对齐），不等 ⇒ 抛 ChainCompositionError。
     特征值不变性（相似 + α）与外部 TF 不变性都数值断言。

  F4（model/physical 口径）保持为**正向一致性断言**（omega_max =
     max|λ|/alpha = 2π·scaled.bandwidth()），不是缺口。

单源真值：run_chain 不重算任何 pass 的判定（closure/split/basis/scaling/
cert/contracts/p7 都是调用 lowering 现有入口），它只负责**排序、坐标
声明与一致性断言**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis import Descriptor

__all__ = ["TypedBoundaryError", "ChainCompositionError", "StageRecord",
           "ChainResult", "run_chain"]

_DUST_REL = 1e-12        # 共轭尘埃清理 + p7 a_tol 同一相对约定
_MASK_ITERS = 4


class TypedBoundaryError(TypeError):
    """链的输入域类型错误：对象不是 descriptor 产线产物。"""


class ChainCompositionError(RuntimeError):
    """链内一致性断言失败（坐标混合 / 边集分裂 / 不动点不收敛）。"""


# ---------------------------------------------------------------------------
# stage 记录
# ---------------------------------------------------------------------------
@dataclass
class StageRecord:
    """一个 chain stage 的记录：坐标声明 + 工件摘要。"""

    name: str
    domain_in: str
    domain_out: str
    summary: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(name=self.name, domain_in=self.domain_in,
                    domain_out=self.domain_out, summary=dict(self.summary),
                    notes=list(self.notes))


# ---------------------------------------------------------------------------
# 链结果
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    """run_chain 的结构化输出：逐 stage 工件 + 各自坐标系 + 最终载体。"""

    stages: list
    descriptor: object              # 下游最终载体（masked，坐标见 coords）
    coords: str                     # "model" | "basis-T(model)"
    scaling: object                 # 最终 ScalingResult（作用于 descriptor）
    scaling_full_record: dict       # F2/F5 全部 scaling 的诚实台账
    cert: object
    closure: object | None
    closure_carried: str            # "full" | "projected"
    split: object | None
    basis: object | None
    interface: object | None        # InterfaceResult 或 None（not applicable）
    interface_status: str           # "evidence (field input)" | "not applicable"
    contracts: object | None        # RouteContractReport
    config: object | None           # p7 Config
    eig_invariance: float | None    # F5: |eig(final scaled) - eig(carried scaled)|
    tf_invariance: float | None     # F5: 外部传递函数最大偏差（cond 归一）
    edge_sets_equal: bool | None    # F5: 契约边集 == coeff 边集
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(
            stages=[s.as_dict() for s in self.stages],
            coords=self.coords,
            scaling=(self.scaling.as_dict() if self.scaling is not None
                     and hasattr(self.scaling, "as_dict") else None),
            scaling_full_record=self.scaling_full_record,
            cert=(self.cert.as_dict() if self.cert is not None
                  and hasattr(self.cert, "as_dict") else repr(self.cert)),
            closure=(self.closure.as_dict() if self.closure is not None
                     and hasattr(self.closure, "as_dict") else None),
            closure_carried=self.closure_carried,
            split=(self.split.as_dict() if self.split is not None
                   and hasattr(self.split, "as_dict") else None),
            basis=(self.basis.as_dict() if self.basis is not None
                   and hasattr(self.basis, "as_dict") else None),
            interface=(self.interface.as_dict() if self.interface is not None
                       and hasattr(self.interface, "as_dict") else None),
            interface_status=self.interface_status,
            contracts=(self.contracts.as_dict() if self.contracts is not None
                       else None),
            config=(self.config.as_dict() if self.config is not None
                    and hasattr(self.config, "as_dict") else None),
            eig_invariance=self.eig_invariance,
            tf_invariance=self.tf_invariance,
            edge_sets_equal=self.edge_sets_equal,
            notes=list(self.notes),
        )


# ---------------------------------------------------------------------------
# 输入边界（F1）
# ---------------------------------------------------------------------------
def _coerce_descriptor_source(source):
    """descriptor 产线对象 → (descriptor, states, meta, grid)。

    接受：Descriptor（合成负载，如实标注）、Discretized、ProblemBundle
    （.descriptor 为 Descriptor 的 bundle）。reduce 的 System IR /
    NormalizedSystem / Normalization 等**符号 lane**对象 ⇒ TypedBoundaryError。
    """
    if isinstance(source, Descriptor):
        return source, None, {"synthetic": True, "source": "raw Descriptor"}, ()
    desc = getattr(source, "descriptor", None)
    if isinstance(desc, Descriptor):
        states = list(getattr(source, "states", None) or [])
        meta = getattr(source, "meta", None)
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta.setdefault("source", type(source).__name__)
        grid = tuple(getattr(source, "grid", None) or ())
        return desc, states, meta, grid
    raise TypedBoundaryError(
        "run_chain: source is %s, which is not a descriptor-producing object "
        "(accepted: analysis.Descriptor, lowering.discretize.Discretized, "
        "bench.problems.ProblemBundle). The reduce System IR / "
        "Normalization lane feeds lowering.normalize (pipeline_normalize), "
        "which consumes the System IR only — the descriptor lane enters the "
        "chain through discretize (spec→descriptor) and gramian_scaling. "
        "Route the symbolic source through reduce→normalize→(linearize/"
        "discretize) first, or pass its Descriptor directly."
        % type(source).__name__)


def _desc_from(source):
    return _coerce_descriptor_source(source)[0]


# ---------------------------------------------------------------------------
# 内部：卫生清理 / 边准入 mask
# ---------------------------------------------------------------------------
def _dust_clean_A(desc: Descriptor) -> Descriptor:
    """共轭尘埃清理：|A_ij| ≤ 1e-12·max|A| 的非对角元置零。

    背景：eigenvector 基的 T⁻¹AT 在精确对角处带 ~1e-14 机器尘埃；逐字
    count_nonzero 会把尘埃当耦合边（旧 e2e 的 12 条"边"里大部分是尘埃）。
    容差与 p7 的 a_tol 同一相对约定（1e-12·max）。真实耦合（0.05、5.0 等）
    远高于容差，不受影响。A 只在 model 坐标上清，清完才进 scaling。
    """
    A = np.asarray(desc.A, dtype=float)
    scale = float(np.max(np.abs(A))) if A.size else 0.0
    if scale <= 0.0:
        return desc
    tol = _DUST_REL * scale
    off = A - np.diag(np.diag(A))
    ii, jj = np.nonzero((np.abs(off) <= tol) & (off != 0.0))
    if ii.size == 0:
        return desc
    A2 = A.copy()
    A2[ii, jj] = 0.0
    return Descriptor(np.asarray(desc.E, dtype=float), A2,
                      np.asarray(desc.B, dtype=float),
                      np.asarray(desc.C, dtype=float),
                      np.asarray(getattr(desc, "D", np.zeros((0, 0))),
                                 dtype=float))


def _p7_edge_mask(scaled: Descriptor) -> np.ndarray:
    """p7 系数图的边准入 mask（a_tol = 1e-12·max|A_v|，p7.py 同口径）。

    返回布尔阵：off-diag |A_v[i,j]| > a_tol。mask 换算回 model 坐标后
    应用于 desc（见 _apply_mask），保证 contracts 与 coeff 图吃同一批边。
    """
    A = np.asarray(scaled.A, dtype=float)
    a_scale = float(np.max(np.abs(A))) if A.size else 0.0
    a_tol = _DUST_REL * a_scale if a_scale > 0 else 0.0
    off = A - np.diag(np.diag(A))
    return np.abs(off) > a_tol


def _apply_mask(desc: Descriptor, mask: np.ndarray) -> Descriptor:
    A = np.asarray(desc.A, dtype=float)
    A2 = A.copy()
    A2[~mask] = 0.0
    np.fill_diagonal(A2, np.diag(A))
    if np.array_equal(A2, A):
        return desc
    return Descriptor(np.asarray(desc.E, dtype=float), A2,
                      np.asarray(desc.B, dtype=float),
                      np.asarray(desc.C, dtype=float),
                      np.asarray(getattr(desc, "D", np.zeros((0, 0))),
                                 dtype=float))


def _eig_cmp(lam_a, lam_b, scale):
    a = np.sort_complex(np.asarray(lam_a, dtype=complex))
    b = np.sort_complex(np.asarray(lam_b, dtype=complex))
    if a.size != b.size:
        return float("inf")
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_chain(source, *, B_hw: float, kappa_hw: float, v_swing: float = 1.0,
              margin: float = 0.8, k_head: float = 3.0,
              closure_N: int | None = None,
              closure_eps: float | None = None,
              closure_forcing=None,
              closure_carry: str = "full",
              basis_delta=None,
              basis_candidates: tuple | None = None,
              field=None, dx: float = 1.0,
              delta_A_norm: float = 1e-3, tol_l2: float = 1e-2,
              sigma_req=None) -> ChainResult:
    """描述符产线负载走整条编译链，返回结构化结果。

    链序（每个 stage 声明坐标系）：

      boundary → closure(P_N) → split → scaling(P3) → basis(select)
        → [F5 复合：尘埃清理 + 重推 scaling + 边准入 mask + 不动点]
        → cert(evaluate_sys on final) → interface(field 域证据) →
        contracts(emit_route_contracts on final) → p7(generate_config on final)
        → F5 断言（契约边集 == coeff 边集）

    closure_carry="projected"：projected descriptor 成为唯一下游载体，
    scaling 在它上面重推（F2 关闭）；"full"：全 descriptor 前进，
    closure 证书作证据。其余参数直通对应 pass。
    """
    from lowering.closure import project_spectral
    from lowering.split import cvsplit
    from lowering.scaling import gramian_scaling
    from lowering.basis import select_basis
    from lowering.budget import evaluate_sys
    from analysis import HardwareSpec

    if closure_carry not in ("full", "projected"):
        raise ValueError("closure_carry ∈ {'full','projected'}, got %r"
                         % closure_carry)
    stages: list = []
    notes: list = []
    scaling_ledger: dict = {}

    # ---- 1. typed boundary（F1）--------------------------------------------
    desc0, states0, meta, _grid = _coerce_descriptor_source(source)
    stages.append(StageRecord(
        "boundary", "source", "model",
        summary=dict(type=type(source).__name__, n=desc0.n,
                     synthetic=bool(meta.get("synthetic", False))),
        notes=["descriptor-producing object accepted; symbolic/reduce-IR "
               "objects raise TypedBoundaryError (F1 closed by construction)"]))

    # ---- 2. closure（P_N + η_closure）--------------------------------------
    cl = None
    if closure_N is not None:
        cl = project_spectral(desc0, int(closure_N), eps=closure_eps,
                              forcing=closure_forcing)
        lam_full = np.sort_complex(desc0.finite_eigs())
        lam_proj = (np.sort_complex(cl.projected.finite_eigs())
                    if cl.projected is not None else np.array([]))
        lam_max_full = float(np.max(np.abs(lam_full))) if lam_full.size else float("nan")
        lam_max_proj = float(np.max(np.abs(lam_proj))) if lam_proj.size else float("nan")
        spectrum_changed = bool(
            cl.projected is not None and np.isfinite(lam_max_full)
            and np.isfinite(lam_max_proj)
            and abs(lam_max_proj - lam_max_full)
            > 1e-9 * max(lam_max_full, 1e-30))
        stages.append(StageRecord(
            "closure", "model", "model",
            summary=dict(N=cl.N, eta_closure=cl.eta_closure,
                         verdict=cl.verdict, projected_n=cl.N_effective,
                         lam_max_full=lam_max_full, lam_max_projected=lam_max_proj,
                         spectrum_changed=spectrum_changed),
            notes=["P_N projection in model coordinates; projected spectrum "
                   "change is the F2 fact, carried honestly"]))
    else:
        spectrum_changed = False
        lam_max_full = lam_max_proj = float("nan")

    # ---- F2：载体决策 --------------------------------------------------------
    if closure_carry == "projected" and cl is not None and cl.projected is not None:
        desc_carried = cl.projected
        carried = "projected"
        carry_note = ("projected descriptor (N=%d) is the ONLY downstream "
                      "carrier; scaling re-derived on it (F2 closed)"
                      % cl.N)
    else:
        desc_carried = desc0
        carried = "full"
        carry_note = ("full descriptor carried; closure certificate rides as "
                      "evidence (projected would change max|lambda| "
                      "%.4g→%.4g — F2 recorded)" % (lam_max_full, lam_max_proj)
                      if spectrum_changed else
                      "full descriptor carried (no spectrum change)")
    notes.append(carry_note)
    # F2 台账：全谱 alpha（在 FULL descriptor 上）与载体 alpha 分开记录
    alpha_full = None
    if cl is not None and cl.projected is not None:
        from lowering.scaling import alpha_for_bandwidth
        alpha_full = float(alpha_for_bandwidth(desc0, B_hw, margin))
    stages.append(StageRecord(
        "carry-decision", "model", "model",
        summary=dict(carried=carried, closure_verdict=(cl.verdict if cl else None),
                     eta=(cl.eta_closure if cl else None),
                     alpha_full=alpha_full),
        notes=[carry_note]))

    # ---- 3. split（快慢分类；证据，不自动采用 reduced）-----------------------
    sp = cvsplit(desc_carried)
    stages.append(StageRecord(
        "split", "model", "model",
        summary=dict(verdict=sp.verdict, n_slow=sp.n_slow, n_fast=sp.n_fast,
                     gap_ratio=sp.gap_ratio, slaving_error=sp.slaving_error),
        notes=["cvsplit verdict rides as evidence; the QSSA lane "
               "(lowering.split) owns any reduced-system adoption"]))

    # ---- 4. scaling（P3，作用在**载体**上）-----------------------------------
    def _scale_and_mask(desc_in):
        """scaling + p7 边准入 mask 的不动点（F5 的构造性关闭）。"""
        sc = gramian_scaling(desc_in, v_swing=v_swing, B_hw=B_hw,
                             margin=margin, k_head=k_head)
        mask = _p7_edge_mask(sc.scaled)
        for _ in range(_MASK_ITERS):
            masked = _apply_mask(desc_in, mask)
            sc2 = gramian_scaling(masked, v_swing=v_swing, B_hw=B_hw,
                                  margin=margin, k_head=k_head)
            mask2 = _p7_edge_mask(sc2.scaled)
            if np.array_equal(mask, mask2):
                return sc2, masked, mask
            mask = mask2
        raise ChainCompositionError(
            "edge-admission mask did not reach a fixed point in %d iterations"
            % _MASK_ITERS)

    sc_init, desc_m, mask_used = _scale_and_mask(desc_carried)
    n_edges_carrier = int(np.count_nonzero(mask_used)
                          - np.count_nonzero(np.diag(mask_used)))
    scaling_ledger["carrier"] = dict(
        tag="carrier", coords="model",
        alpha=sc_init.alpha, method=sc_init.method,
        alpha_full_on_full_descriptor=alpha_full,
        lam_max=float(np.max(np.abs(desc_carried.finite_eigs()))),
        n_edges=n_edges_carrier,
        note="gramian_scaling on the CARRIED descriptor (F2: not on a "
             "pre-projection descriptor when carry=projected)")
    stages.append(StageRecord(
        "scaling", "model", "physical-time",
        summary=dict(alpha=sc_init.alpha, method=sc_init.method,
                     bandwidth_pass=sc_init.bandwidth_pass),
        notes=["E_v = S E S⁻¹, A_v = α⁻¹ S A S⁻¹, B_v = α⁻¹ S B "
               "(scaling.py input contract), C_v = C S⁻¹"]))

    # ---- 5. basis（Γ·amplification 选基）------------------------------------
    if basis_candidates is not None:
        br = select_basis(desc_m, candidates=tuple(basis_candidates),
                          delta_pert=basis_delta)
    else:
        br = select_basis(desc_m, delta_pert=basis_delta)
    if br.winner is not None and br.winner != "identity" and br.T is not None:
        desc_final = _dust_clean_A(br.transformed)
        coords = "basis-T(model)"
        basis_note = ("winner=%s composed INTO the scaled lane: T⁻¹AT "
                      "(dust-cleaned at 1e-12·max|A|) is the descriptor both "
                      "consumers see; scaling re-derived on it (F5 closed)"
                      % br.winner)
    else:
        desc_final = desc_m
        coords = "model"
        basis_note = ("winner=%s (identity): coordinates unchanged"
                      % (br.winner or "none"))
    stages.append(StageRecord(
        "basis", "model", coords,
        summary=dict(winner=br.winner,
                     cond_phi=(br.candidates[br.winner].cond_phi
                               if br.winner else None),
                     candidates=(list(basis_candidates) if basis_candidates
                                 is not None else "default"),
                     reason=br.reason),
        notes=[basis_note]))

    # F5 复合：在最终坐标上重推 scaling + mask 不动点
    sc_final, desc_final_masked, mask_final = _scale_and_mask(desc_final)
    if not np.array_equal(mask_final, mask_used) and coords == "model":
        raise ChainCompositionError("identity-basis mask changed under "
                                    "re-scaling (fixed point unstable)")

    # 特征值不变性（相似 + α：共轭不改谱，只改坐标）
    lam_carried = np.sort_complex(sc_init.scaled.finite_eigs())
    lam_final = np.sort_complex(sc_final.scaled.finite_eigs())
    eig_inv = _eig_cmp(lam_final, lam_carried,
                       float(np.max(np.abs(lam_carried))) if lam_carried.size else 1.0)
    _tol_eig = 1e-6 * max(1.0, float(np.linalg.cond(np.asarray(br.T, dtype=float)))
                          if (br.winner not in (None, "identity") and br.T is not None)
                          else 1.0)
    if eig_inv > _tol_eig:
        raise ChainCompositionError(
            "F5 conjugation broke eigenvalue invariance: %.3e > tol %.3e"
            % (eig_inv, _tol_eig))

    # 外部传递函数不变性（B_v 语义的最终裁判：内部 round-trip 查不出的错）
    w_rel = np.array([0.0, 0.01, 0.1, 1.0])
    bw0 = float(sc_init.scaled.bandwidth())
    H0 = np.asarray(sc_init.scaled.freqresp(w_rel * 2.0 * np.pi * bw0))
    H1 = np.asarray(sc_final.scaled.freqresp(w_rel * 2.0 * np.pi * bw0))
    cond_t = (float(np.linalg.cond(np.asarray(br.T, dtype=float)))
              if (br.winner not in (None, "identity") and br.T is not None)
              else 1.0)
    tf_inv = float(np.max(np.abs(H0 - H1)) / max(1.0, float(np.max(np.abs(H0)))))
    _tol_tf = 1e-6 * max(1.0, cond_t)
    if tf_inv > _tol_tf:
        raise ChainCompositionError(
            "F5 conjugation broke the external transfer function: %.3e > "
            "tol %.3e (B_v = alpha^-1 S_z B_z semantics violated)" % (tf_inv, _tol_tf))

    scaling_ledger["final"] = dict(
        tag="final", coords=coords + "(physical)",
        alpha=sc_final.alpha, method=sc_final.method,
        n_edges=int(np.count_nonzero(mask_final)
                    - np.count_nonzero(np.diag(mask_final))),
        b_contract=("B_v = alpha^-1·S_z·B_z (scaling.py contract applied in "
                    "the FINAL coordinates); external TF preserved "
                    "(max dev %.2e, cond(T)-scaled tol %.2e)" % (tf_inv, _tol_tf)),
        note="alpha invariance under the basis conjugation: "
             "alpha_final/alpha_carrier = %.12f" % (sc_final.alpha / sc_init.alpha))
    stages.append(StageRecord(
        "scaling-final", coords, "physical-time",
        summary=dict(alpha=sc_final.alpha, alpha_ratio=sc_final.alpha / sc_init.alpha,
                     coords=coords, eig_invariance=eig_inv, tf_invariance=tf_inv),
        notes=[basis_note, scaling_ledger["final"]["b_contract"]]))

    # ---- 6. cert（评在最终载体 + 最终 scaling 上）----------------------------
    spec_h = HardwareSpec(kappa_hw=kappa_hw, B_hw=B_hw, sigma_req=sigma_req,
                          delta_A_norm=delta_A_norm, tol_l2=tol_l2)
    cert = evaluate_sys(desc_final_masked, spec_h, scaling=sc_final)
    stages.append(StageRecord(
        "cert", coords + "(physical)", "certificate",
        summary=dict(status=cert.status, binding_axis=cert.binding_axis),
        notes=["evaluate_sys on the SAME masked descriptor + scaling that "
               "contracts/p7 consume (one coordinate system end-to-end)"]))

    # ---- 7. interface（field 域证据；F3 的显式声明边界）----------------------
    ir = None
    if field is not None:
        from lowering.interface import choose_interface
        u = np.asarray(field, dtype=float).ravel()
        if u.size != desc0.n:
            raise ChainCompositionError(
                "interface field input must live on the workload's grid "
                "(len=%d), got %d (domain 'field', not modal coordinates)"
                % (desc0.n, u.size))
        ir = choose_interface(u, dx=dx)
        interface_status = "evidence (field input)"
        stages.append(StageRecord(
            "interface", "field", "contract",
            summary=dict(mode=ir.mode, singular=ir.singular,
                         single_source=ir.contract.single_source,
                         n_interfaces=ir.n_interfaces),
            notes=["INTERFACE INPUT DOMAIN = FIELD: choose_interface consumes "
                   "a physical field on the periodic grid, NOT modal/"
                   "descriptor coordinates. The modal<->field representation "
                   "transform (DESIGN §5.7 T) is an explicit declared "
                   "boundary here — the chain hands it the field the "
                   "workload carries, and the evidence does not feed p7 "
                   "(cross-representation evidence only). F3 closed by "
                   "declaration + typed length check above."]))
    else:
        interface_status = "not applicable"
        stages.append(StageRecord(
            "interface", "field", "contract",
            summary=dict(status="not applicable"),
            notes=["workload carries NO physical field representation "
                   "(no grid/field in spec_meta or field= argument) — "
                   "recorded explicitly, NOT silently skipped. The "
                   "field-domain boundary stands: if a field is provided "
                   "later, interface runs on it in domain 'field'."]))

    # ---- 8. wrapper + contracts + p7（同一 masked descriptor）----------------
    w_desc = desc_final_masked
    w_states = list(states0 or ["x%d" % i for i in range(w_desc.n)])
    if len(w_states) > w_desc.n:
        # projected 载体：模态坐标前 N 个（P_N = 前 N 坐标，closure.py 约定）
        w_states = w_states[:w_desc.n]
        notes.append("projected carrier: state names truncated to the first "
                     "%d modal coordinates (P_N = first-N convention)" % w_desc.n)
    elif len(w_states) < w_desc.n:
        w_states = w_states + ["x%d" % i for i in
                               range(len(w_states), w_desc.n)]
    w_meta = dict(meta or {})

    class _Wrap:                                    # noqa: N801 - 最小 Discretized 形状
        descriptor = w_desc
        states = w_states
        meta = w_meta
        coupling_graph = None

    from lowering.route_contract import emit_route_contracts
    rep = emit_route_contracts(_Wrap, scaling=sc_final)
    stages.append(StageRecord(
        "route-contracts", coords + "(physical)", "contract",
        summary=dict(n_edges=rep.n_edges, n_feedback=rep.n_feedback,
                     omega_max=rep.omega_max),
        notes=["edges derived from the SAME masked descriptor the coeff "
               "graph uses; omega_max = max|lambda|/alpha = 2π·scaled."
               "bandwidth() (physical-time hand-off, F4 positive check)"]))

    from lowering.p7 import generate_config
    cfg = generate_config(_Wrap, cert, sc_final)

    # ---- 9. F5 断言：契约边集 == coeff 边集 ----------------------------------
    contract_pairs = sorted((int(c["i"]), int(c["j"]))
                            for c in (cc.as_dict() for cc in rep.contracts))
    coeff_pairs = sorted((int(e["i"]), int(e["j"])) for e in cfg.coeff_edges)
    equal = contract_pairs == coeff_pairs
    index_ok = all(
        e.get("route_contract") is not None for e in cfg.coeff_edges)
    if not equal:
        raise ChainCompositionError(
            "F5 FAIL: contract edge set %r != coeff edge set %r "
            "(different coordinate systems between consumers)"
            % (contract_pairs, coeff_pairs))
    if coeff_pairs and not index_ok:
        raise ChainCompositionError(
            "F5 FAIL: coeff edges not indexed into route contracts")
    stages.append(StageRecord(
        "p7-generate-config", coords + "(physical)", "config",
        summary=dict(n_elements=len(cfg.elements),
                     n_coeff_edges=len(cfg.coeff_edges),
                     n_contracts=len(cfg.route_contracts),
                     n_flags=len(cfg.flags),
                     edge_sets_equal=equal, contract_indexed=index_ok),
        notes=["F5 closed: coeff edge set == contract edge set (same (i,j) "
               "pairs, same masked descriptor)"]))

    return ChainResult(
        stages=stages, descriptor=desc_final_masked, coords=coords,
        scaling=sc_final, scaling_full_record=scaling_ledger, cert=cert,
        closure=cl, closure_carried=carried, split=sp, basis=br,
        interface=ir, interface_status=interface_status, contracts=rep,
        config=cfg, eig_invariance=eig_inv, tf_invariance=tf_inv,
        edge_sets_equal=equal, notes=notes)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def selftest() -> dict:
    """最小链路自测：合成对角 3 态系统走全链 + F5 恒等基。"""
    A = np.diag([-1.0, -10.0, -100.0])
    res = run_chain(Descriptor(np.eye(3), A, np.eye(3)[:, :1], np.eye(3)),
                    B_hw=1e4, kappa_hw=1e3,
                    closure_N=3, closure_eps=1e-9)
    assert res.coords == "model" and res.edge_sets_equal is True
    assert res.scaling.alpha > 0 and res.cert is not None
    assert res.interface_status == "not applicable"
    doms = [s.domain_in for s in res.stages]
    assert "field" in doms, doms
    return dict(ok=True, alpha=res.scaling.alpha,
                n_stages=len(res.stages))
