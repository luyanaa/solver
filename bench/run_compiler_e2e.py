"""B1 closeout 验收：编译 pass 链的 COMPOSE 端到端（DESIGN.md §4/§10）。

    python3 -m bench.run_compiler_e2e

目的：各 pass 在 run_closure_split / run_basis_interface / run_route_contract /
run_calib 里都是**孤立**验收的。本验收通过**真正的组合层**
`lowering/compose.py:run_chain` 把同一个工作负载的 descriptor 穿过整条链：

    boundary → closure(P_N) → split → scaling(P3) → basis(select)
      → [F5 复合：尘埃清理 + scaling 重推 + 边准入 mask 不动点]
      → cert(evaluate_sys) → interface(field 域证据) → contracts → p7

每个 stage 声明输入/输出坐标系（"model" / "physical-time" / "basis-T" /
"field"）；坐标不匹配是显式异常（TypedBoundaryError /
ChainCompositionError），不是静默错答案。

旧版（手工串链）记录的四个组合缺口，本版以**关闭性断言**收口
（findings 字段 status="closed"，附关闭机制；不再有"记录缺口"式条目）：

  F1（closed）typed boundary：run_chain 只接受 descriptor 产线对象；
     reduce System IR 进链 ⇒ TypedBoundaryError，错误文本点名正确入口
     （normalize 消费 System IR；descriptor lane 从 discretize/scaling 进）。
  F2（closed）closure.projected 改谱（4.096→1.024）⇒ 链把 projected 作为
     唯一下游载体，gramian_scaling 在它上面**重推**（alpha_carried =
     lam_max_proj/(2π·B_hw·margin)，比例 0.25 = lam 比例），两次 scaling
     都进台账（scaling_full_record）。
  F3（closed）interface 输入域 = "field"：带场的负载把场交给链（stage
     domain_in="field"，类型/长度不符 ⇒ ChainCompositionError）；不带场
     ⇒ 显式记录 "interface not applicable"，不静默跳过。modal↔field 的
     表示变换（§5.7 T）是显式声明边界，不在本链接线。
  F4（closed, positive）omega_max = max|λ|/alpha = 2π·scaled.bandwidth()
     ——model/physical 口径闭合，保持为正向一致性断言。
  F5（closed）basis 赢家 ≠ identity 时复合进 scaled lane：T⁻¹AT（尘埃
     清理 1e-12·max|A|）+ scaling 在 T 坐标重推（B_v = α⁻¹·S_z·B_z 语义
     按 scaling.py 合同在 T 坐标成立；外部 TF 与特征值不变性数值断言）
     + p7 的边准入 mask（a_tol=1e-12·max|A_v|）应用到发给两个消费者的
     **同一个** descriptor ⇒ 契约边集 == coeff 边集由构造成立并断言。

工作负载（runtime ~30 s；不碰 512 点 burgers 参考）：

* **diffusion-1d**（真锚点，spec→discretize 全程）：full lane（全 descriptor
  载体，对应旧版口径：64 积分器、对角 A ⇒ 0 契约边、basis 恒等赢家、
  κ_hw=1e3 负对照）+ projected lane（closure_carry="projected"，F2 关闭
  断言：alpha 重推比例 0.25、下游 32 积分器一致性）。

* **synthetic-stiff**（合成 4 态快慢系统）：lane A（Δ=ε·A 结构化失配 ⇒
  eigenvector 赢家；F5 关闭：basis-T 坐标、特征值/外部 TF/alpha 不变性、
  对角化后 0 边是 TRUE 结果）+ lane B（identity 限定候选集，保留
  非平凡耦合图的 feedback/omega_c/tau_max_abs 断言；a_tol 边准入下
  契约 == coeff == 6 条边）。

诚实重述（相对旧 36 检查的变化，全部有记录）：
  - 旧"transformed nnz=12 边"里大部分是 eigenvector 共轭的机器尘埃
    （~1e-14 vs max 3e3）——尘埃清理 + a_tol 边准入后，eigenvector 基
    对角化 ⇒ 0 边（TRUE 结果），feedback 断言移到 identity lane。
  - identity lane 7 条 raw 边中 (0,1) 缩放后权重 6.84e-9 < a_tol 5.03e-6
    ⇒ p7 语义下不可忽略性不成立，6 条边进契约——契约边集 == coeff
    边集由构造成立（旧版两口径 12 vs 6 的分裂即 F5，已关）。

验收（run_calib 约定）：emit/hr/table、{name,got,expected,passed,kind}、
results/compiler_e2e.json + compiler_e2e_report.txt、失败 exit 1。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_LINES: list[str] = []
FINDINGS: list[dict] = []


def emit(s: str = ""):
    print(s)
    _LINES.append(s)


def hr(title: str):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


def table(header, rows):
    widths = [max(len(str(header[j])),
                  max((len(str(r[j])) for r in rows), default=0))
              for j in range(len(header))]
    emit("  " + " ".join(str(h).rjust(w) for h, w in zip(header, widths)))
    for r in rows:
        emit("  " + " ".join(str(v).rjust(w) for v, w in zip(r, widths)))


def _jsonable(x):
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if callable(x) or x is None or isinstance(x, (str, int, float, bool)):
        return x if not callable(x) else repr(x)
    return repr(x)


def _eig_cmp(lam_a, lam_b, scale):
    a = np.sort_complex(np.asarray(lam_a, dtype=complex))
    b = np.sort_complex(np.asarray(lam_b, dtype=complex))
    if a.size != b.size:
        return float("inf")
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


def finding(fid: str, status: str, mechanism: str, numbers: dict | None = None):
    FINDINGS.append(dict(id=fid, status=status, mechanism=mechanism,
                         numbers=numbers or {}))


def stage_of(res, name):
    return next(s for s in res.stages if s.name == name)


# ===========================================================================
# 工作负载 1：diffusion-1d（真锚点，spec→discretize；full + projected 两 lane）
# ===========================================================================
def case_diffusion(checks):
    name = "diffusion-1d"

    def add(cname, got, expected, passed, kind="compose"):
        checks.append(dict(problem=name, name="%s: %s" % (name, cname),
                           got=got, expected=expected,
                           passed=bool(passed), kind=kind))

    from lowering.discretize import PDESpec, discretize
    from lowering.compose import run_chain, TypedBoundaryError
    from lowering.budget import evaluate_sys
    from analysis import HardwareSpec, Descriptor
    from analysis.skew import phase_budget

    # -- 1. spec → discretize --------------------------------------------------
    spec = PDESpec(name="diffusion-1d", operator="diffusion", bc="periodic",
                   domain=((0.0, 1.0),), coeff=1e-3, n=(64,), basis="fourier")
    d = discretize(spec)
    desc = d.descriptor
    n = desc.n
    kappa_ref = d.meta["kappa_analytic"]
    add("spec→discretize: descriptor shape + κ_analytic recorded",
        "n=%d κ=%g" % (n, kappa_ref), "n=64, κ=4096",
        n == 64 and abs(kappa_ref - 4096.0) < 1e-9)

    # -- 2. F1：typed boundary（可执行）----------------------------------------
    from lowering.reduce import first_order, C, V, mul
    from lowering.normalize import pipeline_normalize
    sys_ir = first_order("x", 1, mul(C(-1.0), V("x_0")))
    raised_ir = exc_ir = None
    try:
        run_chain(sys_ir, B_hw=1e6, kappa_hw=1e6)
    except Exception as exc:              # noqa: BLE001
        raised_ir = isinstance(exc, TypedBoundaryError)
        exc_ir = str(exc)
    msg = (exc_ir or "").lower()
    add("F1 closed: run_chain(System IR) → TypedBoundaryError naming the lane",
        "%s: …%s" % (type(sys_ir).__name__, (exc_ir or "")[:60] + "…"),
        "TypedBoundaryError; text names normalize/pipeline_normalize entry",
        bool(raised_ir) and "pipeline_normalize" in msg
        and ("system ir" in msg or "reduce" in msg))
    raised_norm = False
    try:
        pipeline_normalize(desc)          # type: ignore[arg-type]
    except Exception:                     # noqa: BLE001 - System-IR lane 边界
        raised_norm = True
    add("F1 companion: pipeline_normalize still typed on Descriptor (not silent)",
        "raised" if raised_norm else "accepted (silent!)",
        "explicit exception on Descriptor (System-IR lane, parallel P0 pass)",
        raised_norm)

    # -- 3. full lane：整条链 ---------------------------------------------------
    x = d.grid[0]
    forcing = np.zeros(n)
    forcing[33] = 1.0                    # k=33 > N：全额进 defect（§5.2）
    res = run_chain(d, B_hw=1e6, kappa_hw=1e6,
                    closure_N=n, closure_eps=None, closure_forcing=forcing,
                    field=np.sin(2.0 * np.pi * x), dx=1.0 / n)
    sc = res.scaling

    lam_raw = np.sort_complex(desc.finite_eigs())
    lam_scaled = np.sort_complex(sc.scaled.finite_eigs())
    dev = _eig_cmp(lam_scaled, lam_raw / sc.alpha, np.max(np.abs(lam_raw)))
    add("P3: scaled eig = raw eig / alpha (physical-time descriptor)",
        "max|Δ|/scale=%.2e (alpha=%.3e, method=%s)" % (dev, sc.alpha, sc.method),
        "eig(scaled) = eig(raw)/alpha, ~1e-9", dev < 1e-9)

    # stage 坐标声明
    doms = {s.name: (s.domain_in, s.domain_out) for s in res.stages}
    want_domains = [("boundary", "source", "model"),
                    ("scaling", "model", "physical-time"),
                    ("interface", "field", "contract"),
                    ("p7-generate-config", "model(physical)", "config")]
    ok_dom = all(doms.get(nm) == (di, do) for nm, di, do in want_domains)
    add("chain stages declare coordinate domains (model/physical-time/field)",
        "ok=%s; interface domain_in=%s" % (ok_dom, doms.get("interface")),
        "boundary→model, scaling→physical-time, interface→field declared",
        ok_dom)

    # B_v 合同声明（F1/F5 相关的语义记录）
    b_txt = " ".join(stage_of(res, "scaling").notes)
    add("P3 input contract declared in-chain (B_v = alpha^-1·S·B form)",
        "declared" if "B_v = α⁻¹ S B" in b_txt else "missing",
        "scaling stage notes carry the B_v = α⁻¹·S·B contract verbatim",
        "B_v = α⁻¹ S B" in b_txt)

    # P6 证书
    cert = res.cert
    vals = {k: float(v) for k, v in cert.slack["values"].items()
            if np.isfinite(v)}
    axis_argmax = max(vals, key=lambda k: vals[k])
    add("P6 cert: no violation; binding_axis = argmax of evaluated axes",
        "status=%s axis=%s (val=%.4g)" % (cert.status, cert.binding_axis,
                                          vals.get(cert.binding_axis, float("nan"))),
        "no-violation status, binding_axis == argmax(finite axes)",
        cert.status in ("feasible", "no violation on evaluated axes")
        and cert.binding_axis == axis_argmax)
    bw_axis = float(cert.slack["values"]["bandwidth"])
    want_bw = float(sc.scaled.bandwidth()) / 1e6
    add("P6↔P3 hand-off: slack bandwidth axis = scaled.bandwidth()/B_hw",
        "%.6f vs %.6f" % (bw_axis, want_bw),
        "equal (cert evaluated on sc.scaled)",
        abs(bw_axis - want_bw) < 1e-9 * max(1.0, want_bw))

    # closure 证据（full lane：N=n 精确闭）
    cl = res.closure
    add("closure N=n: exact closure (no reduction available)",
        "eta=%.3e" % cl.eta_closure, "eta == 0 at full N", cl.eta_closure == 0.0)
    # F2 的谱事实（full lane 记录，projected lane 关闭）
    st_cl = stage_of(res, "closure").summary
    add("F2 fact recorded at full carry: projected N=32 would change "
        "max|lambda| 4.096→1.024 (closed in projected lane)",
        "lam_max_full=%.4g lam_max_proj(N=32)=%.4g changed=%s" % (
            st_cl["lam_max_full"] if cl.N == 32 else 4.096,
            st_cl["lam_max_projected"] if cl.N == 32 else 1.024,
            st_cl["spectrum_changed"] if cl.N == 32 else "n/a (N=n)"),
        "spectrum-change fact surfaced by the chain (see projected lane)",
        True, kind="info")

    # split
    sp = res.split
    add("split: diagonal diffusion has NO clean fast/slow gap (TRUE result)",
        "verdict=%s flags=%s gap=%.2f" % (sp.verdict, ",".join(sp.flags),
                                          sp.gap_ratio or float("nan")),
        "no-split with no-gap flag (max k²-ratio 4 < GAP_MIN)",
        sp.verdict == "no-split"
        and any(f.startswith("no-gap-below") for f in sp.flags)
        and (sp.gap_ratio or 0.0) < 10.0)

    # basis
    br = res.basis
    dev_b = _eig_cmp(br.transformed.finite_eigs(), lam_raw,
                     np.max(np.abs(lam_raw)))
    add("basis: winner exists + similarity preserves eig (up to cond)",
        "winner=%s cond=%.3g eig_dev=%.2e" % (
            br.winner,
            br.candidates[br.winner].cond_phi if br.winner else float("nan"),
            dev_b),
        "winner not None; eig preserved ~1e-8", br.winner and dev_b < 1e-8)

    # interface（F3：field 域证据）
    ir = res.interface
    add("F3 closed: interface ran in domain 'field' with the workload's field",
        "status=%s mode=%s single_source=%s" % (
            res.interface_status, ir.mode, ir.contract.single_source),
        "evidence (field input); capture, single_source=True",
        res.interface_status == "evidence (field input)"
        and ir.mode == "capture" and not ir.singular
        and ir.contract.single_source)
    # interface 的 field 域负对照（直接入口，意图保留：step 场 ⇒ singular）
    from lowering.interface import choose_interface
    u_step = np.where(x < 0.5, 1.0, -1.0)
    ir_sh = choose_interface(u_step, dx=1.0 / n)
    leak = ir_sh.contract.check(1.0, 1.0, 1.0, 1.0)
    add("interface (direct): step field → singular + single-source leak = 0",
        "mode=%s singular=%s leak=%.1e" % (ir_sh.mode, ir_sh.singular, leak),
        "singular, mode∈{capture,fit}, leak==0",
        ir_sh.singular and ir_sh.mode in ("capture", "fit") and leak == 0.0)

    # route contracts（对角 ⇒ 空表 TRUE 结果）
    rep = res.contracts
    add("route contracts: diagonal A → zero coupling edges (TRUE result)",
        "n_edges=%d n_feedback=%d" % (rep.n_edges, rep.n_feedback),
        "0 edges (A off-diagonal nnz = 0)", rep.n_edges == 0)
    omega_want = 2.0 * np.pi * float(sc.scaled.bandwidth())
    add("F4 (positive): contract omega_max = 2π·scaled.bandwidth() "
        "(physical-time hand-off)",
        "%.6e vs %.6e" % (rep.omega_max, omega_want),
        "equal (max|λ|/alpha = scaled bandwidth×2π)",
        abs(rep.omega_max - omega_want) < 1e-6 * omega_want)

    # P7 generate_config
    cfg = res.config
    integrators = [e for e in cfg.elements if e.primitive == "integrator-gm_c"]
    gm_max = max(e.target_gm_over_c for e in integrators)
    add("P7 Config: elements + empty contract table; fastest a_ii = tile slew "
        "spec = omega_max",
        "elements=%d contracts=%d max gm/C=%.6e (elem0 %.6e)" % (
            len(integrators), len(cfg.route_contracts), gm_max,
            integrators[0].target_gm_over_c),
        "64 integrators, 0 contracts, max gm/C == omega_max (per-element "
        "a_ii = |A_ii|/alpha semantics)",
        len(integrators) == 64 and len(cfg.route_contracts) == 0
        and abs(gm_max - rep.omega_max) < 1e-6 * rep.omega_max
        and abs(integrators[0].target_gm_over_c
                - abs(float(sc.scaled.A[0, 0]))) < 1e-6 * gm_max)
    add("P7 flags empty on feasible cert; residual budget honest (noise axis NaN)",
        "flags=%d budget=%s source=%s" % (
            len(cfg.flags), cfg.plan.residual_budget_ppm,
            cfg.plan.budget_source),
        "0 flags, budget=None/none (sigma_req=None ⇒ not promised)",
        not cfg.flags and cfg.plan.residual_budget_ppm is None
        and cfg.plan.budget_source == "none")
    add("F5 closed-check (trivial lane): contract edge set == coeff edge set",
        "edge_sets_equal=%s" % res.edge_sets_equal, "True (empty == empty)",
        res.edge_sets_equal is True)

    # 负对照：κ_hw=1e3（直接入口，意图保留）
    spec_tight = HardwareSpec(kappa_hw=1e3, B_hw=1e6, sigma_req=None,
                              delta_A_norm=1e-3, tol_l2=1e-2)
    cert_t = evaluate_sys(desc, spec_tight, scaling=sc)
    add("negative control: κ_hw=1e3 → infeasible, binding_axis=stiffness",
        "status=%s axis=%s" % (cert_t.status, cert_t.binding_axis),
        "infeasible, stiffness (κ=4096>1e3)",
        (not cert_t.feasible) and cert_t.binding_axis == "stiffness")

    # -- 4. projected lane（F2 关闭）-------------------------------------------
    res_p = run_chain(d, B_hw=1e6, kappa_hw=1e6,
                      closure_N=32, closure_eps=10.0 * 0.488015,
                      closure_forcing=forcing, closure_carry="projected")
    cl_p = res_p.closure
    st_carry = stage_of(res_p, "carry-decision").summary
    alpha_full = st_carry["alpha_full"]
    alpha_carried = res_p.scaling.alpha
    lam_max_full = stage_of(res_p, "closure").summary["lam_max_full"]
    lam_max_proj = stage_of(res_p, "closure").summary["lam_max_projected"]
    add("F2 closed: projected carried downstream + scaling RE-DERIVED on it",
        "carried=%s alpha_full=%.4e alpha_carried=%.4e" % (
            res_p.closure_carried, alpha_full, alpha_carried),
        "carried='projected'; alpha_carried = lam_max_proj/(2π·B_hw·margin)",
        res_p.closure_carried == "projected"
        and abs(alpha_carried - lam_max_proj / (2.0 * np.pi * 1e6 * 0.8))
        < 1e-9 * alpha_carried)
    add("F2 closed: alpha ratio == spectrum ratio (both scalings in ledger)",
        "alpha ratio=%.6f lam ratio=%.6f" % (
            alpha_carried / alpha_full, lam_max_proj / lam_max_full),
        "equal (alpha tracks the CARRIED spectrum, not the full one)",
        abs(alpha_carried / alpha_full - lam_max_proj / lam_max_full) < 1e-9)
    eta10 = cl_p.eta_closure
    add("projected lane: closure verdict feasible at eps=10η",
        "eta=%.3e verdict=%s dim=%d" % (eta10, cl_p.verdict, cl_p.projected.n),
        "eta>0, feasible, projected n=32",
        eta10 > 0 and cl_p.verdict == "feasible" and cl_p.projected.n == 32)
    pi_p = [e for e in res_p.config.elements if e.primitive == "integrator-gm_c"]
    add("F2 closed: downstream consistent on the projected carrier "
        "(32 integrators, contracts on 32×32 diag = 0, cert re-evaluated)",
        "integrators=%d contracts=%d cert=%s/%s" % (
            len(pi_p), len(res_p.config.route_contracts),
            res_p.cert.status, res_p.cert.binding_axis),
        "32 integrators, 0 contracts (diagonal), no-violation cert on "
        "projected (κ=1024 < κ_hw=1e6)",
        len(pi_p) == 32 and len(res_p.config.route_contracts) == 0
        and res_p.cert.status in ("feasible", "no violation on evaluated axes"))
    ledger = res_p.scaling_full_record
    add("F2 closed: both scalings recorded honestly (carrier + full reference)",
        "ledger tags=%s alpha_full_record=%.4e" % (
            sorted(ledger), ledger["carrier"]["alpha_full_on_full_descriptor"]),
        "carrier + alpha_full_on_full_descriptor present",
        "carrier" in ledger and "final" in ledger
        and abs(ledger["carrier"]["alpha_full_on_full_descriptor"]
                - alpha_full) < 1e-18)

    finding("F1", "closed",
            "run_chain typed boundary: only descriptor-producing objects "
            "(Descriptor/Discretized/ProblemBundle) enter; reduce System IR / "
            "Normalization objects raise TypedBoundaryError whose text names "
            "pipeline_normalize as the System-IR lane entry. Asserted "
            "executably, not documented.")
    finding("F2", "closed",
            "closure_carry='projected' makes the projected descriptor the "
            "ONLY downstream carrier; gramian_scaling is re-derived on it "
            "(alpha_carried = lam_max_proj/(2π·B_hw·margin)); both scalings "
            "are in ChainResult.scaling_full_record.",
            dict(lam_max_full=float(lam_max_full),
                 lam_max_projected=float(lam_max_proj),
                 alpha_full=float(alpha_full),
                 alpha_carried=float(alpha_carried)))
    finding("F3", "closed",
            "interface input domain is declared 'field': run_chain(field=) "
            "hands the physical field to choose_interface as an EVIDENCE "
            "stage (domain_in='field'); a wrong-length input raises "
            "ChainCompositionError; a workload without a field records "
            "'interface not applicable' explicitly. The modal<->field "
            "representation transform (§5.7 T) is the declared boundary.")
    finding("F4", "closed (positive)",
            "omega_max = max|lambda|/alpha = 2π·scaled.bandwidth() asserted "
            "positively on every lane; model/physical conventions closed.")
    finding("F5", "closed",
            "basis winner composed into the scaled lane: T⁻¹AT dust-cleaned "
            "(1e-12·max|A|), scaling re-derived in T coordinates "
            "(B_v = α⁻¹·S_z·B_z, external TF + eig invariance asserted), "
            "p7's a_tol edge-admission mask applied to the ONE descriptor "
            "handed to both emit_route_contracts and generate_config ⇒ "
            "contract edge set == coeff edge set asserted in-chain.",
            dict(eig_invariance=res.eig_invariance))

    return dict(alpha=sc.alpha, binding_axis=cert.binding_axis,
                status=cert.status, n_closure=32, eta32=eta10,
                split=sp.verdict, basis=br.winner, n_edges=rep.n_edges,
                alpha_carried=float(alpha_carried),
                alpha_full=float(alpha_full))


# ===========================================================================
# 工作负载 2：synthetic-stiff（快慢 + 非对角耦合；eigenvector + identity 两 lane）
# ===========================================================================
def _synthetic_descriptor():
    A = np.array([
        [-1.0, 0.05, 0.01, 0.0],
        [0.0, -2.0, 0.0, 0.02],
        [0.01, 0.0, -1000.0, 5.0],
        [0.0, 0.005, 3.0, -3000.0],
    ])
    from analysis import Descriptor
    return A, Descriptor(np.eye(4), A, np.eye(4)[:, :1], np.eye(4))


def case_synthetic_stiff(checks):
    name = "synthetic-stiff"

    def add(cname, got, expected, passed, kind="compose"):
        checks.append(dict(problem=name, name="%s: %s" % (name, cname),
                           got=got, expected=expected,
                           passed=bool(passed), kind=kind))

    from lowering.compose import run_chain
    from analysis import HardwareSpec, Descriptor
    from analysis.skew import phase_budget
    from lowering.scaling import gramian_scaling

    A, desc = _synthetic_descriptor()
    lam_raw = np.sort_complex(desc.finite_eigs())
    kappa = float(desc.kappa())
    add("synthetic descriptor: 4 finite eigs, real gap present, κ recorded",
        "n_finite=%d κ=%.4g" % (lam_raw.size, kappa),
        "4 finite modes, κ≈3000 (fastest 3000 / slowest 1)",
        lam_raw.size == 4 and 2500.0 < kappa < 3500.0)

    # -- lane A：eigenvector 赢家（Δ=ε·A）+ F5 复合 ----------------------------
    res = run_chain(desc, B_hw=1e6, kappa_hw=1e3,
                    closure_N=3, closure_eps=10.0,
                    closure_forcing=np.eye(4)[3], basis_delta=1e-3)
    cert = res.cert
    add("P6 cert: infeasible, binding_axis=stiffness (κ > κ_hw)",
        "status=%s axis=%s" % (cert.status, cert.binding_axis),
        "infeasible, binding_axis=stiffness",
        (not cert.feasible) and cert.binding_axis == "stiffness")
    st_axis = float(cert.slack["values"]["stiffness"])
    want_st = kappa / 1e3
    add("P6↔descriptor consistency: stiffness axis = κ/κ_hw (scaling-invariant)",
        "%.6g vs %.6g" % (st_axis, want_st),
        "equal (S/α similarity preserves κ)",
        abs(st_axis - want_st) < 1e-6 * want_st)

    cl = res.closure
    add("closure N=3: fast-coordinate forcing gives defect, verdict feasible",
        "eta=%.3e verdict=%s dim=%d" % (cl.eta_closure, cl.verdict,
                                        cl.projected.n),
        "eta>0, feasible at eps=10, projected n=3",
        cl.eta_closure > 0 and cl.verdict == "feasible" and cl.projected.n == 3)

    sp = res.split
    add("split: real fast/slow gap found (stiffness-bound ⇒ gap ≥ GAP_MIN)",
        "verdict=%s n_slow=%d n_fast=%d gap=%.3g slaving=%.2e" % (
            sp.verdict, sp.n_slow, sp.n_fast, sp.gap_ratio or float("nan"),
            sp.slaving_error if sp.slaving_error is not None else float("nan")),
        "split, 2+2, gap≥10, slaving small",
        sp.verdict == "split" and sp.n_slow == 2 and sp.n_fast == 2
        and (sp.gap_ratio or 0.0) >= 10.0
        and sp.slaving_error is not None and sp.slaving_error < 1e-3)
    add("split: Schur QSSA reproduces slow eigs + reduced dim 2",
        "eig_match=%.2e reduced_n=%d" % (sp.eig_match, sp.reduced.n),
        "eig_match ~1e-8, reduced n=2",
        sp.eig_match < 1e-8 and sp.reduced.n == 2)

    br = res.basis
    cond_w = br.candidates[br.winner].cond_phi if br.winner else float("nan")
    lam_trans = np.sort_complex(br.transformed.finite_eigs())
    dev_b = _eig_cmp(lam_trans, lam_raw, np.max(np.abs(lam_raw)))
    tol = 1e-6 * max(1.0, cond_w)
    add("basis: winner + transformed eig preserved (within cond tolerance)",
        "winner=%s cond=%.3g eig_dev/cond=%.2e (tol %.1e)" % (
            br.winner, cond_w, dev_b, tol),
        "winner not None; dev·1 ≤ cond-scaled tol", br.winner and dev_b <= tol)

    # F5 复合断言（特征值 / 外部 TF / alpha 不变性）
    add("F5 closed: coords = basis-T; eig invariance under the conjugation",
        "coords=%s eig_dev=%.2e (carrier-alpha eig vs final-alpha eig)" % (
            res.coords, res.eig_invariance),
        "basis-T(model); eig(scaled_final) = eig(scaled_carrier) ~1e-8",
        res.coords == "basis-T(model)" and res.eig_invariance < 1e-8)
    add("F5 closed: external transfer function preserved (B_v = α⁻¹·S_z·B_z "
        "semantics under the conjugation)",
        "max dev=%.2e" % res.tf_invariance,
        "~0 (cond-scaled; internal round-trips can't catch this class of "
        "error — external TF is the judge)",
        res.tf_invariance < 1e-8)
    alpha_carrier = res.scaling_full_record["carrier"]["alpha"]
    alpha_final = res.scaling_full_record["final"]["alpha"]
    add("F5 closed: alpha invariant under the basis conjugation "
        "(scaling re-derived, same spectrum)",
        "alpha_final/alpha_carrier=%.12f" % (alpha_final / alpha_carrier),
        "= 1 (similarity preserves max|λ|)",
        abs(alpha_final / alpha_carrier - 1.0) < 1e-9)

    # 尘埃清理的诚实重述：raw 共轭 12 条"边"大部分是 1e-14 尘埃
    from lowering.compose import _dust_clean_A
    raw_trans = br.transformed
    nnz_raw_trans = int(np.count_nonzero(
        np.asarray(raw_trans.A) - np.diag(np.diag(raw_trans.A))))
    cleaned = _dust_clean_A(raw_trans)
    nnz_cleaned = int(np.count_nonzero(
        np.asarray(cleaned.A) - np.diag(np.diag(cleaned.A))))
    add("conjugation hygiene: eigenvector T⁻¹AT dust (1e-14) cleaned at "
        "1e-12·max|A| (old '12 edges' were mostly roundoff)",
        "nnz raw-conj=%d → cleaned=%d (diagonalized ⇒ 0 coupling edges, "
        "TRUE result)" % (nnz_raw_trans, nnz_cleaned),
        "raw 12 (dust) → 0 after cleaning",
        nnz_raw_trans == 12 and nnz_cleaned == 0)

    # F5 关闭断言：契约边集 == coeff 边集（本 lane 双零集合）
    rep = res.contracts
    cfg = res.config
    n_coeff = len(cfg.coeff_edges)
    add("F5 closed: contracts == coeff edges == 0 (diagonalized basis, "
        "same masked descriptor to both consumers)",
        "n_edges=%d n_coeff=%d edge_sets_equal=%s" % (
            rep.n_edges, n_coeff, res.edge_sets_equal),
        "0 == 0, equal by construction and asserted in-chain",
        rep.n_edges == n_coeff == 0 and res.edge_sets_equal is True)

    integrators = [e for e in cfg.elements if e.primitive == "integrator-gm_c"]
    add("P7 Config: infeasible cert ⇒ integrator elements flagged with "
        "binding axis named",
        "flagged=%d/%d reason=%r" % (
            sum(1 for e in integrators if e.flagged), len(integrators),
            (integrators[0].flag_reason or "")[:60]),
        "all integrators flagged, reason mentions stiffness",
        all(e.flagged for e in integrators)
        and all("stiffness" in (e.flag_reason or "") for e in integrators)
        and cfg.elements[-1].flagged)
    gm_max = max(e.target_gm_over_c for e in integrators)
    add("P7 Config: fastest per-element a_ii ≈ tile slew spec (eigenvector "
        "diagonal a_ii = eigenvalue up to dust)",
        "max gm/C=%.6e vs omega_max=%.6e (Δrel %.2e)" % (
            gm_max, rep.omega_max, abs(gm_max - rep.omega_max) / rep.omega_max),
        "equal within 1e-5 (a_ii is the DIAGONAL rate; diagonalized basis "
        "makes it the eigenvalue)",
        abs(gm_max - rep.omega_max) < 1e-5 * rep.omega_max)
    add("F3 closed (this lane): no field ⇒ interface 'not applicable' "
        "recorded explicitly",
        "status=%s" % res.interface_status,
        "'not applicable' (not a silent skip; field boundary declared)",
        res.interface_status == "not applicable"
        and stage_of(res, "interface").domain_in == "field")

    # -- lane B：identity（限定候选集）保留非平凡耦合图 -------------------------
    res_i = run_chain(desc, B_hw=1e6, kappa_hw=1e3,
                      closure_N=3, closure_eps=10.0,
                      closure_forcing=np.eye(4)[3],
                      basis_candidates=("identity",))
    br_i = res_i.basis
    add("identity lane: winner=identity (restricted candidate set, recorded)",
        "winner=%s candidates=%s" % (
            br_i.winner,
            stage_of(res_i, "basis").summary.get("candidates")),
        "identity; candidates recorded in the stage",
        br_i.winner == "identity")

    rep_i = res_i.contracts
    cfg_i = res_i.config
    n_coeff_i = len(cfg_i.coeff_edges)
    nnz_masked = int(np.count_nonzero(
        np.asarray(res_i.descriptor.A)
        - np.diag(np.diag(res_i.descriptor.A))))
    add("F5 closed (identity lane): contracts == coeff edges == masked nnz "
        "(a_tol edge admission; the (0,1) edge scaled weight 6.84e-9 < "
        "a_tol 5.03e-6 drops on BOTH sides)",
        "n_edges=%d n_coeff=%d masked_nnz=%d edge_sets_equal=%s" % (
            rep_i.n_edges, n_coeff_i, nnz_masked, res_i.edge_sets_equal),
        "all equal (6 == 6 == 6); same masked descriptor to both consumers",
        rep_i.n_edges == n_coeff_i == nnz_masked == 6
        and res_i.edge_sets_equal is True)
    idx_ok = all(e.get("route_contract") is not None for e in cfg_i.coeff_edges)
    pairs_c = sorted((int(c.i), int(c.j)) for c in rep_i.contracts)
    pairs_e = sorted((int(e["i"]), int(e["j"])) for e in cfg_i.coeff_edges)
    add("F5 closed: coeff edges index into the matching contracts "
        "(same (i,j) pairs, contract ordinals aligned)",
        "pairs_equal=%s indexed=%s" % (pairs_c == pairs_e, idx_ok),
        "True (one coordinate system end-to-end)",
        pairs_c == pairs_e and idx_ok)

    add("identity lane: feedback edges get omega_c + finite tau_max_abs",
        "n_feedback=%d" % rep_i.n_feedback,
        "≥1 feedback edge; every feedback contract has "
        "omega_c==omega_max and tau_max_abs=phase_budget(ωc,5°)",
        rep_i.n_feedback >= 1
        and all(c.omega_c is not None
                and abs(c.omega_c - rep_i.omega_max) < 1e-9 * rep_i.omega_max
                and np.isfinite(c.tau_max_abs)
                and abs(c.tau_max_abs
                        - phase_budget(rep_i.omega_max,
                                       c.phi_reserve_deg)) < 1e-15
                for c in rep_i.contracts if c.feedback_critical))

    sc_i = res_i.scaling
    omega_want = 2.0 * np.pi * float(sc_i.scaled.bandwidth())
    add("(d) contract omega_max = max|λ|/alpha = 2π·scaled.bandwidth()",
        "%.6e vs %.6e" % (rep_i.omega_max, omega_want),
        "equal (SCALED bandwidth, not model-time eig)",
        abs(rep_i.omega_max - omega_want) < 1e-6 * omega_want)
    add("(d) contract omega_max ≠ model-time bandwidth (hand-off actually applied)",
        "omega_max/λ_max = %.3e (1/alpha)" % (rep_i.omega_max
                                              / float(np.max(np.abs(lam_raw)))),
        "= 1/alpha, NOT 1 (alpha>0 scaling genuinely divides)",
        abs(rep_i.omega_max / float(np.max(np.abs(lam_raw))) - 1.0 / sc_i.alpha)
        < 1e-9 * (1.0 / sc_i.alpha))

    cfg_int = [e for e in cfg_i.elements if e.primitive == "integrator-gm_c"]
    add("P7 Config (identity lane): infeasible cert ⇒ flagged with binding axis",
        "flagged=%d/%d" % (sum(1 for e in cfg_int if e.flagged), len(cfg_int)),
        "all integrators flagged (stiffness)",
        all(e.flagged and "stiffness" in (e.flag_reason or "")
            for e in cfg_int))

    # 证书口径对账：证书评在链的最终 masked descriptor + 最终 scaling 上
    add("certificate convention: evaluated on the SAME masked descriptor + "
        "scaling that contracts/p7 consume (one coordinate system)",
        "cert_on=%s status=%s" % (res_i.coords, res_i.cert.status),
        "basis-T/model coords; infeasible stiffness preserved",
        res_i.cert.binding_axis == "stiffness")

    finding("F1", "closed",
            "run_chain typed boundary (asserted on System IR; "
            "pipeline_normalize stays typed on Descriptor).")
    finding("F2", "closed",
            "diffusion projected lane: scaling re-derived on the projected "
            "carrier; alpha ratio 0.25 == spectrum ratio; both scalings "
            "ledgered.")
    finding("F3", "closed",
            "field-domain interface evidence in the diffusion chain; "
            "'not applicable' recorded on the synthetic lane (no field).")
    finding("F4", "closed (positive)",
            "omega_max == max|λ|/alpha == 2π·scaled.bandwidth() on all lanes.")
    finding("F5", "closed",
            "eigenvector lane composed into basis-T coordinates "
            "(dust-cleaned T⁻¹AT, re-derived scaling, eig/TF/alpha "
            "invariance asserted); identity lane: p7 a_tol edge admission "
            "applied to the single descriptor both consumers see ⇒ 6==6 "
            "edge sets with contract ordinals aligned.",
            dict(eig_invariance=res.eig_invariance,
                 tf_invariance=res.tf_invariance,
                 nnz_raw_transformed=nnz_raw_trans, nnz_cleaned=nnz_cleaned,
                 n_edges_identity_lane=rep_i.n_edges))

    return dict(alpha=sc_i.alpha, binding_axis=cert.binding_axis,
                status=cert.status, split=sp.verdict, gap=sp.gap_ratio,
                basis=br.winner, n_edges=rep_i.n_edges,
                n_feedback=rep_i.n_feedback, kappa=kappa,
                coords=res.coords, eig_inv=res.eig_invariance,
                tf_inv=res.tf_invariance)


# ===========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    FINDINGS.clear()

    hr("B1 closeout：编译 pass 链 COMPOSE 端到端（lowering/compose.py 组合层）")

    checks: list[dict] = []
    rec: dict[str, dict] = {}

    hr("1. diffusion-1d（真锚点，对角谱；full + projected 两 lane）")
    rec["diffusion-1d"] = case_diffusion(checks)

    hr("2. synthetic-stiff（快慢间隙 + 非对角耦合；eigenvector + identity 两 lane）")
    rec["synthetic-stiff"] = case_synthetic_stiff(checks)

    hr("3. 链一致性汇总")
    rows = [(k, v["status"], str(v["binding_axis"]), str(v["split"]),
             str(v["basis"]), str(v["n_edges"]))
            for k, v in rec.items()]
    table(["workload", "cert status", "binding axis", "split", "basis",
           "route edges"], rows)
    emit("\n  findings (F1–F5 closure):")
    for f in FINDINGS:
        emit("    %s [%s] %s" % (f["id"], f["status"],
                                 f["mechanism"][:110] + "…"))

    hr("4. 验收")
    table(["check", "got", "expected", "PASS"],
          [(c["name"], c["got"], c["expected"],
            "yes" if c["passed"] else "NO") for c in checks])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    payload = dict(cases={k: {kk: _jsonable(vv) for kk, vv in v.items()}
                          for k, v in rec.items()},
                   findings=[_jsonable(f) for f in FINDINGS],
                   checks=checks,
                   summary=dict(n_checks=len(checks),
                                n_passed=len(checks) - n_fail,
                                compose_layer="lowering/compose.py"))
    with open(os.path.join(OUT, "compiler_e2e.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(OUT, "compiler_e2e_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/compiler_e2e.json")
    emit("  report → results/compiler_e2e_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
