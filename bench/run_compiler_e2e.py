"""B1 closeout 验收：编译 pass 链的 COMPOSE 端到端（DESIGN.md §4/§10）。

    python3 -m bench.run_compiler_e2e

目的：各 pass 在 run_closure_split / run_basis_interface / run_route_contract /
run_calib 里都是**孤立**验收的。本验收把**同一个工作负载的 descriptor** 依次
穿过编译器会走的整条链，并断言证书的 binding axis 与关键 verdict 在每一级
**保持一致**——"各 pass 单独对"不等于"链上对"：

    spec → discretize → [normalize lane] → gramian_scaling（.scaled + alpha）
      → evaluate_sys(scaling=) 证书（binding axis + slack）
      → closure.project_spectral（η_closure）
      → split.cvsplit（快慢分类）
      → basis.select_basis（赢家 + 变换后 descriptor）
      → interface.choose_interface（capture/fit + 单源合约）
      → route_contract.emit_route_contracts（逐边四判据）
      → p7.generate_config（最终 Config）

两个工作负载（runtime ~24 s；不碰 512 点 burgers 参考）：

* **diffusion-1d**（真锚点，spec→discretize 全程）：对角 νk²，κ=4096。
  证书无违约，binding_axis=argmax(已评轴)=robustness_A（Γ_proj·δA=1.0；
  bandwidth 轴=0.8=物理时间交接 sc.scaled.bandwidth()/B_hw，数值断言）；
  closure 在 k>N forcing 下 η>0；split 无谱间隙（连续 k² 谱，最大比值
  4<GAP_MIN）—— "对角扩散无干净快慢分" 是 TRUE 结果，按 TRUE 断言；
  basis 恒等赢家（对角谱换基无税可逃）；route 契约空表（A 非对角零元
  —— 真"无布线"）；Config 携带 64 元素（每元素 a_ii=|A_ii|/alpha，
  最快模态 = tile slew spec = omega_max）+ 空契约表。负对照 κ_hw=1e3。

* **synthetic-stiff**（合成 4 态快慢系统，唯一有非对角耦合 + 真谱间隙
  的负载；discretize 没有锚定的刚性 ODE 组合，故直接建 Descriptor 并
  在记录里如实标注 synthetic）：慢块 {-1,-2}、快块 {-1000,-3000}，
  块间小耦合。证书 binding_axis=stiffness（κ≈3000 > κ_hw=1e3）⇒ 下游
  一致性要求：split 找到真快慢间隙（gap=500、slaving 9.9e-9、Schur 复
  现慢特征值 1e-10）、route 契约条数 = **basis 赢家变换后**（本例
  eigenvector 基，Δ=ε·A 结构化失配下赢）descriptor 的非对角非零数
  （12，raw 是 7——换基确实改了耦合图，口径按 transformed 断言）、
  omega_max = max|λ|/alpha = 2π·sc.scaled.bandwidth()（物理时间交接
  (d)：契约用 SCALED 带宽，不是 model-time eig）、infeasible 证书 ⇒
  积分器元素 flagged 且 flag_reason 点名 binding axis。

诚实发现（pass 组合摩擦，随 JSON 一起落盘，见 findings 字段）：

  F1 normalize（lowering.normalize）消费 reduce 的 System IR，不消费
     Discretized/Descriptor——descriptor 产线锚点的无量纲化在 discretize
     的 unit-domain 前置里；两条 P0 lane 并行，编译器 descriptor lane 从
     scaling 进入。按接口边界断言（喂 Descriptor 必须显式 raise）。
  F2 closure 线性路径的 projected（N=32）会改 max|λ|（64→32 模态），
     而 alpha/scaling 是在**投影前**的 descriptor 上算的；若把 projected
     带进下游，route_contract/p7 的 omega_max（用给定 descriptor 的
     lam_max）与 alpha（来自原 lam_max）口径分裂。本验收记录该缺口：
     把 projected 带进下游前必须**重算** scaling。负载 A 携带完整
     descriptor 前进 + closure 证书作证据。
  F3 interface 消费**物理场**（周期网格 u），不消费 descriptor/modal 坐
     标；modal↔field 的 plumb（表示变换 T 应用）是编译器尚未接的缝，
     本验收用锚点 grid 上的真实场（光滑/阶跃）作最小真实输入并记录。
  F4 route_contract/generate_config 从**给定** descriptor 推耦合边；
     证书评在 sc.scaled（物理时间）上而契约的 J 是 model 坐标 A——
     谱半径一致（相似+α 缩放），omega_max 已除 alpha，口径闭合
     （本验收正向断言，非缺口）。
  F5 basis 选择没有复合进 scaling：p7 的系数图消费 scaling.scaled
     （RAW 坐标 + P3 预条件 + a_tol 阈值），而契约表消费传入的
     descriptor（这里 = basis 赢家 transformed A）。赢家 ≠ identity 时
     两者是不同坐标系（12 条 transformed 契约边 vs 6 条 raw 系数边）；
     编译器须先把 T 复合进 scaled descriptor 再发系数图+契约。

36 项验收，~24 s，全过（results/compiler_e2e.json + report）。

run_calib 约定：OPENBLAS 前导、emit/hr/table、{name,got,expected,passed,
kind}、results/compiler_e2e.json + compiler_e2e_report.txt、失败 exit 1。
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
    """排序复谱的最大偏差（相对尺度归一）。"""
    a = np.sort_complex(np.asarray(lam_a, dtype=complex))
    b = np.sort_complex(np.asarray(lam_b, dtype=complex))
    if a.size != b.size:
        return float("inf")
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


# ===========================================================================
# 工作负载 1：diffusion-1d（真锚点，spec→discretize）
# ===========================================================================
def case_diffusion(checks):
    name = "diffusion-1d"

    def add(cname, got, expected, passed, kind="compose"):
        checks.append(dict(problem=name, name="%s: %s" % (name, cname),
                           got=got, expected=expected,
                           passed=bool(passed), kind=kind))

    from lowering.discretize import PDESpec, discretize
    from lowering.scaling import gramian_scaling
    from lowering.budget import evaluate_sys
    from lowering.closure import project_spectral
    from lowering.split import cvsplit
    from lowering.basis import select_basis
    from lowering.interface import choose_interface
    from lowering.route_contract import emit_route_contracts
    from lowering.p7 import generate_config
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

    # -- 2. normalize lane 边界（发现 F1）--------------------------------------
    from lowering.normalize import pipeline_normalize
    raised = False
    try:
        pipeline_normalize(desc)          # type: ignore[arg-type]
    except Exception as exc:              # noqa: BLE001 - 任何异常都证明是 typed 边界
        raised = True
        exc_name = type(exc).__name__
    FINDINGS.append(dict(
        id="F1", workload=name,
        text="normalize (lowering.normalize) consumes the reduce System IR, "
             "not Discretized/Descriptor; nondimensionalization for "
             "descriptor-producing anchors is the discretize unit-domain "
             "precondition. Descriptor lane enters the chain at scaling.",
    ))
    add("normalize lane boundary is typed (Finding F1)",
        "%s on Descriptor" % exc_name if raised else "accepted (silent!)",
        "explicit exception on Descriptor (System-IR lane, parallel P0 pass)",
        raised)

    # -- 3. P3 gramian_scaling -------------------------------------------------
    B_hw = 1e6
    sc = gramian_scaling(desc, v_swing=1.0, B_hw=B_hw)
    lam_raw = np.sort_complex(desc.finite_eigs())
    lam_scaled = np.sort_complex(sc.scaled.finite_eigs())
    dev = _eig_cmp(lam_scaled, lam_raw / sc.alpha, np.max(np.abs(lam_raw)))
    add("P3: scaled eig = raw eig / alpha (physical-time descriptor)",
        "max|Δ|/scale=%.2e (alpha=%.3e, method=%s)" % (dev, sc.alpha, sc.method),
        "eig(scaled) = eig(raw)/alpha, ~1e-9", dev < 1e-9)

    # -- 4/5. P6 证书（评在 sc.scaled 上）--------------------------------------
    spec_d = HardwareSpec(kappa_hw=1e6, B_hw=B_hw, sigma_req=None,
                          delta_A_norm=1e-3, tol_l2=1e-2)
    cert = evaluate_sys(desc, spec_d, scaling=sc)
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
    want_bw = float(sc.scaled.bandwidth()) / B_hw
    add("P6↔P3 hand-off: slack bandwidth axis = scaled.bandwidth()/B_hw",
        "%.6f vs %.6f" % (bw_axis, want_bw),
        "equal (cert evaluated on sc.scaled)",
        abs(bw_axis - want_bw) < 1e-9 * max(1.0, want_bw))

    # -- 6. closure（P_N + η_closure）------------------------------------------
    forcing = np.zeros(n)
    forcing[33] = 1.0                    # k=33 > N：全额进 defect（§5.2）
    cl32 = project_spectral(desc, 32, eps=None, forcing=forcing)
    add("closure N=32: k>N forcing gives full defect, projected dim",
        "eta=%.3e dim=%d" % (cl32.eta_closure, cl32.projected.n),
        "eta>0, projected n=32",
        cl32.eta_closure > 0 and cl32.projected.n == 32)
    cl32j = project_spectral(desc, 32, eps=10.0 * cl32.eta_closure,
                             forcing=forcing)
    add("closure N=32 verdict with eps=10η",
        cl32j.verdict, "feasible", cl32j.verdict == "feasible")
    cl64 = project_spectral(desc, 64, forcing=forcing)
    add("closure N=n: exact closure (no reduction available)",
        "eta=%.3e" % cl64.eta_closure, "eta == 0 at full N", cl64.eta_closure == 0.0)

    # -- 发现 F2：projected 改 max|λ| ⇒ alpha 口径分裂 -------------------------
    lam_proj_max = float(np.max(np.abs(cl32.projected.finite_eigs())))
    lam_raw_max = float(np.max(np.abs(lam_raw)))
    FINDINGS.append(dict(
        id="F2", workload=name,
        text="closure.projected (N=32) changes max|lambda| (%.4g -> %.4g); "
             "alpha/omega_max were derived on the FULL descriptor. Carrying "
             "the projected descriptor downstream requires RE-DERIVING "
             "scaling on it, else route_contract/p7 omega_max and P3 alpha "
             "split into two conventions. Chain carries the full descriptor "
             "+ closure certificate as evidence." % (lam_raw_max, lam_proj_max),
        numbers=dict(lam_max_full=lam_raw_max, lam_max_projected=lam_proj_max),
    ))
    add("finding F2 recorded: projected N changes max|lambda| vs alpha source",
        "lam_max %.4g → %.4g (projected)" % (lam_raw_max, lam_proj_max),
        "recorded; chain carries full descriptor", True,
        kind="finding")

    # -- 7. split（快慢分类）---------------------------------------------------
    sp = cvsplit(desc)
    add("split: diagonal diffusion has NO clean fast/slow gap (TRUE result)",
        "verdict=%s flags=%s gap=%.2f" % (sp.verdict, ",".join(sp.flags),
                                          sp.gap_ratio or float("nan")),
        "no-split with no-gap flag (max k²-ratio 4 < GAP_MIN)",
        sp.verdict == "no-split"
        and any(f.startswith("no-gap-below") for f in sp.flags)
        and (sp.gap_ratio or 0.0) < 10.0)

    # -- 8. basis（Γ·amplification 选基）---------------------------------------
    br = select_basis(desc)
    brd = br.as_dict()
    dev_b = _eig_cmp(br.transformed.finite_eigs(), lam_raw,
                     np.max(np.abs(lam_raw)))
    add("basis: winner exists + similarity preserves eig (up to cond)",
        "winner=%s cond=%.3g eig_dev=%.2e" % (
            br.winner,
            br.candidates[br.winner].cond_phi if br.winner else float("nan"),
            dev_b),
        "winner not None; eig preserved ~1e-8", br.winner and dev_b < 1e-8)
    emit("    basis detail: %s" % brd["reason"])

    # -- 9. interface（capture/fit + 单源合约；发现 F3）------------------------
    x = d.grid[0]
    u_smooth = np.sin(2.0 * np.pi * x)
    ir_s = choose_interface(u_smooth, dx=1.0 / n)
    add("interface: smooth field → capture free + single-source contract",
        "mode=%s singular=%s single_source=%s" % (
            ir_s.mode, ir_s.singular, ir_s.contract.single_source),
        "capture, not singular, single_source=True",
        ir_s.mode == "capture" and not ir_s.singular
        and ir_s.contract.single_source)
    u_step = np.where(x < 0.5, 1.0, -1.0)
    ir_sh = choose_interface(u_step, dx=1.0 / n)
    leak = ir_sh.contract.check(1.0, 1.0, 1.0, 1.0)
    add("interface: step field → singular + single-source leak = 0",
        "mode=%s singular=%s leak=%.1e" % (ir_sh.mode, ir_sh.singular, leak),
        "singular, mode∈{capture,fit}, leak==0",
        ir_sh.singular and ir_sh.mode in ("capture", "fit") and leak == 0.0)
    FINDINGS.append(dict(
        id="F3", workload=name,
        text="interface consumes a physical field on the periodic grid, not "
             "the descriptor/modal coordinates; modal<->field plumbing (the "
             "representation transform T) is compiler work not yet wired. "
             "Minimal real input used: the anchor grid fields (smooth/step).",
    ))
    add("finding F3 recorded: interface needs field input, modal↔field "
        "plumbing unwired",
        "recorded (smooth+step fields used)", "recorded", True,
        kind="finding")

    # -- 10. route contracts（对角 ⇒ 空表 TRUE 结果）---------------------------
    rep = emit_route_contracts(d, scaling=sc)
    add("route contracts: diagonal A → zero coupling edges (TRUE result)",
        "n_edges=%d n_feedback=%d" % (rep.n_edges, rep.n_feedback),
        "0 edges (A off-diagonal nnz = 0)", rep.n_edges == 0)
    omega_want = 2.0 * np.pi * float(sc.scaled.bandwidth())
    add("route report omega_max = 2π·scaled.bandwidth() (physical-time hand-off)",
        "%.6e vs %.6e" % (rep.omega_max, omega_want),
        "equal (max|λ|/alpha = scaled bandwidth×2π)",
        abs(rep.omega_max - omega_want) < 1e-6 * omega_want)

    # -- 11. P7 generate_config -------------------------------------------------
    cfg = generate_config(d, cert, sc)
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

    # -- 12. 负对照：κ_hw=1e3 ---------------------------------------------------
    spec_tight = HardwareSpec(kappa_hw=1e3, B_hw=B_hw, sigma_req=None,
                              delta_A_norm=1e-3, tol_l2=1e-2)
    cert_t = evaluate_sys(desc, spec_tight, scaling=sc)
    add("negative control: κ_hw=1e3 → infeasible, binding_axis=stiffness",
        "status=%s axis=%s" % (cert_t.status, cert_t.binding_axis),
        "infeasible, stiffness (κ=4096>1e3)",
        (not cert_t.feasible) and cert_t.binding_axis == "stiffness")
    return dict(alpha=sc.alpha, binding_axis=cert.binding_axis,
                status=cert.status, n_closure=32, eta32=cl32.eta_closure,
                split=sp.verdict, basis=br.winner, n_edges=rep.n_edges)


# ===========================================================================
# 工作负载 2：synthetic-stiff（快慢 + 非对角耦合）
# ===========================================================================
def case_synthetic_stiff(checks):
    name = "synthetic-stiff"

    def add(cname, got, expected, passed, kind="compose"):
        checks.append(dict(problem=name, name="%s: %s" % (name, cname),
                           got=got, expected=expected,
                           passed=bool(passed), kind=kind))

    from lowering.scaling import gramian_scaling
    from lowering.budget import evaluate_sys
    from lowering.closure import project_spectral
    from lowering.split import cvsplit
    from lowering.basis import select_basis
    from lowering.interface import choose_interface
    from lowering.route_contract import emit_route_contracts
    from lowering.p7 import generate_config
    from analysis import HardwareSpec, Descriptor
    from analysis.skew import phase_budget

    # -- 0. 合成 descriptor（如实标注 synthetic）-------------------------------
    # 慢块 {-1,-2}（上三角 → 坐标归属干净）、快块 {-1000,-3000}、
    # 块间小耦合（A_sf/A_fs 对角注入）——非对角耦合图非空，谱间隙 ~500。
    A = np.array([
        [-1.0, 0.05, 0.01, 0.0],
        [0.0, -2.0, 0.0, 0.02],
        [0.01, 0.0, -1000.0, 5.0],
        [0.0, 0.005, 3.0, -3000.0],
    ])
    desc = Descriptor(np.eye(4), A, np.eye(4)[:, :1], np.eye(4))
    lam_raw = np.sort_complex(desc.finite_eigs())
    kappa = float(desc.kappa())
    add("synthetic descriptor: 4 finite eigs, real gap present, κ recorded",
        "n_finite=%d κ=%.4g" % (lam_raw.size, kappa),
        "4 finite modes, κ≈3000 (fastest 3000 / slowest 1)",
        lam_raw.size == 4 and 2500.0 < kappa < 3500.0)

    # -- 1. P3 scaling ---------------------------------------------------------
    B_hw = 1e6
    sc = gramian_scaling(desc, v_swing=1.0, B_hw=B_hw)
    lam_scaled = np.sort_complex(sc.scaled.finite_eigs())
    dev = _eig_cmp(lam_scaled, lam_raw / sc.alpha, np.max(np.abs(lam_raw)))
    add("P3: scaled eig = raw eig / alpha",
        "max|Δ|/scale=%.2e (alpha=%.3e)" % (dev, sc.alpha),
        "~1e-9 (α = max|λ|/(2π·B_hw·margin))", dev < 1e-9)

    # -- 2. P6 证书：stiffness 绑定 -------------------------------------------
    kappa_hw = 1e3
    spec_b = HardwareSpec(kappa_hw=kappa_hw, B_hw=B_hw, sigma_req=None,
                          delta_A_norm=1e-3, tol_l2=1e-2)
    cert = evaluate_sys(desc, spec_b, scaling=sc)
    add("P6 cert: infeasible, binding_axis=stiffness (κ > κ_hw)",
        "status=%s axis=%s" % (cert.status, cert.binding_axis),
        "infeasible, binding_axis=stiffness",
        (not cert.feasible) and cert.binding_axis == "stiffness")
    st_axis = float(cert.slack["values"]["stiffness"])
    want_st = kappa / kappa_hw
    add("P6↔descriptor consistency: stiffness axis = κ/κ_hw (scaling-invariant)",
        "%.6g vs %.6g" % (st_axis, want_st),
        "equal (S/α similarity preserves κ)",
        abs(st_axis - want_st) < 1e-6 * want_st)

    # -- 3. closure（P_N 线性路径；坐标 3 是快坐标 forcing）--------------------
    forcing = np.zeros(4)
    forcing[3] = 1.0
    cl3 = project_spectral(desc, 3, eps=10.0, forcing=forcing)
    add("closure N=3: fast-coordinate forcing gives defect, verdict feasible",
        "eta=%.3e verdict=%s dim=%d" % (cl3.eta_closure, cl3.verdict,
                                        cl3.projected.n),
        "eta>0, feasible at eps=10, projected n=3",
        cl3.eta_closure > 0 and cl3.verdict == "feasible" and cl3.projected.n == 3)

    # -- 4. split：stiffness 绑定 ⇔ 真快慢间隙（一致性断言 (b)）----------------
    sp = cvsplit(desc)
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

    # -- 5. basis（结构化 Δ=ε·A；赢家 + 变换后 descriptor）---------------------
    br = select_basis(desc, delta_pert=1e-3)
    lam_trans = np.sort_complex(br.transformed.finite_eigs())
    dev_b = _eig_cmp(lam_trans, lam_raw, np.max(np.abs(lam_raw)))
    cond_w = br.candidates[br.winner].cond_phi if br.winner else float("nan")
    tol = 1e-6 * max(1.0, cond_w)
    add("basis: winner + transformed eig preserved (within cond tolerance)",
        "winner=%s cond=%.3g eig_dev/cond=%.2e (tol %.1e)" % (
            br.winner, cond_w, dev_b, tol),
        "winner not None; dev·1 ≤ cond-scaled tol", br.winner and dev_b <= tol)
    emit("    basis detail: %s" % br.as_dict()["reason"])

    # -- 6. interface（发现 F3 同款：给场输入）---------------------------------
    xs = np.linspace(0.0, 1.0, 64, endpoint=False)
    ir = choose_interface(np.sin(2.0 * np.pi * xs), dx=1.0 / 64)
    leak = ir.contract.check(2.0, 2.0, 3.0, 3.0)
    add("interface: smooth synthetic state → capture free, single-source leak 0",
        "mode=%s singular=%s leak=%.1e" % (ir.mode, ir.singular, leak),
        "capture, not singular, leak==0",
        ir.mode == "capture" and not ir.singular and leak == 0.0)

    # -- 7. route contracts on the FINAL transformed descriptor ----------------
    # 一致性断言 (c)：generate_config 看到的是 **basis 赢家的 transformed**
    # descriptor（合成负载走 basis 后的最终形态）；契约条数 = 它的非对角
    # 非零数，不是 raw descriptor 的（本例两者同为 7，但口径按 transformed
    # 断言——若赢家换基改变稀疏性，这里会立刻暴露）。
    trans = br.transformed
    nnz_trans = int(np.count_nonzero(trans.A - np.diag(np.diag(trans.A))))
    nnz_raw = int(np.count_nonzero(A - np.diag(np.diag(A))))
    states_b = ["x_slow0", "x_slow1", "x_fast0", "x_fast1"]

    class _Wrapped:                                # noqa: N801 - 最小 Discretized 形状
        descriptor = trans
        states = states_b
        meta: dict = {}
        coupling_graph = None

    rep = emit_route_contracts(_Wrapped, scaling=sc)
    add("route contracts on FINAL (basis-transformed) descriptor: "
        "count = off-diagonal nnz",
        "n_edges=%d (transformed nnz=%d, raw nnz=%d)" % (
            rep.n_edges, nnz_trans, nnz_raw),
        "n_edges == nnz(transformed.A off-diagonal)",
        rep.n_edges == nnz_trans)
    add("route contracts: feedback edges get omega_c + finite tau_max_abs",
        "n_feedback=%d" % rep.n_feedback,
        "≥1 feedback edge; every feedback contract has "
        "omega_c==omega_max and tau_max_abs=phase_budget(ωc,5°)",
        rep.n_feedback >= 1
        and all(c.omega_c is not None
                and abs(c.omega_c - rep.omega_max) < 1e-9 * rep.omega_max
                and np.isfinite(c.tau_max_abs)
                and abs(c.tau_max_abs
                        - phase_budget(rep.omega_max,
                                       c.phi_reserve_deg)) < 1e-15
                for c in rep.contracts if c.feedback_critical))

    # -- 8. 物理时间交接 (d)：omega_max 用 SCALED 带宽 -------------------------
    omega_want = 2.0 * np.pi * float(sc.scaled.bandwidth())
    add("(d) contract omega_max = max|λ|/alpha = 2π·scaled.bandwidth()",
        "%.6e vs %.6e" % (rep.omega_max, omega_want),
        "equal (SCALED bandwidth, not model-time eig)",
        abs(rep.omega_max - omega_want) < 1e-6 * omega_want)
    add("(d) contract omega_max ≠ model-time bandwidth (hand-off actually applied)",
        "omega_max/λ_max = %.3e (1/alpha)" % (rep.omega_max
                                              / float(np.max(np.abs(lam_raw)))),
        "= 1/alpha, NOT 1 (alpha>0 scaling genuinely divides)",
        abs(rep.omega_max / float(np.max(np.abs(lam_raw))) - 1.0 / sc.alpha)
        < 1e-9 * (1.0 / sc.alpha))

    # -- 9. P7 generate_config on the transformed descriptor -------------------
    cfg = generate_config(_Wrapped, cert, sc)
    n_contracts = len(cfg.route_contracts)
    gm0 = cfg.elements[0].target_gm_over_c
    add("P7 Config: carries elements AND route_contracts; count consistent "
        "with the descriptor generate_config sees (transformed)",
        "elements=%d contracts=%d" % (len(cfg.elements), n_contracts),
        "contracts == %d (transformed nnz) and == emit count" % nnz_trans,
        n_contracts == nnz_trans == rep.n_edges)
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
    add("P7 Config: fastest per-element a_ii ≈ tile slew spec (coupling shift "
        "honestly visible)",
        "max gm/C=%.6e vs omega_max=%.6e (Δrel %.2e)" % (
            gm_max_b := max(e.target_gm_over_c for e in integrators),
            rep.omega_max,
            abs(gm_max_b - rep.omega_max) / rep.omega_max),
        "equal within 1e-5 (a_ii is the DIAGONAL rate; the eigenvalue differs "
        "by the off-diagonal coupling shift)",
        abs(gm_max_b - rep.omega_max) < 1e-5 * rep.omega_max)
    # 系数图口径对账：p7 的 coeff graph 消费 scaling.scaled（raw 坐标 + P3
    # 预条件，显式 a_tol 阈值），契约表来自传入 descriptor（这里 = basis
    # 赢家 transformed）。两者坐标系统不同 ⇒ 记录为组合缺口 F5，并断言
    # 各自口径内部一致。
    A_v = np.asarray(sc.scaled.A, dtype=float)
    a_scale_v = float(np.max(np.abs(A_v)))
    a_tol_v = 1e-12 * a_scale_v if a_scale_v > 0 else 0.0
    nnz_v = int(np.count_nonzero(np.abs(A_v - np.diag(np.diag(A_v))) > a_tol_v))
    src = cfg.source
    FINDINGS.append(dict(
        id="F5", workload=name,
        text="basis selection is not composed into scaling: p7's coefficient "
             "graph consumes scaling.scaled (RAW coordinates + P3 "
             "preconditioning, a_tol-thresholded), while route contracts / "
             "generate_config's coupling graph consume the descriptor passed "
             "in (here the basis winner's transformed A). When winner != "
             "identity these are DIFFERENT coordinate systems (contracts on "
             "12 transformed edges vs %d coeff edges on the raw graph); the "
             "compiler must compose T into the scaled descriptor before "
             "emitting coeff graph + contracts." % nnz_v,
        numbers=dict(n_coeff_edges=src.get("n_coeff_edges"),
                     n_contracts=rep.n_edges),
    ))
    add("P7 Config: coeff edges = scaled off-diag nnz above p7's own a_tol "
        "(Finding F5: raw vs transformed coordinate gap recorded)",
        "n_coeff_edges=%d, scaled nnz(>a_tol)=%d, contracts=%d" % (
            src.get("n_coeff_edges", -1), nnz_v, len(cfg.route_contracts)),
        "n_coeff_edges == a_tol-thresholded scaled nnz (internal consistency; "
        "cross-coordinate equality is F5, not asserted)",
        src.get("n_coeff_edges") == nnz_v)
    return dict(alpha=sc.alpha, binding_axis=cert.binding_axis,
                status=cert.status, split=sp.verdict, gap=sp.gap_ratio,
                basis=br.winner, n_edges=rep.n_edges,
                n_feedback=rep.n_feedback, kappa=kappa)


# ===========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    FINDINGS.clear()

    hr("B1 closeout：编译 pass 链 COMPOSE 端到端（spec→…→generate_config）")

    checks: list[dict] = []
    rec: dict[str, dict] = {}

    hr("1. diffusion-1d（真锚点，对角谱）")
    rec["diffusion-1d"] = case_diffusion(checks)

    hr("2. synthetic-stiff（快慢间隙 + 非对角耦合）")
    rec["synthetic-stiff"] = case_synthetic_stiff(checks)

    hr("3. 链一致性汇总")
    rows = [(k, v["status"], str(v["binding_axis"]), str(v["split"]),
             str(v["basis"]), str(v["n_edges"]))
            for k, v in rec.items()]
    table(["workload", "cert status", "binding axis", "split", "basis",
           "route edges"], rows)
    emit("\n  findings:")
    for f in FINDINGS:
        emit("    F%s [%s] %s" % (f["id"], f["workload"],
                                  f["text"][:100] + "…"))

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
                                n_passed=len(checks) - n_fail))
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
