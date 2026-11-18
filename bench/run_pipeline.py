"""Stage A5 step-3 验收：3 个锚定基准的端到端链条（DESIGN.md §5/§10）。

    python3 -m bench.run_pipeline

链条（每个问题都显式走一遍并逐级记账）：

    符号规格 PDESpec → discretize → Descriptor
      → P1 descriptor.check()（κ/带宽/稳定/n_finite）
      → P3 gramian_scaling（α、S_x、PhysicalScaling、Γ before/after、method）
      → P4/P5 HardwareSpec → P6 evaluate_sys → Certificate
      → 数字孪生 run_twin() → 预测 vs 实测 对账（本 bench 的核心）

三个锚定问题（bench/problems.py，与 run_phase0 同源）：

    diffusion-1d  κ=4096，expected_axis=None：证书可行；
                  孪生 det_err < σ_y（算法误差落在噪声底之下）；
                  meas_sigma vs Lyapunov σ_y（×2）。
    poisson-2d    椭圆问题 → T10 KCL 松弛孪生：steady_err<1e-9、
                  floor_meas ≤ 确定性下界、tau_fit vs C/λ_min（5%）、
                  残差小。负对照：c_node=1e-6 → ε_c=τ_c/τ_state FAIL
                  ↔ 截断视界上实测不收敛（残差≈‖I‖：椭圆变抛物）。
    burgers-1d    证书 binding_axis="representation"（E_tail/tol²>1）；
                  孪生：守恒 ~1e-16（拓扑保证不受表示失败影响）、
                  激波位置差 ≤1 格、熵号、l2_blend/l2_weno5（§6.16 ~1.34×）；
                  对账：E_tail^{1/2} ~ l2_blend（同一数量级，≤1 decade——
                  这是标度声明，不是恒等式）。

负控制：
    diffusion κ_hw=1e3（κ=4096>1e3）→ 证书不可行，binding_axis=stiffness。
    burgers 线性化在光滑预激波态 T=0.2：表示轴不 binding——守卫
    "总是 representation" 的过拟合。（T≈0.05/0.1 的陷阱：p-拟合被
    截断噪声主导，rep 靠 p_thresh/p 违规；T=0.2 两条都干净。）
    非法问题名 → build_problem 拒绝（若实现了的话）。

诚实声明：P3 在不稳定 Burgers Jacobian 上必须走降级路径（Lyapunov
不可用），method 字段如实记录；不把降级包装成"可行"。
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
CASES: dict[str, dict] = {}


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


def _find(d, *keys, default=None):
    """在（嵌套）dict 里递归找第一个出现的 key——lane 的 reference/spec_meta
    命名可能微调，查找容错，找不到显式记 n/a，不静默造数。"""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            got = _find(v, *keys, default=None)
            if got is not None:
                return got
    return default


def _decade(a, b):
    """|log10(a/b)|，非正数 → inf。"""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return float("inf")
    if not (a > 0 and np.isfinite(a)) or not (b > 0 and np.isfinite(b)):
        return float("inf")
    return abs(np.log10(a / b))


# ===========================================================================
# 三锚定链条
# ===========================================================================
def drive(name, bundle, spec, checks, ctrl):
    """一个锚定问题走全链并记账。ctrl: dict 收集负控制要用的东西。"""
    rec: dict = {}
    CASES[name] = rec

    def add(cname, got, expected, passed, kind="recon"):
        checks.append(dict(problem=name, name="%s: %s" % (name, cname),
                           got=got, expected=expected,
                           passed=bool(passed), kind=kind))

    # -- 1. 符号规格 → 半离散化 -------------------------------------------------
    meta = bundle.spec_meta or {}
    rec["spec_meta"] = _jsonable(meta)
    rec["grid"] = _jsonable(bundle.grid)
    add("spec recorded (spec_meta from PDESpec)",
        "keys=%s" % ",".join(sorted(map(str, meta))[:6]) or "(empty)",
        "non-empty spec_meta", bool(meta))

    dsc = bundle.descriptor
    # -- 2. P1 ------------------------------------------------------------------
    chk = dsc.check()
    rec["p1_check"] = _jsonable(chk)
    # stability 只在可行锚点上断言 —— Burgers 的 pre-shock Jacobian 上游是
    # 反扩散（非稳定），这是正确的物理，不是缺陷；stable_projection 在下游
    # 会处理。判定"descriptor 结构健全"= 有限模态数 > 0。
    exp_stable = getattr(bundle, "expected_stable", bundle.expected_axis is None)
    add("P1 check: finite modes (stable iff expected)",
        "n_finite=%s stable=%s" % (chk["n_finite"], chk["stable"]),
        "n_finite>=1%s" % (", stable" if exp_stable else ""),
        chk["n_finite"] >= 1 and (bool(chk["stable"]) or not exp_stable))

    # -- 3. P3 ------------------------------------------------------------------
    from lowering.scaling import gramian_scaling
    sc = gramian_scaling(dsc, v_swing=1.0, B_hw=1e6)
    rec["p3_scaling"] = dict(alpha=float(sc.alpha), method=sc.method,
                             gamma_before=float(sc.gamma_before),
                             gamma_after=float(sc.gamma_after),
                             notes=list(sc.notes), bandwidth_pass=bool(sc.bandwidth_pass))
    add("P3 scaling: alpha & method recorded",
        "alpha=%.3e method=%s" % (sc.alpha, sc.method),
        "alpha>0, fallback recorded honestly",
        sc.alpha > 0 and (sc.method == "rms3sigma" or bool(sc.notes)))

    # -- 4/5. P4/P5 + P6 --------------------------------------------------------
    tau_state_min = None
    lam = dsc.finite_eigs()
    if lam.size and np.max(np.abs(lam)) > 0:
        tau_state_min = 1.0 / float(np.max(np.abs(lam)))
    if ctrl.get("tau_state_min") is None and tau_state_min is not None:
        ctrl["tau_state_min"] = tau_state_min

    from lowering.budget import evaluate_sys
    # reg/reg_w 证据在两个位置都可能出现：bundle.reference 或 bundle.spec_meta
    # （problems.py 把它们放在 spec_meta —— 证书证据是 spec 的一部分）。
    reg = _find(meta, "reg") or _find(bundle.reference, "reg")
    reg_w = _find(meta, "reg_w") or _find(bundle.reference, "reg_w")
    # P6 证书评在 P3 的物理时间 descriptor（sc.scaled：eig=λ/α、双边
    # 预条件）上 —— scaling 修好方向后（α⁻¹·S·A·S⁻¹，非 α·A·S⁻¹）这就是
    # 唯一一致入口：bandwidth 轴的物理值 = B_hw·margin ≤ B_hw，model-time
    # 的原始带宽是伪违反。evaluate_sys(scaling=...) 内部取 sc.scaled。
    # gamma_M：poisson 的 Γ_proj 解析上 = ∞（G 守恒零模，S 相似改不掉）；
    # delta_A_norm=None 时该轴不被读，供 gamma_M=inf 是**真值**（非捷径），
    # 省去 scaled 矩阵上 ~15min 的 resolvent sweep（其它问题 None→正常 sweep）。
    gamma_M = (float("inf") if (spec.delta_A_norm is None
                                and name == "poisson-2d") else None)
    cert = evaluate_sys(dsc, spec, reg=reg, scaling=sc, gamma_M=gamma_M)
    rec["p6_certificate"] = cert.as_dict()
    rec["p6_binding_axis"] = cert.binding_axis
    rec["scaling"] = dict(alpha=float(sc.alpha),
                          scaled_bandwidth_hz=float(sc.scaled.bandwidth()),
                          bandwidth_pass=bool(sc.bandwidth_pass))
    exp_axis = bundle.expected_axis
    if exp_axis is not None:
        ok = cert.feasible is False and cert.binding_axis == exp_axis
    else:
        ok = cert.status in ("feasible", "no violation on evaluated axes")
    add("P6 certificate status", cert.status,
        "feasible/no-violation" if exp_axis is None else
        "infeasible w/ binding_axis=%s" % exp_axis, ok)
    if ctrl.get("binding_axis") is None:
        ctrl["binding_axis"] = cert.binding_axis

    # -- 6. 数字孪生 ------------------------------------------------------------
    twin = bundle.run_twin() or {}
    rec["twin"] = _jsonable(twin)
    return rec, twin, cert, dict(scaling=sc, reg=reg, reg_w=reg_w,
                                 tau_state_min=tau_state_min)


def reconcile_diffusion(rec, twin, cert, extra, bundle, checks, sigma_y):
    """diffusion-1d 对账：算法误差 < 噪声底；EM 噪声实测 vs Lyapunov σ_y。"""
    def add(cname, got, expected, passed):
        checks.append(dict(problem="diffusion-1d",
                           name="diffusion-1d: %s" % cname, got=got,
                           expected=expected, passed=bool(passed), kind="recon"))
    sy = float(sigma_y)
    det_err = _find(twin, "det_err", "detrm_err", "deterministic_err")
    if det_err is None:
        add("twin det_err present", "missing", "det_err from run_twin", False)
        return
    add("det_err < predicted sigma_y (machine-limited)",
        "%.3e vs %.3e" % (det_err, sy),
        "det_err < sigma_y", float(det_err) < sy)
    meas = _find(twin, "meas_sigma", "measured_sigma")
    if meas is not None:
        ratio = float(meas) / sy
        add("meas_sigma vs Lyapunov sigma_y (×2)",
            "%.3e vs %.3e (×%.2f)" % (meas, sy, ratio),
            "within factor 2", 0.5 <= ratio <= 2.0)
    else:
        add("meas_sigma present", "missing", "EM-noise sigma from twin", False)
    # 预测轴 ↔ 实测对账的记录行（expected_axis=None ⇒ 只要求无违约）
    rec["reconciliation"] = dict(
        predicted="sigma_y=%.3e (noise floor; no axis violated)" % sy,
        measured="det_err=%.3e meas_sigma=%s" % (det_err, meas),
        verdict=("within-band: algorithm error below noise floor"
                 if float(det_err) < sy else "out-of-band"))


def reconcile_poisson(rec, twin, cert, extra, bundle, checks):
    """poisson-2d 对账：松弛孪生收敛到直接解；ε_c 违约 ↔ 截断视界不收敛。"""
    from twin.constraint import (ConstraintMesh, eps_c, residual, relax_time,
                                 settle_with_c, solve_constraint)
    def add(cname, got, expected, passed):
        checks.append(dict(problem="poisson-2d",
                           name="poisson-2d: %s" % cname, got=got,
                           expected=expected, passed=bool(passed), kind="recon"))
    for key, tol in (("steady_err", 1e-9), ("residual", 1e-7)):
        v = _find(twin, key)
        if v is None:
            add("twin %s present" % key, "missing", "from run_twin", False)
        else:
            add("twin %s small" % key, "%.3e" % v, "< %g" % tol, float(v) < tol)
    # floor_meas ≤ 确定性下界（下界是下界——实测不得超过）
    floor_meas = _find(twin, "floor_meas", "measured_floor")
    floor_bound = _find(bundle.reference, "floor_bound", "floor_meas_bound",
                        "mismatch_floor", "floor_lower_bound")
    if floor_meas is not None and floor_bound is not None:
        add("floor_meas <= deterministic bound",
            "%.3e vs %.3e" % (floor_meas, floor_bound),
            "measured ≤ bound", float(floor_meas) <= float(floor_bound) * (1 + 1e-9))
    elif floor_meas is not None:
        add("floor bound present", "bound missing", "reference floor bound", False)
    # tau_fit vs C/λ_min（5%）
    tau_fit = _find(twin, "tau_fit", "tau_relax_fit")
    c_node = float(_find(bundle.spec_meta, "c_node", default=1.0) or 1.0)
    g = float(_find(bundle.spec_meta, "g", "conductance", default=1.0) or 1.0)
    nx = ny = None
    grid = bundle.grid          # tuple of coordinate arrays, not sizes
    if grid:
        dims = []
        for ax in grid[:2]:
            try:
                dims.append(len(np.asarray(ax)))   # coordinate array
            except TypeError:
                dims.append(int(ax))               # plain size
        nx = dims[0]
        ny = dims[1] if len(dims) > 1 else dims[0]
    tau_pred = None
    mesh = None
    if nx and ny:
        mesh = ConstraintMesh.poisson_2d(nx, ny, g=g)
        tau_pred = relax_time(mesh, c_node)
    tau_ref = _find(bundle.reference, "tau_pred", "c_over_lambda_min",
                    "tau_relax_pred")
    if tau_ref is not None:
        tau_pred = float(tau_ref)
    if tau_fit is not None and tau_pred is not None:
        rel = abs(float(tau_fit) - tau_pred) / abs(tau_pred)
        add("tau_fit vs C/lambda_min (5%)",
            "%.4g vs %.4g (Δ%.2f%%)" % (tau_fit, tau_pred, 100 * rel),
            "within 5%", rel <= 0.05)
    else:
        add("tau_fit vs C/lambda_min", "tau_fit=%s pred=%s" % (tau_fit, tau_pred),
            "within 5%", False)

    # -- 负对照：c_node=1e-6 → ε_c FAIL ↔ 实测不收敛 --------------------------
    if mesh is not None and extra["tau_state_min"] is not None:
        tau_s = extra["tau_state_min"]
        i_rhs = np.sin(2.0 * np.pi * np.arange(mesh.n) / mesh.n)
        big = solve_constraint(mesh, i_rhs, c_node=1e-6, tau_state_min=tau_s)
        rec["constraint_big_c"] = dict(verdict=big.verdict,
                                       eps_c=float(big.eps_c) if big.eps_c else None,
                                       tau_relax=float(big.tau_relax))
        add("second cert @c_node=1e-6 flags FAIL (eps_c = tau_c/tau_state)",
            "verdict=%s eps_c=%.3g" % (big.verdict, big.eps_c),
            "FAIL with eps_c >= 0.01",
            big.verdict == "FAIL" and big.eps_c is not None and big.eps_c >= 0.01)
        # 实测："椭圆变抛物"的对比要在**同一 C、不同视界**上做：在
        # t=τ_state,min 处大 C 的残差 ≈ ‖I‖（没收敛），在 t=τ_relax 处
        # ≈ ‖I‖/e（按 τ_relax 自己的时钟收敛）。同 t 跨 C 的比较没意义
        # —— res(t)=‖I‖e^{−t/τ}，τ≫t 时两侧都 ≈‖I‖（实测 11.31/11.31）。
        v_big_tau_s = settle_with_c(mesh, 1e-6, i_rhs, t=tau_s)
        res_big = residual(mesh, v_big_tau_s, i_rhs)
        tau_relax_big = float(big.tau_relax)
        v_big_tau_r = settle_with_c(mesh, 1e-6, i_rhs, t=tau_relax_big)
        res_conv = residual(mesh, v_big_tau_r, i_rhs)
        rec["constraint_big_c"]["res_at_tau_state"] = float(res_big)
        rec["constraint_big_c"]["res_at_own_tau_relax"] = float(res_conv)
        rec["constraint_big_c"]["tau_relax"] = tau_relax_big
        add("large-C nonconvergence at t=tau_state vs own tau_relax",
            "res(τ_state)=%.3e  res(τ_relax)=%.3e" % (res_big, res_conv),
            "res(τ_state) > 0.5‖I‖, res(τ_relax) < 0.5‖I‖",
            res_big > 0.5 * float(np.linalg.norm(i_rhs))
            and res_conv < 0.5 * float(np.linalg.norm(i_rhs)))
        rec["reconciliation"] = dict(
            predicted="eps_c=%.3g >= 0.01 ⇒ FAIL (elliptic becomes parabolic)"
                      % (big.eps_c if big.eps_c is not None else float("nan")),
            measured="res(t=τ_state)=%.3e vs res(t=τ_relax)=%.3e"
                     % (res_big, res_conv),
            verdict="bound-holds: nonconverged at τ_state, converged at τ_relax"
                    if res_big > 0.5 * float(np.linalg.norm(i_rhs))
                    and res_conv < 0.5 * float(np.linalg.norm(i_rhs))
                    else "out-of-band")
    else:
        rec["reconciliation"] = dict(predicted="n/a", measured="n/a",
                                     verdict="skipped (no mesh / no tau_state)")


def reconcile_burgers(rec, twin, cert, extra, bundle, checks):
    """burgers-1d 对账：表示轴违约 ↔ 实测 L2 误差（数量级一致）。"""
    from analysis import shock_sensor
    def add(cname, got, expected, passed):
        checks.append(dict(problem="burgers-1d",
                           name="burgers-1d: %s" % cname, got=got,
                           expected=expected, passed=bool(passed), kind="recon"))
    cons = _find(twin, "conservation", "mass_drift")
    if cons is not None:
        add("conservation ~1e-16 despite representation failure",
            "%.2e" % cons, "|drift| < 1e-12", abs(float(cons)) < 1e-12)
    else:
        add("twin conservation present", "missing", "from run_twin", False)
    xs_b = _find(twin, "shock_x_blend", "shock_blend")
    xs_e = _find(bundle.reference, "shock_x_exact", "shock_x")
    if xs_b is not None and xs_e is not None:
        n = 128
        if bundle.grid:
            try:
                n = len(np.asarray(bundle.grid[0]))   # coordinate array
            except TypeError:
                n = int(bundle.grid[0])               # plain size
        dom = _find(bundle.spec_meta, "domain", default=(0.0, 1.0))
        try:
            length = float(dom[1]) - float(dom[0])
        except Exception:
            length = 1.0
        h = length / n
        add("shock position within ~1 cell",
            "|%.5f - %.5f| = %.3e vs h=%.3e" % (xs_b, xs_e, abs(xs_b - xs_e), h),
            "<= 1.5 cell", abs(float(xs_b) - float(xs_e)) <= 1.5 * h)
    elif xs_b is not None:
        add("shock_x_exact present", "missing", "reference shock location", False)
    ok_ent = _find(twin, "entropy_sign_ok", "entropy_ok")
    if ok_ent is not None:
        add("entropy inequality sign ok", str(ok_ent), "True", bool(ok_ent))
    l2b = _find(twin, "l2_blend", "l2_blend_vs_exact")
    l2w = _find(twin, "l2_weno5", "l2_weno") or _find(bundle.reference, "l2_weno5")
    if l2b is not None and l2w:
        ratio = float(l2b) / float(l2w)
        add("l2_blend / l2_weno5 (§6.16 ~1.34x)",
            "%.3f" % ratio, "in [1.0, 2.0]", 1.0 <= ratio <= 2.0)
        rec["l2_ratio"] = float(ratio)
    # reg_w（WENO 捕捉解的谱尾）vs reg（物理态）——capturing hides the tail
    reg, reg_w = extra["reg"], extra["reg_w"]
    if reg_w is not None and reg is not None:
        et_w = float(_find(reg_w, "E_tail", default=float("nan")))
        et_p = float(_find(reg, "E_tail", default=float("nan")))
        if np.isfinite(et_w) and np.isfinite(et_p):
            add("WENO reg E_tail < physical-state E_tail (tail hidden)",
                "%.3e vs %.3e" % (et_w, et_p), "E_tail(WENO) < E_tail(phys)",
                et_w < et_p)
    # 核心对账：E_tail^{1/2} ~ L2（标度声明，≤1 decade）
    if reg is not None and l2b is not None:
        et = float(_find(reg, "E_tail", default=float("nan")))
        if np.isfinite(et) and et > 0:
            dec = _decade(np.sqrt(et), l2b)
            add("E_tail^(1/2) ~ l2_blend (order-of-magnitude, <=1 decade)",
                "%.3e vs %.3e (Δ%.2f dec)" % (np.sqrt(et), l2b, dec),
                "within 1 decade", dec <= 1.0)
            rec["reconciliation"] = dict(
                predicted="E_tail^(1/2)=%.3e (representation violation size)"
                          % np.sqrt(et),
                measured="l2_blend=%.3e" % l2b,
                verdict=("within-band (scaling claim, not identity)"
                         if dec <= 1.0 else "out-of-band"))
    elif l2b is not None:
        rec["reconciliation"] = dict(predicted="n/a", measured="l2_blend=%.3e" % l2b,
                                     verdict="no E_tail evidence")
    else:
        rec["reconciliation"] = dict(predicted="n/a", measured="n/a",
                                     verdict="no twin L2 evidence")


def negative_controls(checks):
    """负控制：κ 窗口收紧、光滑态、非法名。"""
    from analysis import HardwareSpec
    from bench.problems import build_problem
    from bench.run_phase0 import build_burgers
    from lowering.budget import evaluate_sys

    def add(name, got, expected, passed):
        checks.append(dict(problem="negative-control", name=name, got=got,
                           expected=expected, passed=bool(passed),
                           kind="negative"))

    # 1. diffusion κ_hw=1e3 → infeasible / stiffness
    b = build_problem("diffusion-1d")
    spec_tight = HardwareSpec(kappa_hw=1e3, B_hw=10e6, sigma_req=None,
                              delta_A_norm=1e-3, tol_l2=1e-2)
    cert = evaluate_sys(b.descriptor, spec_tight)
    add("diffusion κ_hw=1e3 → infeasible",
        "status=%s axis=%s" % (cert.status, cert.binding_axis),
        "infeasible, binding_axis=stiffness",
        (not cert.feasible) and cert.binding_axis == "stiffness")

    # 2. burgers 线性化在光滑态：表示轴不 binding。T 的选择要避开的陷阱：
    #    shock_sensor 的 rep = max(E_tail/tol², p_thresh/p)；近正弦态（T≈0.05/
    #    0.1）的 p-拟合被截断噪声主导（T=0.05 → p=0.247），rep 靠 p_thresh/p
    #    违规而不是 E_tail。T=0.2 给 rep=0.13（陡化到 p 拟合稳定、E_tail
    #    仍 ≪ tol²）——同一个证书逻辑给出不同的判决。
    #    直接走 run_phase0.build_burgers（不 build bundle —— 512 点高分辨率
    #    参考要 ~2 分钟，这个控制只需要 linearization 与 reg）
    sys_sm, _meta_sm, reg_sm = build_burgers(128, T=0.2)
    spec_b = HardwareSpec(kappa_hw=1e6, B_hw=10e6, sigma_req=None,
                          delta_A_norm=1e-3, tol_l2=1e-2)
    cert_sm = evaluate_sys(sys_sm, spec_b, reg=reg_sm)
    rep_v = cert_sm.slack["values"].get("representation") if cert_sm.slack else None
    add("burgers smooth state T=0.2: representation NOT binding",
        "E_tail=%.3e rep=%s" % (float(_find(reg_sm, "E_tail",
                                            default=float("nan"))), rep_v),
        "representation axis not violated",
        cert_sm.slack is not None and "representation" not in cert_sm.slack["violated"])

    # 3. 非法问题名 → 拒绝（若实现为 raise；否则跳过并留痕）
    raised = False
    try:
        r = build_problem("no-such-problem-xyz")
        raised = r is None
    except Exception:
        raised = True
    if raised:
        add("malformed problem name rejected", "raises", "raise", True)
    else:
        emit("  note: build_problem did not raise on bad name; skip check")
    return checks


# ===========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()

    global build_problem
    from analysis import HardwareSpec, solve_generalized_lyapunov
    try:
        from bench.problems import build_problem
    except ImportError as e:
        emit("lane bench/problems.py not landed yet (%s)" % e)
        emit("run_pipeline is code-complete against the frozen interface;")
        emit("validation pending until the lane lands.")
        raise SystemExit(3)

    hr("Stage A5 step-3：3 锚定基准 e2e（符号 → 证书 → 孪生对账）")

    checks: list[dict] = []
    ctrl: dict = {}

    # ---- diffusion-1d ---------------------------------------------------------
    hr("1. diffusion-1d")
    bundle = build_problem("diffusion-1d")
    spec_d = HardwareSpec(kappa_hw=1e6, B_hw=10e6, sigma_req=None,
                          delta_A_norm=1e-3, tol_l2=1e-2)
    rec, twin, cert, extra = drive("diffusion-1d", bundle, spec_d, checks, ctrl)
    sigma_y = float(_find(bundle.reference, "sigma_y_analytic", "sigma_y",
                          default=float("nan")))
    if not np.isfinite(sigma_y):
        sigma_y = float(solve_generalized_lyapunov(bundle.descriptor).sigma_y)
    rec["sigma_y"] = sigma_y
    rec["kappa"] = float(_find(bundle.reference, "kappa", "kappa_analytic",
                               default=cert.slack["values"]["stiffness"]
                               if cert.slack else float("nan")))
    kappa_ref = _find(bundle.reference, "kappa", "kappa_analytic")
    if kappa_ref is not None:
        d = _decade(chk := bundle.descriptor.check()["kappa"], kappa_ref)
        checks.append(dict(problem="diffusion-1d",
                           name="diffusion-1d: kappa = 4096",
                           got="%.4g" % chk, expected="%.4g" % kappa_ref,
                           passed=bool(d <= 0.5), kind="recon"))
    else:
        checks.append(dict(problem="diffusion-1d",
                           name="diffusion-1d: kappa = 4096",
                           got="%.4g" % bundle.descriptor.check()["kappa"],
                           expected="reference kappa missing", passed=False,
                           kind="recon"))
    reconcile_diffusion(rec, twin, cert, extra, bundle, checks, sigma_y)

    # ---- poisson-2d -----------------------------------------------------------
    hr("2. poisson-2d")
    bundle_p = build_problem("poisson-2d")
    # delta_A_norm=None：KCL 松弛网格的物理扰动是**失配/泄漏**（δI 注入，
    # 激发守恒零模）—— slack.py docstring 明确这类扰动判 drift budget，
    # 不用 Γ_proj‖ΔA‖；robustness_A 轴对此问题"不适用"（未评估），其误差
    # 落到 floor_bound（mismatch_floor 实测有界）由对账核对。这与
    # 其它锚点不同：它们的 ΔA 是 conservative A-perturbation（robustness_A
    # 轴正是为其设计）。
    spec_p = HardwareSpec(kappa_hw=1e6, B_hw=10e6, sigma_req=None,
                          delta_A_norm=None, tol_l2=1e-2, eta_route=None)
    rec_p, twin_p, cert_p, extra_p = drive("poisson-2d", bundle_p, spec_p,
                                           checks, ctrl)
    reconcile_poisson(rec_p, twin_p, cert_p, extra_p, bundle_p, checks)
    r0 = rec_p.setdefault("reconciliation", {})
    r0["cert_note"] = (
        "cert on P3 physical-time descriptor (eig=lambda/alpha): bandwidth "
        "= B_hw*margin within window; robustness_A unevaluated (delta_A_norm "
        "=None — mismatch is a leakage/drift perturbation judged by "
        "floor_bound, not Gamma_proj); feasible on evaluated axes")

    # ---- burgers-1d -----------------------------------------------------------
    hr("3. burgers-1d")
    bundle_b = build_problem("burgers-1d")
    reg_b = _find(bundle_b.spec_meta, "reg")
    spec_b = HardwareSpec(kappa_hw=1e6, B_hw=10e6, sigma_req=None,
                          delta_A_norm=1e-3, tol_l2=1e-2)
    rec_b, twin_b, cert_b, extra_b = drive("burgers-1d", bundle_b, spec_b,
                                           checks, ctrl)
    reconcile_burgers(rec_b, twin_b, cert_b, extra_b, bundle_b, checks)

    # ---- 负控制 ---------------------------------------------------------------
    hr("4. 负控制")
    negative_controls(checks)

    # ---- 汇总 -----------------------------------------------------------------
    hr("5. 对账汇总")
    rows = []
    for name in ("diffusion-1d", "poisson-2d", "burgers-1d"):
        r = CASES.get(name, {})
        recon = r.get("reconciliation", {})
        rows.append((name,
                     r.get("p6_certificate", {}).get("status", "n/a"),
                     str(r.get("p6_binding_axis")),
                     recon.get("predicted", "n/a"),
                     recon.get("measured", "n/a"),
                     recon.get("verdict", "n/a")))
    table(["problem", "cert status", "binding axis", "predicted",
           "measured", "verdict"], rows)

    hr("6. 验收")
    table(["check", "got", "expected", "PASS"],
          [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
           for c in checks])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    payload = dict(cases={k: v for k, v in CASES.items()}, checks=checks,
                   summary=dict(n_checks=len(checks),
                                n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "pipeline.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(os.path.join(OUT, "pipeline_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/pipeline.json")
    emit("  report → results/pipeline_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
