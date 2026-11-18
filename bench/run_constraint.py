"""T10 验收：constraint fabric（KCL/Poisson 松弛）+ substrate 映射。

    python3 -m bench.run_constraint

算例（DESIGN.md §3.4 / §3.7 合同）：

    1  Poisson 1D     N=64 Dirichlet −∇²u=f, f=sin πx ⇒ u=sin πx/π²；
                      G·V=I 的稳态 O(h²) 命中解析解。
    2  τ_relax        C_node/λ_min(G) 公式 vs 解析逐模松弛的慢模衰减拟合，
                      比值 5% 内（§3.4 的 τ_relax ~ C_y/λ_min(A)）。
    3  κ 缩放         poisson_1d κ(G) ~ N²：N=32 vs 64 比值 ≈4±20%。
    4  Descriptor     kcl_relaxation_descriptor → analysis 广义特征值
                      = −λ_mesh/C_node，A 稳定（复用已验证的分析栈）。
    5  mismatch 底噪  确定性界 ‖δV‖∞ ≤ ‖G⁻¹‖∞·‖δI‖∞ 在 10 个种子上成立；
                      底噪随 κ 增长（N=128/N=32 ≈ κ 比值）；lam·ΔV 项只在
                      ΔV≠0 时起作用；eps_sys 共模 = η·V*。
    6  ε_c 时序       τ_constraint/τ_state,min ≪ 1 合同：一个通过、两个构造失败。
    7  Neumann 泄漏   neumann-mixed（单点 g_ref 接地）非奇异，λ_min 是 g_ref 量级。
    8  非对称守卫     有向 gm 网络触发 (G+Gᵀ)/2 + 全谱横坐标检查路径，
                      monotone_spd=False 如实报告，全谱 Re<0 才算稳定。
    9  流体映射       Hagen–Poiseuille R_f；两节点链压力解析命中；
                      G 保持 graph-Laplacian 结构（§3.7 同构可执行化）。
    10 负控制         纯 Neumann（无 g_ref）⇒ 零模被标记为 SINGULAR，
                      绝不静默求逆。

诚实声明：settle() 对非对称 G 用对称部分做衰减、G⁻¹I 做渐近线——小扰动
下成立的近似，bench 只在对称 mesh 上用它做定量拟合；mismatch 模型是
coupling.py InterfaceModel 误差类别（lam·ΔV + Pelgrom + 共模）的线性化
静态版本，不是 SPICE 蒙特卡洛。
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from twin.constraint import (ConstraintMesh, ConstraintResult, eps_c,
                             kcl_relaxation_descriptor, load_lambda_table,
                             mismatch_floor, relax_time, relaxation_modes,
                             residual, settle_with_c, solve_constraint)
from twin.substrate import (KAPPA_HW, FLUIDIC_TO_ELECTRICAL, TransducerCost,
                            fluidic_mesh_to_conductance, fluidic_resistance)

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
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


checks: list[dict] = []


def add(name, got, expected, passed, kind="exact"):
    checks.append(dict(name=name, got=got, expected=expected,
                       passed=bool(passed), kind=kind))


def _poisson_1d_rhs(n):
    """I_rhs = h²·f for f = sin(πx), interior nodes of a unit interval."""
    h = 1.0 / (n + 1)
    x = h * np.arange(1, n + 1)
    return h * h * np.sin(np.pi * x), x, h


# ---------------------------------------------------------------------------
def case_poisson_1d():
    hr("CASE 1  Poisson 1D: steady state vs analytic sin(πx)/π²")
    n = 64
    mesh = ConstraintMesh.poisson_1d(n, g=1.0)
    i_rhs, x, h = _poisson_1d_rhs(n)
    u_exact = np.sin(np.pi * x) / np.pi ** 2
    v_star = np.linalg.solve(mesh.G, i_rhs)
    err = float(np.max(np.abs(v_star - u_exact)))
    tol = 1e-2 * float(np.max(np.abs(u_exact)))
    add("poisson1d max|V*−u|", "%.3e" % err, "< %.3e" % tol, err < tol, "bound")
    add("poisson1d symmetric SPD", mesh.is_symmetric, True, mesh.is_symmetric)
    table(["quantity", "value"], [
        ["N", n], ["h", "%.4e" % h], ["max err", "%.3e" % err],
        ["max|u|", "%.4f" % u_exact.max()],
    ])


def case_tau_relax():
    hr("CASE 2  τ_relax = C_node/λ_min(G) vs fitted slow-mode decay")
    n = 64
    mesh = ConstraintMesh.poisson_1d(n)
    c_node = 1e-12
    i_rhs, x, _ = _poisson_1d_rhs(n)
    lam, Q = np.linalg.eigh(0.5 * (mesh.G + mesh.G.T))
    lam_min = lam[0]
    tau_formula = c_node / lam_min
    t = np.linspace(0.0, 20.0 * tau_formula, 41)
    V = settle_with_c(mesh, c_node, i_rhs, t)
    v_star = np.linalg.solve(mesh.G, i_rhs)
    a = (V - v_star[None, :]) @ Q[:, 0]          # slow-mode amplitude a_0(t)
    # fit log-slope over the tail
    mask = a > 1e-14 * max(abs(a[0]), 1e-30)
    slope = np.polyfit(t[mask], np.log(np.abs(a[mask])), 1)[0]
    tau_fit = -1.0 / slope
    ratio = tau_fit / tau_formula
    add("tau_relax fit/formula", "%.4f" % ratio, "1 ± 0.05",
        abs(ratio - 1.0) < 0.05, "ratio")
    add("tau_relax value", "%.3e s" % tau_formula, "C/λ_min", True, "info")
    # residual R(t)=‖A·V(t)−b‖: 0 at large t, = ‖G·v0−I‖ at t=0
    r0 = residual(mesh, V[0], i_rhs)
    r_end = residual(mesh, V[-1], i_rhs)
    r0_expected = float(np.linalg.norm(mesh.G @ np.zeros(n) - i_rhs))
    add("residual t=0 = ‖G·0−I‖", "%.3e" % r0, "%.3e" % r0_expected,
        abs(r0 - r0_expected) <= 1e-9 * max(r0_expected, 1e-30))
    add("residual t_end → 0", "%.3e" % r_end, "< 1e-6·r0", r_end < 1e-6 * r0_expected)
    table(["quantity", "value"], [
        ["λ_min(G)", "%.4e" % lam_min], ["τ formula", "%.4e s" % tau_formula],
        ["τ fitted", "%.4e s" % tau_fit], ["ratio", "%.4f" % ratio],
        ["R(0)", "%.3e" % r0], ["R(end)", "%.3e" % r_end],
    ])


def case_kappa_scaling():
    hr("CASE 3  κ(G) ~ N² scaling (N=32 vs N=64)")
    k32 = relaxation_modes(ConstraintMesh.poisson_1d(32))["kappa_G"]
    k64 = relaxation_modes(ConstraintMesh.poisson_1d(64))["kappa_G"]
    ratio = k64 / k32
    add("κ ratio N=64/32", "%.3f" % ratio, "≈4 ±20%", abs(ratio - 4.0) < 0.8, "ratio")
    add("κ growth ~N²", "ratio %.3f" % ratio, "(65/33)²=%.3f" % (65 / 33) ** 2,
        abs(ratio - (65 / 33) ** 2) < 0.2, "info")
    table(["N", "κ(G)", "λ_min"], [
        [32, "%.3f" % k32, "%.4e" % relaxation_modes(ConstraintMesh.poisson_1d(32))["lam_min_sym"]],
        [64, "%.3f" % k64, "%.4e" % relaxation_modes(ConstraintMesh.poisson_1d(64))["lam_min_sym"]],
    ])


def case_descriptor():
    hr("CASE 4  Descriptor handoff → analysis stack")
    n = 64
    mesh = ConstraintMesh.poisson_1d(n)
    c_node = 1e-12
    desc = kcl_relaxation_descriptor(mesh, c_node)
    lam_mesh = np.linalg.eigvalsh(mesh.G)
    w = desc.finite_eigs()
    re_max = float(np.max(w.real))
    add("descriptor A stable (Re eig < 0)", "%.3e" % re_max, "< 0", re_max < 0)
    # sign convention: A = −G/C_node ⇒ eigenvalues −λ_mesh/C_node
    dev = float(np.max(np.abs(np.sort(w.real) - np.sort(-lam_mesh) / c_node)))
    add("eigvals = −λ_mesh/C_node", "%.3e" % dev, "≈0 (<1e-9 rel)", dev < 1e-9 * (lam_mesh.max() / c_node))
    add("E = C_node·I (regular)", desc.is_regular(), True, desc.is_regular())
    check = desc.check()
    add("descriptor.check() n", check["n"], n, check["n"] == n)
    table(["field", "value"], [
        ["n", check["n"]], ["max Re eig", "%.4e" % re_max],
        ["is_regular", desc.is_regular()],
    ])


def case_mismatch():
    hr("CASE 5  mismatch floor: deterministic bound, κ growth, λ·ΔV term")
    n = 64
    mesh = ConstraintMesh.poisson_1d(n)
    i_rhs, x, _ = _poisson_1d_rhs(n)
    lam_table = load_lambda_table()
    emit("  λ table (results/device_extract.json): %s" %
         ", ".join("%s=%.4g" % kv for kv in sorted(lam_table.items())))

    # (a) deterministic bound over 10 seeded draws, dev-dominant model
    ok = True
    floors = []
    for seed in range(10):
        mf = mismatch_floor(mesh, i_rhs, lam=0.0, eps_dev=0.01, seed=seed)
        floors.append(mf["floor_inf"])
        ok &= mf["floor_inf"] <= mf["bound_inf"]
    add("‖δV‖∞ ≤ ‖G⁻¹‖∞‖δI‖∞ (10 seeds)", "all hold" if ok else "violated",
        "all hold", ok)

    # (b) floor grows with κ: N=32 vs N=128, same relative eps_dev, same
    # ABSOLUTE injected current (unit uniform source) so ‖δI‖ matches and
    # the 1/λ_min amplification is what grows the floor
    mesh32 = ConstraintMesh.poisson_1d(32)
    mesh128 = ConstraintMesh.poisson_1d(128)
    ir32 = np.ones(32)
    ir128 = np.ones(128)
    f32 = mismatch_floor(mesh32, ir32, lam=0.0, eps_dev=0.01, seed=0)["floor_inf"]
    f128 = mismatch_floor(mesh128, ir128, lam=0.0, eps_dev=0.01, seed=0)["floor_inf"]
    k32 = relaxation_modes(mesh32)["kappa_G"]
    k128 = relaxation_modes(mesh128)["kappa_G"]
    floor_ratio = f128 / f32
    kappa_ratio = k128 / k32
    add("floor ratio N=128/32", "%.2f" % floor_ratio,
        "≈κ ratio %.2f (×0.5–2)" % kappa_ratio,
        0.5 * kappa_ratio <= floor_ratio <= 2.0 * kappa_ratio, "ratio")

    # (c) lam·ΔV term only matters when ΔV ≠ 0
    v_star = np.linalg.solve(mesh.G, i_rhs)
    dv = v_star - v_star[np.argmax(np.abs(i_rhs))]   # drain deviation vs reference node
    f_lam0 = mismatch_floor(mesh, i_rhs, lam=lam_table["ihp-hbt"], eps_dev=0.0,
                            delta_v=dv, seed=0)["floor_inf"]
    f_lam970 = mismatch_floor(mesh, i_rhs, lam=lam_table["ihp-sg13g2"], eps_dev=0.0,
                              delta_v=dv, seed=0)["floor_inf"]
    add("lam systematic grows with λ (ΔV≠0)", "%.2e → %.2e" % (f_lam0, f_lam970),
        "increases", f_lam970 > 3.0 * f_lam0)
    f_dev_lam = mismatch_floor(mesh, i_rhs, lam=lam_table["ihp-sg13g2"], eps_dev=0.01,
                               seed=0)["floor_inf"]
    f_dev_only = mismatch_floor(mesh, i_rhs, lam=lam_table["ihp-sg13g2"], eps_dev=0.01,
                                delta_v=np.zeros(n), seed=0)["floor_inf"]
    add("eps_dev-dominant when ΔV=0 (λ term inert)", "%.4f" % (f_dev_lam / f_dev_only),
        "1.0 (identical draws)", abs(f_dev_lam / f_dev_only - 1.0) < 1e-12)

    # (d) eps_sys common-mode: δI = η·I ⇒ δV = η·V* (relative error on the solution)
    mf_sys = mismatch_floor(mesh, i_rhs, lam=0.0, eps_dev=0.0, eps_sys=0.01, seed=0)
    pred = 0.01 * float(np.max(np.abs(v_star)))
    add("eps_sys floor ≈ η·eps_sys·‖V*‖∞", "%.3e" % mf_sys["floor_inf"],
        "O(%.3e)" % pred,
        0.3 * pred <= mf_sys["floor_inf"] <= 3.0 * pred, "order")
    table(["model", "floor ‖δV‖∞", "dominant mode"], [
        ["eps_dev=0.01 (N=32)", "%.3e" % f32, str(mismatch_floor(mesh32, ir32, 0.0, 0.01, seed=0)["dominant_mode"])],
        ["eps_dev=0.01 (N=128)", "%.3e" % f128, "—"],
        ["λ=0.0138, ΔV≠0", "%.3e" % f_lam0, "—"],
        ["λ=0.970, ΔV≠0", "%.3e" % f_lam970, "—"],
        ["eps_sys=0.01", "%.3e" % mf_sys["floor_inf"], str(mf_sys["dominant_mode"])],
    ])


def case_eps_c():
    hr("CASE 6  ε_c = τ_constraint/τ_state,min timing contract (§3.4)")
    n = 64
    mesh = ConstraintMesh.poisson_1d(n)
    c_good, tau_state = 1e-13, 1e-8
    eps_good = eps_c(mesh, c_good, tau_state)
    add("ε_c pass case (<0.01)", "%.4f" % eps_good, "< 0.01", eps_good < 0.01)
    eps_big_c = eps_c(mesh, 1e-9, tau_state)
    add("ε_c fail: C_node too large", "%.4f" % eps_big_c, "≥ 0.01 → FAIL flag",
        eps_big_c >= 0.01)
    res = solve_constraint(mesh, _poisson_1d_rhs(n)[0], 1e-9,
                           external_state_timescale=tau_state)
    add("verdict flags timing violation", res.verdict, "FAIL", res.verdict == "FAIL")
    add("ε_c fail: state too fast (τ_state=1e-12)", "%.4f" % eps_c(mesh, c_good, 1e-12),
        "≥ 0.01", eps_c(mesh, c_good, 1e-12) >= 0.01)
    table(["case", "τ_constraint", "ε_c", "verdict"], [
        ["C=1e-13, τ_state=1e-8", "%.3e" % relax_time(mesh, c_good), "%.4f" % eps_good,
            solve_constraint(mesh, _poisson_1d_rhs(n)[0], c_good, tau_state).verdict],
        ["C=1e-9, τ_state=1e-8", "%.3e" % relax_time(mesh, 1e-9), "%.4f" % eps_big_c,
            res.verdict],
    ])


def case_neumann():
    hr("CASE 7  Neumann leakage: single weak ground keeps G nonsingular")
    n = 32
    g_ref = 1e-3
    mesh = ConstraintMesh.poisson_1d(n, bc="neumann-mixed", g_ref=g_ref)
    modes = relaxation_modes(mesh)
    add("neumann-mixed nonsingular", not modes["singular"], True,
        not modes["singular"])
    lam_min = modes["lam_min_sym"]
    add("λ_min is g_ref-scale", "%.3e" % lam_min,
        "0 < λ_min < g_ref", 0 < lam_min < g_ref)
    add("λ_min ≫ 0 but ≪ λ_max", "%.2e ≪ %.2e" % (lam_min, modes["lam_max_sym"]),
        "κ large (leakage-only zero-mode lifting)", modes["kappa_G"] > 1e3, "info")
    # negative control: pure Neumann (no g_ref) — zero mode must be FLAGGED
    bad = ConstraintMesh.from_edges([(i, i + 1, 1.0) for i in range(n - 1)],
                                    bc="neumann", name="pure-neumann")
    bmodes = relaxation_modes(bad)
    res = solve_constraint(bad, np.ones(n), 1e-12)
    add("pure Neumann zero mode detected", bmodes["singular"], True, bmodes["singular"])
    add("solve_constraint verdict SINGULAR (never inverted)", res.verdict, "SINGULAR",
        res.verdict == "SINGULAR")
    table(["mesh", "λ_min", "κ(G)", "verdict"], [
        ["neumann-mixed g_ref=1e-3", "%.3e" % lam_min, "%.3e" % modes["kappa_G"], "PASS"],
        ["pure Neumann (no gnd)", "%.3e" % bmodes["lam_min_sym"], "inf", "SINGULAR"],
    ])


def case_nonsymmetric():
    hr("CASE 8  non-symmetric guard: (G+Gᵀ)/2 + full-spectrum abscissa")
    # A KCL-conserving two-terminal element is symmetric by construction, so
    # genuine non-symmetry in a KCL mesh comes from controlled sources: node
    # i+1 receives g_m·V_i and node i draws nothing (§3.4: "电阻 mesh 的
    # M-矩阵性质不能直接搬到有源 gm 网络"). Small gm: non-monotone but stable.
    n = 32
    base = ConstraintMesh.poisson_1d(n)

    def cs_mesh(gm):
        G = base.G.copy()
        for i in range(n - 1):
            G[i + 1, i] -= gm      # controlled source g_m·V_i → node i+1
        return ConstraintMesh(G=G, bc="dirichlet", name="gm-source")

    ns = cs_mesh(0.005)
    modes = relaxation_modes(ns)
    add("non-symmetric mesh detected", ns.is_symmetric, False,
        ns.is_symmetric is False)
    add("monotone_spd correctly False", modes["monotone_spd"], False,
        modes["monotone_spd"] is False)
    add("full-spectrum abscissa < 0 (stable)", "%.3e" % modes["spectral_abscissa"],
        "< 0", modes["spectral_abscissa"] < 0)
    add("stable_full uses full eig (not sym part)", modes["stable_full"], True,
        modes["stable_full"])
    add("λ_min taken on symmetric part", "%.3e" % modes["lam_min_sym"],
        "= λ_min((G+Gᵀ)/2)",
        abs(modes["lam_min_sym"] - np.linalg.eigvalsh(0.5 * (ns.G + ns.G.T))[0]) < 1e-12)
    # guard must also catch instability: larger gm closes a destabilizing loop
    un = cs_mesh(0.05)
    umodes = relaxation_modes(un)
    add("destabilizing gm caught by full spectrum",
        umodes["stable_full"], False, umodes["stable_full"] is False)
    add("destabilizing abscissa > 0", "%.3e" % umodes["spectral_abscissa"],
        "> 0", umodes["spectral_abscissa"] > 0)
    table(["mesh", "monotone_spd", "abscissa(−G)", "stable_full"], [
        ["gm=0.005 (stable, non-monotone)", str(modes["monotone_spd"]),
            "%.3e" % modes["spectral_abscissa"], str(modes["stable_full"])],
        ["gm=0.05 (unstable)", str(umodes["monotone_spd"]),
            "%.3e" % umodes["spectral_abscissa"], str(umodes["stable_full"])],
    ])


def case_fluidic():
    hr("CASE 9  fluidic substrate → same graph Laplacian (§3.7)")
    # Hagen–Poiseuille formula check
    l, r, mu = 1e-2, 1e-4, 1e-3
    R = fluidic_resistance(l, r, mu)
    R_exact = 8.0 * mu * l / (np.pi * r ** 4)
    add("R_f = 8μl/(πr⁴)", "%.4e" % R, "%.4e" % R_exact,
        abs(R - R_exact) < 1e-9 * R_exact)

    # two-node chain: reservoir (−1) — R1 — node0 — R2 — node1 — R3(reservoir);
    # Q injected AT node0 (current source): R1 is a return path, R2+R3 the
    # outlet leg ⇒ P0 = Q/(1/R1 + 1/(R2+R3)), P1 = P0·R3/(R2+R3)
    R1 = fluidic_resistance(1e-2, 1e-4, 1e-3)
    R2 = fluidic_resistance(2e-2, 1e-4, 1e-3)   # twice the length ⇒ 2·R1
    R3 = R2
    G = fluidic_mesh_to_conductance([(-1, 0, R1), (0, 1, R2), (1, -1, R3)],
                                    n_nodes=2)
    Q = 1e-12
    P = np.linalg.solve(G, np.array([Q, 0.0]))
    p0_exact = Q / (1.0 / R1 + 1.0 / (R2 + R3))
    p1_exact = p0_exact * R3 / (R2 + R3)
    add("two-node chain P0 = Q·(R1∥(R2+R3))", "%.4e" % P[0], "%.4e" % p0_exact,
        abs(P[0] - p0_exact) < 1e-9 * p0_exact)
    add("two-node chain P1 = P0·R3/(R2+R3)", "%.4e" % P[1], "%.4e" % p1_exact,
        abs(P[1] - p1_exact) < 1e-9 * p1_exact)
    offdiag_ok = bool(np.allclose(G, G.T) and np.all(G <= 1e-30 + 0.0) is not None
                      and np.all(G - np.diag(np.diag(G)) <= 1e-30))
    add("fluidic G structure (sym, off-diag ≤ 0)", str(offdiag_ok), "True",
        offdiag_ok)
    add("mapping table P↔V, Q↔I, R_f↔R", str(FLUIDIC_TO_ELECTRICAL),
        "{'P':'V','Q':'I','R_f':'R','C_f':'C','L_R':'G'}",
        FLUIDIC_TO_ELECTRICAL == {"P": "V", "Q": "I", "R_f": "R",
                                  "C_f": "C", "L_R": "G"})
    # the fluidic G must be consumable by the constraint machinery as-is
    modes = relaxation_modes(ConstraintMesh(G=G, bc="dirichlet", name="fluidic-2n"))
    add("fluidic G → relaxation_modes works (SPD)", modes["monotone_spd"], True,
        modes["monotone_spd"])
    table(["quantity", "value"], [
        ["R1 (l=1cm, r=100µm)", "%.4e Pa·s/m³" % R1],
        ["R2 (l=2cm)", "%.4e" % R2],
        ["P0/P1 analytic", "%.4f" % (p0_exact / p1_exact)],
    ])


def case_2d_and_misc():
    hr("CASE 10  2D mesh sanity + substrate constants")
    mesh = ConstraintMesh.poisson_2d(8, 8, g=1.0)
    modes = relaxation_modes(mesh)
    add("poisson_2d(8×8) SPD monotone", modes["monotone_spd"], True,
        modes["monotone_spd"])
    G2 = mesh.G.reshape(8, 8, 8, 8)
    interior = G2[1:7, 1:7].reshape(36, 64)
    rs_int = float(np.max(np.abs(interior.sum(axis=1))))
    add("poisson_2d interior row sums ≈ 0", "%.3e" % rs_int, "≈0",
        rs_int < 1e-12)
    add("κ_hw windows present (electronic/mems/fluidic/photonic)",
        str(sorted(KAPPA_HW)), "4 domains", len(KAPPA_HW) == 4)
    add("photonic window is 1e-12..1e-9", str(KAPPA_HW["photonic"]),
        "(1e-12, 1e-9)", KAPPA_HW["photonic"] == (1e-12, 1e-9))
    tc = TransducerCost(gain=1.0, noise_rms=1e-6, bandwidth_hz=1e6, energy=1e-12,
                        notes="stub")
    d = tc.as_dict()
    add("TransducerCost.as_dict contract fields", str(sorted(d)),
        "gain/noise_rms/bandwidth_hz/energy", all(k in d for k in
        ("gain", "noise_rms", "bandwidth_hz", "energy")))


def main():
    emit("T10 constraint fabric acceptance (DESIGN.md §3.4/§3.7)")
    case_poisson_1d()
    case_tau_relax()
    case_kappa_scaling()
    case_descriptor()
    case_mismatch()
    case_eps_c()
    case_neumann()
    case_nonsymmetric()
    case_fluidic()
    case_2d_and_misc()

    hr("VERDICT")
    rows = [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
            for c in checks]
    table(["check", "got", "expected", "pass"], rows)
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))
    summary = dict(n_checks=len(checks), n_passed=len(checks) - n_fail)
    CASES["constraint"] = dict(checks=checks, summary=summary)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "constraint.json"), "w") as fh:
        json.dump(_jsonable(CASES), fh, indent=1)
    with open(os.path.join(OUT, "constraint_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/constraint.json, results/constraint_report.txt")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
