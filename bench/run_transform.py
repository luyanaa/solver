"""Phase 4.5 验收：adaptive representation 与 TransformContract（DESIGN.md §5.7–§5.9）。

四组，对应 DESIGN 的 4.5A–4.5E：

  A  WENO 三算子：sensor → σ → conservative face flux；守恒与无 chatter
  B  lifting wavelet：W / W_inv / W_star；isometry、PR、adjoint 分开验证
  C  transposeable edge：A↔Aᵀ 与离散 adjoint 梯度
  D  scale-dependent poles：mode C 与 mode g_m 的 g_m/噪声 trade-off
  E  Gram calibration：G_T = T̃*T̃，G_T^{-1/2} 白化

运行：python3 -m bench.run_transform
输出：results/transform.json、results/transform_report.txt
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from twin.transform import (
    LiftingWavelet, SigmaTracker, TransposeableEdge,
    adjoint_ode_gradient, blend_face_flux, central_face_flux,
    conservation_drift, count_threshold_crossings, edge_matrix,
    finite_difference_gradient, gram_calibrate, isometry_error,
    muscl_face_flux, quadratic_gradient, rhs_from_face_flux, rk4_matrix,
    scale_time_error, sigma_from_indicator,
    smoothness_indicator, stationary_haar, stationary_haar_inverse,
    weno5_face_flux,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
_LINES = []
CASES = {}


def emit(s=""):
    print(s)
    _LINES.append(s)


def hr(title):
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    emit(fmt.format(*headers))
    emit("  ".join("-" * w for w in widths))
    for row in rows:
        emit(fmt.format(*[str(v) for v in row]))


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _rk4(u, dt, rhs):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# 4.5A WENO 三算子
# ---------------------------------------------------------------------------
def case_weno():
    n = 256
    length = 1.0
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    u0 = 0.5 + 0.5 * np.sin(2.0 * np.pi * x)
    flux = lambda v: 0.5 * v * v

    t_end = 0.5
    cfl = 0.2
    dt = cfl * dx / max(abs(u0))
    nsteps = int(np.ceil(t_end / dt))
    dt = t_end / nsteps

    def rhs_ref(u):
        a = max(float(np.max(np.abs(u))), 1e-12)
        return rhs_from_face_flux(weno5_face_flux(u, flux, a), dx)

    def rhs_adapt(u, sig):
        a = max(float(np.max(np.abs(u))), 1e-12)
        Fs = central_face_flux(u, flux)
        Fw = muscl_face_flux(u, flux, a)
        return rhs_from_face_flux(blend_face_flux(Fs, Fw, sig), dx)

    u_ref = u0.copy()
    u_ad = u0.copy()
    tracker = SigmaTracker(beta0=0.15, k=16.0, hysteresis=0.08,
                           max_rate=400.0, release_rate=10.0)
    sig_hist = []
    for _ in range(nsteps):
        u_ref = _rk4(u_ref, dt, rhs_ref)
        s = smoothness_indicator(u_ad)
        sig = tracker.update(s, dt)
        sig_hist.append(sig.copy())
        u_ad = _rk4(u_ad, dt, lambda v: rhs_adapt(v, sig))

    s_final = smoothness_indicator(u_ad)
    sig_final = sigma_from_indicator(s_final, beta0=0.15, k=16.0)
    i_ad = int(np.argmax(np.abs(np.diff(u_ad))))
    i_ref = int(np.argmax(np.abs(np.diff(u_ref))))
    x_ad = (i_ad + 0.5) * dx
    x_ref = (i_ref + 0.5) * dx

    # 静态 sensor：理想 step 与光滑 sine 各测一次；避开周期边界
    u_step = (x > 0.5).astype(float)
    s_step = smoothness_indicator(u_step)
    sig_step = sigma_from_indicator(s_step, beta0=0.15, k=16.0)
    i_step = n // 2

    u_sine = np.sin(2.0 * np.pi * x)
    s_sine = smoothness_indicator(u_sine)
    sig_sine = sigma_from_indicator(s_sine, beta0=0.15, k=16.0)
    smooth_mask = (np.abs(x - 0.5) > 0.1) & (np.abs(x - 0.5) < 0.45)

    hist = np.asarray(sig_hist)
    crossings = count_threshold_crossings(hist, 0.5)

    return dict(
        n=n, nsteps=nsteps, dt=dt,
        cfl=cfl,
        shock_sigma_static=float(sig_step[i_step]),
        smooth_sigma_static=float(np.max(sig_sine[smooth_mask])),
        shock_sigma_final=float(sig_final[i_ad]),
        smooth_sigma_final=float(np.max(sig_final[np.abs(x - x_ad) > 0.15])),
        conservation_drift=float(conservation_drift(u_ad, u0, dx)),
        conservation_drift_ref=float(conservation_drift(u_ref, u0, dx)),
        shock_x_ad=x_ad, shock_x_ref=x_ref,
        shock_x_error=abs(x_ad - x_ref),
        l2_vs_weno5=float(np.sqrt(np.mean((u_ad - u_ref) ** 2))),
        max_cell_crossings=int(np.max(crossings)),
        mean_cell_crossings=float(np.mean(crossings)),
    )


# ---------------------------------------------------------------------------
# 4.5B lifting wavelet contract
# ---------------------------------------------------------------------------
def case_lifting():
    rng = np.random.default_rng(7)
    n = 64
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    w_haar = LiftingWavelet(n, "haar", levels=3)
    w_cdf = LiftingWavelet(n, "cdf53", levels=3)

    a, details = stationary_haar(x, levels=3)
    a_shift, details_shift = stationary_haar(np.roll(x, 5), levels=3)
    trans_err = max(
        float(np.max(np.abs(np.roll(details[j], 5) - details_shift[j])))
        for j in range(3)
    )
    rec = stationary_haar_inverse(a, details)

    return dict(
        n=n,
        haar_isometry=w_haar.isometry_error(),
        haar_pr=w_haar.pr_error(),
        haar_dot=w_haar.dot_error(x, y),
        haar_noise_gain=w_haar.noise_gain,
        haar_condition=w_haar.condition,
        cdf_isometry=w_cdf.isometry_error(),
        cdf_pr=w_cdf.pr_error(),
        cdf_adjoint_error=w_cdf.adjoint_error(),
        cdf_condition=w_cdf.condition,
        stationary_translation_error=trans_err,
        stationary_inverse_error=float(np.max(np.abs(rec - x))),
        stationary_redundancy=len(details) + 1,
    )


# ---------------------------------------------------------------------------
# 4.5C transposeable edge / adjoint
# ---------------------------------------------------------------------------
def case_adjoint():
    rng = np.random.default_rng(11)
    n = 8
    edges = [TransposeableEdge(int(i), int(j), float(g))
             for i, j, g in zip(
                 rng.integers(0, n, 24),
                 rng.integers(0, n, 24),
                 rng.normal(0.0, 1.0, 24))]
    A = edge_matrix(edges, n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    lhs = float(np.dot(A @ x, y))
    rhs = float(np.dot(x, A.T @ y))
    trans_err = abs(lhs - rhs) / max(abs(lhs) + abs(rhs), 1e-30)

    b = rng.standard_normal(n)
    g_an = quadratic_gradient(A, b, x)
    g_fd = finite_difference_gradient(
        lambda z: 0.5 * float(np.sum((A @ z - b) ** 2)), x)
    grad_err = float(np.max(np.abs(g_an - g_fd))
                     / max(np.max(np.abs(g_fd)), 1e-30))

    Aode = -np.diag([1.0, 2.0, 3.0]) + 0.1 * rng.standard_normal((3, 3))
    x0 = rng.standard_normal(3)
    C = rng.standard_normal((2, 3))
    y2 = rng.standard_normal(2)
    T = 1.0
    nsteps = 100
    adj = adjoint_ode_gradient(Aode, x0, T, nsteps, C, y2)

    M = rk4_matrix(Aode, T / nsteps)

    def cost(z):
        xT = np.linalg.matrix_power(M, nsteps) @ z
        return 0.5 * float(np.sum((C @ xT - y2) ** 2))

    g_ode_fd = finite_difference_gradient(cost, x0)
    ode_err = float(np.max(np.abs(adj["grad"] - g_ode_fd))
                    / max(np.max(np.abs(g_ode_fd)), 1e-30))

    return dict(
        n=n, n_edges=len(edges),
        transpose_identity_error=trans_err,
        quadratic_grad_fd_error=grad_err,
        adjoint_ode_fd_error=ode_err,
        checkpoint_bytes=adj["checkpoint_bytes"],
        checkpoint_states=adj["checkpoint_states"],
    )


# ---------------------------------------------------------------------------
# 4.5D scale-dependent poles
# ---------------------------------------------------------------------------
def case_scale():
    levels = 6
    pC = scale_time_error(levels, mode="C")
    pgm = scale_time_error(levels, mode="gm")
    pn = scale_time_error(levels, mode="none")

    def pole_err(p):
        return float(np.max(np.abs(p["omega"] / p["lambda_model"] - 1.0)))

    return dict(
        levels=levels,
        C_pole_spread=pC["pole_spread"],
        gm_pole_spread=pgm["pole_spread"],
        C_pole_error=pole_err(pC),
        gm_pole_error=pole_err(pgm),
        C_model_time_error=pC["max_rel_error"],
        gm_model_time_error=pgm["max_rel_error"],
        none_model_time_error=pn["max_rel_error"],
        C_gm_spread=pC["gm_spread"],
        C_noise_spread=pC["noise_spread"],
        gm_gm_spread=pgm["gm_spread"],
        gm_noise_spread=pgm["noise_spread"],
        C_omega=pC["omega"],
        gm_omega=pgm["omega"],
    )


# ---------------------------------------------------------------------------
# 4.5E Gram calibration
# ---------------------------------------------------------------------------
def case_gram():
    rng = np.random.default_rng(19)
    n = 64
    w_haar = LiftingWavelet(n, "haar", levels=3)
    w_cdf = LiftingWavelet(n, "cdf53", levels=3)

    W_haar_hw = w_haar.W + 0.002 * rng.standard_normal((n, n))
    g_haar = gram_calibrate(W_haar_hw)
    # 同一个右 Gram 的 K，如果错误地左乘，不会正交化；正确的左标定要用左 Gram
    naive_left = g_haar["K"] @ W_haar_hw
    g_haar_left = gram_calibrate(W_haar_hw, side="left")

    W_cdf_hw = w_cdf.W + 0.01 * rng.standard_normal((n, n))
    g_cdf = gram_calibrate(W_cdf_hw)

    return dict(
        haar_isometry_before=g_haar["isometry_before"],
        haar_isometry_after=g_haar["isometry_after"],
        haar_pr_after=g_haar["pr_after"],
        haar_noise_gain=g_haar["noise_gain"],
        haar_cond=g_haar["cond"],
        haar_adjoint_before=g_haar["adjoint_before"],
        haar_adjoint_after=g_haar["adjoint_after"],
        naive_left_isometry=isometry_error(naive_left),
        haar_left_isometry_after=g_haar_left["isometry_after"],
        cdf_isometry_before=g_cdf["isometry_before"],
        cdf_isometry_after=g_cdf["isometry_after"],
        cdf_pr_after=g_cdf["pr_after"],
        cdf_noise_gain=g_cdf["noise_gain"],
        cdf_cond=g_cdf["cond"],
        cdf_adjoint_before=g_cdf["adjoint_before"],
        cdf_adjoint_after=g_cdf["adjoint_after"],
    )


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------
def acceptance():
    checks = []

    def chk(name, ok, value, threshold, case):
        checks.append(dict(name=name, ok=bool(ok), value=_jsonable(value),
                           threshold=threshold, case=case))
        emit(f"[{'PASS' if ok else 'FAIL'}] {name}: {value} "
             f"(threshold {threshold})")

    w = CASES["weno"]
    emit("4.5A WENO 三算子")
    chk("static shock σ", w["shock_sigma_static"] > 0.8,
        f"{w['shock_sigma_static']:.4f}", ">0.8", "weno")
    chk("static smooth σ", w["smooth_sigma_static"] < 0.2,
        f"{w['smooth_sigma_static']:.4f}", "<0.2", "weno")
    chk("adaptive conservation drift", w["conservation_drift"] < 1e-8,
        f"{w['conservation_drift']:.3e}", "<1e-8", "weno")
    chk("shock position error", w["shock_x_error"] < 0.03,
        f"{w['shock_x_error']:.4f}", "<0.03", "weno")
    chk("L2 vs WENO5", w["l2_vs_weno5"] < 0.05,
        f"{w['l2_vs_weno5']:.3e}", "<0.05", "weno")
    chk("max cell crossings", w["max_cell_crossings"] <= 3,
        w["max_cell_crossings"], "<=3", "weno")
    chk("final smooth σ", w["smooth_sigma_final"] < 0.3,
        f"{w['smooth_sigma_final']:.4f}", "<0.3", "weno")

    l = CASES["lifting"]
    emit()
    emit("4.5B lifting wavelet")
    chk("Haar isometry", l["haar_isometry"] < 5e-3,
        f"{l['haar_isometry']:.3e}", "<0.5%", "lifting")
    chk("Haar PR", l["haar_pr"] < 2e-3,
        f"{l['haar_pr']:.3e}", "<0.2%", "lifting")
    chk("Haar adjoint dot", l["haar_dot"] < 1e-3,
        f"{l['haar_dot']:.3e}", "<1e-3", "lifting")
    chk("CDF53 PR", l["cdf_pr"] < 2e-3,
        f"{l['cdf_pr']:.3e}", "<0.2%", "lifting")
    chk("CDF53 W_inv != W_star", l["cdf_adjoint_error"] > 1e-2,
        f"{l['cdf_adjoint_error']:.3e}", ">0.01", "lifting")
    chk("stationary translation", l["stationary_translation_error"] < 1e-12,
        f"{l['stationary_translation_error']:.3e}", "<1e-12", "lifting")
    chk("stationary inverse", l["stationary_inverse_error"] < 1e-12,
        f"{l['stationary_inverse_error']:.3e}", "<1e-12", "lifting")

    a = CASES["adjoint"]
    emit()
    emit("4.5C transposeable edge / adjoint")
    chk("A↔Aᵀ identity", a["transpose_identity_error"] < 1e-12,
        f"{a['transpose_identity_error']:.3e}", "<1e-12", "adjoint")
    chk("quadratic grad vs FD", a["quadratic_grad_fd_error"] < 1e-6,
        f"{a['quadratic_grad_fd_error']:.3e}", "<1e-6", "adjoint")
    chk("adjoint ODE grad vs FD", a["adjoint_ode_fd_error"] < 1e-6,
        f"{a['adjoint_ode_fd_error']:.3e}", "<1e-6", "adjoint")
    chk("checkpoint reported", a["checkpoint_bytes"] > 0,
        a["checkpoint_bytes"], ">0", "adjoint")

    s = CASES["scale"]
    emit()
    emit("4.5D scale-dependent poles")
    chk("mode C pole spread", abs(s["C_pole_spread"] - 4 ** (s["levels"] - 1)) < 1e-9,
        f"{s['C_pole_spread']:.1f}", f"{4 ** (s['levels'] - 1)}", "scale")
    chk("mode gm pole spread", abs(s["gm_pole_spread"] - 4 ** (s["levels"] - 1)) < 1e-9,
        f"{s['gm_pole_spread']:.1f}", f"{4 ** (s['levels'] - 1)}", "scale")
    chk("mode C pole error", s["C_pole_error"] < 0.05,
        f"{s['C_pole_error']:.3e}", "<5%", "scale")
    chk("mode gm pole error", s["gm_pole_error"] < 0.05,
        f"{s['gm_pole_error']:.3e}", "<5%", "scale")
    chk("mode C model time", s["C_model_time_error"] < 1e-12,
        f"{s['C_model_time_error']:.3e}", "<1e-12", "scale")
    chk("mode gm model time", s["gm_model_time_error"] < 1e-12,
        f"{s['gm_model_time_error']:.3e}", "<1e-12", "scale")
    chk("no-normalization fails", s["none_model_time_error"] > 0.1,
        f"{s['none_model_time_error']:.3f}", ">0.1", "scale")
    chk("mode C trades noise", abs(s["C_gm_spread"] - 1.0) < 1e-12
        and abs(s["C_noise_spread"] - 4 ** (s["levels"] - 1)) < 1e-9,
        f"gm {s['C_gm_spread']:.1f}, noise {s['C_noise_spread']:.1f}",
        "gm=1, noise=4^L", "scale")
    chk("mode gm trades g_m", abs(s["gm_gm_spread"] - 4 ** (s["levels"] - 1)) < 1e-9
        and abs(s["gm_noise_spread"] - 1.0) < 1e-12,
        f"gm {s['gm_gm_spread']:.1f}, noise {s['gm_noise_spread']:.1f}",
        "gm=4^L, noise=1", "scale")

    g = CASES["gram"]
    emit()
    emit("4.5E Gram calibration")
    chk("Haar isometry before", g["haar_isometry_before"] > 5e-3,
        f"{g['haar_isometry_before']:.3e}", ">0.5%", "gram")
    chk("Haar isometry after", g["haar_isometry_after"] < 5e-3,
        f"{g['haar_isometry_after']:.3e}", "<0.5%", "gram")
    chk("Haar PR after", g["haar_pr_after"] < 2e-3,
        f"{g['haar_pr_after']:.3e}", "<0.2%", "gram")
    chk("Haar noise gain", g["haar_noise_gain"] < 1.05,
        f"{g['haar_noise_gain']:.4f}", "<1.05", "gram")
    chk("Haar adjoint after", g["haar_adjoint_after"] < 5e-3,
        f"{g['haar_adjoint_after']:.3e}", "<0.5%", "gram")
    chk("naive left fails", g["naive_left_isometry"] > 1e-2,
        f"{g['naive_left_isometry']:.3f}", ">0.01", "gram")
    chk("left calibration after", g["haar_left_isometry_after"] < 5e-3,
        f"{g['haar_left_isometry_after']:.3e}", "<0.5%", "gram")
    chk("CDF adjoint reduced", g["cdf_adjoint_after"] < g["cdf_adjoint_before"] / 10.0,
        f"{g['cdf_adjoint_before']:.3e} -> {g['cdf_adjoint_after']:.3e}",
        ">10x", "gram")

    return checks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    hr("Phase 4.5 — adaptive representation / TransformContract")

    CASES["weno"] = case_weno()
    hr("4.5A WENO 三算子：sensor → σ → face flux")
    table(["quantity", "value"], [
        ["N / steps / dt", f"{CASES['weno']['n']} / "
                           f"{CASES['weno']['nsteps']} / "
                           f"{CASES['weno']['dt']:.2e}"],
        ["σ at static shock", f"{CASES['weno']['shock_sigma_static']:.4f}"],
        ["σ smooth (static)", f"{CASES['weno']['smooth_sigma_static']:.4f}"],
        ["σ at final shock", f"{CASES['weno']['shock_sigma_final']:.4f}"],
        ["σ smooth (final)", f"{CASES['weno']['smooth_sigma_final']:.4f}"],
        ["conservation drift", f"{CASES['weno']['conservation_drift']:.3e}"],
        ["shock x (adaptive)", f"{CASES['weno']['shock_x_ad']:.4f}"],
        ["shock x (WENO5)", f"{CASES['weno']['shock_x_ref']:.4f}"],
        ["L2 vs WENO5", f"{CASES['weno']['l2_vs_weno5']:.3e}"],
        ["max cell crossings", f"{CASES['weno']['max_cell_crossings']}"],
    ])

    CASES["lifting"] = case_lifting()
    hr("4.5B lifting wavelet：W / W_inv / W_star")
    table(["quantity", "value"], [
        ["Haar isometry", f"{CASES['lifting']['haar_isometry']:.3e}"],
        ["Haar PR", f"{CASES['lifting']['haar_pr']:.3e}"],
        ["Haar adjoint dot", f"{CASES['lifting']['haar_dot']:.3e}"],
        ["Haar noise gain", f"{CASES['lifting']['haar_noise_gain']:.4f}"],
        ["CDF53 isometry", f"{CASES['lifting']['cdf_isometry']:.3e}"],
        ["CDF53 PR", f"{CASES['lifting']['cdf_pr']:.3e}"],
        ["CDF53 W_inv != W_star", f"{CASES['lifting']['cdf_adjoint_error']:.3e}"],
        ["stationary translation", f"{CASES['lifting']['stationary_translation_error']:.3e}"],
        ["stationary inverse", f"{CASES['lifting']['stationary_inverse_error']:.3e}"],
    ])

    CASES["adjoint"] = case_adjoint()
    hr("4.5C transposeable edge / adjoint")
    table(["quantity", "value"], [
        ["A↔Aᵀ identity", f"{CASES['adjoint']['transpose_identity_error']:.3e}"],
        ["quadratic grad vs FD", f"{CASES['adjoint']['quadratic_grad_fd_error']:.3e}"],
        ["adjoint ODE grad vs FD", f"{CASES['adjoint']['adjoint_ode_fd_error']:.3e}"],
        ["checkpoint bytes", f"{CASES['adjoint']['checkpoint_bytes']}"],
    ])

    CASES["scale"] = case_scale()
    hr("4.5D scale-dependent poles：mode C vs mode g_m")
    table(["mode", "pole spread", "gm spread", "noise spread",
           "pole err", "model-time err"], [
        ["C", f"{CASES['scale']['C_pole_spread']:.1f}",
         f"{CASES['scale']['C_gm_spread']:.1f}",
         f"{CASES['scale']['C_noise_spread']:.1f}",
         f"{CASES['scale']['C_pole_error']:.2e}",
         f"{CASES['scale']['C_model_time_error']:.2e}"],
        ["gm", f"{CASES['scale']['gm_pole_spread']:.1f}",
         f"{CASES['scale']['gm_gm_spread']:.1f}",
         f"{CASES['scale']['gm_noise_spread']:.1f}",
         f"{CASES['scale']['gm_pole_error']:.2e}",
         f"{CASES['scale']['gm_model_time_error']:.2e}"],
        ["none", "-", "1.0", "1.0", "-",
         f"{CASES['scale']['none_model_time_error']:.2e}"],
    ])

    CASES["gram"] = case_gram()
    hr("4.5E Gram calibration：G_T = T̃*T̃")
    table(["quantity", "Haar", "CDF53"], [
        ["isometry before", f"{CASES['gram']['haar_isometry_before']:.3e}",
         f"{CASES['gram']['cdf_isometry_before']:.3e}"],
        ["isometry after", f"{CASES['gram']['haar_isometry_after']:.3e}",
         f"{CASES['gram']['cdf_isometry_after']:.3e}"],
        ["PR after", f"{CASES['gram']['haar_pr_after']:.3e}",
         f"{CASES['gram']['cdf_pr_after']:.3e}"],
        ["adjoint before", f"{CASES['gram']['haar_adjoint_before']:.3e}",
         f"{CASES['gram']['cdf_adjoint_before']:.3e}"],
        ["adjoint after", f"{CASES['gram']['haar_adjoint_after']:.3e}",
         f"{CASES['gram']['cdf_adjoint_after']:.3e}"],
        ["naive left isometry", f"{CASES['gram']['naive_left_isometry']:.3f}",
         "-"],
        ["left calibration after", f"{CASES['gram']['haar_left_isometry_after']:.3e}",
         "-"],
        ["cond(G_T)", f"{CASES['gram']['haar_cond']:.3f}",
         f"{CASES['gram']['cdf_cond']:.3f}"],
        ["noise gain", f"{CASES['gram']['haar_noise_gain']:.4f}",
         f"{CASES['gram']['cdf_noise_gain']:.4f}"],
    ])

    hr("ACCEPTANCE")
    checks = acceptance()
    passed = sum(c["ok"] for c in checks)
    emit()
    emit(f"RESULT: {passed}/{len(checks)} checks passed")

    os.makedirs(OUT, exist_ok=True)
    payload = dict(
        cases=CASES,
        checks=checks,
        passed=passed,
        total=len(checks),
    )
    with open(os.path.join(OUT, "transform.json"), "w") as f:
        json.dump(_jsonable(payload), f, indent=2)
    with open(os.path.join(OUT, "transform_report.txt"), "w") as f:
        f.write("\n".join(_LINES) + "\n")

    emit()
    emit(f"wrote {os.path.abspath(os.path.join(OUT, 'transform.json'))}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
