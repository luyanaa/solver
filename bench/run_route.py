"""Phase 1.6 验收：route 是 transfer function，不是长度（DESIGN.md §3.2、§6.15）。

    python3 -m bench.run_route

七组算例：

    wire         distributed vs lumped：同 DC 斜率，不同 t50 / f_3dB
    buffers      fanout 深度同时受相位预算与 N_b σ_b² 限制
    all-pass     pad 延迟不能用 RC：同 DC delay，500 MHz 处 RC 已衰减 15%
    deskew       group delay 对齐后 skew→0，但绝对延迟不变
    feedback     零 skew ≠ 零动力学误差：匹配的 150 ps 仍超 5° 预算
    crosstalk    C_m 的串扰地板，以及 1 mV/1 V 对应的 C_m 上限
    sensitivity  从 J 自动算 τ_tol：1/|J_ij| 与 2Q resonance

`route_closure` 就是 §2.8 四条 closure 的 per-edge 版本：gain、phase、
noise、feedback absolute delay。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import solve_banded

from twin.route import (AllPass, PureDelay, RouteEdge, RouteSpec, Wire,
                        buffer_chain, coupled_nets, deskew, edge_metrics,
                        group_delay, j_h, j_tau, j_tau_group, latency_class,
                        route_closure, sensitivity_summary)

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


# ---------------------------------------------------------------------------
# 测量夹具：分布式 RC 线的阶跃响应
# ---------------------------------------------------------------------------
def _diffusion_t50(tau_rc: float, n: int = 400, nsteps: int = 2000,
                   t_end: float = 1.5) -> float:
    """rc ∂v/∂t = ∂²v/∂x²，v(0)=1，∂v/∂x(1)=0；返回 v(1) 到 0.5 的时间/τ_rc。

    这是分布式 RC 线的物理模型（L→0 的 telegrapher），只用于测量
    §3.2 引用的 0.38 系数，不是求解器。
    """
    h = 1.0 / n
    dt = t_end * tau_rc / nsteps
    alpha = dt / (tau_rc * h * h)
    ab = np.zeros((3, n))
    ab[0, 1:] = -alpha
    ab[1, :] = 1.0 + 2.0 * alpha
    ab[2, :-1] = -alpha
    ab[2, -2] = -2.0 * alpha          # 开路的 ghost node
    v = np.zeros(n)
    b = np.empty(n)
    for k in range(nsteps):
        b[:] = v
        b[0] += alpha                 # v0 = 1
        vn = solve_banded((1, 1), ab, b)
        if vn[-1] >= 0.5:
            frac = (0.5 - v[-1]) / (vn[-1] - v[-1])
            return float((k + frac) * dt / tau_rc)
        v = vn
    return float("nan")


# ---------------------------------------------------------------------------
# 算例
# ---------------------------------------------------------------------------
def case_wire():
    w = Wire(2000, 0.2, 0.2)
    lattice = np.logspace(np.log10(2 * np.pi * 1e5), np.log10(2 * np.pi * 20e9), 400)
    dist = RouteEdge((w,), "distributed")
    lump = RouteEdge((Wire(2000, 0.2, 0.2, "lumped"),), "lumped")
    md = edge_metrics(dist, lattice)
    ml = edge_metrics(lump, lattice)
    CASES["wire"] = dict(
        R=w.R, C=w.C, tau_rc=w.tau_rc,
        gd_dc=float(group_delay(dist, 2 * np.pi * 1e6)),
        gd_dc_lump=float(group_delay(lump, 2 * np.pi * 1e6)),
        t50_fd=_diffusion_t50(w.tau_rc),
        t50_dist=w.delay50,
        t50_lump=float(np.log(2.0) / 2.0 * w.tau_rc),
        f3_db=md["f_3db"], f3_lump=ml["f_3db"],
    )


def case_buffers():
    om = 2 * np.pi * 100e6
    spec = RouteSpec(om, eps_gain=1e-2, eps_phase_deg=3.0, noise_budget=50e-6)
    out = {}
    for d in (1, 4):
        e = buffer_chain(d, gain=0.999, f_p=2e9, tau=0.0, sigma=10e-6,
                         name="depth%d" % d)
        c = route_closure(e, spec)
        c["eta_zero"] = edge_metrics(e, np.array([om]), tau_ideal=0.0)["eta"]
        c["eta_declared"] = edge_metrics(e, np.array([om]),
                                         tau_ideal=e.latency)["eta"]
        out[d] = c
    CASES["buffers"] = out


def case_allpass():
    tau = 200e-12
    band = np.logspace(np.log10(2 * np.pi * 1e5),
                       np.log10(2 * np.pi * 500e6), 300)
    ap = RouteEdge((AllPass(tau),), "allpass")
    rc = RouteEdge((Wire(2000, 0.2, 0.5, "lumped"),), "rc")
    ma = edge_metrics(ap, band, tau_ideal=tau)
    mr = edge_metrics(rc, band, tau_ideal=tau)
    wide = np.logspace(np.log10(2 * np.pi * 1e5), np.log10(2 * np.pi * 2e9), 300)
    rc_f3 = edge_metrics(rc, wide)["f_3db"]
    CASES["allpass"] = dict(
        tau=tau,
        ap_gain=float(ma["gain_err"].max()),
        ap_phase=float(ma["phase_err_deg"].max()),
        ap_jh=j_h(ap, band, tau_ideal=tau),
        ap_jtau=j_tau(ap, band, tau),
        ap_f3=ma["f_3db"],
        rc_gain=float(mr["gain_err"].max()),
        rc_phase=float(mr["phase_err_deg"].max()),
        rc_jh=j_h(rc, band, tau_ideal=tau),
        rc_jtau=j_tau(rc, band, tau),
        rc_f3=rc_f3,
    )


def case_deskew():
    edges = [RouteEdge((PureDelay(20e-12),), "e20"),
             RouteEdge((PureDelay(45e-12),), "e45"),
             RouteEdge((Wire(2000, 0.2, 0.15),), "wire60")]
    om0 = 2 * np.pi * 200e6
    r = deskew(edges, om0)
    band = np.logspace(np.log10(2 * np.pi * 1e5),
                       np.log10(2 * np.pi * 500e6), 300)
    spec1 = RouteSpec(2 * np.pi * 100e6, feedback_critical=True,
                      omega_c=2 * np.pi * 100e6, phi_reserve_deg=1.0)
    spec5 = RouteSpec(2 * np.pi * 100e6, feedback_critical=True,
                      omega_c=2 * np.pi * 100e6, phi_reserve_deg=5.0)
    CASES["deskew"] = dict(
        omega0=om0, target=r["target"],
        before=r["before"], after=r["after"],
        skew_before=r["skew_before"], skew_after=r["skew_after"],
        jt_before=j_tau_group(edges, band),
        jt_after=j_tau_group(r["edges"], band),
        lat_after=[e.latency for e in r["edges"]],
        closure_1deg=route_closure(r["edges"][0], spec1),
        closure_5deg=route_closure(r["edges"][0], spec5),
    )


def case_feedback():
    om_c = 2 * np.pi * 100e6
    e150 = RouteEdge((PureDelay(150e-12),), "150ps")
    e130 = RouteEdge((PureDelay(130e-12),), "130ps")
    # 每条 edge 把自己的 latency 声明进 H_ref：phase 检查放行，
    # 只有 absolute-delay 检查能区分它们。
    spec150 = RouteSpec(om_c, tau_ideal=150e-12, feedback_critical=True,
                        omega_c=om_c, phi_reserve_deg=5.0)
    spec130 = RouteSpec(om_c, tau_ideal=130e-12, feedback_critical=True,
                        omega_c=om_c, phi_reserve_deg=5.0)
    CASES["feedback"] = dict(
        tau_max=route_closure(e130, spec130)["tau_max_abs"],
        r150=route_closure(e150, spec150),
        r130=route_closure(e130, spec130),
        phase_lag_150=float(np.degrees(om_c * 150e-12)),
        phase_lag_130=float(np.degrees(om_c * 130e-12)),
        matched_skew=abs(group_delay(e150, om_c) - group_delay(e150, om_c)),
    )


def case_crosstalk():
    om = 2 * np.pi * 500e6
    wt = 1.0 / (400 * 220e-15)
    p20 = coupled_nets(400, 200, 20)
    p2 = coupled_nets(400, 200, 2)
    p01 = coupled_nets(400, 200, 0.1)
    H = p20.H(om)
    eta = float(np.linalg.svd(H - np.eye(2), compute_uv=False)[0])
    CASES["crosstalk"] = dict(
        omega_trans=wt, f_trans=wt / (2 * np.pi),
        xt_hf=p20.crosstalk_hf, xt_hf_num=float(p20.crosstalk(100 * wt)),
        xt_500m=float(p20.crosstalk(om)),
        xt_hf_2f=p2.crosstalk_hf, xt_hf_01f=p01.crosstalk_hf,
        db_20=float(20 * np.log10(p20.crosstalk_hf)),
        db_2=float(20 * np.log10(p2.crosstalk_hf)),
        eta_500m=eta,
        v20=p20.crosstalk_hf * 1.0, v2=p2.crosstalk_hf * 1.0,
        v01=p01.crosstalk_hf * 1.0,
    )


def case_sensitivity():
    a_s = 2 * np.pi * 1e6
    a_f = 2 * np.pi * 200e6
    w0 = 2 * np.pi * 50e6
    Q = 10.0
    gam = w0 / (2 * Q)
    J = np.array([[-a_s, 0.05 * a_f, 0.0, 0.0],
                  [0.0, -a_f, 0.1 * a_f, 0.0],
                  [0.0, 0.0, -gam, w0],
                  [0.0, 0.0, -w0, -gam]])
    band = np.logspace(5, np.log10(2 * np.pi * 2e9), 2000)
    s = sensitivity_summary(J, band, 1.0)
    CASES["sensitivity"] = dict(
        J=J, Q=Q, omega0=w0, a_slow=a_s, a_fast=a_f,
        tau_tol=s["tau_tol"], hf_tau_tol=s["hf_tau_tol"],
        worst_omega=s["worst_omega"],
    )


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------
def acceptance():
    checks = []

    def add(name, got, expected, passed):
        checks.append(dict(name=name, got=got, expected=expected,
                           passed=bool(passed)))

    w = CASES["wire"]
    add("wire DC group delay = RC/2", "%.2f ps" % (w["gd_dc"] * 1e12),
        "80.00 ps", abs(w["gd_dc"] / (w["tau_rc"] / 2) - 1.0) < 1e-3)
    add("wire t50 by diffusion", "%.4f RC" % w["t50_fd"], "0.3795 RC",
        abs(w["t50_fd"] / 0.3795 - 1.0) < 0.01)
    add("distributed t50 > lumped", "%.2f vs %.2f ps"
        % (w["t50_dist"] * 1e12, w["t50_lump"] * 1e12), ">",
        w["t50_dist"] > w["t50_lump"])
    add("distributed f3 > lumped f3", "%.2f vs %.2f GHz"
        % (w["f3_db"] / 2 / np.pi / 1e9, w["f3_lump"] / 2 / np.pi / 1e9),
        "f3_dist > f3_lump", w["f3_db"] > w["f3_lump"])

    b = CASES["buffers"]
    add("buffer depth1 closure", b[1]["verdict"], True, b[1]["verdict"])
    add("buffer depth4 closure", b[4]["verdict"], False, not b[4]["verdict"])
    add("depth1 phase < 3°", "%.3f°" % b[1]["max_phase_err_deg"], "< 3°",
        b[1]["max_phase_err_deg"] < 3.0)
    add("depth4 phase > 3°", "%.3f°" % b[4]["max_phase_err_deg"], "> 3°",
        b[4]["max_phase_err_deg"] > 3.0)
    add("depth4 noise", "%.1f µV" % (b[4]["noise_rms"] * 1e6), "< 50 µV",
        b[4]["noise_rms"] * 1e6 < 50.0)
    add("depth1 class", b[1]["latency_class"], "L2", b[1]["latency_class"] == "L2")
    add("depth4 class", b[4]["latency_class"], "L3", b[4]["latency_class"] == "L3")
    add("depth4 eta declared vs 0", "%.4f vs %.4f"
        % (b[4]["eta_declared"], b[4]["eta_zero"]),
        "declared < 5%, zero > 15%",
        b[4]["eta_declared"] < 0.05 and b[4]["eta_zero"] > 0.15)

    a = CASES["allpass"]
    add("all-pass gain flat", "%.2e" % a["ap_gain"], "< 1e-12",
        a["ap_gain"] < 1e-12)
    add("RC gain error at 500 MHz", "%.3f" % a["rc_gain"], "> 0.1",
        a["rc_gain"] > 0.1)
    add("all-pass phase < RC phase", "%.3f° vs %.3f°"
        % (a["ap_phase"], a["rc_phase"]), "ap < rc",
        a["ap_phase"] < a["rc_phase"])
    add("all-pass J_H / RC J_H", "%.1f×" % (a["rc_jh"] / a["ap_jh"]), "> 10×",
        a["rc_jh"] / a["ap_jh"] > 10.0)
    add("all-pass J_tau / RC J_tau", "%.1f×" % (a["rc_jtau"] / a["ap_jtau"]),
        "> 5×", a["rc_jtau"] / a["ap_jtau"] > 5.0)
    add("all-pass f3 = inf", a["ap_f3"], "inf", np.isinf(a["ap_f3"]))

    d = CASES["deskew"]
    add("deskew skew before", "%.1f ps" % (d["skew_before"] * 1e12), "> 35 ps",
        d["skew_before"] * 1e12 > 35.0)
    add("deskew skew after", "%.2e ps" % (d["skew_after"] * 1e12), "< 0.01 ps",
        d["skew_after"] * 1e12 < 0.01)
    add("deskew group spread", "%.2e → %.2e ps²"
        % (d["jt_before"] * 1e24, d["jt_after"] * 1e24), "> 5e3×",
        d["jt_before"] / d["jt_after"] > 5e3)
    add("deskew leaves absolute delay", "%.2f° @100 MHz"
        % float(np.degrees(2 * np.pi * 100e6 * d["target"])), "2.15°",
        abs(float(np.degrees(2 * np.pi * 100e6 * d["target"])) - 2.15) < 0.1)
    add("deskewed loop fails 1° reserve", d["closure_1deg"]["pass_absolute"],
        False, not d["closure_1deg"]["pass_absolute"])
    add("deskewed loop passes 5° reserve", d["closure_5deg"]["pass_absolute"],
        True, d["closure_5deg"]["pass_absolute"])

    f = CASES["feedback"]
    add("feedback τ_max", "%.1f ps" % (f["tau_max"] * 1e12), "138.9 ps",
        abs(f["tau_max"] * 1e12 - 138.9) < 0.1)
    add("150 ps fails 5°", f["r150"]["pass_absolute"], False,
        not f["r150"]["pass_absolute"])
    add("130 ps passes 5°", f["r130"]["pass_absolute"], True,
        f["r130"]["pass_absolute"])
    add("matched 150 ps still fails", "skew=%.1f ps"
        % (f["matched_skew"] * 1e12),
        "absolute fails", f["matched_skew"] < 1e-15
        and not f["r150"]["pass_absolute"])
    add("padding cannot fix feedback", f["r150"]["pad_can_fix"], False,
        not f["r150"]["pad_can_fix"])

    x = CASES["crosstalk"]
    add("crosstalk HF limit", "%.4f" % x["xt_hf_num"], "%.4f" % x["xt_hf"],
        abs(x["xt_hf_num"] / x["xt_hf"] - 1.0) < 1e-2)
    add("shield improvement", "%.2f dB" % (x["db_20"] - x["db_2"]), "> 19 dB",
        x["db_20"] - x["db_2"] > 19.0)
    add("20 fF vs 1 mV budget", "%.1f mV" % (x["v20"] * 1e3), "> 1 mV",
        x["v20"] * 1e3 > 1.0)
    add("0.1 fF vs 1 mV budget", "%.2f mV" % (x["v01"] * 1e3), "< 1 mV",
        x["v01"] * 1e3 < 1.0)

    s = CASES["sensitivity"]
    tol = s["tau_tol"]
    ratio = tol[0, 0] / tol[1, 1]
    add("slow/fast τ_tol ratio", "%.1f" % ratio, "200",
        abs(ratio - 200.0) / 200.0 < 0.02)
    res = s["hf_tau_tol"][0, 0] / tol[0, 0]
    add("resonance shrinks budget", "%.2f×" % res, "2Q = 20×",
        abs(res / 20.0 - 1.0) < 0.02)
    add("resonant cross edge", "%.2f ps" % (tol[2, 3] * 1e12),
        "< 14 ps", tol[2, 3] < tol[0, 0] / 10)
    order = [tol[1, 1], tol[1, 2], tol[0, 1], tol[0, 0]]
    add("τ_tol ordered by |J|", " < ".join("%.2f" % (v * 1e12) for v in order),
        "increasing", all(order[i] < order[i + 1] for i in range(3)))
    wf = s["worst_omega"][0, 0] / s["omega0"]
    add("worst ω at resonance", "%.3f ω0" % wf, "≈ 50 MHz",
        abs(wf - 1.0) < 0.03)

    return checks


def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()

    hr("Phase 1.6 验收：route = H(s)，不是 wire length")

    case_wire()
    case_buffers()
    case_allpass()
    case_deskew()
    case_feedback()
    case_crosstalk()
    case_sensitivity()

    hr("A. RC wire：distributed vs lumped")
    w = CASES["wire"]
    table(["quantity", "value"],
          [("R", "%.1f Ω" % w["R"]),
           ("C", "%.1f fF" % w["C"]),
           ("RC", "%.1f ps" % (w["tau_rc"] * 1e12)),
           ("DC group delay", "%.2f ps (both)" % (w["gd_dc"] * 1e12)),
           ("t50 distributed (measured)", "%.2f ps = %.4f RC"
            % (w["t50_fd"] * w["tau_rc"] * 1e12, w["t50_fd"])),
           ("t50 distributed (0.3795RC)", "%.2f ps" % (w["t50_dist"] * 1e12)),
           ("t50 lumped (ln2·RC/2)", "%.2f ps" % (w["t50_lump"] * 1e12)),
           ("f3 distributed", "%.2f GHz" % (w["f3_db"] / 2 / np.pi / 1e9)),
           ("f3 lumped", "%.2f GHz" % (w["f3_lump"] / 2 / np.pi / 1e9))])
    emit("  同一个 DC 斜率（RC/2）不等于同一个波形：lumped π 的 t50 短 8.7%、")
    emit("  −3 dB 带宽低 18%；单极点模型不能同时当 delay 和 bandwidth 的模型。")

    hr("B. Buffered fanout：深度 × 相位 × 噪声")
    b = CASES["buffers"]
    table(["depth", "latency", "class", "gain err", "phase err", "noise", "verdict"],
          [(d, "%.1f ps" % (b[d]["latency"] * 1e12), b[d]["latency_class"],
            "%.3f%%" % (b[d]["max_gain_err"] * 100),
            "%.2f°" % b[d]["max_phase_err_deg"],
            "%.1f µV" % (b[d]["noise_rms"] * 1e6),
            "pass" if b[d]["verdict"] else "FAIL") for d in (1, 4)])
    emit("  g=0.999、f_p=2 GHz、σ=10 µV、budget 3°/1%：4 级在 100 MHz 已经 11.5°，")
    emit("  而 gain 只有 0.9%。限制 fanout 深度的是相位，不是增益，也不是噪声。")
    emit("  eta（τ_ideal=0 → 声明 latency）：depth4 %.1f%% → %.2f%%"
         % (b[4]["eta_zero"] * 100, b[4]["eta_declared"] * 100))

    hr("C. 补延迟：all-pass vs RC")
    a = CASES["allpass"]
    table(["quantity", "all-pass", "RC (1 pole)"],
          [("gain error", "%.2e" % a["ap_gain"], "%.2f%%" % (a["rc_gain"] * 100)),
           ("phase error @500 MHz", "%.3f°" % a["ap_phase"], "%.3f°" % a["rc_phase"]),
           ("J_H", "%.2e" % a["ap_jh"], "%.2e" % a["rc_jh"]),
           ("J_τ", "%.1f ps²" % (a["ap_jtau"] * 1e24), "%.1f ps²" % (a["rc_jtau"] * 1e24)),
           ("f_3dB", "inf", "%.2f GHz" % (a["rc_f3"] / 2 / np.pi / 1e9))])
    emit("  τ=200 ps：all-pass 的幅度严格为 1，RC 在 500 MHz 已 −1.4 dB。")
    emit("  用 RC 补 delay 等于用幅度换相位；J_H 差 %.0f×。" % (a["rc_jh"] / a["ap_jh"]))

    hr("D. Deskew：把 skew 修到零，绝对延迟不动")
    d = CASES["deskew"]
    table(["quantity", "value"],
          [("ω0", "%.0f MHz" % (d["omega0"] / 2 / np.pi / 1e6)),
           ("before / after skew", "%.1f ps → %.2e ps"
            % (d["skew_before"] * 1e12, d["skew_after"] * 1e12)),
           ("group spread", "%.1f → %.2e ps²"
            % (d["jt_before"] * 1e24, d["jt_after"] * 1e24)),
           ("common latency", "%.2f ps" % (d["target"] * 1e12)),
           ("loop phase @100 MHz", "%.2f°"
            % float(np.degrees(2 * np.pi * 100e6 * d["target"]))),
           ("1° reserve (27.8 ps)", "FAIL"),
           ("5° reserve (138.9 ps)", "pass")])
    emit("  对齐后三条 edge 都是 ~60 ps：skew=0，但闭环仍看到 2.15°。")
    emit("  所以 deskew 修的是 matched，不是 instantaneous（§2.8）。")

    hr("E. Feedback：绝对延迟预算")
    f = CASES["feedback"]
    table(["edge", "phase lag @100 MHz", "τ_max", "verdict"],
          [("130 ps", "%.2f°" % f["phase_lag_130"],
            "%.1f ps" % (f["tau_max"] * 1e12), "pass"),
           ("150 ps", "%.2f°" % f["phase_lag_150"],
            "%.1f ps" % (f["tau_max"] * 1e12), "FAIL")])
    emit("  两条 150 ps 的路径 skew=0；它们仍然同时违反 5° 预算。")
    emit("  feedback_critical 的 edge 不能靠 pad 解决：pad_can_fix=False。")

    hr("F. Crosstalk：C_m 的地板")
    x = CASES["crosstalk"]
    table(["C_m", "HF crosstalk", "dB", "1 V → victim"],
          [("20 fF", "%.4f" % x["xt_hf"], "%.2f" % x["db_20"],
            "%.1f mV" % (x["v20"] * 1e3)),
           ("2 fF", "%.4f" % x["xt_hf_2f"], "%.2f" % x["db_2"],
            "%.2f mV" % (x["v2"] * 1e3)),
           ("0.1 fF", "%.5f" % x["xt_hf_01f"],
            "%.1f" % (20 * np.log10(x["xt_hf_01f"])),
            "%.2f mV" % (x["v01"] * 1e3))])
    emit("  高频极限 H_12/H_11 = C_m/(C_2+C_m)=%.4f；transition 在 %.1f GHz。"
         % (x["xt_hf"], x["f_trans"] / 1e9))
    emit("  1 mV/1 V 的预算要求 C_m < 0.11 fF；20 dB 屏蔽不够，要 40 dB 或差分。")
    emit("  500 MHz 处 2×2 η（σ_max(H−I)）= %.3f。" % x["eta_500m"])

    hr("G. 从 J 自动算 τ_tol")
    s = CASES["sensitivity"]
    tol = s["tau_tol"]
    names = {(0, 0): "slow self", (1, 1): "fast self", (2, 3): "osc cross",
             (1, 2): "fast←osc", (0, 1): "slow←fast"}
    table(["edge", "|J_ij|", "τ_tol", "worst ω", "HF limit"],
          [(names[ij], "%.2e" % abs(s["J"][ij]), "%.2f ps" % (tol[ij] * 1e12),
            "%.1f MHz" % (s["worst_omega"][ij] / 2 / np.pi / 1e6),
            "%.1f ps" % (s["hf_tau_tol"][ij] * 1e12)) for ij in names])
    emit("  高频极限 τ_tol = ε/|J_ij|：慢模态 2.8 ns，快模态 13.9 ps，差 200×。")
    emit("  Q=10 的 50 MHz 共振把每条 edge 的预算再压 2Q=20×；排序不变。")
    emit("  这就是 sensitivity-driven routing 的输入：不需要手写 critical path。")

    hr("H. 验收")
    checks = acceptance()
    table(["check", "got", "expected", "PASS"],
          [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
           for c in checks])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    payload = dict(cases=CASES, checks=checks,
                   summary=dict(n_checks=len(checks),
                                n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "route.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(os.path.join(OUT, "route_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/route.json")
    emit("  report → results/route_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
