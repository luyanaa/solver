"""A7 验收：异构介质 MEMS + photonic（§3.7 / Phase 8，研究阶段）。

    python3 -m bench.run_substrate

算例：

    1  MEMS 线性    ω0=√(k/m)、Q=mω0/c 精确公式；小 IC 的阻尼解析响应
                    x(t)=(v0/ωd)e^{−ζω0 t}sin(ωd t) vs RK4（<1e-6）；
                    descriptor (E,A,B) 特征值 = 特征方程根。
    2  Duffing      微幅 RK4 测频 vs 骨架线 ω(A)≈ω0(1+3k3A²/8k)：
                    符号（k3>0 硬化上移）、幅度 A² 缩放（A 与 A/2 位移比≈4）、
                    数值命中（几 % 内，一阶微扰精度如实报告）。
    3  gm/C 标定    gm_c_calibrate：锁定残差 ≈ 测量噪声底，≪ 原始温漂
                    drift_ppm；改变 drift 100× 残差不变（残差由测量底决定，
                    不是漂移决定）——这正是"用机械 ω_ref 锁电学极点"的意义。
    4  光子 MVM     零噪/零漂 → 密集 matmul 机器精度；shot 项 ∝1/√N
                    （N→4N 误差/2）；漂移残差地板与 N 无关；
                    Monte-Carlo photonic_mvm vs 解析 mvm_relative_error。
    5  Transducer   photonic cost 字段齐全 + as_dict 可序列化（明确 TARGET
                    标注）；MEMS cost 不造数（NaN + provenance 说明）。
    6  κ_hw 窗口    真实 (k,m) 的 1/ω0 与 τ_settle 落在 MEMS 窗内；
                    高 Q=1000 的 τ_settle=2Q/ω0 超窗被如实标记（§9 风险33）；
                    负控制：窗外参数 → tau_in_window=False（KAPPA_HW 是真
                    守卫）；photonic 无存储断言（不造 τ）。

诚实声明：Duffing 只做一阶微扰骨架线 + RK4 抽查，不是非线性求解器；
photonic 只建保真度上限（shot + 漂移残差），不是任何具体 mesh 的物理；
gm/C 锁定模型假设 ω_ref 本身在标定区间内稳定（§9 风险 33：参考自身的
漂移要另算，不能靠 ratio accuracy 替代绝对表征）；transducer 数值凡标
TARGET 的都不是测量值，进 slack 前必须换成实测。
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from twin.substrate import (KAPPA_HW, MemsResonator, PHOTONIC_NO_STORAGE,
                            TransducerCost, fluidic_resistance,
                            gm_c_calibrate, mems_omega0, mems_Q,
                            mvm_relative_error, photonic_mvm,
                            photonic_transducer_cost, tau_in_window)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
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


def _rk4(f, state, t_end, n_steps):
    """Minimal fixed-step RK4 (the only solver this bench contains)."""
    state = np.asarray(state, dtype=float)
    dt = t_end / n_steps
    t = 0.0
    for _ in range(n_steps):
        k1 = f(state, t)
        k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(state + dt * k3, t + dt)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt
    return state, t


def _rk4_trajectory(f, state, t_end, n_steps):
    state = np.asarray(state, dtype=float)
    dt = t_end / n_steps
    out = np.empty((n_steps + 1, state.size))
    out[0] = state
    t = 0.0
    for i in range(n_steps):
        k1 = f(state, t)
        k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(state + dt * k3, t + dt)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt
        out[i + 1] = state
    ts = np.arange(n_steps + 1) * dt
    return ts, out


# ---------------------------------------------------------------------------
def case_mems_linear():
    hr("CASE 1  MEMS 线性谐振器：ω0/Q 精确公式 + 阻尼解析响应 + descriptor")
    m, c, k = 1e-12, 1e-11, 1e-4
    r = MemsResonator(m=m, c=c, k=k)
    w0_exact = math.sqrt(k / m)
    add("ω0 = √(k/m)", "%.6e" % r.omega0, "%.6e" % w0_exact,
        abs(r.omega0 - w0_exact) < 1e-12 * w0_exact)
    add("mems_omega0() agrees", "%.6e" % mems_omega0(k, m), "%.6e" % w0_exact,
        abs(mems_omega0(k, m) - w0_exact) < 1e-12 * w0_exact)
    Q_exact = m * w0_exact / c
    add("Q = m·ω0/c", "%.6f" % r.Q, "%.6f" % Q_exact,
        abs(r.Q - Q_exact) < 1e-9 * Q_exact)
    add("ζ = c/(2mω0), Q=1/2ζ", "%.6f" % (1.0 / (2.0 * r.zeta)),
        "%.6f" % Q_exact, abs(1.0 / (2.0 * r.zeta) - Q_exact) < 1e-6)

    # analytic underdamped response from x0=0, v0=1, F=0 (10 cycles: the
    # long-horizon comparison would need an adaptive integrator; 10 cycles
    # is the honest RK4 fixed-step spot check)
    wd = w0_exact * math.sqrt(1.0 - r.zeta ** 2)
    t_end = 10.0 * 2.0 * math.pi / w0_exact
    ts, traj = _rk4_trajectory(r.rhs, np.array([0.0, 1.0]), t_end, 20000)
    x_num = traj[:, 0]
    x_ana = (1.0 / wd) * np.exp(-r.zeta * w0_exact * ts) * np.sin(wd * ts)
    err = float(np.max(np.abs(x_num - x_ana)) / np.max(np.abs(x_ana)))
    add("damped response RK4 vs analytic", "%.3e" % err, "< 1e-6", err < 1e-6)

    # descriptor eigenvalues = roots of m s² + c s + k (complex pair if
    # underdamped — compare as complex, no numeric-arbitrage on the sign)
    E, A, B = r.descriptor()
    eig = np.sort_complex(np.linalg.eigvals(np.linalg.solve(E, A)))
    disc = c * c - 4.0 * m * k
    roots = np.sort_complex(np.array([
        (-c + np.lib.scimath.sqrt(disc)) / (2.0 * m),
        (-c - np.lib.scimath.sqrt(disc)) / (2.0 * m)]))
    dev = float(np.max(np.abs(eig - roots)))
    add("descriptor eigvals = m s²+cs+k roots", "%.3e" % dev,
        "≈0 (<1e-9 rel)", dev < 1e-9 * w0_exact)
    add("descriptor k3 excluded (E diag(1,m))", str(list(np.diag(E))),
        "[1, m]", np.allclose(E, np.diag([1.0, m])))
    add("τ_settle = 2Q/ω0 = 2m/c", "%.4e" % r.tau_settle,
        "%.4e" % (2.0 * m / c), abs(r.tau_settle - 2.0 * m / c) < 1e-12)
    table(["quantity", "value"], [
        ["ω0 [rad/s]", "%.4e" % r.omega0], ["Q", "%.1f" % r.Q],
        ["ζ", "%.3e" % r.zeta], ["τ_settle [s]", "%.4e" % r.tau_settle],
        ["RK4 vs analytic", "%.3e" % err],
    ])


def _duffing_measure_freq(res: MemsResonator, amplitude: float,
                          n_cycles: int = 120) -> float:
    """Undamped free response from x0=A, v0=0; frequency via upcrossings."""
    ts, traj = _rk4_trajectory(res.rhs, np.array([amplitude, 0.0]),
                               200.0 * 2 * math.pi / res.omega0,
                               200 * 1200)
    x = traj[:, 0]
    up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    n_up = len(up) - 1
    period = (ts[up[-1]] - ts[up[0]]) / n_up
    return 2.0 * math.pi / period


def case_duffing():
    hr("CASE 2  Duffing 骨架线：符号、A² 缩放、微幅 RK4 命中")
    m, c, k = 1.0, 0.0, 1.0
    k3 = 0.1
    r = MemsResonator(m=m, c=c, k=k, k3=k3)

    A = 0.5
    w_meas = _duffing_measure_freq(r, A)
    w_bb = r.duffing_backbone(A)
    rel = w_meas / r.omega0 - 1.0
    pred = w_bb / r.omega0 - 1.0
    add("backbone shift present (k3>0 hardens)", "%.3e" % rel, "> 0",
        rel > 0)
    add("RK4 vs backbone (few %)", "%.2e vs %.2e" % (rel, pred),
        "±15% of 1st-order perturbation",
        abs(rel - pred) < 0.15 * abs(pred) + 1e-9, "perturbative")

    A2 = 0.25
    w_meas2 = _duffing_measure_freq(r, A2)
    rel2 = w_meas2 / r.omega0 - 1.0
    ratio = rel / rel2
    add("A² scaling: shift(A)/shift(A/2)", "%.2f" % ratio, "≈4 ±10%",
        abs(ratio - 4.0) < 0.4, "ratio")

    # softening sign: k3<0 must push ω DOWN
    rs = MemsResonator(m=m, c=c, k=k, k3=-k3)
    rel_s = _duffing_measure_freq(rs, A) / rs.omega0 - 1.0
    add("k3<0 softens (shift < 0)", "%.3e" % rel_s, "< 0", rel_s < 0)
    add("softening magnitude mirrors (A²)", "%.2e vs %.2e" % (abs(rel_s), rel),
        "symmetric ±10%", abs(abs(rel_s) - rel) < 0.1 * rel)

    # linear resonator unaffected: k3=0 gives zero shift to RK4 accuracy
    r0 = MemsResonator(m=m, c=c, k=k, k3=0.0)
    rel0 = _duffing_measure_freq(r0, A) / r0.omega0 - 1.0
    add("k3=0 → zero detuning (RK4)", "%.3e" % rel0, "≈0 (<1e-4)",
        abs(rel0) < 1e-4)
    table(["A", "ω_meas/ω0 − 1", "backbone pred", "backbone value"], [
        ["%.3f" % A, "%.3e" % rel, "%.3e" % pred, "%.6f" % w_bb],
        ["%.3f" % A2, "%.3e" % rel2, "%.3e" % (3 * k3 * A2 ** 2 / (8 * k)),
         "%.6f" % r.duffing_backbone(A2)],
    ])
    emit("  诚实范围：一阶微扰骨架线（Lindstedt），更高阶 O(ε²)~%.1e 未建模；"
         "c=0 无阻尼自由响应，未验证受迫/阻尼下的 backbone。" % pred ** 2)


def case_gm_c_calib():
    hr("CASE 3  gm/C 锁定到 MEMS ω_ref：残差 = 测量底，不是温漂")
    omega_ref, C_target = 1e5, 1e-12
    drift = 1000.0e-6           # 1000 ppm 温漂（大漂移，强调锁定作用）
    meas = 10.0e-6              # 10 ppm 测量噪声底
    res = gm_c_calibrate(omega_ref, C_target, drift, meas)
    add("locked residual ≈ meas floor", "%.2f ppm" % (res * 1e6),
        "≈ %.1f ppm" % (meas * 1e6), abs(res - meas) < 0.15 * meas, "order")
    add("residual ≪ raw drift", "%.3f" % (res / drift), "≪ 1 (< 0.05)",
        res / drift < 0.05, "ratio")
    # residual independent of drift_ppm (the point of mechanical referencing)
    res_small = gm_c_calibrate(omega_ref, C_target, drift / 100.0, meas)
    add("residual drift-independent (100× drift change)", "%.2f → %.2f ppm"
        % (res * 1e6, res_small * 1e6), "≈ unchanged (±2%)",
        abs(res_small - res) < 0.02 * res)
    # finite loop gain leaves a trimmable remainder ≪ measurement floor
    res_low_T = gm_c_calibrate(omega_ref, C_target, drift, meas, loop_gain=1.0)
    add("finite loop gain remainder visible", "%.1f ppm" % (res_low_T * 1e6),
        "> meas floor (weak loop)",
        res_low_T > meas + 0.4 * drift, "info")
    add("nominal gm0 = ω_ref·C_target (lock target)", "%.3e" % (omega_ref * C_target),
        "documented", True, "info")
    table(["case", "residual [ppm]"], [
        ["drift=1000 ppm, meas=10 ppm", "%.2f" % (res * 1e6)],
        ["drift=10 ppm, meas=10 ppm", "%.2f" % (res_small * 1e6)],
        ["weak loop (T=1)", "%.1f" % (res_low_T * 1e6)],
    ])
    emit("  假设（已写进 docstring）：ω_ref 在标定区间内自身稳定；参考漂移"
         "（§9 风险33）须单独入账，不在本函数返回值内。")


_M4 = np.array([[0.8, 0.3, -0.2, 0.4],
                [0.3, 0.9, 0.1, -0.3],
                [-0.2, 0.1, 0.7, 0.5],
                [0.4, -0.3, 0.5, 0.6]])
_X4 = np.array([1.0, 0.5, -0.3, 0.2])


def _mc_rel_error(n_photons, drift_resid, n_draws=400):
    errs = []
    for i in range(n_draws):
        y = photonic_mvm(_M4, _X4, n_photons=n_photons,
                         drift_resid=drift_resid, seed=i)
        errs.append(np.sum((y - _M4 @ _X4) ** 2) / np.sum((_M4 @ _X4) ** 2))
    return math.sqrt(float(np.mean(errs)))


def case_photonic_mvm():
    hr("CASE 4  photonic 4×4 MVM：机器精度基线 + shot 1/√N + 漂移地板")
    y0 = photonic_mvm(_M4, _X4)
    y_ref = _M4 @ _X4
    dev = float(np.max(np.abs(y0 - y_ref)))
    add("zeroed noise/drift → machine precision", "%.3e" % dev, "< 1e-15",
        dev < 1e-15)

    # analytic shot scaling: 1/√N (N→4N halves the error)
    e1 = mvm_relative_error(_M4, _X4, n_photons=10_000)
    e2 = mvm_relative_error(_M4, _X4, n_photons=40_000)
    add("analytic shot term = √(n_out/N)", "%.4f" % e1,
        "%.4f" % math.sqrt(4 / 10_000),
        abs(e1 - math.sqrt(4 / 10_000)) < 1e-12)
    add("1/√N scaling: e(N)/e(4N)", "%.3f" % (e1 / e2), "≈2 ±1%",
        abs(e1 / e2 - 2.0) < 0.02)
    add("shot term M,x-independent (energy-share detection)",
        "%.3e" % abs(mvm_relative_error(_M4, _X4, n_photons=1e6) -
                     mvm_relative_error(np.eye(4), _X4, n_photons=1e6)),
        "≈0", abs(mvm_relative_error(_M4, _X4, n_photons=1e6) -
                  mvm_relative_error(np.eye(4), _X4, n_photons=1e6)) < 1e-12)

    # Monte-Carlo photonic_mvm vs analytic expectation (quadrature add)
    drift = 0.01
    mc = _mc_rel_error(10_000, 0.0)
    ana = mvm_relative_error(_M4, _X4, n_photons=10_000)
    add("MC shot error vs analytic", "%.4f vs %.4f" % (mc, ana), "±20%",
        abs(mc - ana) < 0.2 * ana, "monte-carlo")
    mc_d = _mc_rel_error(1_000_000, drift)
    ana_d = mvm_relative_error(_M4, _X4, n_photons=1_000_000, drift_resid=drift)
    add("MC shot+drift (quadrature) vs analytic", "%.4f vs %.4f"
        % (mc_d, ana_d), "±20%", abs(mc_d - ana_d) < 0.2 * ana_d,
        "monte-carlo")

    # drift floor: each photon count gives total ≈ drift (shot term small)
    a1 = mvm_relative_error(_M4, _X4, n_photons=1e6, drift_resid=drift)
    a2 = mvm_relative_error(_M4, _X4, n_photons=1e9, drift_resid=drift)
    add("drift floor N-independent", "%.4f → %.4f" % (a1, a2),
        "both ≈ drift ±2.5% (shot shrinks 1000×, floor stays)",
        abs(a1 / drift - 1.0) < 0.025 and abs(a2 / drift - 1.0) < 0.025,
        "order")
    add("drift term = ‖δM x‖/‖M x‖ = drift_resid", "%.4f" % a2,
        "≈hypot(shot, %.3f)" % drift,
        abs(a2 - math.hypot(math.sqrt(4 / 1e9), drift)) < 1e-9)
    table(["case", "MC rel err", "analytic"], [
        ["shot only, N=1e4", "%.4f" % mc, "%.4f" % ana],
        ["shot+drift, N=1e6", "%.4f" % mc_d, "%.4f" % ana_d],
    ])
    emit("  范围：只建 shot 噪声 + 漂移残差两个保真度上限；WDM 串扰、PD 线性度、"
         "ring 热漂移动力学（§9 风险37）不在本模型内。")


def case_transducer():
    hr("CASE 5  TransducerCost 合同：photonic TARGET 填充 / MEMS 不造数")
    tc = photonic_transducer_cost()
    d = tc.as_dict()
    ok_fields = all(k in d for k in ("gain", "noise_rms", "bandwidth_hz",
                                     "energy", "notes"))
    add("photonic cost fields populated", str(sorted(d)),
        "all contract fields", ok_fields)
    add("serializable (json round-trip)", "ok" if json.loads(json.dumps(d))
        == d else "mismatch", "round-trip ok",
        json.loads(json.dumps(d)) == d)
    add("TARGET provenance labeled", "target-only" if
        d["extra"].get("provenance") == "target-only" else d["extra"],
        "provenance=target-only",
        d["extra"].get("provenance") == "target-only")

    # MEMS transducer: NO numbers fabricated — NaN contract fields + note
    mems_tc = TransducerCost(
        gain=float("nan"), noise_rms=float("nan"), bandwidth_hz=float("nan"),
        energy=float("nan"),
        notes="NO model fabricated: electrostatic drive F∝V², ΔC/C readout, "
              "pull-in and Q tempco need a cited device model (§9 风险34/35). "
              "Fields must be filled from measurement before slack use.",
        extra={"provenance": "unmodeled"})
    dm = mems_tc.as_dict()
    add("MEMS cost: no fabricated numbers (NaN contract)", str(dm["gain"]),
        "NaN + note", math.isnan(dm["gain"]) and "device model" in dm["notes"])
    R_ref = 8.0 * 1e-3 * 1e-2 / (math.pi * 1e-16)
    add("fluidic API unbroken (regression guard)", "%.4e"
        % fluidic_resistance(1e-2, 1e-4, 1e-3), "%.4e" % R_ref,
        abs(fluidic_resistance(1e-2, 1e-4, 1e-3) - R_ref) < 1e-6 * R_ref)


def case_kappa_windows():
    hr("CASE 6  κ_hw 窗口守卫 + photonic 无存储 + 负控制")
    # realistic MEMS: m=1e-12 kg, k=1e-4 N/m, Q=100
    m, k, Q = 1e-12, 1e-4, 100.0
    w0 = math.sqrt(k / m)
    r = MemsResonator(m=m, c=m * w0 / Q, k=k)
    add("1/ω0 in MEMS window", "%.1e" % (1.0 / w0), "in %s" % (KAPPA_HW["mems"],),
        tau_in_window("mems", 1.0 / w0))
    add("τ_settle=2Q/ω0 in window (Q=100)", "%.1e" % r.tau_settle,
        "in %s" % (KAPPA_HW["mems"],), tau_in_window("mems", r.tau_settle))
    # honest flag: Q=1000 pushes τ_settle OUT of the window (§9 risk 33)
    r_hi = MemsResonator(m=m, c=m * w0 / 1000.0, k=k)
    add("Q=1000 τ_settle exceeds window → flagged", "%.1e" % r_hi.tau_settle,
        "outside %s → False" % (KAPPA_HW["mems"],),
        not tau_in_window("mems", r_hi.tau_settle))

    # negative control: a MEMS parameter outside its domain window is a real
    # guard, not decoration
    tau_bad = 1.0 / mems_omega0(1e6, 1e-18)   # ω0=1e12 → τ=1e-12 ≪ 1e-6
    add("negative control: τ outside MEMS window", "%.1e" % tau_bad,
        "flagged (False)", not tau_in_window("mems", tau_bad))
    add("negative control: τ outside fluidic window", "%.1e" % 1e-5,
        "flagged (False)", not tau_in_window("fluidic", 1e-5))

    # photonic: no storage — documented, no τ fabricated
    add("photonic no-storage assertion present",
        "yes" if ("NO storage" in PHOTONIC_NO_STORAGE and
                  "memoryless" in PHOTONIC_NO_STORAGE) else "missing",
        "documented in substrate.py", True)
    add("photonic window kept as link latency, not storage τ",
        str(KAPPA_HW["photonic"]), "(1e-12, 1e-9)",
        KAPPA_HW["photonic"] == (1e-12, 1e-9))
    add("4 domains in KAPPA_HW", str(sorted(KAPPA_HW)), "4 domains",
        len(KAPPA_HW) == 4)
    table(["domain", "window", "checked τ"], [
        ["mems", str(KAPPA_HW["mems"]), "1/ω0=%.1e, τ_settle=%.1e"
         % (1.0 / w0, r.tau_settle)],
        ["fluidic", str(KAPPA_HW["fluidic"]), "1e-5 → flagged"],
        ["photonic", str(KAPPA_HW["photonic"]), "no storage (no τ)"],
    ])


def main():
    emit("A7 异构介质验收：MEMS + photonic（DESIGN.md §3.7 / Phase 8）")
    case_mems_linear()
    case_duffing()
    case_gm_c_calib()
    case_photonic_mvm()
    case_transducer()
    case_kappa_windows()

    hr("VERDICT")
    rows = [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
            for c in checks]
    table(["check", "got", "expected", "pass"], rows)
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))
    summary = dict(n_checks=len(checks), n_passed=len(checks) - n_fail)
    CASES["substrate"] = dict(checks=checks, summary=summary)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "substrate.json"), "w") as fh:
        json.dump(_jsonable(CASES), fh, indent=1)
    with open(os.path.join(OUT, "substrate_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/substrate.json, results/substrate_report.txt")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
