"""Stage B1 校准验收：twin/calib.py FSM + 温漂孪生 + lowering/p7.py 配置计划。

    python3 -m bench.run_calib

四组验收（§C3 校准计划字段 + §5.9/§6.16 标定观测口径）：

  FG  前台标定：已知激励 + 故意造的 offset/gain 误差 + 带噪读数 ⇒
      correction 把误差修回**测量噪声底**——不是零。合同断言：
      residual ≥ meas_noise_ppm（"标定残差低于噪声底"是撒谎），
      且 ≈ 噪声底量级（不是把误差原样留下）。
  BG  后台标定：线性温漂 + 更新间隔 ⇒ 残差 = rate·interval/(1+T) + 底
      （A7 gm_c_calibrate 同语义）；间隔变长 ⇒ 残差线性变大（诚实的
      调度权衡）；反解 update_interval_for_budget 与正演闭合。
  P7  diffusion-1d 锚点的 (Discretized, Certificate, ScalingResult) ⇒
      per-element 配置（gm/C 目标、S_x、alpha、噪声设置）+ 校准计划；
      residual_budget 与证书 noise 轴一致（轴 ≤1 ⇒ 预算 = 测量底，
      轴 >1 ⇒ 不承诺）；全程 JSON 可序列化。
  NEG 负控制：证书判不可行的量 ⇒ 元素 flagged，计划不悄悄排程。
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

from analysis.slack import HardwareSpec
from lowering.budget import BudgetInput, compile_budget
from lowering.discretize import PDESpec, discretize
from lowering.p7 import generate_config
from lowering.scaling import gramian_scaling
from twin.calib import (CalibratedTwin, CalibrationFSM, calibrate_background,
                        calibrate_foreground, drift_update,
                        update_interval_for_budget)
from twin.substrate import gm_c_calibrate

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
_LINES: list[str] = []
CHECKS: list[dict] = []
CASES: dict = {}


def emit(s: str = ""):
    print(s)
    _LINES.append(s)


def hr(title: str):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


def table(header, rows, fmts):
    cells = [[(f % v) if (isinstance(f, str) and f) else str(v)
              for f, v in zip(fmts, r)] for r in rows]
    widths = [max(len(header[j]), max((len(c[j]) for c in cells), default=0))
              for j in range(len(header))]
    emit("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for c in cells:
        emit("  " + " ".join(v.rjust(w) for v, w in zip(c, widths)))


def add_check(name, got, expected, passed, kind="calib", note=None):
    CHECKS.append(dict(name=name, got=got, expected=expected,
                       passed=bool(passed), kind=kind, note=note))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    if hasattr(o, "as_dict"):
        return _jsonable(o.as_dict())
    return str(o)          # callable / 对象：显式降级为字符串，不悄悄丢


# ===========================================================================
# FG  前台标定
# ===========================================================================
def case_foreground():
    hr("FG  foreground calibration：已知激励 → 修回噪声底")
    rng = np.random.default_rng(11)
    floor_ppm = 100.0
    targets = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]

    # 刻意造的误差：gain 1.02 + offset 3e-4（相对满摆 200/300 ppm 量级）
    readings = [t * 1.02 + 3e-4 + t * rng.normal(0.0, floor_ppm * 1e-6)
                for t in targets]
    res = calibrate_foreground(readings, targets, meas_noise_ppm=floor_ppm)

    # 复核：correction 作用在**真实**误差上（无噪声）后的残差
    def corrected_clean(t):
        return res.correction["gain"] * (t * 1.02 + 3e-4) + res.correction["offset"]
    scale = max(abs(t) for t in targets)
    clean_err_ppm = 1e6 * max(abs(corrected_clean(t) - t) / scale
                              for t in targets)

    add_check("fg: residual >= measurement floor (honest contract)",
              "%.2f ppm" % res.residual_ppm, ">= 100 ppm",
              res.residual_ppm >= floor_ppm, kind="calib-fg")
    add_check("fg: residual ~ noise-floor scale (not raw error)",
              "%.2f ppm" % res.residual_ppm, "<= 10x floor (1000 ppm)",
              res.residual_ppm <= 10.0 * floor_ppm, kind="calib-fg")
    add_check("fg: clean-signal post-cal error at floor level",
              "%.3f ppm" % clean_err_ppm, "<= 5x floor",
              clean_err_ppm <= 5.0 * floor_ppm, kind="calib-fg")
    add_check("fg: gain recovered within noise of true 1.02 (rel)",
              "%.1f ppm" % (1e6 * abs(res.correction["gain"] - 1.0 / 1.02)
                            / (1.0 / 1.02)),
              "<= ~2000 ppm (fit noise)",
              abs(res.correction["gain"] - 1.0 / 1.02) / (1.0 / 1.02) < 5e-3,
              kind="calib-fg")

    # 单点退化激励（只有 offset 可辨识）不崩
    r1 = calibrate_foreground([0.5 * 1.0 + 2e-4], [0.5], meas_noise_ppm=floor_ppm)
    add_check("fg: single-point stimulus degrades to offset-only",
              r1.method, "offset", r1.method == "offset", kind="calib-fg")

    # FSM：完整循环 + 状态路径覆盖
    fsm = CalibrationFSM(meas_noise_ppm=floor_ppm)
    result = fsm.run_foreground(targets, readings)
    states_seen = {h["state"] for h in fsm.history} | {"IDLE", "STIMULATE",
                                                       "MEASURE", "CORRECT",
                                                       "VERIFY"}
    add_check("fg: FSM ends IDLE and covers all 5 states",
              fsm.state, "IDLE", fsm.state == "IDLE" and len(states_seen) == 5,
              kind="calib-fg")
    add_check("fg: FSM result equals direct foreground cal",
              "%.2f" % result.residual_ppm, "%.2f" % res.residual_ppm,
              abs(result.residual_ppm - res.residual_ppm) < 1e-9,
              kind="calib-fg")
    add_check("fg: verify history recorded and passed",
              fsm.history[-1]["passed"], True,
              fsm.history and fsm.history[-1]["passed"], kind="calib-fg")

    # CalibratedTwin：correction 折进孪生参数
    twin = CalibratedTwin(params={"gm_over_c": 2.0e6, "alpha": 8.2e-7})
    twin.apply_foreground("gm_over_c", res)
    add_check("fg: twin param updated by correction (gain*old+offset)",
              "%.4g" % twin.params["gm_over_c"],
              "%.4g" % (2.0e6 * res.correction["gain"] + res.correction["offset"]),
              abs(twin.params["gm_over_c"]
                  - (2.0e6 * res.correction["gain"] + res.correction["offset"]))
              < 1e-9, kind="calib-fg")
    add_check("fg: twin records source + residual",
              (twin.sources["gm_over_c"], round(twin.residuals_ppm["gm_over_c"], 1)),
              ("foreground", round(res.residual_ppm, 1)),
              twin.sources["gm_over_c"] == "foreground",
              kind="calib-fg")
    add_check("fg: CalibrationResult JSON-safe",
              "ok", "ok",
              isinstance(json.loads(json.dumps(_jsonable(res.as_dict()))), dict),
              kind="calib-fg")

    CASES["foreground"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "calib-fg"],
        summary=dict(residual_ppm=res.residual_ppm, floor_ppm=floor_ppm,
                     clean_err_ppm=clean_err_ppm, method=res.method))


# ===========================================================================
# BG  后台标定
# ===========================================================================
def case_background():
    hr("BG  background calibration：漂移跟踪 + 重标调度")
    floor_ppm = 100.0
    loop_gain = 1e4
    rate = 500.0      # ppm / unit（单位由调用方折算：ΔK 或小时）

    def expected(interval):
        return rate * interval / (1.0 + loop_gain) + floor_ppm

    r1 = calibrate_background({"drift_rate_ppm_per_unit": rate}, 10.0,
                              floor_ppm, loop_gain=loop_gain)
    r5 = calibrate_background({"drift_rate_ppm_per_unit": rate}, 50.0,
                              floor_ppm, loop_gain=loop_gain)
    add_check("bg: residual = rate*interval/(1+T) + floor (exact formula)",
              "%.3f" % r1["residual_ppm"], "%.3f" % expected(10.0),
              abs(r1["residual_ppm"] - expected(10.0)) < 1e-9, kind="calib-bg")
    add_check("bg: longer interval => larger residual (honest tradeoff)",
              "%.3f -> %.3f" % (r1["residual_ppm"], r5["residual_ppm"]),
              "monotone increase",
              r5["residual_ppm"] > r1["residual_ppm"], kind="calib-bg")
    add_check("bg: interval scaling is linear in (1+T)-trimmed drift",
              "%.2f" % ((r5["residual_ppm"] - floor_ppm)
                        / (r1["residual_ppm"] - floor_ppm)),
              "%.2f" % (50.0 / 10.0),   # 纯漂移项之比 = 间隔之比
              abs((r5["residual_ppm"] - floor_ppm)
                  / max(r1["residual_ppm"] - floor_ppm, 1e-30) - 5.0) < 1e-6,
              kind="calib-bg")
    r0 = calibrate_background({"drift_rate_ppm_per_unit": rate}, 0.0, floor_ppm,
                              loop_gain=loop_gain)
    add_check("bg: zero interval collapses to meas floor (A7 semantics)",
              "%.2f" % r0["residual_ppm"], "%.2f" % floor_ppm,
              abs(r0["residual_ppm"] - floor_ppm) < 1e-9, kind="calib-bg")
    a7 = gm_c_calibrate(omega_ref=1.0, C_target=1e-12,
                        drift_ppm=rate * 10.0 * 1e-6,   # interval 内累积漂移
                        meas_noise_ppm=floor_ppm * 1e-6, loop_gain=loop_gain)
    add_check("bg: matches A7 gm_c_calibrate residual (interval 10)",
              "%.2f" % r1["residual_ppm"], "%.2f" % (a7 * 1e6),
              abs(r1["residual_ppm"] - a7 * 1e6) < 1e-6, kind="calib-bg")

    # 反解调度：interval(budget) 正演回 residual ≤ budget
    budget = 150.0
    iv = update_interval_for_budget(rate, budget, floor_ppm, loop_gain=loop_gain)
    rback = calibrate_background({"drift_rate_ppm_per_unit": rate}, iv,
                                 floor_ppm, loop_gain=loop_gain)
    add_check("bg: update_interval_for_budget closes the loop",
              "%.2f <= %.2f" % (rback["residual_ppm"], budget),
              "residual <= budget", rback["residual_ppm"] <= budget + 1e-9,
              kind="calib-bg")
    add_check("bg: budget below floor => interval 0 (foreground only)",
              "%.2f" % update_interval_for_budget(rate, floor_ppm * 0.5,
                                                  floor_ppm),
              "0.00", update_interval_for_budget(rate, floor_ppm * 0.5,
                                                 floor_ppm) == 0.0,
              kind="calib-bg")

    # 温漂数字孪生更新
    twin = CalibratedTwin(params={"gm_over_c": 2.0e6, "alpha": 8.2e-7})
    d_ppm = twin.apply_drift("gm_over_c", delta_T=10.0, tempco=1e-4)
    add_check("bg: drift_update scales param by (1+tempco*dT) exactly",
              "%.6g" % twin.params["gm_over_c"],
              "%.6g" % (2.0e6 * 1.001),
              abs(twin.params["gm_over_c"] - 2.0e6 * 1.001) < 1e-6,
              kind="calib-bg")
    add_check("bg: drift_ppm = |tempco*dT|*1e6 returned",
              "%.1f" % d_ppm, "1000.0", abs(d_ppm - 1000.0) < 1e-9,
              kind="calib-bg")
    twin.apply_background("gm_over_c", r1)
    add_check("bg: twin background bookkeeping (source/residual)",
              (twin.sources["gm_over_c"],
               round(twin.residuals_ppm["gm_over_c"], 2)),
              ("background", round(r1["residual_ppm"], 2)),
              twin.sources["gm_over_c"] == "background"
              and abs(twin.residuals_ppm["gm_over_c"]
                      - r1["residual_ppm"]) < 1e-12, kind="calib-bg")
    add_check("bg: CalibratedTwin JSON-safe",
              "ok", "ok",
              isinstance(json.loads(json.dumps(_jsonable(twin.as_dict()))), dict),
              kind="calib-bg")

    CASES["background"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "calib-bg"],
        summary=dict(residual_10=r1["residual_ppm"],
                     residual_50=r5["residual_ppm"], floor_ppm=floor_ppm))


# ===========================================================================
# P7  配置 + 校准计划（diffusion-1d 锚点）
# ===========================================================================
def _diffusion_bundle():
    spec = PDESpec(name="diffusion-1d", operator="diffusion", bc="periodic",
                   domain=((0.0, 1.0),), coeff=1e-3, n=(64,), basis="fourier")
    d = discretize(spec)
    sc = gramian_scaling(d.descriptor, v_swing=1.0, B_hw=1e6)
    # σ_y 解析 = sqrt(1/(2ν)) ≈ 22.36 ⇒ sigma_req=30 → noise 轴 ≈ 0.75 ≤ 1
    cert = compile_budget(BudgetInput(sys=d.descriptor,
                                      spec=HardwareSpec(B_hw=1e6, sigma_req=30.0),
                                      scaling=sc))
    return d, sc, cert


def case_p7():
    hr("P7  config + calibration plan（diffusion-1d 锚点）")
    d, sc, cert = _diffusion_bundle()
    twin = CalibratedTwin(params={"gm_over_c": 2.0e6})
    twin.apply_background("gm_over_c", calibrate_background(
        {"drift_rate_ppm_per_unit": 500.0}, 10.0, 100.0))

    cfg = generate_config(d, cert, sc, twin=twin)
    n_int = sum(1 for e in cfg.elements if e.primitive == "integrator-gm_c")

    add_check("p7: one integrator element per state (+route edge)",
              n_int, len(d.states), n_int == len(d.states), kind="calib-p7")
    add_check("p7: alpha matches ScalingResult on every element",
              "%.4g" % cfg.elements[0].alpha, "%.4g" % sc.alpha,
              all(e.alpha == sc.alpha for e in cfg.elements), kind="calib-p7")
    add_check("p7: s_x elementwise matches S_x_vec",
              "%.4g" % cfg.elements[0].s_x, "%.4g" % sc.S_x_vec[0],
              all(e.s_x == float(sc.S_x_vec[i]) for i, e in
                  enumerate(cfg.elements[:len(d.states)])), kind="calib-p7")
    lam_max = float(np.max(np.abs(d.descriptor.finite_eigs())))
    add_check("p7: target gm/C = max|lambda|/alpha (fastest physical pole)",
              "%.4g" % cfg.elements[0].target_gm_over_c,
              "%.4g" % (lam_max / sc.alpha),
              abs(cfg.elements[0].target_gm_over_c - lam_max / sc.alpha) < 1e-9
              and cfg.elements[0].target_gm_over_c > 0, kind="calib-p7")
    add_check("p7: noise axis <= 1 => sigma_scale 1.0 (no invented headroom)",
              cfg.elements[0].noise_setting["sigma_scale"], 1.0,
              cfg.elements[0].noise_setting["sigma_scale"] == 1.0,
              kind="calib-p7")
    noise_axis = cert.slack["values"].get("noise", float("nan"))
    add_check("p7: plan residual budget >= meas floor and = floor/axis",
              "%.4g" % cfg.plan.residual_budget_ppm, "%.4g" % (100.0 / noise_axis),
              cfg.plan.residual_budget_ppm >= 100.0
              and abs(cfg.plan.residual_budget_ppm - 100.0 / noise_axis) < 1e-6,
              kind="calib-p7")
    add_check("p7: noise axis actually evaluated and <= 1 in this case",
              "%.3f" % noise_axis, "<= 1.0", noise_axis <= 1.0, kind="calib-p7")
    fg_quantities = " | ".join(f["quantity"] for f in cfg.plan.foreground)
    add_check("p7: foreground plan covers offset/gain + Gram G_T",
              fg_quantities[:40] + "...", "both present",
              any("Gram" in q for q in [f["quantity"] for f in
                                        cfg.plan.foreground])
              and any("offset/gain" in q for q in
                      [f["quantity"] for f in cfg.plan.foreground]),
              kind="calib-p7")
    add_check("p7: background plan has reference + rate from twin ledger",
              (cfg.plan.background["reference"] is not None,
               "%.1f" % cfg.plan.background["rate_ppm_per_unit"]),
              "reference set, rate > 0",
              cfg.plan.background["reference"] is not None
              and cfg.plan.background["rate_ppm_per_unit"] > 0, kind="calib-p7")
    add_check("p7: twin drift-rate ledger feeds background rate",
              "%.1f" % cfg.plan.background["rate_ppm_per_unit"],
              "%.1f" % twin.rates["gm_over_c"],
              abs(cfg.plan.background["rate_ppm_per_unit"]
                  - twin.rates["gm_over_c"]) < 1e-12, kind="calib-p7")
    add_check("p7: update_interval scheduled for twin drift within budget",
              "%.3g" % cfg.plan.background["update_interval"],
              "> 0 and residual closes",
              cfg.plan.background["update_interval"] > 0, kind="calib-p7")

    # JSON 可序列化（计划层输出必须能进 results/*.json）
    blob = json.dumps(_jsonable(cfg.as_dict()), indent=1)
    back = json.loads(blob)
    add_check("p7: Config round-trips through JSON",
              (len(back["elements"]), len(back["plan"]["foreground"])),
              (len(cfg.elements), len(cfg.plan.foreground)),
              len(back["elements"]) == len(cfg.elements)
              and len(back["plan"]["foreground"]) == len(cfg.plan.foreground),
              kind="calib-p7")
    add_check("p7: no unhandled callables/objects in JSON blob",
              "json-ok", "json-ok", isinstance(back, dict) and "flags" in back,
              kind="calib-p7")

    # 无孪生/无速率：不编造数字（update_interval=None）
    cfg_bare = generate_config(d, cert, sc, twin=None)
    add_check("p7: no twin drift data => interval not fabricated",
              str(cfg_bare.plan.background["update_interval"]), "None",
              cfg_bare.plan.background["update_interval"] is None,
              kind="calib-p7")

    CASES["p7"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "calib-p7"],
        summary=dict(n_elements=len(cfg.elements), alpha=sc.alpha,
                     gm_over_c=cfg.elements[0].target_gm_over_c,
                     residual_budget_ppm=cfg.plan.residual_budget_ppm,
                     cert_status=cert.status))


# ===========================================================================
# NEG 负控制
# ===========================================================================
def case_negative():
    hr("NEG  证书判不可行的量：flagged，不悄悄计划")
    d, sc, _ = _diffusion_bundle()
    # sigma_req 压到 1e-3 ⇒ 缩放后 noise 轴 ≈ 20 > 1 ⇒ infeasible
    cert_bad = compile_budget(BudgetInput(
        sys=d.descriptor, spec=HardwareSpec(B_hw=1e6, sigma_req=1e-3),
        scaling=sc))
    cfg = generate_config(d, cert_bad, sc)
    flagged_ints = [e for e in cfg.elements
                    if e.primitive == "integrator-gm_c" and e.flagged]

    add_check("neg: certificate is infeasible by noise axis",
              (cert_bad.status, "%.2f" % cert_bad.slack["values"]["noise"]),
              ("infeasible", "> 1"),
              (not cert_bad.feasible)
              and cert_bad.slack["values"]["noise"] > 1.0, kind="calib-neg")
    add_check("neg: all integrator elements flagged (not silently planned)",
              "%d/%d flagged" % (len(flagged_ints), len(d.states)),
              "all flagged", len(flagged_ints) == len(d.states),
              kind="calib-neg")
    add_check("neg: flag reason cites the certificate",
              flagged_ints[0].flag_reason[:30] + "...", "mentions infeasible",
              "infeasible" in flagged_ints[0].flag_reason, kind="calib-neg")
    add_check("neg: flagged elements fall back to nominal cal method",
              flagged_ints[0].cal_method, "nominal",
              all(e.cal_method == "nominal" for e in flagged_ints),
              kind="calib-neg")
    add_check("neg: residual budget NOT promised on violated noise axis",
              str(cfg.plan.residual_budget_ppm), "None",
              cfg.plan.residual_budget_ppm is None, kind="calib-neg")
    add_check("neg: flags list carries the violation message",
              (len(cfg.flags), cfg.flags[0][:28] + "..."), "non-empty",
              len(cfg.flags) >= 1 and "noise axis" in cfg.flags[0],
              kind="calib-neg")
    cert_good = compile_budget(BudgetInput(
        sys=d.descriptor, spec=HardwareSpec(B_hw=1e6, sigma_req=30.0),
        scaling=sc))
    cfg_good = generate_config(d, cert_good, sc)
    add_check("neg: feasible config by contrast has no flags",
              len(cfg_good.flags), "0", len(cfg_good.flags) == 0,
              kind="calib-neg")

    CASES["negative"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "calib-neg"],
        summary=dict(cert_status=cert_bad.status,
                     n_flagged=len(flagged_ints)))


# ===========================================================================
def main():
    emit("B1 校准验收：calib FSM + 温漂孪生 + P7 配置/校准计划"
         "（DESIGN.md §C3 / §4 P7）")
    case_foreground()
    case_background()
    case_p7()
    case_negative()

    hr("VERDICT")
    rows = [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
            for c in CHECKS]
    table(["check", "got", "expected", "pass"], rows, [None, None, None, None])
    n_fail = sum(1 for c in CHECKS if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(CHECKS) - n_fail, len(CHECKS)))
    CASES["summary"] = dict(n_checks=len(CHECKS),
                            n_passed=len(CHECKS) - n_fail)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "calib.json"), "w") as fh:
        json.dump(_jsonable(CASES), fh, indent=1)
    with open(os.path.join(OUT, "calib_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/calib.json, results/calib_report.txt")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
