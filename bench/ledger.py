"""七项假设敏感度账本（DESIGN.md Stage-A 退出条件 7）。

    python3 -m bench.ledger

退出条件 7 的原文：κ_hw、α、噪声 PSD、失配、寄生、校准残差，各配一个
敏感度数字。本模块把它们做成结构化账本：

    {name, nominal_value, sensitivity, sensitivity_definition,
     source, falsification_observation}

规则：敏感度数字要么**现场算**（用既有机制，不新增拟合），要么**引用**
results/*.json 里已验收的数；不编造。输出 results/ledger.json。

七项（第 7 项是把 §1.2 (B) 的 α 与 §9 风险 3 的 γ_eff 拆开记录）：
  1  kappa_hw           频窗宽度              → phase0.json（4.86% 翻判）
  2  alpha_B_hw         时间缩放/频窗位置     → scaling margin=0.8 + pipeline.json
  3  noise_psd          B_noise/B_req 带宽亏损 → sde.json + bandwidth_drop_factor
  4  mismatch           Pelgrom ε_dev 失配地板 → twin.constraint 现场算（线性）
  5  parasitics         路由相位/电平预算     → route.json（2.86°/级）
  6  calibration_resid  Gram 标定残差 → isometry → transform.json + §6.16
  7  excess_noise_gamma 过剩噪声因子 γ_eff/α  → σ²=αkT/C 公式（log 敏感度 1/2）
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

__all__ = ["build_ledger", "main"]

_LINES: list[str] = []


def emit(s: str = ""):
    print(s)
    _LINES.append(s)


def _results_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results")


def _load(name: str) -> dict:
    with open(os.path.join(_results_dir(), name)) as fh:
        return json.load(fh)


def _sens_kappa_hw() -> tuple[float, str]:
    """κ_hw 的敏感度：diffusion N=1024 的贴边余量。

    slack_stiffness = κ_t/κ_hw 线性于 1/κ_hw；κ=1.048576e6 @ κ_hw=1e6 ⇒
    翻判点 Δκ_hw = −4.86%（§6.10 已记录"若 κ_hw 取 1.05e6 则刚好可行"）。
    """
    p0 = _load("phase0.json")
    kappa = float(p0["cases"]["diffusion N=1024"]["check"]["kappa"])
    kappa_hw = float(p0["hardware"]["kappa_hw"])
    flip = (kappa / kappa_hw - 1.0)          # 0.0486：−4.86% 即翻判
    return flip, ("slack_stiffness = κ_t/κ_hw linear in 1/κ_hw; "
                  "κ(diffusion N=1024)=%.4g @ κ_hw=%.4g → flips verdict at "
                  "Δκ_hw = −%.2f%%" % (kappa, kappa_hw, 100 * flip))


def _sens_alpha_bw() -> tuple[float, str]:
    """α/B_hw 的敏感度：P3 缩放后带宽轴贴在 margin=0.8 上。

    alpha = max|λ|/(2π·B_hw·margin) 使 scaled.bandwidth() = B_hw·margin，
    slack_bandwidth(post-scaling) = margin = 0.8 ⇒ B_hw 缩 20% 翻判。
    """
    margin = 0.8
    pl = _load("pipeline.json")
    bw_raw = float(pl["cases"]["diffusion-1d"]["p6_certificate"]["slack"]["values"]["bandwidth"])
    return margin, ("scaled bandwidth axis sits at margin=%.1f of the window "
                    "(raw %.2f → post-scaling %.1f·B_hw/B_hw); a %.0f%% B_hw "
                    "reduction flips the verdict" % (margin, bw_raw, margin, 100 * (1 - margin)))


def _sens_noise_psd() -> tuple[float, str]:
    """噪声 PSD 带宽假设：B_noise/B_req=10 处的方差亏损及其斜率。

    实测 6.69%（sde.json A_bandwidth），预测 6.35%（(2/π)arctan 砖墙）；
    斜率 d(drop)/d(ratio) = (2/π)/(1+ratio²) 在 ratio=10 处现场算。
    """
    from twin.noise import bandwidth_drop_factor
    sde = _load("sde.json")["cases"]["A_bandwidth"]
    ratio = 10.0
    drop = 1.0 - bandwidth_drop_factor(ratio)
    slope = (2.0 / np.pi) / (1.0 + ratio * ratio)      # per unit B_noise/B_req
    return slope, ("variance deficit d/dr[1−(2/π)arctan r] = %.4g per unit "
                   "B_noise/B_req at r=%d (measured %.2f%% deficit at r=10, "
                   "predicted %.2f%%)"
                   % (slope, int(ratio), 100 * sde["drop_pct"], 100 * sde["pred_drop_pct"]))


def _sens_mismatch() -> tuple[float, str]:
    """失配 ε_dev 的敏感度：地板线性于 ε_dev，并被 1/λ_min 放大。

    现场算：poisson_1d(32) 网格，ε=1% vs 2% ⇒ floor 比值 ≈2（线性），
    每 1% 失配的地板绝对值也报。κ 增长项引 constraint.json（N=128/32 比值 8.61）。
    """
    from twin.constraint import ConstraintMesh, mismatch_floor
    n = 32
    mesh = ConstraintMesh.poisson_1d(n)
    x = (np.arange(n) + 0.5) / n
    i_rhs = np.sin(np.pi * x)                # 单位源量级
    f1 = float(mismatch_floor(mesh, i_rhs, lam=0.0, eps_dev=0.01, seed=1234)["floor_inf"])
    f2 = float(mismatch_floor(mesh, i_rhs, lam=0.0, eps_dev=0.02, seed=1234)["floor_inf"])
    slope = f1 / 0.01                        # δV floor per unit ε_dev
    lin = f2 / f1
    cons = _load("constraint.json")["constraint"]["checks"]
    kappa_ratio = next(c["got"] for c in cons if c["name"] == "floor ratio N=128/32")
    return slope, ("mismatch floor linear in ε_dev: floor(2%%)/floor(1%%)=%.3f, "
                   "floor=%.3g per 1%% ε_dev; amplified 1/λ_min (N=128/32 floor "
                   "ratio %s, results/constraint.json)" % (lin, f1, kappa_ratio))


def _sens_parasitics() -> tuple[float, str]:
    """寄生的敏感度：路由相位误差每 buffer 级 2.86°（3° 预算 → 1 级余量）。"""
    rt = _load("route.json")["cases"]["buffers"]
    d1 = float(rt["1"]["max_phase_err_deg"])
    d4 = float(rt["4"]["max_phase_err_deg"])
    per_level = d4 / 4.0
    return per_level, ("route phase error %.2f°/buffer level (depth1 %.2f°, "
                       "depth4 %.2f° vs 3° budget → 1-level margin); "
                       "η_route per level %.2f%%"
                       % (per_level, d1, d4, 100 * rt["1"]["eta"]))


def _sens_calibration() -> tuple[float, str]:
    """校准残差的敏感度：isometry 对 Gram 残差 ~17/unit（§6.16 实测）。

    右 Gram 标定后 isometry=3.46e-15；2% 残差扰动时 0.342（DESIGN §5.9/§6.16
    实测记录）⇒ d(isometry)/d(residual) ≈ 17。左乘陷阱引 transform.json。
    """
    tf = _load("transform.json")["cases"]["gram"]
    iso = float(tf["haar_isometry_after"])
    iso_at_2pct = 0.342                    # DESIGN.md §6.16 measured
    slope = iso_at_2pct / 0.02
    return slope, ("isometry after right-Gram %.2e; at 2%% calibration residual "
                   "%.3f (§6.16 measured) → d(isometry)/d(residual)≈%.0f/unit; "
                   "left-multiply trap: naive isometry %.3f"
                   % (iso, iso_at_2pct, slope, tf["naive_left_isometry"]))


def _sens_excess_noise() -> tuple[float, str]:
    """过剩噪声因子 γ_eff（α）的敏感度：σ_y ∝ √α（σ²=αkT/C，§1.2 (B)）。"""
    return 0.5, ("σ_y² = α·kT/C linear in α, σ_y ∝ √α → d log σ_y/d log α = 1/2 "
                 "(exact, formula-derived §1.2(B)); §9 risk 3: real γ_eff "
                 "5–10× nominal ⇒ SNR budget optimistic by 7–10 dB")


def build_ledger(out_dir: str | None = None, emit=None) -> dict:
    """构造七项假设账本；emit 可注入调用方的打印管道。"""
    em = emit or (lambda s="": print(s))
    entries = [
        dict(name="kappa_hw", display="κ_hw 频窗宽度",
             nominal_value=float(_load("phase0.json")["hardware"]["kappa_hw"]),
             sensitivity=_sens_kappa_hw()[0], sensitivity_definition=_sens_kappa_hw()[1],
             source="results/phase0.json (diffusion N=1024 κ=1.0486e6)",
             falsification_observation="measure κ_hw on silicon (τ_min parasitic, "
             "τ_max leakage/1f): §9 risk 1 — real window may be 1e4, halving the "
             "feasible PDE set; a workload flips at −4.86%"),
        dict(name="alpha_B_hw", display="α 时间缩放 / B_hw 频窗位置",
             nominal_value=10e6,
             sensitivity=_sens_alpha_bw()[0], sensitivity_definition=_sens_alpha_bw()[1],
             source="lowering/scaling.py margin=0.8 + results/pipeline.json "
                    "(bandwidth 0.08 raw → 0.8 post-scaling)",
             falsification_observation="§6.15 PhysicalScaling verdict flip: comparing "
             "raw model bandwidth to B_hw without α is the error; 20% B_hw shrink "
             "flips every α-scaled workload"),
        dict(name="noise_psd", display="噪声 PSD / B_noise 带宽",
             nominal_value=10.0,
             sensitivity=_sens_noise_psd()[0], sensitivity_definition=_sens_noise_psd()[1],
             source="results/sde.json A_bandwidth + twin/noise.bandwidth_drop_factor",
             falsification_observation="§9 risk 8: B_noise insufficient ⇒ OU variance "
             "systematically low ≥6% at B_noise/B_req=10 — a bias, not noise; "
             "PSD convention (two-sided) mix-up is a 3 dB error (§1.3)"),
        dict(name="mismatch", display="器件失配 ε_dev（Pelgrom）",
             nominal_value=0.01,
             sensitivity=_sens_mismatch()[0], sensitivity_definition=_sens_mismatch()[1],
             source="twin/constraint.mismatch_floor (live) + results/constraint.json",
             falsification_observation="P0-D contract: Γ_proj‖ΔA‖ is only valid for "
             "conservative ΔA; leakage mismatch must be budgeted by drift "
             "(∝1/λ_min), else the robustness axis silently under-reports"),
        dict(name="parasitics", display="寄生 / 路由相位预算",
             nominal_value=3.0,
             sensitivity=_sens_parasitics()[0], sensitivity_definition=_sens_parasitics()[1],
             source="results/route.json buffers (2.86°/level) + crosstalk floor "
                    "9.09% (1 mV/V C_m budget)",
             falsification_observation="per-edge passes but the loop fails (§6.15: "
             "deskew 3e-7 ps yet 2.15° absolute delay violates a 1° reserve) — "
             "per-edge contracts are necessary, not sufficient"),
        dict(name="calibration_residual", display="校准残差（Gram 标定）",
             nominal_value=0.02,
             sensitivity=_sens_calibration()[0], sensitivity_definition=_sens_calibration()[1],
             source="results/transform.json gram case + DESIGN §6.16 measured 0.342@2%",
             falsification_observation="§9 risk 2: θ(t) drifts with temperature ⇒ "
             "ΔA time-varying, H2/H∞ need periodic recompute; wrong left-Gram "
             "multiplication leaves isometry at 0.032 silently"),
        dict(name="excess_noise_gamma", display="过剩噪声因子 γ_eff / α",
             nominal_value=1.0,
             sensitivity=_sens_excess_noise()[0], sensitivity_definition=_sens_excess_noise()[1],
             source="§1.2 (B) σ²=αkT/C formula (exact log-sensitivity 1/2) + §9 risk 3",
             falsification_observation="real OTA γ_eff and excess noise 5–10× nominal "
             "⇒ every σ_y in the certificates is optimistic by 7–10 dB; "
             "re-measure per-primitive (§C1), never reuse nominal"),
    ]
    ledger = dict(
        n_assumptions=len(entries),
        rule="sensitivity numbers computed from existing machinery or cited from "
             "results/*.json acceptance records; no new fits",
        assumptions=entries,
    )
    em("")
    em("  七项假设账本（nominal → sensitivity）：")
    for a in entries:
        em("    %-22s nominal=%-9g sens=%.4g  [%s]" %
           (a["name"], a["nominal_value"], a["sensitivity"],
            a["sensitivity_definition"][:72] + ("..." if len(a["sensitivity_definition"]) > 72 else "")))
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "ledger.json"), "w") as fh:
            json.dump(ledger, fh, indent=1, default=float)
        em("\n  wrote results/ledger.json")
    return ledger


def main():
    hr = lambda t: emit("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)
    hr("七项假设敏感度账本（Stage-A 退出条件 7）")
    build_ledger(out_dir=_results_dir(), emit=emit)
    with open(os.path.join(_results_dir(), "ledger_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    main()
