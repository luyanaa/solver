"""P6 证书聚合层验收：budget.py 的结构化 Certificate/Rejection 语义。

    python3 -m bench.run_compiler

DESIGN.md §4 P6："可行性证书 slack + 随机证书"。budget.py 本身不产生
任何新判定——它聚合 P0–P5 已算出的证据（slack / manifold-γ 窗 / loop
‖S‖∞ / multirate ρ_sup,k_max / sde noise_realizability），按三条规则出
状态：结构化拒绝 ⇒ rejected（原因逐条列出）；只有 slack 轴违规 ⇒
infeasible（binding_axis）；有未评估（NaN）轴 ⇒ 绝不写 "feasible"。

算例：

    A  可行证书    diffusion N=64（E=I, A=diag(−νk²), B=e1, C=I），六轴
                   全部给足输入（σ_req/ΔA/Δu/η_route/reg）⇒ 全轴有限、
                   无违规 ⇒ status="feasible"（verbatim，非"no violation"）。
    B  不可行      同一系统 κ_hw=1e3：κ_t=N²=4096 > 1e3 ⇒ infeasible，
                   binding_axis="stiffness"。
    C/D manifold   空窗（γ_min≥γ_max）与 chosen 出窗 ⇒ rejected，原因
                   逐字含 "manifold"（§4.3：Baumgarte 窗口合同）。
    E  loop        run_route case_loop 的同一 J=[[−a,−b],[b,−a]]
                   （a=2π·5 MHz, b=2π·50 MHz）：小延迟 τ=1e-10 S_peak≈15
                   不拒绝；τ=1.02·τ_ex（τ_ex=(π/2−φ)/|λ|）S_peak 冲上
                   ~500 ⇒ rejected。基线 = 空 edge_map（H=1）的 loop_margins。
    F  multirate   twin.multirate 的双岛分裂：弱耦合 ρ_sup≈0.74<1、
                   k_max≥1 ⇒ 不拒绝；强耦合 ρ_sup≈1.48>1 ⇒ rejected。
    G  D 馈通      sys.D≠0 ⇒ Lyapunov 判 H2=∞ ⇒ noise 轴 inf ⇒
                   infeasible（不是 rejected——slack 轴违规路径）。
    H  度量 Γ      守恒系统（周期二阶差分，has_imaginary）+ metric=
                   "energy"：peak_with_projection 的投影+度量组合路径
                   （resolvent.py 修复的回归测试）必须可运行且 Γ_proj 有限。
    I  assumptions 评估了哪些项，就要列出哪些假设合同（§1.2/§4.3/§4.4）。
    J  NaN≠可行    sigma_req=None ⇒ noise 轴未评估 ⇒ status 必须是
                   "no violation on evaluated axes"，不是 "feasible"。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from analysis.descriptor import Descriptor
from analysis.slack import HardwareSpec
from lowering.budget import S_PEAK_THRESHOLD, BudgetInput, compile_budget, evaluate_sys
from twin.multirate import k_max, wr_spectral_radius
from twin.route import PureDelay, RouteEdge, loop_margins

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
# 测试系统（与 run_phase0.build_diffusion 同一构造；不依赖 normalize/scaling）
# ---------------------------------------------------------------------------
def build_diffusion(n: int, nu: float = 1e-3):
    k = np.arange(1, n + 1, dtype=float)
    return Descriptor(np.eye(n), np.diag(-nu * k * k),
                      np.eye(n)[:, :1], np.eye(n))


# 六轴全部给足的 spec：σ_y=sqrt(1/(2ν))≈22.4、‖H‖∞=1/ν=1e3、Γ=1/ν_min=1e3
SPEC_FULL = HardwareSpec(kappa_hw=1e6, sigma_req=100.0, delta_A_norm=1e-4,
                         delta_input_norm=1e-4, eta_route=0.05)
REG_FULL = {"p": 6.0, "E_tail": 1e-6}


def case_certificate():
    sys = build_diffusion(64)
    cert = compile_budget(BudgetInput(sys=sys, spec=SPEC_FULL, reg=REG_FULL))
    CASES["feasible"] = cert.as_dict()
    # B：同一系统，κ_hw 收紧到 1e3（κ_t=N²=4096 超窗）
    cert_b = evaluate_sys(sys, HardwareSpec(kappa_hw=1e3))
    CASES["infeasible_kappa"] = cert_b.as_dict()
    # J：NaN ≠ 可行——sigma_req=None ⇒ noise 轴未评估
    cert_j = evaluate_sys(sys, HardwareSpec())
    CASES["nan_not_feasible"] = cert_j.as_dict()


def case_rejections():
    # C：manifold 空窗
    c = evaluate_sys(build_diffusion(16), HardwareSpec(),
                     manifold=dict(gamma_min=1e5, gamma_max=1e3, chosen=None))
    CASES["manifold_empty"] = c.as_dict()
    # D：chosen 出窗（10 < γ_min=100）
    c = evaluate_sys(build_diffusion(16), HardwareSpec(),
                     manifold=dict(gamma_min=100.0, gamma_max=1e4, chosen=10.0))
    CASES["manifold_outside"] = c.as_dict()


def _edges_uniform(tau):
    return {(i, j): RouteEdge((PureDelay(tau),), "e%d%d" % (i, j))
            for i in range(2) for j in range(2)}


def case_loop():
    """E：loop ‖S‖∞ 聚合。同一 J 与 τ_ex 闭式（run_route case_loop）。"""
    w = np.logspace(np.log10(2 * np.pi * 1e5), np.log10(2 * np.pi * 2e9), 1200)
    a = 2 * np.pi * 5e6
    b = 2 * np.pi * 50e6
    J = np.array([[-a, -b], [b, -a]])
    lam = complex(-a, b)
    phi = np.arctan2(b, a)
    tau_ex = (np.pi / 2 - phi) / abs(lam)
    # loop 轴隔离：B_hw 放宽，让 bandwidth/stiffness 都远低于 1
    spec = HardwareSpec(kappa_hw=1e6, B_hw=1e9)
    m_base = loop_margins(J, {}, w, with_first_order=False)
    m_small = loop_margins(J, _edges_uniform(1e-10), w, with_first_order=False)
    m_big = loop_margins(J, _edges_uniform(1.02 * tau_ex), w, with_first_order=False)
    CASES["loop"] = dict(
        tau_ex=float(tau_ex),
        S_base=float(m_base["S_peak"]), S_small=float(m_small["S_peak"]),
        S_big=float(m_big["S_peak"]),
        cert_small=evaluate_sys(Descriptor(np.eye(2), J, np.eye(2)[:, :1], np.eye(2)),
                                spec, loop=m_small).as_dict(),
        cert_big=evaluate_sys(Descriptor(np.eye(2), J, np.eye(2)[:, :1], np.eye(2)),
                              spec, loop=m_big).as_dict(),
    )


def case_multirate():
    """F：twin.multirate 直接构造收敛/发散分裂（run_multirate 双岛模式）。"""
    E4 = np.eye(4)
    OSC = np.array([[-0.2, 5.0], [-5.0, -0.2]])
    A_diag = np.block([[OSC, np.zeros((2, 2))], [np.zeros((2, 2)), OSC]])
    OG = np.logspace(np.log10(0.03), np.log10(30.0), 121)

    def build_split(c):
        A_off = np.zeros((4, 4))
        A_off[:2, 2:4] = c * np.eye(2)
        A_off[2:4, :2] = c * np.eye(2)
        return A_off

    out = {}
    for tag, c in (("convergent", 0.15), ("divergent", 0.30)):
        r = wr_spectral_radius(E4, A_diag, build_split(c), OG)
        mr = dict(rho_sup=r["rho_sup"], omega_peak=r["omega_peak"],
                  eps_replay=1e-3,
                  k_max=k_max(r["rho_sup"], r["rho_sup"], 1e-3, 1.0))
        cert = evaluate_sys(build_diffusion(16), HardwareSpec(), multirate=mr)
        out[tag] = dict(multirate=mr, rho_eig_sup=r["rho_eig_sup"],
                        cert=cert.as_dict())
    CASES["multirate"] = out


def case_feedthrough():
    """G：sys.D≠0 ⇒ H2=∞ ⇒ noise 轴 inf ⇒ infeasible（hinf_norm 不吃 D，
    预先传入 hinf=0 隔离被测路径）。"""
    sys = Descriptor(np.eye(2), np.array([[-1.0, 0.0], [0.0, -2.0]]),
                     np.eye(2)[:, :1], np.array([[1.0, 0.0]]), np.array([[1.0]]))
    cert = compile_budget(BudgetInput(sys=sys,
                                      spec=HardwareSpec(sigma_req=1.0),
                                      hinf=0.0))
    CASES["feedthrough"] = cert.as_dict()


def case_metric():
    """H：守恒系统 + metric="energy" 的投影路径（resolvent 修复回归）。"""
    Ec = np.diag([1.0, 1e-2, 1.0, 1.0, 1.0, 1.0])
    n6 = Ec.shape[0]
    D2 = np.zeros((n6, n6))
    for i in range(n6):
        D2[i, (i - 1) % n6] += -1.0
        D2[i, (i + 1) % n6] += -1.0
        D2[i, i] += 2.0
    sys = Descriptor(Ec, -0.5 * D2, np.zeros((n6, 0)), np.zeros((0, n6)))
    cert_e = compile_budget(BudgetInput(sys=sys, spec=HardwareSpec(),
                                        metric="energy", hinf=0.0))
    cert_b = compile_budget(BudgetInput(sys=sys, spec=HardwareSpec(), hinf=0.0))
    CASES["metric"] = dict(
        has_imaginary=bool(sys.has_imaginary()),
        gamma_M=float(cert_e.items["resolvent"]["value"]["Gamma_proj"]),
        gamma_bare=float(cert_b.items["resolvent"]["value"]["Gamma_proj"]),
        cert=cert_e.as_dict(),
    )


def case_sde():
    """随机证书：noise_realizability 的 pass/reject 两档进聚合。"""
    from analysis.sde import noise_realizability
    good = noise_realizability(dict(B_noise=1e8, B_req=1e6, tau_c=1e-9,
                                    D=1e-3, S_xi=1e-6))
    bad = noise_realizability(dict(B_noise=5e5, B_req=1e6, tau_c=1e-9,
                                   D=1e-3, S_xi=1e-6))
    cert_g = evaluate_sys(build_diffusion(16), HardwareSpec(),
                          sde_cert=good.__dict__)
    cert_b = evaluate_sys(build_diffusion(16), HardwareSpec(),
                          sde_cert=bad.__dict__)
    CASES["sde"] = dict(good=cert_g.as_dict(), bad=cert_b.as_dict())


def case_assumptions():
    """I：评估了 slack/loop/multirate/manifold ⇒ 对应假设合同必须出现。"""
    cert = evaluate_sys(build_diffusion(16), SPEC_FULL, reg=REG_FULL,
                        loop=dict(S_peak=10.0),
                        multirate=dict(rho_sup=0.5, k_max=3, eps_replay=1e-3),
                        manifold=dict(gamma_min=1.0, gamma_max=1e4, chosen=10.0))
    CASES["assumptions"] = cert.as_dict()


# ---------------------------------------------------------------------------
def acceptance():
    checks = []

    def add(name, got, expected, passed):
        checks.append(dict(name=name, got=got, expected=expected,
                           passed=bool(passed)))

    f = CASES["feasible"]
    add("feasible: status verbatim 'feasible'", f["status"], "feasible",
        f["status"] == "feasible")
    add("feasible: no structural rejection", f["rejection_reasons"] or "-", "-",
        not f["rejection_reasons"] and f["feasible"])
    add("feasible: slack block with 7 axes",
        ",".join(sorted(f["slack"]["values"])),
        "bandwidth,noise,representation,robustness_A,robustness_io,routing,stiffness",
        len(f["slack"]["values"]) == 7)
    add("feasible: binding_axis reported (max finite axis)",
        f["binding_axis"], "representation",
        f["binding_axis"] == "representation")

    b = CASES["infeasible_kappa"]
    add("kappa infeasible: status", b["status"], "infeasible",
        b["status"] == "infeasible")
    add("kappa infeasible: binding_axis", b["binding_axis"], "stiffness",
        b["binding_axis"] == "stiffness")

    me = CASES["manifold_empty"]
    add("manifold empty window → rejected", me["status"], "rejected",
        me["status"] == "rejected" and len(me["rejection_reasons"]) == 1
        and "manifold" in me["rejection_reasons"][0])
    mo = CASES["manifold_outside"]
    add("manifold chosen outside window → rejected", mo["status"], "rejected",
        mo["status"] == "rejected" and "manifold" in mo["rejection_reasons"][0])

    lp = CASES["loop"]
    add("loop baseline S_peak finite (H=1 edges)", "%.4g" % lp["S_base"],
        "< %.4g" % S_PEAK_THRESHOLD, np.isfinite(lp["S_base"])
        and lp["S_base"] < S_PEAK_THRESHOLD)
    add("loop small delay: S_peak finite, no rejection",
        "%.4g" % lp["S_small"], "< %.4g, pass" % S_PEAK_THRESHOLD,
        np.isfinite(lp["S_small"]) and lp["S_small"] < S_PEAK_THRESHOLD
        and not lp["cert_small"]["rejection_reasons"])
    add("loop delayed past τ_ex: S_peak blows up → rejected",
        "%.4g" % lp["S_big"], "≥ %.4g, rejected" % S_PEAK_THRESHOLD,
        lp["S_big"] >= S_PEAK_THRESHOLD
        and lp["cert_big"]["status"] == "rejected"
        and "loop" in lp["cert_big"]["rejection_reasons"][0])
    add("loop: S_peak(delayed) ≫ S_peak(baseline)",
        "%.4g vs %.4g" % (lp["S_big"], lp["S_base"]), "≥ 10×",
        lp["S_big"] >= 10.0 * lp["S_base"])

    mr = CASES["multirate"]
    mc, md = mr["convergent"], mr["divergent"]
    add("multirate convergent: ρ_sup<1, k_max≥1, no rejection",
        "ρ=%.4f k=%d" % (mc["multirate"]["rho_sup"], mc["multirate"]["k_max"]),
        "<1, ≥1, pass",
        mc["multirate"]["rho_sup"] < 1.0 and mc["multirate"]["k_max"] >= 1
        and mc["cert"]["status"] != "rejected")
    add("multirate divergent: ρ_sup≥1 → rejected",
        "ρ=%.4f" % md["multirate"]["rho_sup"], "rejected",
        md["multirate"]["rho_sup"] > 1.0 and md["cert"]["status"] == "rejected"
        and any("multirate" in r for r in md["cert"]["rejection_reasons"]))
    add("multirate k_max recorded on both branches",
        "%d/%s" % (mc["multirate"]["k_max"], md["multirate"]["k_max"]),
        "both recorded", mc["multirate"]["k_max"] is not None
        and md["multirate"]["k_max"] is not None)

    ft = CASES["feedthrough"]
    add("D feedthrough: H2=∞ → noise axis inf → infeasible",
        "%s (noise=%s)" % (ft["status"], ft["slack"]["values"]["noise"]),
        "infeasible, binding=noise",
        ft["status"] == "infeasible" and ft["binding_axis"] == "noise"
        and ft["slack"]["values"]["noise"] == float("inf"))

    mt = CASES["metric"]
    add("metric path: projected Γ_M finite on conservation system",
        "%.4g (projected=%s)" % (mt["gamma_M"], mt["has_imaginary"]),
        "finite", np.isfinite(mt["gamma_M"]) and mt["has_imaginary"])
    add("metric matters: Γ_M ≠ bare Γ_proj",
        "%.4g vs %.4g" % (mt["gamma_M"], mt["gamma_bare"]), "ratio ≠ 1",
        abs(mt["gamma_M"] / mt["gamma_bare"] - 1.0) > 1e-6)

    sd = CASES["sde"]
    add("sde_cert good: pass (B_noise/B_req≥100)",
        sd["good"]["status"], "no violation on evaluated axes",
        sd["good"]["status"] == "no violation on evaluated axes"
        and not sd["good"]["rejection_reasons"])
    add("sde_cert bad (B_noise/B_req<10) → rejected",
        sd["bad"]["status"], "rejected",
        sd["bad"]["status"] == "rejected"
        and any("sde_cert" in r for r in sd["bad"]["rejection_reasons"]))

    asm = CASES["assumptions"]
    a_list = asm["assumptions"]
    need = ["ΔA declared conservative", "τ_c/τ_min≪1", "loop_margins",
            "σ_max WR bound"]
    add("assumptions non-empty and cover evaluated contracts",
        "%d items" % len(a_list),
        "contains: " + "; ".join(n[:20] for n in need),
        len(a_list) > 0 and all(any(n in s for s in a_list) for n in need))

    nf = CASES["nan_not_feasible"]
    add("NaN axis ⇒ NOT verbatim 'feasible'",
        nf["status"], "no violation on evaluated axes",
        nf["status"] == "no violation on evaluated axes"
        and nf["status"] != "feasible")
    return checks


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()

    hr("P6 验收：可行性证书聚合层（budget.py，DESIGN.md §4）")

    case_certificate()
    case_rejections()
    case_loop()
    case_multirate()
    case_feedthrough()
    case_metric()
    case_sde()
    case_assumptions()

    hr("A. 状态矩阵")
    rows = []
    for name in ("feasible", "infeasible_kappa", "manifold_empty",
                 "manifold_outside", "feedthrough", "nan_not_feasible"):
        c = CASES[name]
        rows.append((name, c["status"], c["binding_axis"] or "-",
                     "; ".join(c["rejection_reasons"]) or "-",
                     str(c["feasible"])))
    table(["case", "status", "binding_axis", "rejections", "feasible"], rows)

    hr("B. loop ‖S‖∞（τ_ex=%.4e s，阈值 %g）"
       % (CASES["loop"]["tau_ex"], S_PEAK_THRESHOLD))
    lp = CASES["loop"]
    table(["edge_map", "S_peak", "certificate status"],
          [("H=1 baseline", "%.4g" % lp["S_base"], "-"),
           ("τ=1e-10 s", "%.4g" % lp["S_small"],
            lp["cert_small"]["status"]),
           ("τ=1.02·τ_ex", "%.4g" % lp["S_big"], lp["cert_big"]["status"])])
    emit("  τ_ex=(π/2−φ)/|λ| 是标量 DDE 精确边界；刚过界时 S_peak 沿网格冲到"
         " ~10²——聚合层按阈值拒绝，且不把 loop_margins 当精确根（assumptions 里留痕）。")

    hr("C. multirate 双岛分裂")
    mr = CASES["multirate"]
    table(["split", "ρ_sup", "ρ_eig_sup", "k_max", "certificate status"],
          [("convergent (c=0.15)", "%.4f" % mr["convergent"]["multirate"]["rho_sup"],
            "%.4f" % mr["convergent"]["rho_eig_sup"],
            str(mr["convergent"]["multirate"]["k_max"]),
            mr["convergent"]["cert"]["status"]),
           ("divergent (c=0.30)", "%.4f" % mr["divergent"]["multirate"]["rho_sup"],
            "%.4f" % mr["divergent"]["rho_eig_sup"],
            str(mr["divergent"]["multirate"]["k_max"]),
            mr["divergent"]["cert"]["status"])])
    emit("  收敛分支进证书不产生拒绝；发散分支按 §4.3 合同拒绝（ρ_sup≥1）。")

    hr("D. 度量 Γ 投影路径（守恒系统，has_imaginary=%s）"
       % CASES["metric"]["has_imaginary"])
    mt = CASES["metric"]
    emit("  Γ_M(energy)=%.6g vs bare Γ_proj=%.6g —— 投影+度量组合路径"
         "（resolvent.py 修复）在聚合层可用。" % (mt["gamma_M"], mt["gamma_bare"]))

    hr("E. 验收")
    checks = acceptance()
    table(["check", "got", "expected", "PASS"],
          [(c["name"], str(c["got"]), str(c["expected"]),
            "yes" if c["passed"] else "NO") for c in checks])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    payload = dict(cases={k: _jsonable(v) for k, v in CASES.items()},
                   checks=checks,
                   summary=dict(n_checks=len(checks), n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "compiler.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(OUT, "compiler_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/compiler.json")
    emit("  report → results/compiler_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
