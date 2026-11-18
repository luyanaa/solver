"""T10 验收（追加）：ε_c 时序语义修复 —— 外部时标 + 证书可见性。

    python3 -m bench.run_constraint_timing

P0 finding（已确认的两处缺陷，本验收将其钉死为合同）：

    1  ε_c 的分母是**外部受约束动力学**的时标（NS 的 τ_fluid,min、MD 的
       粒子时标），必须由 workload 显式声明——不能从约束子系统自己的极点
       推导（约束网络对自身最快模态的比值没有物理意义）。纯 Poisson 稳态
       没有外部动态 ⇒ 无 ε_c 可算 ⇒ 判 (solve_deadline 达到? residual
       < tol?) 的事实，不伪造比值。
    2  约束时序判决必须到达 P6 Certificate：ε_c FAIL / deadline 未收敛
       ⇒ 证书 status="rejected"（不再"no violation on evaluated axes"而
       FAIL 埋在 side-record 里）；纯稳态事实 ⇒ 证书带 constraint 字段但
       不被标成 infeasible（也不是 feasible 的时序判决）。

算例：

    A  签名/语义     eps_c = τ_relax/τ_external（显式声明）；非法时标 raise；
                      solve_constraint 的 ε_c 与 eps_c() 一致。
    B  纯稳态         external_state_timescale=None：
                      (i) 无 deadline ⇒ verdict STEADY、eps_c=None、不判决；
                      (ii) 有 deadline 且收敛 ⇒ PASS + converged=True；
                      (iii) 构造 deadline 太短 ⇒ FAIL + converged=False
                           （不是 infeasible，是收敛事实）。
    C  耦合 ε_c       慢外部时标（τ_ext=1000·τ_relax）⇒ ε_c=1e-3 PASS；
                      快外部时标（τ_ext=50·τ_relax）⇒ ε_c=0.02 FAIL，
                      note 指向 §3.4。
    D  证书可见性     evaluate_sys(constraint=...)：FAIL ⇒ rejected/不可行/
                      理由点名 constraint；STEADY ⇒ 证书仍"no violation on
                      evaluated axes"（事实不是判决）；PASS ⇒ 证书不变；
                      畸形证据（无 verdict）⇒ rejected；负控制：把快外部
                      时标塞进证书 ⇒ 证书必须反映它。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from twin.constraint import (ConstraintMesh, eps_c, kcl_relaxation_descriptor,
                             relax_time, solve_constraint)
from lowering.budget import evaluate_sys
from analysis import HardwareSpec

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_LINES: list[str] = []

checks: list[dict] = []


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


def add(name, got, expected, passed, kind="exact"):
    checks.append(dict(name=name, got=got, expected=expected,
                       passed=bool(passed), kind=kind))


def _jsonable(x):
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
def _mesh():
    m = ConstraintMesh.poisson_1d(64)
    m.c_node = 1e-13
    return m


def _spec_neutral():
    # 中性六轴（bandwidth 轴不 binding）：kappa_hw 大、B_hw 大、无 σ_req、
    # 无 ΔA —— 证书状态只由 constraint 证据驱动，隔离被测语义。
    return HardwareSpec(kappa_hw=1e6, B_hw=1e15, sigma_req=None,
                        delta_A_norm=None, tol_l2=1e-2)


def case_signature():
    hr("CASE A  ε_c 签名/语义：外部时标显式声明")
    mesh = _mesh()
    c = 1e-13
    tau_relax = relax_time(mesh, c)
    tau_ext = 1e-8
    eps = eps_c(mesh, c, tau_ext)
    add("ε_c formula = τ_relax / τ_external", "%.6e" % eps,
        "%.6e" % (tau_relax / tau_ext), abs(eps - tau_relax / tau_ext) < 1e-15)
    add("ε_c < 0.01 for slow external dynamics", "%.4f" % eps, "< 0.01", eps < 0.01)
    raised = False
    try:
        eps_c(mesh, c, None)
    except ValueError:
        raised = True
    add("external_state_timescale=None raises (no bogus ε_c)",
        "raises" if raised else "returns", "ValueError", raised)
    raised = False
    try:
        eps_c(mesh, c, -1.0)
    except ValueError:
        raised = True
    add("non-positive external timescale raises", "raises" if raised else "returns",
        "ValueError", raised)
    res = solve_constraint(mesh, np.ones(mesh.n), c, tau_ext)
    add("solve_constraint ε_c matches eps_c()", "%.6e" % res.eps_c,
        "%.6e" % eps, abs(res.eps_c - eps) < 1e-15)
    add("solve_constraint records declared timescale",
        "%.3e" % res.external_state_timescale, "%.3e" % tau_ext,
        res.external_state_timescale == tau_ext)
    add("coupled record is not steady_state", str(res.steady_state), "False",
        res.steady_state is False)
    table(["quantity", "value"], [
        ["τ_relax", "%.3e" % tau_relax],
        ["τ_external (declared)", "%.3e" % tau_ext],
        ["ε_c", "%.4f" % eps],
    ])


def case_steady():
    hr("CASE B  纯稳态：无 ε_c，deadline+residual 事实")
    mesh = _mesh()
    c = 1e-13
    tau_relax = relax_time(mesh, c)
    i_rhs = np.ones(mesh.n)
    # (i) no deadline: fact only, no feasibility verdict
    res = solve_constraint(mesh, i_rhs, c)
    add("steady no-deadline: eps_c is None", str(res.eps_c), "None",
        res.eps_c is None)
    add("steady no-deadline: verdict STEADY (fact, not PASS/FAIL)",
        res.verdict, "STEADY", res.verdict == "STEADY")
    add("steady no-deadline: flagged steady_state", str(res.steady_state), "True",
        res.steady_state is True)
    add("steady no-deadline: note explains no-ε_c policy",
        any("no ε_c" in n or "no ε_c" in n for n in res.notes), True,
        any("no ε_c" in n for n in res.notes))
    # (ii) deadline met (T = 40·τ_relax ⇒ e^{-40} below roundoff); no explicit
    # residual_tol here so the DEFAULT (1e-6·||I||₂) is what's tested
    res_ok = solve_constraint(mesh, i_rhs, c, solve_deadline=40.0 * tau_relax)
    add("steady deadline met: verdict PASS", res_ok.verdict, "PASS",
        res_ok.verdict == "PASS")
    add("steady deadline met: converged=True", str(res_ok.converged), "True",
        res_ok.converged is True)
    add("steady deadline met: residual below tol", "%.3e" % res_ok.residual_at_deadline,
        "< %.3e" % res_ok.residual_tol,
        res_ok.residual_at_deadline < res_ok.residual_tol)
    add("steady residual_tol defaults to 1e-6·||I||",
        "%.3e" % res_ok.residual_tol, "%.3e" % (1e-6 * float(np.linalg.norm(i_rhs))),
        abs(res_ok.residual_tol - 1e-6 * float(np.linalg.norm(i_rhs))) < 1e-18)
    # (iii) deadline too short (constructed failure: big C, tiny T)
    res_bad = solve_constraint(mesh, i_rhs, 1e-9, solve_deadline=1e-9,
                               residual_tol=1e-6)
    add("steady deadline missed: verdict FAIL (constructed)", res_bad.verdict,
        "FAIL", res_bad.verdict == "FAIL")
    add("steady deadline missed: converged=False", str(res_bad.converged),
        "False", res_bad.converged is False)
    add("steady deadline missed: residual ≈ ||I|| (unconverged)",
        "%.3f·||I||" % (res_bad.residual_at_deadline / float(np.linalg.norm(i_rhs))),
        "≥ 0.5·||I||", res_bad.residual_at_deadline > 0.5 * float(np.linalg.norm(i_rhs)))
    add("steady deadline missed: note names the fact (not infeasibility)",
        any("not converged at deadline" in n for n in res_bad.notes), True,
        any("not converged at deadline" in n for n in res_bad.notes))
    table(["case", "C_node", "T_deadline", "res(T)", "verdict"], [
        ["no deadline", "%.1e" % c, "—", "—", res.verdict],
        ["T=40τ_relax", "%.1e" % c, "%.2e" % (40 * tau_relax),
            "%.3e" % res_ok.residual_at_deadline, res_ok.verdict],
        ["constructed miss", "1e-9", "1e-9",
            "%.3e" % res_bad.residual_at_deadline, res_bad.verdict],
    ])


def case_coupled():
    hr("CASE C  耦合 ε_c：显式外部时标的 PASS / FAIL")
    mesh = _mesh()
    c = 1e-13
    tau_relax = relax_time(mesh, c)
    i_rhs = np.ones(mesh.n)
    ok = solve_constraint(mesh, i_rhs, c,
                          external_state_timescale=1000.0 * tau_relax)
    add("slow external dynamics: ε_c = 1e-3", "%.6f" % ok.eps_c, "≈ 1e-3",
        abs(ok.eps_c - 1e-3) < 1e-9)
    add("slow external dynamics: verdict PASS", ok.verdict, "PASS", ok.verdict == "PASS")
    bad = solve_constraint(mesh, i_rhs, c,
                           external_state_timescale=50.0 * tau_relax)
    add("fast external dynamics: ε_c = 0.02 ≥ 0.01", "%.6f" % bad.eps_c, "≥ 0.01",
        bad.eps_c >= 0.01)
    add("fast external dynamics: verdict FAIL", bad.verdict, "FAIL", bad.verdict == "FAIL")
    add("FAIL note cites §3.4 timing contract",
        any("§3.4" in n for n in bad.notes), True, any("§3.4" in n for n in bad.notes))
    # external timescale FASTER than the constraint's own relaxation
    very_bad = solve_constraint(mesh, i_rhs, c, external_state_timescale=0.1 * tau_relax)
    add("external dynamics faster than relaxation ⇒ FAIL",
        "verdict=%s ε_c=%.1f" % (very_bad.verdict, very_bad.eps_c), "FAIL, ε_c > 1",
        very_bad.verdict == "FAIL" and very_bad.eps_c > 1.0)
    table(["case", "τ_external/τ_relax", "ε_c", "verdict"], [
        ["slow", "1000", "%.4f" % ok.eps_c, ok.verdict],
        ["fast", "50", "%.4f" % bad.eps_c, bad.verdict],
        ["faster than relax", "0.1", "%.1f" % very_bad.eps_c, very_bad.verdict],
    ])


def case_certificate():
    hr("CASE D  约束判决到达 Certificate（不再埋 side-record）")
    mesh = _mesh()
    c = 1e-13
    tau_relax = relax_time(mesh, c)
    i_rhs = np.ones(mesh.n)
    spec = _spec_neutral()
    dsc = kcl_relaxation_descriptor(mesh, c)
    # (1) FAIL → rejected / infeasible / reason names constraint
    bad = solve_constraint(mesh, i_rhs, c,
                           external_state_timescale=50.0 * tau_relax)
    cert_bad = evaluate_sys(dsc, spec, gamma_M=1.0, hinf=1.0,
                            constraint=bad.as_dict())
    add("constraint FAIL ⇒ cert status rejected", cert_bad.status, "rejected",
        cert_bad.status == "rejected")
    add("constraint FAIL ⇒ cert infeasible", str(cert_bad.feasible), "False",
        cert_bad.feasible is False)
    add("rejection reason names the constraint fabric",
        any("constraint" in r for r in cert_bad.rejection_reasons), True,
        any("constraint" in r for r in cert_bad.rejection_reasons))
    add("cert carries constraint item with rejected verdict",
        cert_bad.items.get("constraint", {}).get("verdict"), "rejected",
        cert_bad.items.get("constraint", {}).get("verdict") == "rejected")
    add("cert.as_dict() exposes the constraint field",
        (cert_bad.as_dict().get("constraint") or {}).get("verdict"), "FAIL",
        (cert_bad.as_dict().get("constraint") or {}).get("verdict") == "FAIL")
    # (2) STEADY fact ⇒ cert unchanged (not marked infeasible, not a timing pass)
    steady = solve_constraint(mesh, i_rhs, c)
    cert_st = evaluate_sys(dsc, spec, gamma_M=1.0, hinf=1.0,
                           constraint=steady.as_dict())
    add("pure-steady fact ⇒ cert stays 'no violation on evaluated axes'",
        cert_st.status, "no violation on evaluated axes",
        cert_st.status == "no violation on evaluated axes")
    add("steady fact recorded as pass-with-fact, not a feasibility verdict",
        cert_st.items["constraint"]["detail"],
        "detail states 'not a feasibility verdict'",
        "not a feasibility verdict" in cert_st.items["constraint"]["detail"])
    add("steady fact keeps eps_c None in the cert",
        str((cert_st.constraint or {}).get("eps_c")), "None",
        (cert_st.constraint or {}).get("eps_c") is None)
    # (3) coupled PASS ⇒ cert unchanged
    okc = solve_constraint(mesh, i_rhs, c,
                           external_state_timescale=1000.0 * tau_relax)
    cert_ok = evaluate_sys(dsc, spec, gamma_M=1.0, hinf=1.0,
                           constraint=okc.as_dict())
    add("coupled ε_c PASS ⇒ cert unchanged", cert_ok.status,
        "no violation on evaluated axes",
        cert_ok.status == "no violation on evaluated axes")
    # (4) malformed evidence ⇒ rejected (never silently ignored)
    cert_mal = evaluate_sys(dsc, spec, gamma_M=1.0, hinf=1.0,
                            constraint={"eps_c": 123.0})
    add("malformed constraint evidence (no verdict) ⇒ rejected",
        cert_mal.status, "rejected", cert_mal.status == "rejected")
    # (5) negative control: fast external timescale inside the certificate
    very_bad = solve_constraint(mesh, i_rhs, c, external_state_timescale=0.1 * tau_relax)
    cert_neg = evaluate_sys(dsc, spec, gamma_M=1.0, hinf=1.0,
                            constraint=very_bad.as_dict())
    add("negative control: ε_c>1 external timescale ⇒ cert rejected",
        cert_neg.status, "rejected", cert_neg.status == "rejected")
    table(["constraint evidence", "cert status", "feasible"], [
        ["ε_c=0.02 (FAIL)", cert_bad.status, str(cert_bad.feasible)],
        ["pure steady (STEADY)", cert_st.status, str(cert_st.feasible)],
        ["ε_c=1e-3 (PASS)", cert_ok.status, str(cert_ok.feasible)],
        ["malformed", cert_mal.status, str(cert_mal.feasible)],
        ["ε_c>1 (negative)", cert_neg.status, str(cert_neg.feasible)],
    ])


def main():
    os.makedirs(OUT, exist_ok=True)
    hr("T10 ε_c 时序语义验收（外部时标 + 证书可见性）")
    case_signature()
    case_steady()
    case_coupled()
    case_certificate()
    hr("验收汇总")
    table(["check", "got", "expected", "PASS"],
          [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
           for c in checks])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))
    payload = dict(checks=_jsonable(checks),
                   summary=dict(n_checks=len(checks),
                                n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "constraint_timing.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(os.path.join(OUT, "constraint_timing_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/constraint_timing.json")
    emit("  report → results/constraint_timing_report.txt")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
