"""Phase 1.5 验收：互连延迟的一阶 descriptor 模型（DESIGN.md §6.14）。

    python3 -m bench.run_skew

四个算例：

    uniform 2×2      均匀 τ：广义特征值有闭式 λ/(1+τλ)
    scalar DDE       精确根 vs 一阶展开；τ_crit 的 2/π 差
    diffusion chain  κ 随 τ 的漂移；一阶稳定边界
    phase budget     1°/10° @100 MHz 对应多少 ps

结论的用法：`E=I+A∘T` 可以直接进 analysis/descriptor.py；但它只在
ωτ≪1 时可信。标量 DDE 的精确边界是 |a|τ=π/2，一阶模型给 1，
乐观 2/π=0.637 倍。所以 §2.8 的 `ω_max Δτ_route ≪ 1` 不能省。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from analysis.skew import (phase_budget, scalar_dde_root, scalar_tau_crit,
                           skew_descriptor, skew_report, uniform_delay_poles,
                           uniform_tau_crit)

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


def _match(x, ref):
    """把 x 的每个元素配到 ref 中最近的一个。"""
    out = []
    for v in np.atleast_1d(x):
        j = int(np.argmin(np.abs(np.atleast_1d(ref) - v)))
        out.append(np.atleast_1d(ref)[j])
    return np.asarray(out)


# ---------------------------------------------------------------------------
# 算例
# ---------------------------------------------------------------------------
def case_uniform():
    A = np.array([[-1.0, 0.5], [0.2, -3.0]])
    tau = 0.01
    ideal = np.linalg.eigvals(A)
    p_desc = uniform_delay_poles(A, tau)
    p_skew = skew_descriptor(A, tau).finite_eigs()
    # 两只极点按最近距离配对
    p_skew_m = _match(p_desc, p_skew)
    formula_err = float(np.max(np.abs(p_skew_m - p_desc) / np.abs(p_desc)))
    lead = float(np.max(np.abs((p_skew_m - ideal) / tau / (-ideal ** 2) - 1.0)))
    r = skew_report(A, tau)
    CASES["uniform"] = dict(
        A=A, tau=tau, ideal=ideal, poles_desc=p_desc, poles_skew=p_skew_m,
        formula_err=formula_err, leading_err=lead,
        kappa_ideal=r["kappa_ideal"], kappa_skew=r["kappa_skew"],
        gamma_ideal=r["gamma_ideal"], gamma_skew=r["gamma_skew"],
        E_cond=r["E_cond"], tau_crit=r["tau_crit_first_order"])


def case_scalar():
    a = -1.0
    rows = []
    for tau in (1e-3, 1e-2, 1e-1):
        s_desc = a / (1.0 + tau * a)
        s_dde = scalar_dde_root(a, tau)
        rows.append(dict(tau=tau, s_desc=s_desc, s_dde=s_dde,
                         err=abs(s_dde - s_desc) / abs(s_dde)))
    exact = scalar_tau_crit(a)
    first = uniform_tau_crit(np.array([[a]]))
    CASES["scalar"] = dict(a=a, rows=rows, tau_crit_exact=exact,
                           tau_crit_first=first, ratio=first / exact)


def case_diffusion():
    n = 8
    A = -2.0 * np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    tau = 5e-3
    r = skew_report(A, tau)
    tcrit = r["tau_crit_first_order"]
    stable_below = bool(skew_descriptor(A, 0.9 * tcrit).is_stable())
    stable_above = bool(skew_descriptor(A, 1.1 * tcrit).is_stable())
    CASES["diffusion"] = dict(
        n=n, tau=tau, kappa_ideal=r["kappa_ideal"],
        kappa_skew=r["kappa_skew"], gamma_ideal=r["gamma_ideal"],
        gamma_skew=r["gamma_skew"], E_cond=r["E_cond"],
        tau_crit=tcrit, stable_below=stable_below, stable_above=stable_above)


def case_shear():
    A = np.array([[-1.0, 10.0], [0.0, -2.0]])
    tau = 1e-3
    r = skew_report(A, tau)
    CASES["shear"] = dict(
        A=A, tau=tau, kappa_ideal=r["kappa_ideal"],
        kappa_skew=r["kappa_skew"], gamma_ideal=r["gamma_ideal"],
        gamma_skew=r["gamma_skew"], max_rel_shift=r["max_rel_shift"])


def case_nonuniform():
    a = np.array([-1.0, -2.0, -5.0])
    tau = np.array([1e-3, 5e-3, 1e-2])
    rows = []
    for ai, ti in zip(a, tau):
        pole = ai / (1.0 + ti * ai)          # diagonal E, diagonal A
        s_dde = scalar_dde_root(ai, ti)
        rows.append(dict(a=ai, tau=ti, pole=float(np.real(pole)),
                         dde=float(np.real(s_dde)),
                         err=abs(s_dde - pole) / abs(s_dde)))
    CASES["nonuniform"] = dict(rows=rows)


def case_phase():
    omega = 2.0 * np.pi * 100e6
    CASES["phase"] = dict(f=100e6,
                          tau_1deg=phase_budget(omega, 1.0),
                          tau_10deg=phase_budget(omega, 10.0))


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------
def acceptance():
    checks = []

    def add(name, got, expected, passed):
        checks.append(dict(name=name, got=got, expected=expected,
                           passed=bool(passed)))

    u = CASES["uniform"]
    add("uniform pole formula", "%.2e" % u["formula_err"], "< 1e-12",
        u["formula_err"] < 1e-12)
    add("uniform leading −τλ²", "%.2e" % u["leading_err"], "< 5e-2",
        u["leading_err"] < 5e-2)

    s = CASES["scalar"]
    for row in s["rows"]:
        limit = {1e-3: 1e-5, 1e-2: 1e-3, 1e-1: 2e-2}[row["tau"]]
        add("scalar DDE τ=%.0e" % row["tau"], "%.2e" % row["err"],
            "< %.0e" % limit, row["err"] < limit)
    add("scalar τ_crit ratio", "%.6f" % s["ratio"], "2/π = 0.636620",
        abs(s["ratio"] - 2.0 / np.pi) < 1e-6)

    d = CASES["diffusion"]
    add("diffusion κ grows", "%.3f → %.3f" % (d["kappa_ideal"], d["kappa_skew"]),
        "κ_skew > κ_ideal", d["kappa_skew"] > d["kappa_ideal"])
    add("diffusion E well-conditioned", "%.3f" % d["E_cond"], "< 1.5",
        d["E_cond"] < 1.5)
    add("diffusion stable below τ_crit", d["stable_below"], True,
        d["stable_below"])
    add("diffusion unstable above τ_crit", d["stable_above"], False,
        not d["stable_above"])

    ph = CASES["phase"]
    add("phase budget 1°", "%.2f ps" % (ph["tau_1deg"] * 1e12), "27.78 ps",
        abs(ph["tau_1deg"] * 1e12 - 27.78) < 0.05)
    add("phase budget 10°", "%.2f ps" % (ph["tau_10deg"] * 1e12), "277.8 ps",
        abs(ph["tau_10deg"] * 1e12 - 277.8) < 0.5)

    nu = CASES["nonuniform"]
    worst = max(r["err"] for r in nu["rows"])
    add("nonuniform diagonal roots", "%.2e" % worst, "< 5e-3", worst < 5e-3)

    sh = CASES["shear"]
    add("shear Γ finite", "%.3f → %.3f" % (sh["gamma_ideal"], sh["gamma_skew"]),
        "finite", np.isfinite(sh["gamma_skew"]))
    add("shear pole shift", "%.2e" % sh["max_rel_shift"], "> 0",
        sh["max_rel_shift"] > 0)

    return checks


def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()

    hr("Phase 1.5 验收：互连延迟 → E=I+A∘T")

    case_uniform()
    case_scalar()
    case_diffusion()
    case_shear()
    case_nonuniform()
    case_phase()

    hr("A. 均匀延迟：闭式极点与一阶展开")
    u = CASES["uniform"]
    table(["quantity", "value"],
          [("ideal λ", np.array2string(u["ideal"], precision=6)),
           ("descriptor λ", np.array2string(u["poles_desc"], precision=6)),
           ("formula err", "%.2e" % u["formula_err"]),
           ("leading −τλ² err", "%.2e" % u["leading_err"]),
           ("κ ideal → skew", "%.3f → %.3f" % (u["kappa_ideal"], u["kappa_skew"])),
           ("Γ ideal → skew", "%.3f → %.3f" % (u["gamma_ideal"], u["gamma_skew"])),
           ("cond(E)", "%.3f" % u["E_cond"]),
           ("τ_crit (first order)", "%.4f" % u["tau_crit"])])

    hr("B. 标量 DDE：一阶展开的误差与稳定边界")
    s = CASES["scalar"]
    table(["τ", "s_descriptor", "s_DDE", "rel err"],
          [("%.0e" % r["tau"], "%.6f" % r["s_desc"].real,
            "%.6f" % r["s_dde"].real, "%.2e" % r["err"]) for r in s["rows"]])
    emit("  exact τ_crit = π/2 = %.4f；E=I+τA 给 %.4f；ratio = %.4f（= 2/π）"
         % (s["tau_crit_exact"], s["tau_crit_first"], s["ratio"]))

    hr("C. 扩散链：κ 与稳定边界")
    d = CASES["diffusion"]
    table(["quantity", "value"],
          [("n", d["n"]),
           ("τ", "%.1e" % d["tau"]),
           ("κ ideal → skew", "%.3f → %.3f" % (d["kappa_ideal"], d["kappa_skew"])),
           ("Γ ideal → skew", "%.3f → %.3f" % (d["gamma_ideal"], d["gamma_skew"])),
           ("cond(E)", "%.3f" % d["E_cond"]),
           ("τ_crit (first order)", "%.4f" % d["tau_crit"]),
           ("stable @ 0.9 τ_crit", d["stable_below"]),
           ("stable @ 1.1 τ_crit", d["stable_above"])])

    hr("D. 非均匀延迟（对角系统）")
    table(["a", "τ", "descriptor", "DDE", "rel err"],
          [("%.1f" % r["a"], "%.0e" % r["tau"], "%.6f" % r["pole"],
            "%.6f" % r["dde"], "%.2e" % r["err"]) for r in CASES["nonuniform"]["rows"]])

    hr("E. 相位预算")
    ph = CASES["phase"]
    emit("  f = 100 MHz：1° → %.2f ps；10° → %.2f ps"
         % (ph["tau_1deg"] * 1e12, ph["tau_10deg"] * 1e12))
    emit("  这就是 §2.8 的 ω_max Δτ_route ≪ 1 换算成几何/延迟预算后的样子。")

    hr("F. 非正规 shear")
    sh = CASES["shear"]
    table(["quantity", "value"],
          [("A", np.array2string(sh["A"])),
           ("τ", "%.1e" % sh["tau"]),
           ("κ ideal → skew", "%.3f → %.3f" % (sh["kappa_ideal"], sh["kappa_skew"])),
           ("Γ ideal → skew", "%.3f → %.3f" % (sh["gamma_ideal"], sh["gamma_skew"])),
           ("max rel pole shift", "%.2e" % sh["max_rel_shift"])])

    hr("G. 验收")
    checks = acceptance()
    table(["check", "got", "expected", "PASS"],
          [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
           for c in checks])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    payload = dict(cases={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                              for kk, vv in v.items()} for k, v in CASES.items()},
                   checks=checks,
                   summary=dict(n_checks=len(checks), n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "skew.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(os.path.join(OUT, "skew_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/skew.json")
    emit("  report → results/skew_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
