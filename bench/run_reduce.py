"""Phase 1 前端验收：方程 → canonical IR（DESIGN.md §1.5、§7 Phase 1）。

    python3 -m bench.run_reduce

五个算例，分别压一条边界：

    Lorenz      polynomial，3 状态，degree 2
    Duffing     polynomial，cos 增广后 4 状态，degree 3；s²+c² 漂移 < 1e-3
    HH          rational，6 个 exp 增广 + 3 个分母；标准 HH 不是 polynomial
    Bessel J0   rational，1/t；t≥1 可增广，t=0 拒绝
    Gamma       rejected，differentially transcendental

HH 的结论是对 DESIGN.md 早期措辞的修正：Lorenz/Duffing 落到同一 sparse
polynomial IR；标准 Hodgkin–Huxley 的 α/β 含 1/(1−exp)，只能落到 rational
IR，需要 divider 原语或额外的多项式拟合。不要把"常见"当成"polynomial"。
"""

from __future__ import annotations

import json
import math
import os

# 与其它 bench 一致：本机 OpenBLAS 默认多线程在这些小矩阵上反而慢。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.special import jv, jvp

from lowering.reduce import (C, System, V, add, besselj, complete_env, cos,
                             div, eval_expr, eval_system, exp, first_order,
                             gamma, mul, pow, reduce_system, sub)

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


def table(header, rows, fmts):
    cells = [[(f % v) if (isinstance(f, str) and f) else str(v)
              for f, v in zip(fmts, r)] for r in rows]
    widths = [max(len(header[j]), max((len(c[j]) for c in cells), default=0))
              for j in range(len(header))]
    emit("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for c in cells:
        emit("  " + " ".join(v.rjust(w) for v, w in zip(c, widths)))


def _fmt(v, spec="%.3g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "inf" if v == float("inf") else "n/a"
    return spec % v


# ---------------------------------------------------------------------------
# 算例
# ---------------------------------------------------------------------------
def lorenz():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    sys = System(
        ["x", "y", "z"],
        {"x": add(mul(sigma, V("y")), mul(-sigma, V("x"))),
         "y": add(mul(rho, V("x")), mul(-1.0, V("x"), V("z")),
                  mul(-1.0, V("y"))),
         "z": add(mul(V("x"), V("y")), mul(-beta, V("z")))},
    )
    return reduce_system(sys)


def duffing():
    delta, alpha, beta, gam, omega = 0.2, -1.0, 1.0, 0.3, 1.2
    rhs = add(mul(-delta, V("x_1")), mul(-alpha, V("x_0")),
              mul(-beta, pow(V("x_0"), 3)),
              mul(gam, cos(mul(omega, V("t")))))
    return reduce_system(first_order("x", 2, rhs))


def hh():
    """标准 Hodgkin–Huxley（I=0，C=1）。单位：mV、ms、µA/cm²。"""
    V_, m, h, n = V("V"), V("m"), V("h"), V("n")

    def rate(num, den):
        return div(num, den)

    a_m = rate(mul(0.1, add(V_, 40)),
               sub(1, exp(mul(div(C(-1), C(10)), add(V_, 40)))))
    b_m = mul(4, exp(mul(div(C(-1), C(18)), add(V_, 65))))
    a_h = mul(0.07, exp(mul(div(C(-1), C(20)), add(V_, 65))))
    b_h = div(1, add(1, exp(mul(div(C(-1), C(10)), add(V_, 35)))))
    a_n = rate(mul(0.01, add(V_, 10)),
               sub(1, exp(mul(div(C(-1), C(10)), add(V_, 10)))))
    b_n = mul(0.125, exp(mul(div(C(-1), C(80)), add(V_, 65))))

    rhs = {
        "V": sub(0, add(mul(120, pow(m, 3), h, sub(V_, 50)),
                        mul(36, pow(n, 4), sub(V_, -77)),
                        mul(0.3, sub(V_, -54.387)))),
        "m": sub(mul(a_m, sub(1, m)), mul(b_m, m)),
        "h": sub(mul(a_h, sub(1, h)), mul(b_h, h)),
        "n": sub(mul(a_n, sub(1, n)), mul(b_n, n)),
    }
    return reduce_system(System(["V", "m", "h", "n"], rhs))


def bessel0():
    return System(["x", "t"], {"x": besselj(0, V("t")), "t": C(1.0)})


def hh_reference(Vv, mv, hv, nv):
    def am(v):
        return 0.1 * (v + 40) / (1.0 - math.exp(-(v + 40) / 10.0))

    def bm(v):
        return 4.0 * math.exp(-(v + 65) / 18.0)

    def ah(v):
        return 0.07 * math.exp(-(v + 65) / 20.0)

    def bh(v):
        return 1.0 / (1.0 + math.exp(-(v + 35) / 10.0))

    def an(v):
        return 0.01 * (v + 10) / (1.0 - math.exp(-(v + 10) / 10.0))

    def bn(v):
        return 0.125 * math.exp(-(v + 65) / 80.0)

    return {
        "V": -(120 * mv ** 3 * hv * (Vv - 50) + 36 * nv ** 4 * (Vv + 77)
               + 0.3 * (Vv + 54.387)),
        "m": am(Vv) * (1 - mv) - bm(Vv) * mv,
        "h": ah(Vv) * (1 - hv) - bh(Vv) * hv,
        "n": an(Vv) * (1 - nv) - bn(Vv) * nv,
    }


# ---------------------------------------------------------------------------
# 积分
# ---------------------------------------------------------------------------
def rhs_vec(system: System, y: np.ndarray) -> np.ndarray:
    env = dict(zip(system.states, y))
    d = eval_system(system, env)
    return np.array([d[s] for s in system.states], dtype=float)


def integrate(system: System, y0, t0, t1, h):
    y = np.asarray(y0, dtype=float)
    nstep = int(round((t1 - t0) / h))
    for _ in range(nstep):
        k1 = rhs_vec(system, y)
        k2 = rhs_vec(system, y + 0.5 * h * k1)
        k3 = rhs_vec(system, y + 0.5 * h * k2)
        k4 = rhs_vec(system, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


def _initial(red, base: dict) -> np.ndarray:
    env = complete_env(red, base)
    return np.array([env[s] for s in red.system.states], dtype=float)


def duffing_drift(red, perturb: float = 0.0):
    system = red.system
    if perturb:
        s_name = red.augmentations[0]["states"][0]
        eq = dict(system.rhs)
        eq[s_name] = mul(C(1.0 + perturb), eq[s_name])
        system = System(list(system.states), eq,
                        list(system.invariants), system.time)
    y = _initial(red, {"x_0": 0.1, "x_1": 0.0, "t": 0.0})
    s_name, c_name = red.augmentations[0]["states"]
    i_s = system.states.index(s_name)
    i_c = system.states.index(c_name)
    h = 1e-3
    worst = 0.0
    for _ in range(20000):
        k1 = rhs_vec(system, y)
        k2 = rhs_vec(system, y + 0.5 * h * k1)
        k3 = rhs_vec(system, y + 0.5 * h * k2)
        k4 = rhs_vec(system, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        worst = max(worst, abs(y[i_s] ** 2 + y[i_c] ** 2 - 1.0))
    return worst


def bessel_check(red):
    env0 = complete_env(red, {"x": 0.0, "t": 1.0})
    y0 = np.array([env0[s] for s in red.system.states], dtype=float)
    y = integrate(red.system, y0, 1.0, 10.0, 1e-3)
    env = dict(zip(red.system.states, y))
    j_name, k_name = red.augmentations[0]["states"]
    ref_j = float(jv(0, 10.0))
    ref_k = float(jvp(0, 10.0))
    return dict(t=float(env["t"]), j=float(env[j_name]), k=float(env[k_name]),
                ref_j=ref_j, ref_k=ref_k,
                j_err=abs(env[j_name] - ref_j) / abs(ref_j),
                k_err=abs(env[k_name] - ref_k) / max(abs(ref_k), 1e-300))


def eval_poly(poly, env: dict, states) -> float:
    out = 0.0
    for mono, coeff in poly.items():
        term = float(coeff)
        for s, p in zip(states, mono):
            if p:
                term *= env[s] ** p
        out += term
    return out


def canonical_error(red, env: dict) -> float:
    """canonical 有理式在 env 上重建 RHS 的相对误差（验证 _poly_frac / 约分）。"""
    ref = eval_system(red.system, env)
    worst = 0.0
    for s in red.system.states:
        num = eval_poly(red.canonical.num[s], env, red.canonical.states)
        den = eval_poly(red.canonical.den[s], env, red.canonical.states)
        got = num / den
        worst = max(worst, abs(got - ref[s]) / max(abs(ref[s]), 1e-12))
    return worst


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_case(name: str, red):
    c = dict(name=name, kind=red.kind, level=red.level, ok=bool(red.ok),
             reason=red.reason, n=len(red.system.states) if red.system else 0,
             states=list(red.system.states) if red.system else [],
             augmentations=[(a["kind"], str(a["arg"]), list(a["states"]))
                            for a in red.augmentations],
             domain=red.domain)
    if red.canonical is not None:
        c.update(degree=red.canonical.degree,
                 n_monomials=red.canonical.n_monomials,
                 dense_monomials=red.canonical.dense_monomials,
                 sparsity=red.canonical.sparsity,
                 divisions=list(red.canonical.divisions),
                 invariants=list(red.canonical.invariants),
                 canonical=red.canonical.as_dict())
    return c


def acceptance() -> list[dict]:
    checks: list[dict] = []

    def add_check(name, got, expected, passed, kind="value"):
        checks.append(dict(name=name, got=got, expected=expected,
                           passed=bool(passed), kind=kind))

    lor = CASES["Lorenz"]
    add_check("Lorenz kind", lor["kind"], "polynomial", lor["kind"] == "polynomial")
    add_check("Lorenz states", lor["n"], 3, lor["n"] == 3)
    add_check("Lorenz degree", lor["degree"], 2, lor["degree"] == 2)
    add_check("Lorenz monomials", lor["n_monomials"], 7, lor["n_monomials"] == 7)
    add_check("Lorenz sparse", lor["sparsity"], "< 1",
              lor["n_monomials"] < lor["dense_monomials"])

    duf = CASES["Duffing"]
    add_check("Duffing kind", duf["kind"], "polynomial",
              duf["kind"] == "polynomial")
    add_check("Duffing states", duf["n"], 4, duf["n"] == 4)
    add_check("Duffing degree", duf["degree"], 3, duf["degree"] == 3)
    add_check("Duffing monomials", duf["n_monomials"], 7,
              duf["n_monomials"] == 7)
    drift = CASES["Duffing"]["invariant_drift"]
    add_check("Duffing s²+c² drift", drift, "< 1e-3", drift < 1e-3, "abs")

    for key in ("Lorenz", "Duffing", "HH", "Bessel"):
        err = CASES[key]["canonical_err"]
        add_check("%s canonical rebuild" % key, err, "< 1e-10",
                  err < 1e-10, "abs")

    hh_c = CASES["HH"]
    add_check("HH kind", hh_c["kind"], "rational", hh_c["kind"] == "rational")
    add_check("HH states", hh_c["n"], 10, hh_c["n"] == 10)
    add_check("HH divisions", len(hh_c["divisions"]), 3,
              len(hh_c["divisions"]) == 3)
    add_check("HH RHS rel err", hh_c["rhs_err"], "< 1e-10",
              hh_c["rhs_err"] < 1e-10, "abs")
    add_check("HH sparse", hh_c["sparsity"], "< 0.01",
              hh_c["sparsity"] < 0.01)

    gam = CASES["Gamma"]
    add_check("Gamma rejected", gam["kind"], "rejected", gam["kind"] == "rejected")

    bes = CASES["Bessel"]
    add_check("Bessel kind", bes["kind"], "rational", bes["kind"] == "rational")
    add_check("Bessel domain", bes["ok"], True, bool(bes["ok"]))
    add_check("Bessel J0 rel err", bes["j_err"], "< 1e-6",
              bes["j_err"] < 1e-6, "abs")
    add_check("Bessel at x=0 rejected", bes["rejected_zero"],
              "rejected", bes["rejected_zero"] == "rejected")

    shared = (lor["kind"] == duf["kind"] == "polynomial"
              and lor["sparsity"] < 0.5 and duf["sparsity"] < 0.5)
    add_check("Lorenz/Duffing share polynomial IR", "polynomial",
              "polynomial", shared)

    return checks


def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()

    hr("Phase 1 前端验收：方程 → canonical IR")

    # ---- 逐例 ----
    red = lorenz()
    CASES["Lorenz"] = run_case("Lorenz", red)
    CASES["Lorenz"]["canonical_err"] = canonical_error(
        red, {"x": 1.2, "y": -0.7, "z": 2.3})

    red_d = duffing()
    CASES["Duffing"] = run_case("Duffing", red_d)
    CASES["Duffing"]["invariant_drift"] = duffing_drift(red_d)
    CASES["Duffing"]["invariant_drift_perturbed"] = duffing_drift(red_d, 1e-3)
    CASES["Duffing"]["canonical_err"] = canonical_error(
        red_d, complete_env(red_d, {"x_0": 0.3, "x_1": -0.2, "t": 0.4}))

    red_h = hh()
    CASES["HH"] = run_case("HH", red_h)
    env = complete_env(red_h, {"V": -60.0, "m": 0.05, "h": 0.6, "n": 0.32})
    got = eval_system(red_h.system, env)
    ref = hh_reference(-60.0, 0.05, 0.6, 0.32)
    err = max(abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-12)
              for k in ("V", "m", "h", "n"))
    CASES["HH"]["rhs_err"] = float(err)
    CASES["HH"]["canonical_err"] = canonical_error(red_h, env)

    red_b = reduce_system(bessel0(), bessel_arg_min=1.0)
    CASES["Bessel"] = run_case("Bessel", red_b)
    CASES["Bessel"].update(bessel_check(red_b))
    CASES["Bessel"]["canonical_err"] = canonical_error(
        red_b, complete_env(red_b, {"x": 0.0, "t": 1.0}))
    CASES["Bessel"]["rejected_zero"] = reduce_system(
        bessel0(), bessel_arg_min=0.0).kind

    CASES["Gamma"] = run_case(
        "Gamma", reduce_system(System(["x"], {"x": gamma(V("x"))})))

    hr("A. 分类与 canonical IR")
    table(["system", "kind", "level", "n", "degree", "monomials", "dense",
           "sparsity", "divisions", "note"],
          [(k, c["kind"], c["level"], c["n"], c.get("degree", "-"),
            c.get("n_monomials", "-"), c.get("dense_monomials", "-"),
            _fmt(c.get("sparsity")), len(c.get("divisions", [])),
            (c["reason"][:36] + "…") if c.get("reason") else "")
           for k, c in CASES.items()],
          [""] * 10)

    hr("B. 状态与增广")
    for key in ("Lorenz", "Duffing", "HH", "Bessel"):
        c = CASES[key]
        emit("  %-8s states = %s" % (key, ", ".join(c["states"])))
        for kind, arg, states in c["augmentations"]:
            emit("           + %-7s %-24s → %s" % (kind, arg, ", ".join(states)))
    emit("  Gamma    rejected: %s" % CASES["Gamma"]["reason"])

    hr("C. 验收")
    checks = acceptance()
    table(["check", "got", "expected", "PASS"],
          [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
           for c in checks],
          ["", "%s", "%s", ""])
    n_fail = sum(1 for c in checks if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    hr("D. 读法")
    duf_c = CASES["Duffing"]
    emit("  - Lorenz/Duffing 都落到 polynomial IR：n=%d/%d，monomials=%d/%d，"
         "dense=%d/%d。"
         % (CASES["Lorenz"]["n"], duf_c["n"],
            CASES["Lorenz"]["n_monomials"], duf_c["n_monomials"],
            CASES["Lorenz"]["dense_monomials"], duf_c["dense_monomials"]))
    emit("    Duffing 的 cos(ωt) 增广成 (s,c)，s²+c²−1=0；RK4 的实测最大漂移 %.2e。"
         % duf_c["invariant_drift"])
    emit("    把 s· 乘以 (1+1e-3) 后漂移升到 %.2e——不变流形不是自稳定的。"
         % duf_c["invariant_drift_perturbed"])
    emit("  - 标准 HH 是 rational，不是 polynomial：6 个 exp 增广 + %d 个分母"
         % len(CASES["HH"]["divisions"]))
    emit("    （%s）。α/β 的 1/(1−exp) 必须用 divider 原语或额外拟合；"
         % "; ".join(CASES["HH"]["divisions"]))
    emit("    这是对 §7 Phase 1 早期措辞的修正。")
    emit("    HH 的 degree=%d 是因为 exp 状态继承 V̇ 的复杂度；sparsity=%.1e，"
         % (CASES["HH"]["degree"], CASES["HH"]["sparsity"]))
    emit("    但 numerator monomials=%d，仍远小于 dense=%d。"
         % (CASES["HH"]["n_monomials"], CASES["HH"]["dense_monomials"]))
    emit("  - Bessel J0 可精确增广成有理系统（1/t），t≥1 时相对误差 %.1e；"
         % CASES["Bessel"]["j_err"])
    emit("    t=0 被拒绝：1/x 增广在 x=0 处失效。Gamma 没有有限阶代数增广。")

    payload = dict(cases=CASES, checks=checks,
                   summary=dict(n_checks=len(checks), n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "reduce.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    with open(os.path.join(OUT, "reduce_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/reduce.json")
    emit("  report → results/reduce_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
