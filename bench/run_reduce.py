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
from fractions import Fraction

# 与其它 bench 一致：本机 OpenBLAS 默认多线程在这些小矩阵上反而慢。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.special import jv, jvp

from lowering.reduce import (C, System, V, add, besselj, complete_env, cos,
                             div, eval_expr, eval_system, exp, first_order,
                             gamma, mul, pow, reduce_system,
                             reduce_with_primitives, sub, tanh)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_LINES: list[str] = []
CASES: dict[str, dict] = {}
REDS: dict[str, object] = {}       # Reduced 对象（domain 检查要看原始 aug dict）


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


def duffing_system():
    delta, alpha, beta, gam, omega = 0.2, -1.0, 1.0, 0.3, 1.2
    rhs = add(mul(-delta, V("x_1")), mul(-alpha, V("x_0")),
              mul(-beta, pow(V("x_0"), 3)),
              mul(gam, cos(mul(omega, V("t")))))
    return first_order("x", 2, rhs)


def duffing():
    return reduce_system(duffing_system())


def hh_system():
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
    return System(["V", "m", "h", "n"], rhs)


def hh():
    return reduce_system(hh_system())


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


def shared_denominator():
    """a/x + b/x 形：add 分支同分母必须先合并分子，den 不能被平方。"""
    sys = System(["x", "y", "z"], {
        "x": add(div(V("x"), V("z")), div(V("y"), V("z"))),
        "y": C(0.0), "z": C(0.0)})
    red = reduce_system(sys)
    den = red.canonical.den["x"]
    monos = sorted(den.items())
    ok = (len(monos) == 1 and monos[0][0] == (0, 0, 1)
          and monos[0][1] == Fraction(1))
    num_ok = (len(red.canonical.num["x"]) == 2
              and red.canonical.num["x"].get((1, 0, 0)) == Fraction(1)
              and red.canonical.num["x"].get((0, 1, 0)) == Fraction(1))
    return dict(kind="polynomial", level=0, n=3, monos=monos,
                num=dict(red.canonical.num["x"]),
                ok=bool(ok and num_ok))


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


def manifold_variance(gamma, *, offset=1e-3, noise=3e-6, nstep=2000, h=1e-3,
                      seed=7):
    """P1-manifold-stoch：Baumgarte 法向的随机验收（有偏漂移 + 噪声）。

    注入 (a) 常值法向 offset：rhs[s] += ε·s, rhs[c] += ε·c——即 ε_off=ε 乘
    单位法向 (s,c)，流形上 δ̇ = 2ε，无稳定化时 δ ∝ t 线性增长；
    (b) 每步高斯扰动 σ_n 加到 s、c——连续时间对应 D=σ_n²/h 的白噪声，
    无稳定化时 Var(δ) ∝ t（Brown）。γ>0 时 δ̇ ≈ −2γδ + 2ε + 白噪声，
    稳态 E[g²] = a/γ + b/γ²（噪声项 /γ，offset 项 /γ²）。

    返回 first/last 四分之一区间的 mean(g²) 与全程 max|g|。
    """
    red = reduce_system(duffing_system(), manifold_gamma=(gamma or None))
    system = red.system
    s_name, c_name = red.augmentations[0]["states"]
    eq = dict(system.rhs)
    if offset:
        eq[s_name] = add(eq[s_name], mul(C(offset), V(s_name)))
        eq[c_name] = add(eq[c_name], mul(C(offset), V(c_name)))
    sys2 = System(list(system.states), eq, list(system.invariants),
                  system.time)
    i_s = sys2.states.index(s_name)
    i_c = sys2.states.index(c_name)
    y = _initial(red, {"x_0": 0.1, "x_1": 0.0, "t": 0.0})
    rng = np.random.default_rng(seed)
    q = nstep // 4
    g2_first: list[float] = []
    g2_last: list[float] = []
    max_abs = 0.0
    for step in range(nstep):
        k1 = rhs_vec(sys2, y)
        k2 = rhs_vec(sys2, y + 0.5 * h * k1)
        k3 = rhs_vec(sys2, y + 0.5 * h * k2)
        k4 = rhs_vec(sys2, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if noise:
            y[i_s] += noise * rng.standard_normal()
            y[i_c] += noise * rng.standard_normal()
        g = y[i_s] ** 2 + y[i_c] ** 2 - 1.0
        max_abs = max(max_abs, abs(g))
        if step < q:
            g2_first.append(g * g)
        elif step >= nstep - q:
            g2_last.append(g * g)
    return dict(g2_first=float(np.mean(g2_first)),
                g2_last=float(np.mean(g2_last)),
                max_abs=float(max_abs))


def _jac_spectral_radius(system: System, y: np.ndarray) -> float:
    """增广 Jacobian 的 max|λ|（中心差分，小系统直接满秩算）。"""
    n = len(y)
    y = np.asarray(y, dtype=float)
    J = np.zeros((n, n))
    for j in range(n):
        dy = 1e-6 * max(abs(y[j]), 1.0)
        yp = y.copy(); yp[j] += dy
        ym = y.copy(); ym[j] -= dy
        J[:, j] = (rhs_vec(system, yp) - rhs_vec(system, ym)) / (2.0 * dy)
    return float(np.max(np.abs(np.linalg.eigvals(J))))


def _augmented_radius(gamma: float) -> float:
    red = reduce_system(duffing_system(), manifold_gamma=gamma)
    y = _initial(red, {"x_0": 0.1, "x_1": 0.0, "t": 0.0})
    return _jac_spectral_radius(red.system, y)


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
    drift_s = CASES["Duffing"]["invariant_drift_stabilized"]
    drift_p = CASES["Duffing"]["invariant_drift_perturbed"]
    add_check("Duffing stabilized drift (γ=100, injected) < 1% of bare",
              "%.2e vs %.2e" % (drift_s, drift_p), "stabilized < 0.01·bare",
              drift_s < 0.01 * drift_p and drift_s < 1e-5, "abs")
    add_check("Duffing aug records stabilized/gamma",
              "%s, %s" % (CASES["Duffing"]["aug_stabilized"],
                          CASES["Duffing"]["aug_gamma"]),
              "True, 100.0",
              CASES["Duffing"]["aug_stabilized"]
              and CASES["Duffing"]["aug_gamma"] == 100.0)

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
    add_check("HH augmented monomials", hh_c["n_monomials"], 65,
              hh_c["n_monomials"] == 65)
    add_check("HH augmented degree", hh_c["degree"], 6, hh_c["degree"] == 6)

    nat = CASES["HH"]["native"]
    add_check("HH native states", nat["n_states"], 4, nat["n_states"] == 4)
    add_check("HH native primitives (exp tiles)", nat["n_primitives"], 6,
              nat["n_primitives"] == 6)
    add_check("HH native degree < augmented degree",
              "%d vs %d" % (nat["degree"], hh_c["degree"]),
              "native < augmented", nat["degree"] < hh_c["degree"])

    man = CASES["Manifold"]
    c0 = man["combined"]["0.0"]
    cg0 = man["combined"][str(man["g0"])]
    grow = c0["g2_last"] / max(c0["g2_first"], 1e-300)
    add_check("Manifold γ=0: E[g²] grows (last/first)", "%.1f" % grow,
              "≥ 10 (no saturation)", grow >= 10.0)
    add_check("Manifold γ=γ0 plateau vs γ=0", "%.2e vs %.2e"
              % (cg0["g2_last"], c0["g2_last"]),
              "plateau < 0.1·unstable", cg0["g2_last"] < 0.1 * c0["g2_last"])
    fit = man["fit"]
    add_check("Manifold two-term fit a/γ + b/γ²",
              "a=%.2e b=%.2e resid=%.1e"
              % (fit["a"], fit["b"], fit["rel_resid"]),
              "a>0, b>0, rel resid < 0.2",
              fit["a"] > 0 and fit["b"] > 0 and fit["rel_resid"] < 0.2)
    r_off = (man["offset_only"]["10.0"]["g2_last"]
             / man["offset_only"]["100.0"]["g2_last"])
    add_check("Manifold offset-only ∝ 1/γ² (γ0/10γ0)", "%.1f" % r_off,
              "≈ 100 (E ∝ b/γ²)", 30.0 <= r_off <= 300.0)
    r_nz = (man["noise_only"]["10.0"]["g2_last"]
            / man["noise_only"]["100.0"]["g2_last"])
    add_check("Manifold noise-only ∝ 1/γ (γ0/10γ0)", "%.1f" % r_nz,
              "≈ 10 (E ∝ a/γ)", 4.0 <= r_nz <= 30.0)
    p1000 = man["plateau_1000x"]
    add_check("Manifold γ=1000× plateau bounded", "%.2e" % p1000["max_abs"],
              "max|g| < 1e-3", p1000["max_abs"] < 1e-3)
    sp = man["spectral"]
    add_check("Manifold γ=1000× pushes max|λ|",
              "%.3g vs %.3g" % (sp["radius_1000x"], sp["radius_gamma0"]),
              "> 100×", sp["radius_1000x"] > 100.0 * sp["radius_gamma0"])
    add_check("Manifold feasible γ interval non-empty",
              "[%.3g, %.3g]" % (sp["gamma_min"], sp["gamma_max"]),
              "γ_min < γ0 < γ_max (no forced native verdict)",
              sp["gamma_min"] < man["g0"] < sp["gamma_max"])

    gam = CASES["Gamma"]
    add_check("Gamma rejected", gam["kind"], "rejected", gam["kind"] == "rejected")

    sd = CASES["SharedDen"]
    add_check("a/x + b/x denominator not squared", "%.3f·x + %.3f·y over %s"
              % (sd["num"].get((1, 0, 0), 0), sd["num"].get((0, 1, 0), 0),
                 sd["monos"]),
              "numerator summed, den = z^1 (not z^2)", sd["ok"])

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

    tanh_c = CASES["Tanh"]
    add_check("Tanh kind", tanh_c["kind"], "polynomial",
              tanh_c["kind"] == "polynomial")
    # 域证书检查直接看 red.augmentations（run_case 只保留 kind/arg/states）。
    dom_tanh = [a.get("domain") for a in REDS["Tanh"].augmentations if a["kind"] == "tanh"]
    add_check("Tanh augmentation records domain (h, −1, 1)",
              "%s" % (dom_tanh,), "[('h1', -1.0, 1.0)]",
              dom_tanh == [("h1", -1.0, 1.0)])
    dom_exp = [a.get("domain") for a in REDS["HH"].augmentations if a["kind"] == "exp"]
    add_check("HH exp augmentations record domain (e_k, 0, None)",
              "%d × %s" % (len(dom_exp), dom_exp[0] if dom_exp else "-"),
              "6 × (e_k, 0.0, None)",
              len(dom_exp) == 6 and all(
                  isinstance(d, tuple) and len(d) == 3 and d[1] == 0.0
                  and d[2] is None for d in dom_exp))

    return checks


def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()
    REDS.clear()

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

    # Baumgarte 流形投影（P1-manifold-stabilization）：γ=100 的稳定化版本，
    # 同样的注入扰动（rhs[s] × (1+1e-3)）下法向偏差被拉回流形。注意这不是
    # 衰减到零：持续扰动下 δ̇ = 0.002·s·c·u̇ − 2γ(s²+c²)δ 有稳态平衡
    # δ_ss ∝ 1/γ；γ 越大越贴回流形。
    MANIFOLD_GAMMA = 100.0
    red_ds = reduce_system(duffing_system(), manifold_gamma=MANIFOLD_GAMMA)
    CASES["Duffing"]["stabilized_gamma"] = MANIFOLD_GAMMA
    CASES["Duffing"]["aug_stabilized"] = bool(
        red_ds.augmentations[0].get("stabilized"))
    CASES["Duffing"]["aug_gamma"] = red_ds.augmentations[0].get("gamma")
    CASES["Duffing"]["invariant_drift_stabilized"] = duffing_drift(red_ds, 1e-3)

    CASES["SharedDen"] = shared_denominator()

    # ---- P1-domain：增广状态的域证书（硬件映射提示，review 修复）。
    # tanh 增广 h 满足 ḣ=(1−h²)u̇：|h|>1 时系数反号，负反馈变正反馈——
    # 硬件 offset/噪声会让增广状态逃逸。augmentation dict 必须记录
    # domain=(h, −1, 1)；exp 记 (e, 0, None)；sin/cos 的域由不变流形蕴含。
    red_tanh = reduce_system(System(["x"], {"x": tanh(V("x"))}))
    CASES["Tanh"] = run_case("Tanh", red_tanh)
    REDS["Tanh"] = red_tanh

    red_h = hh()
    REDS["HH"] = red_h
    CASES["HH"] = run_case("HH", red_h)
    env = complete_env(red_h, {"V": -60.0, "m": 0.05, "h": 0.6, "n": 0.32})
    got = eval_system(red_h.system, env)
    ref = hh_reference(-60.0, 0.05, 0.6, 0.32)
    err = max(abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-12)
              for k in ("V", "m", "h", "n"))
    CASES["HH"]["rhs_err"] = float(err)
    CASES["HH"]["canonical_err"] = canonical_error(red_h, env)

    # ---- P1-native-primitives：增广 vs 原生 tile（exp/tanh 不增广，留作
    # 原子 gm 律：亚阈 MOS exp、差分对 tanh）的成本对照。默认路径不受影响。----
    CASES["HH"]["native"] = reduce_with_primitives(hh_system()).as_dict()

    # ---- P1-manifold-stoch：Baumgarte 法向的随机验收。注入 (a) 常值法向
    # offset ε_off=1e-3·(单位法向)，(b) 每步高斯扰动 σ_n。无稳定化时法向
    # 特征值为 0：offset 使 δ ∝ t 线性、噪声使 Var(δ) ∝ t（Brown）；γ>0
    # 时 E[g²](∞) = a/γ + b/γ²。σ_n 取 3e-6 使两项在 γ0 附近同量级、可分辨。----
    MAN_G0 = 100.0
    MAN_TOL = 1e-2          # 法向偏差容差：流形上 1% 的 g 误差
    MAN_KAPPA = 1e6         # κ_hw 预算（max|λ| 上限）
    man = dict(g0=MAN_G0, tol=MAN_TOL, kappa_hw=MAN_KAPPA,
               combined={}, offset_only={}, noise_only={})
    for g in (0.0, 10.0, 25.0, 50.0, 100.0, 200.0):
        man["combined"][str(g)] = manifold_variance(g)
    man["offset_only"]["100.0"] = manifold_variance(100.0, noise=0.0)
    man["offset_only"]["10.0"] = manifold_variance(10.0, noise=0.0)
    man["noise_only"]["100.0"] = manifold_variance(100.0, offset=0.0)
    man["noise_only"]["10.0"] = manifold_variance(10.0, offset=0.0)

    # 两项拟合限制在 2γh ≪ 1 的区间：每步脉冲式注入的噪声在 2γh → ∞ 时
    # Var(Δ) → σ²（每步衰减 R(−2γh) → 0），1/γ 律只在 2γh ≪ 1 成立。
    gs = np.array([10.0, 25.0, 50.0, 100.0])
    E = np.array([man["combined"][str(g)]["g2_last"] for g in gs])
    X = np.stack([1.0 / gs, 1.0 / gs ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(X, E, rcond=None)
    man["fit"] = dict(a=float(coef[0]), b=float(coef[1]),
                      rel_resid=float(np.max(np.abs(X @ coef - E))
                                      / max(float(np.max(E)), 1e-300)))

    b_off = man["offset_only"]["100.0"]["g2_last"] * 100.0 ** 2
    a_nz = man["noise_only"]["100.0"]["g2_last"] * 100.0
    r_lo = _augmented_radius(100.0)
    r_hi = _augmented_radius(1e5)
    man["spectral"] = dict(radius_gamma0=r_lo, radius_1000x=r_hi,
                           gamma_min=max(math.sqrt(b_off / MAN_TOL ** 2),
                                         a_nz / MAN_TOL),
                           gamma_max=1e5 * (MAN_KAPPA / r_hi))
    # 1000× 的 γ 在 h=1e-3 下 RK4 本身失稳（γh=100），按稳定步长 h=1e-6 积分
    man["plateau_1000x"] = manifold_variance(1e5, h=1e-6)
    CASES["Manifold"] = man


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
           for k, c in CASES.items() if "kind" in c],
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

    hr("E. Baumgarte 流形随机验收（P1-manifold-stoch）")
    man = CASES["Manifold"]
    emit("  注入：offset ε=1e-3·(单位法向) + 每步高斯 σ_n=3e-6；RK4，h=1e-3，"
         "2000 步。E[g²](∞) ≈ a/γ + b/γ²。")
    rows = [(g, d["g2_first"], d["g2_last"], d["max_abs"])
            for g, d in sorted(man["combined"].items(), key=lambda kv: float(kv[0]))]
    table(["γ", "E[g²] first ¼", "E[g²] last ¼", "max|g|"], rows,
          ["%s", "%.3e", "%.3e", "%.3e"])
    fit = man["fit"]
    emit("  两项拟合：E[g²] = %.3e/γ + %.3e/γ²，最大相对残差 %.1e。"
         % (fit["a"], fit["b"], fit["rel_resid"]))
    emit("  分项标度（2γh≪1 区间，γ=10 → 100）：offset-only E 比值 = %.1f"
         "（≈100 → 1/γ²）；noise-only = %.1f（≈10 → 1/γ，离散脉冲注入带来的"
         " ~15%% 压缩）。"
         % (man["offset_only"]["10.0"]["g2_last"]
            / man["offset_only"]["100.0"]["g2_last"],
            man["noise_only"]["10.0"]["g2_last"]
            / man["noise_only"]["100.0"]["g2_last"]))
    sp = man["spectral"]
    emit("  γ 可行区间：γ_min(offset ≤ %.0e) = %.3g，γ_max(κ_hw=%.0e) = %.3g"
         % (man["tol"], sp["gamma_min"], man["kappa_hw"], sp["gamma_max"]))
    emit("    max|λ|(γ=γ0) = %.3g，max|λ|(γ=1000γ0) = %.3g（%s）。区间%s空——"
         "对这组参数稳定化可行，不触发“必须用原生原语”的判决。"
         % (sp["radius_gamma0"], sp["radius_1000x"],
            "%.0f×" % (sp["radius_1000x"] / sp["radius_gamma0"]),
            "非" if sp["gamma_min"] < man["g0"] < sp["gamma_max"] else "为"))
    emit("    γ=1000× 在 h=1e-3 下 RK4 失稳（γh=100），按 h=1e-6 积分："
         "max|g| = %.2e（有界）。" % man["plateau_1000x"]["max_abs"])

    hr("F. 增广 vs 原生原语成本（P1-native-primitives，HH）")
    nat = CASES["HH"]["native"]
    aug = CASES["HH"]
    table(["路径", "states", "monomials", "degree", "附加"],
          [["augmented (exp→6 状态)", aug["n"], aug["n_monomials"],
            aug["degree"], "6 divider / gm 乘法"],
           ["native (exp/tanh = 原子 tile)", nat["n_states"],
            nat["n_monomials"], nat["degree"],
            "%d 次原生 NL 求值" % nat["n_primitives"]]],
          ["", "%d", "%d", "%d", ""])
    emit("  native 的 degree=%d 来自 HH 自身的门控幂（n⁴·(V+77)、m³h·(V−50)），"
         % nat["degree"])
    emit("  不是增广引入的；增广把 6 个 exp 变成 6 个带状态、继承 V̇ 全部复杂度"
         "的积分器。")
    emit("  breakeven：原生 tile 用 1 次跨导律求值换掉每 exp 两个状态 + 乘法器，"
         "代价是该原语不再是多项式环内的精确代数。")

    # SharedDen 的 tuple-keyed num/monos 无法直接 json.dump；验收已跑完，
    # 序列化前换成字符串键。manifold 的 float 键同理。
    CASES["SharedDen"] = dict(
        CASES["SharedDen"],
        num={str(k): float(v) for k, v in CASES["SharedDen"]["num"].items()},
        monos=[[list(k), float(v)] for k, v in CASES["SharedDen"]["monos"]])

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
