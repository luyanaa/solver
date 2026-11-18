"""Stage B1 P7 系数图 + 行为级网表 round-trip 验收。

    python3 -m bench.run_p7_netlist

P0-3 的判据：generate_config 必须编码**动力学**，不只是一张速度 spec。
对角系统 diag(-1,-100) 是试金石——旧 config 只输出一个全局
max|λ|/alpha，分辨不出 −1 和 −100 两个模态；新 config 必须为每个
积分器给出**各自**的有效自系数 a_ii（物理 rad/s），并完整展开
A 边/B 输入/C 抽头的系数图。round-trip：config → to_behavioral →
重构有效 A → 特征值必须 ≈ scaling.scaled 的特征值（重构是精确装配，
唯一容差来自浮点）。

三组验收：
  D2   diag(-1,-100)（+ 耦合对照 [[-1,0.5],[0,-100]]）：逐元素系数、
       B/C 图、契约消费、round-trip eig；
  DENSE 4×4 稠密稳定系统：满系数图 + round-trip；
  DIFF diffusion-1d 锚点：64 个**互不相同**的模态系数（spread=64²），
       tile 速度 spec 仍在 source 里，cal_bindings 契约。
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

from analysis.descriptor import Descriptor
from analysis.slack import HardwareSpec
from lowering.budget import BudgetInput, compile_budget
from lowering.discretize import PDESpec, discretize
from lowering.netlist import to_behavioral
from lowering.p7 import C_BANK_CODES, generate_config
from lowering.scaling import gramian_scaling

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


def add_check(name, got, expected, passed, kind, note=None):
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
    return str(o)


def _bundle(A, B, C):
    desc = Descriptor(np.eye(A.shape[0]), np.asarray(A, float),
                      np.asarray(B, float), np.asarray(C, float))
    sc = gramian_scaling(desc, v_swing=1.0, B_hw=1e6)
    cert = compile_budget(BudgetInput(
        sys=desc, spec=HardwareSpec(B_hw=1e6, sigma_req=30.0), scaling=sc))
    return desc, sc, cert


def _eig_err(net, sc):
    lam_rec = np.sort_complex(net.eigs())
    lam_ref = np.sort_complex(sc.scaled.finite_eigs())
    denom = max(float(np.max(np.abs(lam_ref))), 1e-30)
    return float(np.max(np.abs(lam_rec - lam_ref)) / denom)


# ===========================================================================
# D2  diag(-1,-100) + 耦合对照
# ===========================================================================
def case_d2():
    hr("D2  diag(-1,-100)：逐元素有效系数 + round-trip")
    # 纯对角：两个模态速率差 100×，旧的全局 max|λ|/alpha 分辨不出
    d, sc, cert = _bundle(np.diag([-1.0, -100.0]),
                          np.array([[1.0], [1.0]]), np.array([[1.0, 0.0]]))
    cfg = generate_config(d, cert, sc)
    net = to_behavioral(cfg, name="diag2")
    ints = [e for e in cfg.elements if e.primitive == "integrator-gm_c"]
    a_v = np.asarray(sc.scaled.A, dtype=float)
    diag_abs = np.abs(np.diag(a_v))

    add_check("d2: two integrators emit TWO DISTINCT effective coefficients",
              "%.4g / %.4g rad/s" % (ints[0].target_gm_over_c,
                                     ints[1].target_gm_over_c),
              "ratio 100 (not one global max|lambda|/alpha)",
              abs(ints[1].target_gm_over_c / ints[0].target_gm_over_c - 100.0)
              < 1e-6 * 100.0, kind="p7net-d2")
    add_check("d2: a_ii signed and equals scaled A diagonal (a_ii/alpha)",
              ("%.4g" % ints[0].a_ii, "%.4g" % ints[1].a_ii),
              ("%.4g" % a_v[0, 0], "%.4g" % a_v[1, 1]),
              abs(ints[0].a_ii - a_v[0, 0]) < 1e-12
              and abs(ints[1].a_ii - a_v[1, 1]) < 1e-12, kind="p7net-d2")
    add_check("d2: target_gm_over_c = |a_ii| per element",
              "ok" if all(abs(itg.target_gm_over_c - abs(itg.pole)) < 1e-12
                          for itg in net.integrators) else "mismatch",
              "ok", all(abs(itg.target_gm_over_c - abs(itg.pole)) < 1e-12
                        for itg in net.integrators), kind="p7net-d2")
    add_check("d2: diagonal system => zero off-diagonal coeff edges",
              len(cfg.coeff_edges), 0, len(cfg.coeff_edges) == 0,
              kind="p7net-d2")
    add_check("d2: B input graph emitted (2 ports, one per state)",
              len(cfg.b_ports), 2, len(cfg.b_ports) == 2, kind="p7net-d2")
    # 逐端口精确对拍（上面的聚合 check 只保证数量）
    B_v = np.asarray(sc.scaled.B, dtype=float)
    ok_b = all(any(p["i"] == i and abs(p["gain"] - B_v[i, p["input"]]) < 1e-12
                   for p in cfg.b_ports)
               for i in range(B_v.shape[0]) for p_col in [0]
               if abs(B_v[i, 0]) > 0)
    add_check("d2: each b_port gain matches B_v[i,0] exactly",
              "ok" if ok_b else "mismatch", "ok", ok_b, kind="p7net-d2")
    C_v = np.asarray(sc.scaled.C, dtype=float)
    ok_c = (len(cfg.c_taps) == 1 and cfg.c_taps[0]["i"] == 0
            and abs(cfg.c_taps[0]["gain"] - C_v[0, 0]) < 1e-12)
    add_check("d2: C taps emitted (1 tap, gain = C_v[0,0])",
              "%.4g" % (cfg.c_taps[0]["gain"] if cfg.c_taps else float("nan")),
              "%.4g" % C_v[0, 0], ok_c, kind="p7net-d2")

    # round-trip：重构 A 的特征值 ≈ scaled 特征值
    err = _eig_err(net, sc)
    add_check("d2: round-trip eig error (reconstructed vs scaled) < 1e-9",
              "%.2e" % err, "<= 1e-9", err < 1e-9, kind="p7net-d2")

    # 耦合对照：A[0,1] = 0.5 ⇒ 一条边 + 契约消费。B 激励两个状态
    #（只激励 x0 会让 x1 近似不可控，S_1 爆炸把边增益缩到 1e-296 ——
    # 那是诚实的 scaling 物理量，但作 demo 会淹没在 a_tol 之下）。
    d2, sc2, cert2 = _bundle(np.array([[-1.0, 0.5], [0.0, -100.0]]),
                             np.array([[1.0], [1.0]]),
                             np.array([[1.0, 1.0]]))
    cfg2 = generate_config(d2, cert2, sc2)
    net2 = to_behavioral(cfg2, name="coup2")
    add_check("d2: coupled system emits the off-diagonal edge (j=1 -> i=0)",
              (len(cfg2.coeff_edges),
               (cfg2.coeff_edges[0]["src"], cfg2.coeff_edges[0]["dst"])
               if cfg2.coeff_edges else None),
              "1 edge, src=state1 dst=state0",
              len(cfg2.coeff_edges) == 1
              and cfg2.coeff_edges[0]["src"] == "x1"
              and cfg2.coeff_edges[0]["dst"] == "x0", kind="p7net-d2")
    a2 = np.asarray(sc2.scaled.A, dtype=float)
    add_check("d2: edge gain = physical A_v[0,1] (alpha^-1 S A S^-1 entry)",
              "%.4g" % cfg2.coeff_edges[0]["gain"], "%.4g" % a2[0, 1],
              abs(cfg2.coeff_edges[0]["gain"] - a2[0, 1]) < 1e-12,
              kind="p7net-d2")
    k = cfg2.coeff_edges[0]["route_contract"]
    ok_link = (k is not None and len(cfg2.route_contracts) > k
               and cfg2.route_contracts[k].i == 0
               and cfg2.route_contracts[k].j == 1)
    add_check("d2: coeff edge consumes an existing route contract (i,j match)",
              ("contract %s" % k, len(cfg2.route_contracts)),
              "linked, (i=0,j=1)", ok_link, kind="p7net-d2")
    err2 = _eig_err(net2, sc2)
    add_check("d2: coupled round-trip eig error < 1e-9",
              "%.2e" % err2, "<= 1e-9", err2 < 1e-9, kind="p7net-d2")
    add_check("d2: route contracts present in Config.route_contracts",
              len(cfg2.route_contracts), ">= 1", len(cfg2.route_contracts) >= 1,
              kind="p7net-d2")

    CASES["d2"] = dict(checks=[c for c in CHECKS if c["kind"] == "p7net-d2"],
                       summary=dict(
                           a_ii=[float(i.pole) for i in net.integrators],
                           coeff_spread=float(diag_abs[1] / diag_abs[0]),
                           eig_err_diag=err, eig_err_coupled=err2,
                           alpha=sc.alpha))


# ===========================================================================
# DENSE  4x4 稠密稳定系统
# ===========================================================================
def case_dense():
    hr("DENSE  4x4 稠密系统：满系数图 + round-trip")
    rng = np.random.default_rng(13)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    A = Q @ np.diag([-0.2, -1.0, -3.0, -10.0]) @ Q.T
    d, sc, cert = _bundle(A, rng.standard_normal((4, 2)),
                          rng.standard_normal((2, 4)))
    cfg = generate_config(d, cert, sc)
    net = to_behavioral(cfg, name="dense4")
    a_v = np.asarray(sc.scaled.A, dtype=float)
    off = a_v - np.diag(np.diag(a_v))
    n_off = int(np.count_nonzero(np.abs(off) > 1e-12 * np.max(np.abs(a_v))))

    add_check("dense: every off-diagonal nonzero emitted as an edge",
              len(cfg.coeff_edges), n_off, len(cfg.coeff_edges) == n_off,
              kind="p7net-dense")
    ok_g = all(abs(e["gain"] - a_v[e["i"], e["j"]]) < 1e-12
               for e in cfg.coeff_edges)
    add_check("dense: every edge gain equals the physical A_v entry",
              "ok" if ok_g else "mismatch", "ok", ok_g, kind="p7net-dense")
    B_v = np.asarray(sc.scaled.B, dtype=float)
    C_v = np.asarray(sc.scaled.C, dtype=float)
    ok_b = (len(cfg.b_ports) == int(np.count_nonzero(B_v))
            and all(abs(p["gain"] - B_v[p["i"], p["input"]]) < 1e-12
                    for p in cfg.b_ports))
    ok_c = (len(cfg.c_taps) == int(np.count_nonzero(C_v))
            and all(abs(t["gain"] - C_v[t["tap"], t["i"]]) < 1e-12
                    for t in cfg.c_taps))
    add_check("dense: B/C graphs emitted elementwise",
              (len(cfg.b_ports), len(cfg.c_taps)),
              (int(np.count_nonzero(B_v)), int(np.count_nonzero(C_v))),
              ok_b and ok_c, kind="p7net-dense")
    # 重构 B/C 与 scaled 的直接对拍（比逐元素更硬：整个矩阵）
    ok_B = np.allclose(net.effective_B(), B_v, rtol=0, atol=1e-12)
    ok_C = np.allclose(net.effective_C(), C_v, rtol=0, atol=1e-12)
    add_check("dense: effective_B/effective_C reconstruct B_v/C_v",
              ("%.1e" % float(np.max(np.abs(net.effective_B() - B_v))),
               "%.1e" % float(np.max(np.abs(net.effective_C() - C_v)))),
              "<= 1e-12", ok_B and ok_C, kind="p7net-dense")
    err = _eig_err(net, sc)
    add_check("dense: round-trip eig error < 1e-8",
              "%.2e" % err, "<= 1e-8", err < 1e-8, kind="p7net-dense")
    blob = json.loads(json.dumps(_jsonable(net.as_dict())))
    add_check("dense: BehavioralNetlist round-trips through JSON",
              (len(blob["integrators"]), len(blob["edges"])),
              (4, len(net.edges)),
              len(blob["integrators"]) == 4
              and len(blob["edges"]) == len(net.edges), kind="p7net-dense")
    ok_bind = all(
        any(b["element"] == e.name and b["method"] == "foreground"
            and b["reference"] for b in cfg.cal_bindings)
        for e in net.integrators if not e.flagged)
    add_check("dense: cal_bindings bind every unflagged integrator to a reference",
              len(cfg.cal_bindings), ">= 2*4 unflagged fg+bg", ok_bind,
              kind="p7net-dense")

    CASES["dense"] = dict(checks=[c for c in CHECKS if c["kind"] == "p7net-dense"],
                          summary=dict(n_edges=len(cfg.coeff_edges),
                                       eig_err=err, alpha=sc.alpha))


# ===========================================================================
# DIFF  diffusion-1d 锚点：64 个互不相同的模态系数
# ===========================================================================
def case_diff():
    hr("DIFF  diffusion-1d：64 模态逐元素系数 + tile 速度 spec")
    spec = PDESpec(name="diffusion-1d", operator="diffusion", bc="periodic",
                   domain=((0.0, 1.0),), coeff=1e-3, n=(64,), basis="fourier")
    d = discretize(spec)
    sc = gramian_scaling(d.descriptor, v_swing=1.0, B_hw=1e6)
    cert = compile_budget(BudgetInput(
        sys=d.descriptor, spec=HardwareSpec(B_hw=1e6, sigma_req=30.0),
        scaling=sc))
    cfg = generate_config(d, cert, sc)
    net = to_behavioral(cfg, name="diff64")
    a_v = np.asarray(sc.scaled.A, dtype=float)
    diag_abs = np.abs(np.diag(a_v))
    spread = float(diag_abs.max() / diag_abs.min())

    add_check("diff: per-element coefficients distinct across 64 modes "
              "(spread = n^2 = 4096)",
              "%.1f" % spread, "4096", abs(spread - 4096.0) < 1e-6 * 4096.0,
              kind="p7net-diff")
    ints_d = [e for e in cfg.elements if e.primitive == "integrator-gm_c"]
    add_check("diff: element k targets its OWN mode speed nu*k^2/alpha "
              "(not the global max)",
              ("%.4g" % ints_d[0].target_gm_over_c,
                "%.4g" % ints_d[-1].target_gm_over_c),
              "min != max; max = tile_speed_spec",
              ints_d[0].target_gm_over_c < ints_d[-1].target_gm_over_c
              and abs(ints_d[-1].target_gm_over_c
                      - cfg.source["tile_speed_spec"]) < 1e-9,
              kind="p7net-diff")
    lam_max = float(np.max(np.abs(d.descriptor.finite_eigs())))
    add_check("diff: tile_speed_spec kept in source (= max|lambda|/alpha)",
              "%.4g" % cfg.source["tile_speed_spec"],
              "%.4g" % (lam_max / sc.alpha),
              abs(cfg.source["tile_speed_spec"] - lam_max / sc.alpha) < 1e-9,
              kind="p7net-diff")
    add_check("diff: coeff graph encoded flag is True (descriptor present)",
              cfg.source["coeff_graph_encoded"], True,
              cfg.source["coeff_graph_encoded"] is True, kind="p7net-diff")
    err = _eig_err(net, sc)
    add_check("diff: round-trip eig error < 1e-9 (64 modes)",
              "%.2e" % err, "<= 1e-9", err < 1e-9, kind="p7net-diff")
    c_code = cfg.elements[0].C_code
    add_check("diff: C_code selects nearest bank entry and is recorded",
              (c_code, "%.4g" % C_BANK_CODES[c_code]),
              "index of nearest to C_bank=1pF",
              0 <= c_code < len(C_BANK_CODES)
              and C_BANK_CODES[c_code] == min(
                  C_BANK_CODES, key=lambda c: abs(c - 1e-12)),
              kind="p7net-diff")

    CASES["diff"] = dict(checks=[c for c in CHECKS if c["kind"] == "p7net-diff"],
                         summary=dict(spread=spread, eig_err=err,
                                      tile_speed=cfg.source["tile_speed_spec"],
                                      alpha=sc.alpha))


# ===========================================================================
def main():
    emit("B1 P7 系数图 + 行为级网表 round-trip 验收（P0-3）")
    case_d2()
    case_dense()
    case_diff()

    hr("VERDICT")
    rows = [(c["name"], c["got"], c["expected"], "yes" if c["passed"] else "NO")
            for c in CHECKS]
    table(["check", "got", "expected", "pass"], rows, [None, None, None, None])
    n_fail = sum(1 for c in CHECKS if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(CHECKS) - n_fail, len(CHECKS)))
    CASES["summary"] = dict(n_checks=len(CHECKS),
                            n_passed=len(CHECKS) - n_fail)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p7_netlist.json"), "w") as fh:
        json.dump(_jsonable(CASES), fh, indent=1)
    with open(os.path.join(OUT, "p7_netlist_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/p7_netlist.json, results/p7_netlist_report.txt")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
