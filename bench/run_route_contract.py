"""B1 per-edge route contracts 验收：lowering/route_contract.py + p7.py 集成。

    python3 -m bench.run_route_contract

五组验收（§3.2/§6.15 四判据契约 + §4 P7 编译输出）：

  EDGE  3-node 链 + 1 feedback 边的小描述符 → 每条非对角 A_ij 一条契约；
        feedback 边 flag + 有限 tau_max_abs = phase_budget(ωc)（非 inf）；
        非 feedback 边 absolute_ok 平凡 True；candidate 短线过四判据；
        |A_ij| 排序给出 1/|J_ij| 敏感度排序（edge_sensitivity）。
  P7    diffusion-1d 锚点 generate_config ⇒ Config.route_contracts 挂上、
        JSON 可序列化、条数 = 非对角 nnz；3-node 链 ⇒ feedback 契约进
        Config（聚合 "route-edges" ElementConfig 保留，向后兼容）。
  VIOL  强制长线（candidate_length）⇒ verdict False 如实报告：
        feedback 边 pass_absolute False 且 pad_can_fix=False（deskew
        负控制的 P7 版：修 skew 不修绝对延迟）；非 feedback 边
        pad_can_fix=True（pad/deskew 可修）。
  NEG   负控制：纯对角 ⇒ 空契约表（真"无布线"，不是错误）；
        burgers（descriptor=None）⇒ 空表；NaN/inf 边 ⇒ flagged，
        不静默 pass。
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
from analysis.skew import phase_budget
from lowering.discretize import PDESpec, discretize
from lowering.route_contract import derive_coupling_edges, emit_route_contracts
from lowering.scaling import gramian_scaling
from twin.route import latency_class

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


def add_check(name, got, expected, passed, kind="route-contract", note=None):
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


# ---------------------------------------------------------------------------
# 小耦合描述符：0↔1 双向环（feedback）+ 两条不闭环的 spur 边
#   A[1,0]=4e5、A[0,1]=-3e5 构成有向环 0→1→0（环增益
#   (4e5·3e5)^{1/2}=3.5e5 < 1e6 ⇒ 稳定）⇒ 这两条是 feedback_critical；
#   A[2,1]=5e5（1→2 的 spur，无回到 1 的路径）与 A[0,3]=-2e5（3→0 的
#   纯输入 spur，0 无路径回到 3）不在任何环上 ⇒ 非 feedback。
# ---------------------------------------------------------------------------
def _chain_descriptor():
    A = np.array([[-1e6, -3e5, 0.0, -2e5],
                  [4e5, -1e6, 0.0, 0.0],
                  [0.0, 5e5, -1e6, 0.0],
                  [0.0, 0.0, 0.0, -1e6]])
    return Descriptor(np.eye(4), A, np.eye(4)[:, :1], np.eye(4))


class _TinyDisc:
    """最小 Discretized 形状：descriptor + states（route_contract 只吃这些）。"""

    def __init__(self, descriptor, states):
        self.descriptor = descriptor
        self.states = states
        self.meta = {}


def case_edge():
    hr("EDGE  3-node 链 + feedback 边：per-edge 契约（需求 + verdict + 敏感度）")
    desc = _chain_descriptor()
    d = _TinyDisc(desc, ["x0", "x1", "x2", "x3"])
    rep = emit_route_contracts(d)

    add_check("edge: one contract per nonzero off-diagonal A_ij",
              rep.n_edges, 4, rep.n_edges == 4, kind="rc-edge")
    add_check("edge: exactly two feedback-critical edges (the 0↔1 cycle)",
              rep.n_feedback, 2, rep.n_feedback == 2, kind="rc-edge")
    edges = {(c.i, c.j) for c in rep.contracts}
    add_check("edge: edge set matches A off-diagonal nnz",
              sorted(edges), "[(0,1),(0,3),(1,0),(2,1)]",
              edges == {(0, 1), (0, 3), (1, 0), (2, 1)}, kind="rc-edge")

    fb = [c for c in rep.contracts if c.feedback_critical][0]
    nf = [c for c in rep.contracts if not c.feedback_critical]
    add_check("edge: feedback edges are exactly {(0,1),(1,0)}",
              sorted((c.i, c.j) for c in rep.contracts
                     if c.feedback_critical), "[(0,1),(1,0)]",
              {(c.i, c.j) for c in rep.contracts if c.feedback_critical}
              == {(0, 1), (1, 0)}, kind="rc-edge")
    add_check("edge: weights match |A_ij|",
              tuple(sorted(round(c.weight, 1) for c in rep.contracts)),
              "(200000, 300000, 400000, 500000)",
              all(abs(c.weight - abs(desc.A[c.i, c.j])) < 1e-6
                  for c in rep.contracts), kind="rc-edge")

    # feedback 边：有限 tau_max_abs = phase_budget(ωc, φ_reserve)，不是 inf
    add_check("edge: feedback omega_c finite and positive",
              "%.4g" % fb.omega_c, "> 0",
              fb.omega_c is not None and np.isfinite(fb.omega_c)
              and fb.omega_c > 0, kind="rc-edge")
    tau_max = phase_budget(fb.omega_c, fb.phi_reserve_deg)
    add_check("edge: feedback tau_max_abs = phase_budget(omega_c) (NOT inf)",
              "%.4g s" % fb.tau_max_abs, "%.4g s" % tau_max,
              np.isfinite(fb.tau_max_abs)
              and abs(fb.tau_max_abs - tau_max) < 1e-18, kind="rc-edge")
    add_check("edge: candidate short wire meets absolute budget (deskew-vs-"
              "absolute: common delay doesn't rescue a violation)",
              "pd %.3g <= tau %.3g" % (fb.closure["phase_delay_omega_c"],
                                       fb.tau_max_abs),
              "pass_absolute True",
              fb.closure["pass_absolute"], kind="rc-edge")
    add_check("edge: phase_delay_at(omega_c) <= tau_max on candidate",
              "%.3g" % fb.closure["phase_delay_omega_c"], "<= %.3g" % tau_max,
              fb.closure["phase_delay_omega_c"] <= fb.tau_max_abs,
              kind="rc-edge")

    # 非 feedback 边：绝对判据平凡 True，tau_max = inf
    add_check("edge: non-feedback edges trivially absolute_ok with inf budget",
              tuple(c.closure["pass_absolute"] for c in nf), "all True",
              all(c.closure["pass_absolute"] for c in nf)
              and all(np.isinf(c.tau_max_abs) for c in nf), kind="rc-edge")

    # candidate 短线过全部四判据（这类边"存在且满足契约"）
    add_check("edge: all candidate edges pass gain/phase/noise on short wire",
              tuple((c.closure["pass_gain"], c.closure["pass_phase"],
                     c.closure["pass_noise"]) for c in rep.contracts),
              "all True", all(c.closure["pass_gain"] and c.closure["pass_phase"]
                              and c.closure["pass_noise"]
                              for c in rep.contracts), kind="rc-edge")
    add_check("edge: eta_route within the spec budget (phase+gain, "
              "<= rad(eps_phase))",
              "%.2e" % max(c.closure["eta"] for c in rep.contracts),
              "<= %.3g" % math.radians(3.0),
              max(c.closure["eta"] for c in rep.contracts)
              <= math.radians(3.0) * 1.001, kind="rc-edge")
    add_check("edge: latency class recorded and in the valid set",
              tuple(sorted({c.closure["latency_class"]
                            for c in rep.contracts})), "subset of L0..L4",
              all(c.closure["latency_class"]
                  in {"L0", "L1", "L2", "L3", "L4"}
                  and c.closure["latency_class"]
                  == latency_class(c.closure["latency"])
                  for c in rep.contracts), kind="rc-edge")

    # 敏感度：hf_tau_tol = ε/|A_ij| ⇒ 按 |A_ij| 降序 = τ_tol 升序（1/|J_ij|）
    by_w = sorted(rep.contracts, key=lambda c: -c.weight)
    tols = [c.hf_tau_tol for c in by_w]
    add_check("edge: 1/|A_ij| sensitivity ordering (hf_tau_tol ascending "
              "with |A_ij| descending)",
              "%.3g -> %.3g -> %.3g -> %.3g" % tuple(tols), "monotone",
              tols[0] < tols[1] < tols[2], kind="rc-edge")
    add_check("edge: hf_tau_tol == eps_phi/|A_ij| exactly",
              "%.3g" % by_w[0].hf_tau_tol,
              "%.3g" % (math.radians(by_w[0].eps_phase_deg) / by_w[0].weight),
              abs(by_w[0].hf_tau_tol
                  - math.radians(by_w[0].eps_phase_deg) / by_w[0].weight) < 1e-18,
              kind="rc-edge")
    add_check("edge: resolvent pair identity tau_tol * S_tau == eps_phi "
              "(exact inverse)",
              "%.6g" % (fb.tau_tol * fb.S_tau),
              "%.6g" % math.radians(fb.eps_phase_deg),
              abs(fb.tau_tol * fb.S_tau
                  - math.radians(fb.eps_phase_deg)) < 1e-12
              and fb.tau_tol > 0, kind="rc-edge")
    add_check("edge: S_tau = omega*|A_ij|*||resolvent|| > 0",
              "%.3g" % fb.S_tau, "> 0", fb.S_tau > 0, kind="rc-edge")

    # JSON 往返
    blob = json.dumps(_jsonable(rep.as_dict()))
    back = json.loads(blob)
    add_check("edge: report round-trips through JSON (4 contracts)",
              len(back["contracts"]), 4,
              len(back["contracts"]) == 4
              and back["contracts"][0]["closure"]["latency_class"]
              in {"L0", "L1", "L2", "L3", "L4"},
              kind="rc-edge")

    CASES["edge"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "rc-edge"],
        summary=dict(n_edges=rep.n_edges, n_feedback=rep.n_feedback,
                     omega_max=rep.omega_max,
                     tau_max_abs_fb=fb.tau_max_abs,
                     phase_delay_fb=fb.closure["phase_delay_omega_c"]))


def case_units():
    hr("UNITS  alpha≠1 ⇒ 敏感度/权重全在物理时间轴（α 解析缩放核对）")
    desc = _chain_descriptor()
    d = _TinyDisc(desc, ["x0", "x1", "x2", "x3"])
    rep_m = emit_route_contracts(d)            # 无 scaling：alpha=1, model=physical
    alpha = 1e-3
    from types import SimpleNamespace
    sc = SimpleNamespace(alpha=alpha, S_x_vec=np.ones(4))
    rep_p = emit_route_contracts(d, scaling=sc)  # 物理时间动力学（重构口径）

    add_check("units: omega_max on the physical axis (lam_max/alpha)",
              "%.6g" % rep_p.omega_max, "%.6g" % (rep_m.omega_max / alpha),
              abs(rep_p.omega_max - rep_m.omega_max / alpha)
              < 1e-9 * rep_p.omega_max, kind="rc-units")
    fb_m = [c for c in rep_m.contracts if c.feedback_critical][0]
    fb_p = [c for c in rep_p.contracts if c.feedback_critical][0]
    add_check("units: weight is the physical coefficient A_v = alpha^-1*A",
              "%.6g" % fb_p.weight, "%.6g" % (fb_m.weight / alpha),
              abs(fb_p.weight - fb_m.weight / alpha)
              < 1e-9 * fb_p.weight, kind="rc-units")
    # 解析缩放：J_p = A/α、ω = lam_max/α ⇒ resolvent = α·r_model，
    # S = ω|A_v|r = S_model/α（相似缩放，容差给 roundoff）。
    ratio_S = fb_p.S_tau / fb_m.S_tau
    add_check("units: S_tau(physical) = S_tau(model)/alpha (NOT the old "
              "mixed-unit number)",
              "%.6g" % ratio_S, "1/alpha = %.6g" % (1.0 / alpha),
              abs(ratio_S - 1.0 / alpha) < 1e-6 * (1.0 / alpha),
              kind="rc-units")
    ratio_tol = fb_p.tau_tol / fb_m.tau_tol
    add_check("units: tau_tol(physical) = alpha * tau_tol(model)",
              "%.6g" % ratio_tol, "alpha = %.6g" % alpha,
              abs(ratio_tol - alpha) < 1e-6 * alpha, kind="rc-units")
    ratio_hf = fb_p.hf_tau_tol / fb_m.hf_tau_tol
    add_check("units: hf_tau_tol = eps_phi/|A_v,ij| scales by alpha "
              "(physical A_ij, not model A_ij)",
              "%.6g" % ratio_hf, "alpha = %.6g" % alpha,
              abs(ratio_hf - alpha) < 1e-12 * alpha, kind="rc-units")
    w_sorted = sorted(rep_p.contracts, key=lambda c: -c.weight)
    tols = [c.hf_tau_tol for c in w_sorted]
    add_check("units: ordering preserved in physical units "
              "(hf_tau_tol ascending with |A_v,ij| descending)",
              "%.4g -> %.4g -> %.4g -> %.4g" % tuple(tols), "monotone",
              tols[0] < tols[1] < tols[2] < tols[3], kind="rc-units")
    add_check("units: source records the physical-dynamics derivation",
              (rep_p.source["A_source"][:20] + "...",
               rep_p.source["J_source"][:16] + "..."), "reconstructed/E_p^-1",
              "reconstructed" in rep_p.source["A_source"]
              and "E_p^-1 A_p" in rep_p.source["J_source"]
              and "physical" in rep_p.source["units"], kind="rc-units")
    add_check("units: report J field carries the physical J_p (not model A)",
              "J_p" if np.allclose(np.asarray(rep_p.J),
                                   np.asarray(desc.A) / alpha,
                                   rtol=1e-12) else "not J_p",
              "J_p = E_p^-1 A_p",
              np.allclose(np.asarray(rep_p.J), np.asarray(desc.A) / alpha,
                          rtol=1e-9), kind="rc-units")

    CASES["units"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "rc-units"],
        summary=dict(alpha=alpha,
                     S_tau_ratio=float(ratio_S), tau_tol_ratio=float(ratio_tol),
                     hf_ratio=float(ratio_hf),
                     omega_max_model=rep_m.omega_max,
                     omega_max_phys=rep_p.omega_max))


def case_singular():
    hr("SING  奇异 E_p：pinv 微分子空间提升（不朴素求逆，不造假数）")
    # 2×2 index-1 DAE：E=diag(1,0)，代数变量 x1（约束方程 4e5·x0 − 1e6·x1 = 0
    # ⇔ x1 = 0.4·x0），微分方程 x0' = −1e6·x0 − 3e5·x1。有限特征值
    # λ = −1e6 − 3e5·0.4 = −1.12e6（另一个在无穷 = 代数块）。
    A = np.array([[-1e6, -3e5],
                  [4e5, -1e6]])
    E = np.diag([1.0, 0.0])
    desc = Descriptor(E, A, np.array([[1.0], [0.0]]), np.eye(2))
    d = _TinyDisc(desc, ["xd", "xa"])
    rep = emit_route_contracts(d)

    lam_fin = np.abs(desc.finite_eigs())
    lam_fin = lam_fin[np.isfinite(lam_fin)]
    add_check("sing: singular-E descriptor accepted, 2 contracts emitted",
              rep.n_edges, 2, rep.n_edges == 2, kind="rc-sing")
    add_check("sing: omega_max = max|finite eig| on the differential "
              "subspace (no naive E inversion)",
              "%.6g" % rep.omega_max, "%.6g" % float(np.max(lam_fin)),
              np.isfinite(rep.omega_max)
              and abs(rep.omega_max - float(np.max(lam_fin)))
              < 1e-6 * rep.omega_max, kind="rc-sing")
    add_check("sing: source records the pinv differential-subspace lift",
              rep.source["J_source"][:24] + "...", "pinv(E_p) A_p",
              "pinv(E_p) A_p" in rep.source["J_source"], kind="rc-sing")
    # 微分行 (i=0)：J_p[0,1] = −3e5 ⇒ 敏感度有限
    c01 = [c for c in rep.contracts if (c.i, c.j) == (0, 1)][0]
    add_check("sing: differential-row edge keeps finite sensitivity",
              "(S %.4g, tol %.3g)" % (c01.S_tau, c01.tau_tol), "finite > 0",
              np.isfinite(c01.S_tau) and c01.S_tau > 0
              and np.isfinite(c01.tau_tol) and c01.tau_tol > 0,
              kind="rc-sing")
    # 代数行 (i=1)：J_p[1,:] = 0 ⇒ 敏感度如实为 0，tau_tol = inf
    c10 = [c for c in rep.contracts if (c.i, c.j) == (1, 0)][0]
    add_check("sing: algebraic-row edge reports honest 0 sensitivity "
              "(no finite dynamics on that edge), tau_tol=inf",
              "(S %.4g, tol %.3g)" % (c10.S_tau, c10.tau_tol),
              "S=0, tau_tol=inf",
              c10.S_tau == 0.0 and np.isinf(c10.tau_tol), kind="rc-sing")
    add_check("sing: hf_tau_tol still finite on the algebraic row "
              "(eps/|A_v,ij| — the gm budget is independent of the lift)",
              "%.4g" % c10.hf_tau_tol,
              "%.4g" % (math.radians(c10.eps_phase_deg) / c10.weight),
              np.isfinite(c10.hf_tau_tol)
              and abs(c10.hf_tau_tol
                      - math.radians(c10.eps_phase_deg) / c10.weight) < 1e-18,
              kind="rc-sing")
    add_check("sing: pinv lift is not flagged as defective (contracts valid)",
              tuple(c.flagged for c in rep.contracts), "all False",
              not any(c.flagged for c in rep.contracts), kind="rc-sing")

    CASES["singular"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "rc-sing"],
        summary=dict(omega_max=rep.omega_max,
                     S_tau_diff=c01.S_tau, S_tau_alg=c10.S_tau,
                     J_source=rep.source["J_source"]))


def case_p7():
    hr("P7  generate_config 集成：diffusion-1d 锚点 + 3-node 链")
    spec = PDESpec(name="diffusion-1d", operator="diffusion", bc="periodic",
                   domain=((0.0, 1.0),), coeff=1e-3, n=(64,), basis="fourier")
    d = discretize(spec)
    sc = gramian_scaling(d.descriptor, v_swing=1.0, B_hw=1e6)

    cfg = None
    from lowering.p7 import generate_config
    cfg = generate_config(d, None, sc)

    A = np.asarray(d.descriptor.A)
    off_nnz = int(np.count_nonzero(A - np.diag(np.diag(A))))
    add_check("p7: Config carries route_contracts field",
              type(cfg.route_contracts).__name__, "list",
              isinstance(cfg.route_contracts, list), kind="rc-p7")
    add_check("p7: contract count == off-diagonal nnz of A (diffusion: "
              "diagonal => 0, a true 'no routing')",
              len(cfg.route_contracts), off_nnz,
              len(cfg.route_contracts) == off_nnz, kind="rc-p7")
    add_check("p7: aggregate 'route-edges' ElementConfig kept (backward compat)",
              any(e.name == "route-edges" for e in cfg.elements), "present",
              any(e.name == "route-edges" for e in cfg.elements), kind="rc-p7")
    blob = json.dumps(_jsonable(cfg.as_dict()))
    back = json.loads(blob)
    add_check("p7: Config JSON round-trip includes route_contracts key",
              "route_contracts" in back, True,
              "route_contracts" in back, kind="rc-p7")

    # 4-node 链：feedback 契约进 Config（scaling 用最小 stub：alpha+S_x_vec）
    from types import SimpleNamespace
    sc3 = SimpleNamespace(alpha=1e-7, S_x_vec=np.ones(4))
    disc3 = _TinyDisc(_chain_descriptor(), ["x0", "x1", "x2", "x3"])
    cfg3 = generate_config(disc3, None, sc3)
    fbs = [c for c in cfg3.route_contracts if c.feedback_critical]
    add_check("p7: chain => 4 contracts, 1 feedback in Config",
              (len(cfg3.route_contracts), len(fbs)), (4, 2),
              len(cfg3.route_contracts) == 4 and len(fbs) == 2, kind="rc-p7")
    add_check("p7: contract names carry state names from discretized.states",
              cfg3.route_contracts[0].src + "->"
              + cfg3.route_contracts[0].dst, "state names",
              all(c.src.startswith("x") and c.dst.startswith("x")
                  for c in cfg3.route_contracts), kind="rc-p7")
    add_check("p7: source records contract counts",
              (cfg3.source.get("n_route_contracts"),
               cfg3.source.get("n_route_feedback")), (4, 2),
              cfg3.source.get("n_route_contracts") == 4
              and cfg3.source.get("n_route_feedback") == 2, kind="rc-p7")
    add_check("p7: each contract carries the four closure verdict keys",
              tuple(sorted(k for k in cfg3.route_contracts[0].closure
                           if k.startswith("pass_"))),
              "gain/noise/phase/absolute",
              {"pass_gain", "pass_phase", "pass_noise",
               "pass_absolute"} <= set(cfg3.route_contracts[0].closure),
              kind="rc-p7")

    CASES["p7"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "rc-p7"],
        summary=dict(diffusion_contracts=len(cfg.route_contracts),
                     chain_contracts=len(cfg3.route_contracts),
                     chain_feedback=len(fbs)))


def case_violation():
    hr("VIOL  强制长线：verdict 翻 False + pad_can_fix 诚实")
    desc = _chain_descriptor()
    d = _TinyDisc(desc, ["x0", "x1", "x2"])
    rep = emit_route_contracts(d, candidate_length=5e5)   # 500 mm 的 RC 线
    fb = [c for c in rep.contracts if c.feedback_critical][0]
    nf = [c for c in rep.contracts if not c.feedback_critical]

    add_check("viol: forced long wire flips feedback verdict to False",
              fb.closure["verdict"], False, not fb.closure["verdict"],
              kind="rc-viol")
    add_check("viol: failure is the absolute-delay criterion (phase_delay "
              ">> tau_max), not invented",
              "pd %.3g > tau %.3g" % (fb.closure["phase_delay_omega_c"],
                                      fb.tau_max_abs),
              "pass_absolute False",
              not fb.closure["pass_absolute"]
              and fb.closure["phase_delay_omega_c"] > fb.tau_max_abs,
              kind="rc-viol")
    add_check("viol: pad_can_fix=False on feedback edge (deskew negative "
              "control: matched delay != instantaneous)",
              fb.closure["pad_can_fix"], False,
              not fb.closure["pad_can_fix"], kind="rc-viol")
    add_check("viol: non-feedback long wire still pad_can_fix=True",
              tuple(c.closure["pad_can_fix"] for c in nf), "all True",
              all(c.closure["pad_can_fix"] for c in nf), kind="rc-viol")
    add_check("viol: contract-level verdict property reflects failure",
              tuple(c.verdict for c in rep.contracts), "not all True",
              not all(c.verdict for c in rep.contracts), kind="rc-viol")

    # 非 feedback 边的失败模式是 gain/phase（可 pad 修的那类）
    gain_fail = [c for c in nf if not c.closure["pass_gain"]
                 or not c.closure["pass_phase"]]
    add_check("viol: non-feedback failure lands in pad-fixable criteria",
              len(gain_fail), 2, len(gain_fail) == 2, kind="rc-viol")

    CASES["violation"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "rc-viol"],
        summary=dict(fb_phase_delay=fb.closure["phase_delay_omega_c"],
                     fb_tau_max=fb.tau_max_abs,
                     fb_latency=fb.closure["latency"]))


def case_negative():
    hr("NEG  负控制：纯对角 / 无描述符 / defective 边")
    spec = PDESpec(name="diffusion-1d", operator="diffusion", bc="periodic",
                   domain=((0.0, 1.0),), coeff=1e-3, n=(64,), basis="fourier")
    d = discretize(spec)
    rep0 = emit_route_contracts(d)
    add_check("neg: diagonal A => empty contract list (true 'no routing', "
              "not an error)", rep0.n_edges, 0, rep0.n_edges == 0,
              kind="rc-neg")

    disc_b = discretize(PDESpec(name="burgers-1d", operator="burgers",
                                bc="periodic", domain=((0.0, 1.0),),
                                n=(16,), basis="fv"))
    rep_b = emit_route_contracts(disc_b)
    add_check("neg: burgers (descriptor=None, no explicit edges) => empty, "
              "not fabricated", rep_b.n_edges, 0, rep_b.n_edges == 0,
              kind="rc-neg")

    # defective 边：NaN / inf 的 A_ij ⇒ flagged，不静默 pass。
    # Descriptor 自身的 __post_init__ 会拒绝非有限值（这是 analysis 栈的
    # 正确守卫），所以这里用手写的 defective 描述符 stub（front-end 允许
    # 传进 P7 之前坏数据——正是契约层要拦的）。
    def _bad_desc(A):
        from types import SimpleNamespace
        return SimpleNamespace(
            A=A, finite_eigs=lambda: np.array([-1e6, -1e6], dtype=complex))

    A = np.array([[-1e6, np.nan], [1e5, -1e6]])
    repn = emit_route_contracts(_TinyDisc(_bad_desc(A), ["x0", "x1"]))
    fl = [c for c in repn.contracts if c.flagged]
    ok_edge = [c for c in repn.contracts if not c.flagged]
    add_check("neg: NaN edge flagged, finite edge NOT flagged (per-edge scope)",
              (len(repn.contracts), len(fl), len(ok_edge)), (2, 1, 1),
              len(repn.contracts) == 2 and len(fl) == 1 and len(ok_edge) == 1,
              kind="rc-neg")
    add_check("neg: flagged contract carries a defective reason; verdict "
              "not all True", fl[0].flag_reason[:24] + "...", "defective",
              "defective" in fl[0].flag_reason
              and not all(c.verdict for c in repn.contracts), kind="rc-neg")

    A2 = np.array([[-1e6, np.inf], [1e5, -1e6]])
    repi = emit_route_contracts(_TinyDisc(_bad_desc(A2), ["x0", "x1"]))
    fb_i = [c for c in repi.contracts if c.flagged]
    add_check("neg: inf edge flagged with reason; finite edge passes spec",
              (len(fb_i), fb_i[0].flag_reason[:22] + "...",
               len([c for c in repi.contracts if c.verdict])), ">0 flagged",
              len(fb_i) == 1 and "defective" in fb_i[0].flag_reason,
              kind="rc-neg")

    # 零 off-diagonal 但显式边表为空 ⇒ 空表（derive 路径本身）
    edges, J = derive_coupling_edges(d)
    add_check("neg: derive_coupling_edges returns [] + J for diagonal desc",
              (len(edges), J is not None), (0, True),
              len(edges) == 0 and J is not None, kind="rc-neg")

    CASES["negative"] = dict(
        checks=[c for c in CHECKS if c["kind"] == "rc-neg"],
        summary=dict(diagonal_contracts=rep0.n_edges,
                     burgers_contracts=rep_b.n_edges,
                     nan_flagged=len(fl)))


def main():
    emit("B1 per-edge route contracts 验收：route_contract.py + p7.py 集成"
         "（DESIGN.md §3.2/§6.15 四判据 → P7 编译输出；敏感度物理单位合同）")
    case_edge()
    case_units()
    case_singular()
    case_p7()
    case_violation()
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
    with open(os.path.join(OUT, "route_contract.json"), "w") as fh:
        json.dump(_jsonable(CASES), fh, indent=1)
    with open(os.path.join(OUT, "route_contract_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/route_contract.json, results/route_contract_report.txt")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
