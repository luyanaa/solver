"""B1 验收：lowering/basis.py（选基）+ lowering/interface.py（接口 selector）。

    python3 -m bench.run_basis_interface

四个 CASE（run_reduce/run_stagea 口径：OPENBLAS 前置、emit/hr/table、
{name,got,expected,passed,kind}、results/basis_interface.json + report、
失败 exit 1）：

  CASE 1  非正规 shear（§8 行）最坏情况换基判定：特征向量基被
          选择性淘汰——Γ 降到 1.000 但 κ(Φ)=cond(V)=20.05，总预算
          20.05 > 恒等/Schur 基 5.12。chooser 选的是**更低的总预算**，
          不是更低的原始 Γ。与 results/stagea.json、skew.json 对账。
  CASE 2  同一 shear 系统、给定失配结构 Δ=ε·A（系数成比例漂移）：
          特征向量基翻身赢（amp=‖Λ‖/‖A‖≈0.22<1）——chooser 不退化，
          判定依赖失配模型，delta_pert 必须显式进 API。
  CASE 3  近正规系统（cond(V)≪20）Δ=ε·A：特征向量基以小优势赢
          （Γ 的收益 > cond(V) 的税）。最坏情况下仍然不赢（不变式
          ‖R_x‖ ≤ cond(T)‖R_z‖ ⇒ 纯相似变换预算 ≥ 恒等基）。
  CASE 4  接口 selector：光滑 ⇒ capture 免费（无特殊接口）；
          真 step ⇒ 单源合约 + 守恒账本与 coupling.py 的 dM/dt
          恒等式一致（single-source 泄漏≈机器精度，mismatch=1e-2
          泄漏 ≈ ε·|F|）；ν_num 越过阈值时判定翻转。
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from analysis import Descriptor
from analysis.resolvent import resolvent_peak
from lowering.basis import select_basis
from lowering.interface import InterfaceContract, choose_interface, mass_budget
from twin.coupling import CoupledSystem, InterfaceModel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
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


def add_check(name, got, expected, passed, kind="basis_interface", note=None):
    CHECKS.append(dict(name=name, got=got, expected=expected,
                       passed=bool(passed), kind=kind,
                       **({"note": note} if note else {})))


def _print_cards(title, res):
    rows = []
    for name, c in res.candidates.items():
        rows.append((name, "%.6g" % c.Gamma if np.isfinite(c.Gamma) else "n/a",
                     "%.6g" % c.cond_phi if np.isfinite(c.cond_phi) else "n/a",
                     c.nnz, "%.6g" % c.amplification if np.isfinite(c.amplification) else "n/a",
                     "%.6g" % c.total_budget if np.isfinite(c.total_budget) else "n/a",
                     "*" if c.selected else ""))
    table(["candidate", "Gamma", "cond(Phi)", "nnz", "amp", "budget", "win"],
          rows, ["", "%s", "%s", "%d", "%s", "%s", "%s"])


def _shear():
    A = np.array([[-1.0, 10.0], [0.0, -2.0]])
    return Descriptor(np.eye(2), A, np.array([[1.0], [0.0]]), np.eye(2))


# ==========================================================================
# CASE 1  shear 最坏情况：特征向量基被淘汰（§8 行的 chooser 版）
# ==========================================================================
def case_shear_worst():
    hr("CASE 1  非正规 shear（worst-case delta_pert=None）：特征向量基被淘汰")
    sys0 = _shear()
    res = select_basis(sys0)
    _print_cards("shear worst-case", res)
    emit("  %s" % res.reason)
    cands = res.candidates
    gamma_id = float(resolvent_peak(sys0)["Gamma"])
    c_eig = cands["eigenvector"]
    c_sch = cands["schur"]
    c_gm = cands["gramian"]
    win = cands[res.winner]

    # 对账：results/stagea.json（run_stagea CASE 1）与 results/skew.json
    stagea = json.load(open(os.path.join(OUT, "stagea.json")))["cases"]["shear"]
    skew = json.load(open(os.path.join(OUT, "skew.json")))["cases"]["shear"]
    g_ref = float(skew["gamma_ideal"])

    add_check("identity Gamma matches run_skew gamma_ideal (±2%)",
              "%.4f vs %.4f" % (gamma_id, g_ref), "±2%",
              abs(gamma_id / g_ref - 1.0) < 0.02)
    add_check("identity Gamma matches run_stagea gamma (±2%)",
              "%.4f vs %.4f" % (gamma_id, float(stagea["gamma"])), "±2%",
              abs(gamma_id / float(stagea["gamma"]) - 1.0) < 0.02)
    add_check("eigenvector Gamma drops to diagonal value 1.0",
              "%.6f" % c_eig.Gamma, "1.0 (stagea gamma_eigvec)",
              abs(c_eig.Gamma - float(stagea["gamma_eigvec"])) < 1e-6)
    add_check("eigenvector tax κ(Φ)=cond(V)=20.05 (stagea-measured)",
              "%.4f vs %.4f" % (c_eig.cond_phi, float(stagea["cond_V"])),
              "±2%", abs(c_eig.cond_phi / float(stagea["cond_V"]) - 1.0) < 0.02)
    add_check("eigenvector NOT selected on shear (budget 20.05 loses)",
              "winner=%s" % res.winner, "not eigenvector",
              res.winner != "eigenvector",
              note="§8: 禁止病态特征基——chooser 按总预算淘汰，不按原始 Γ")
    add_check("winner budget < eigenvector budget (20.05)",
              "%.4f < %.4f" % (win.total_budget, c_eig.total_budget),
              "lower total budget", win.total_budget < c_eig.total_budget)
    add_check("schur keeps honest Γ (= identity Γ, orthogonal invariance)",
              "%.4f vs %.4f" % (c_sch.Gamma, gamma_id), "= Γ_identity (1e-9)",
              abs(c_sch.Gamma - gamma_id) < 1e-9)
    add_check("invariant: every similarity budget ≥ Γ_identity (‖R_x‖≤cond·‖R_z‖)",
              "min budget %.6g vs Γ_id %.6g" %
              (min(c.total_budget for c in cands.values()
                   if np.isfinite(c.total_budget)), gamma_id),
              "≥ Γ_id(1−1e-6)",
              min(c.total_budget for c in cands.values()
                  if np.isfinite(c.total_budget)) >= gamma_id * (1.0 - 1e-6),
              note="最坏情况下恒等基在纯相似变换里不可能被打败——这是定理不是经验")
    add_check("sparsity reported: identity nnz ≤ schur nnz",
              "%d ≤ %d" % (win.nnz, c_sch.nnz), "monotone", win.nnz <= c_sch.nnz)
    add_check("gramian candidate honest on shear (degenerate state floored, "
              "no free Γ in model time)",
              "%.4f" % c_gm.Gamma, "≈ Γ_id",
              abs(c_gm.Gamma - gamma_id) / gamma_id < 1e-6,
              note="stagea 的 Γ=3.98e-8 是 α-scaled 物理时间量；换基账本在 "
                   "model time 记，时间缩放属于 scaling.py（P3）")
    CASES["shear_worst"] = dict(
        winner=res.winner, delta_pert=res.delta_pert,
        candidates={k: v.as_dict() for k, v in cands.items()},
        gamma_identity=gamma_id, stagea_crosscheck=dict(
            gamma=stagea["gamma"], gamma_eigvec=stagea["gamma_eigvec"],
            cond_V=stagea["cond_V"]))
    return res, win


# ==========================================================================
# CASE 2  shear + Δ=ε·A：特征向量基翻身（chooser 不退化）
# ==========================================================================
def case_shear_structured():
    hr("CASE 2  shear + Δ=ε·A（系数成比例漂移）：特征向量基翻身赢")
    sys0 = _shear()
    eps = 1e-2
    res = select_basis(sys0, delta_pert=eps)
    _print_cards("shear eps*A", res)
    emit("  %s" % res.reason)
    cands = res.candidates
    c_eig = cands["eigenvector"]
    c_id = cands["identity"]

    add_check("structured delta_pert: eigenvector selected",
              "winner=%s" % res.winner, "eigenvector",
              res.winner == "eigenvector")
    add_check("eigenvector amplification ‖Λ‖/‖A‖ < 1 (Δ diagonalized)",
              "%.4f" % c_eig.amplification, "< 1", c_eig.amplification < 1.0)
    add_check("eigenvector total budget < identity budget",
              "%.4f < %.4f" % (c_eig.total_budget, c_id.total_budget),
              "lower", c_eig.total_budget < c_id.total_budget)
    add_check("chooser verdict depends on the mismatch model (not degenerate)",
              "worst: %s / eps*A: %s" %
              (CASES["shear_worst"]["winner"], res.winner),
              "different winner for different delta_pert",
              CASES["shear_worst"]["winner"] != res.winner,
              note="delta_pert 进 API 的原因：判定 = f(失配结构)，不是常数")
    CASES["shear_structured"] = dict(
        winner=res.winner, delta_pert=res.delta_pert,
        candidates={k: v.as_dict() for k, v in cands.items()})
    return res


# ==========================================================================
# CASE 3  近正规系统：特征向量基以小优势赢；worst-case 仍不赢
# ==========================================================================
def case_near_normal():
    hr("CASE 3  近正规系统（cond(V)≪20）：特征向量基小优势赢 / worst-case 不赢")
    A = np.array([[-2.0, 0.5], [0.1, -1.0]])
    sys0 = Descriptor(np.eye(2), A, np.array([[1.0], [1.0]]), np.eye(2))
    gamma_id = float(resolvent_peak(sys0)["Gamma"])

    res_w = select_basis(sys0)
    _print_cards("near-normal worst-case", res_w)
    c_w = res_w.candidates["eigenvector"]
    add_check("near-normal worst-case: eigenvector still NOT selected",
              "winner=%s" % res_w.winner, "not eigenvector",
              res_w.winner != "eigenvector",
              note="不变式成立：纯相似变换预算 ≥ 恒等基，与条件数无关")

    eps = 1e-2
    res = select_basis(sys0, delta_pert=eps)
    _print_cards("near-normal eps*A", res)
    c_e = res.candidates["eigenvector"]
    add_check("near-normal structured: eigenvector selected",
              "winner=%s" % res.winner, "eigenvector",
              res.winner == "eigenvector")
    add_check("near-normal: cond(V) small (≪ shear's 20.05)",
              "%.3f" % c_e.cond_phi, "< 1.5", c_e.cond_phi < 1.5,
              note="近正规系统：换基税接近 1，Γ 的收益 1.075→1.050 能覆盖税")
    add_check("near-normal structured: eigenvector budget < identity budget",
              "%.4f < %.4f" % (c_e.total_budget, gamma_id), "lower",
              c_e.total_budget < gamma_id)
    add_check("Γ drop covers the tax: Γ_id/Γ_eig > cond(V)·(amp correction)",
              "Γ_eig·amp=%.4f < Γ_id=%.4f" % (c_e.total_budget, gamma_id),
              "budget decreases", c_e.total_budget < gamma_id)
    CASES["near_normal"] = dict(
        winner_worst=res_w.winner, winner_structured=res.winner,
        candidates_worst={k: v.as_dict() for k, v in res_w.candidates.items()},
        candidates={k: v.as_dict() for k, v in res.candidates.items()},
        gamma_identity=gamma_id)
    return res_w, res


# ==========================================================================
# CASE 4  接口 selector：光滑 / step / 翻转 / 守恒账本
# ==========================================================================
def case_interface():
    hr("CASE 4  接口 selector：光滑免费 / 单源合约 / ν_num 阈值翻转 / 守恒")

    # (a) 光滑状态：capture 免费，无特殊接口
    n = 64
    x = np.arange(n) * 2 * np.pi / n
    smooth = np.sin(x)
    r_s = choose_interface(smooth)
    emit("  smooth sin: mode=%s, index=%s, nu_num=%.3g, reason: %s"
         % (r_s.mode, r_s.shared_flux_index, r_s.nu_num_expected, r_s.reason))
    add_check("smooth state: no special interface (capture, free)",
              "mode=%s, idx=%s, nu_num=%.2e" %
              (r_s.mode, r_s.shared_flux_index, r_s.nu_num_expected),
              "capture / None / ~0",
              r_s.mode == "capture" and r_s.shared_flux_index is None
              and r_s.nu_num_expected < 1e-9)
    add_check("smooth state: single-source contract still attached",
              "%s" % r_s.contract.single_source, "True",
              r_s.contract.single_source)

    # (b) 真激波（单前沿 tanh，周期绕回处唯一跳变）：M_int=1，单源合约
    front = -np.tanh(3.0 * x)          # u: +1 ... −1，绕回面单个 2-jump
    r_b = choose_interface(front)
    emit("  tanh front ±1: mode=%s, M_int=%d, face=%d, F=%.4f, nu_num=%.3g, "
         "remesh=%.3g" % (r_b.mode, r_b.n_interfaces, r_b.shared_flux_index,
                          r_b.shared_flux_value, r_b.nu_num_expected,
                          r_b.remesh_cost))
    emit("  reason: %s" % r_b.reason)
    i_f = r_b.shared_flux_index
    f_expected = 0.5 * (0.5 * front[i_f] ** 2 + 0.5 * front[(i_f + 1) % n] ** 2) \
        - 0.5 * float(np.max(np.abs(front))) * (front[(i_f + 1) % n] - front[i_f])
    add_check("shock front: singular detected, M_int=1",
              "singular=%s, M_int=%d" % (r_b.singular, r_b.n_interfaces),
              "True / 1", r_b.singular and r_b.n_interfaces == 1)
    add_check("shock front: single-source face at the jump (Rusanov value)",
              "face=%d, F=%.4f" % (r_b.shared_flux_index, r_b.shared_flux_value),
              "index set, F = Rusanov(uL,uR) ±1e-9",
              r_b.shared_flux_index is not None
              and abs(r_b.shared_flux_value - f_expected) < 1e-9)
    add_check("shock ±1: nu_num=0.5·α·dx·σ > remesh unit ⇒ fit",
              "mode=%s, nu_num=%.3g, remesh=%.3g" %
              (r_b.mode, r_b.nu_num_expected, r_b.remesh_cost),
              "fit (nu_num > 0.25)",
              r_b.mode == "fit" and r_b.nu_num_expected > r_b.remesh_cost)

    # (c) 翻转：弱激波 capture；同一强激波 + 高 remesh 单价 capture
    weak = 0.2 * front
    r_w = choose_interface(weak)
    r_t = choose_interface(front, remesh_cost=10.0)
    emit("  weak step ±0.2: mode=%s, nu_num=%.3g (vs 0.25)" %
         (r_w.mode, r_w.nu_num_expected))
    emit("  strong step + remesh_cost=10: mode=%s" % r_t.mode)
    add_check("flip: weak shock (nu_num < threshold) -> capture",
              "mode=%s, nu_num=%.3g" % (r_w.mode, r_w.nu_num_expected),
              "capture", r_w.mode == "capture")
    add_check("flip: raising the threshold flips fit -> capture (same state)",
              "mode=%s" % r_t.mode, "capture", r_t.mode == "capture",
              note="capture-vs-fit 的判定随 ν_num/remesh 单价比翻转——阈值是显式参数")

    # (d) 守恒账本：与 coupling.py 的 dM/dt 恒等式对账
    contract = r_b.contract
    leak = contract.check(1.0, 1.0, 2.0, 2.0)
    add_check("contract: single-source dM/dt exactly zero (topological)",
              "%.1e" % leak, "0.0", leak == 0.0)
    add_check("contract: mass_budget identity matches coupling.py rhs",
              "%.1f" % mass_budget(1.0, 3.0, 2.0, 5.0), "(3−1)−(5−2) = −1",
              abs(mass_budget(1.0, 3.0, 2.0, 5.0) - (-1.0)) < 1e-15)

    # (e) CoupledSystem 实跑：single-source 泄漏≈机器精度 vs mismatch 泄漏
    T, dt = 0.2, 1e-3
    s1 = CoupledSystem(p=8, n_cells=32, model=InterfaceModel(single_source=True))
    r1 = s1.run(s1.smooth_ic(), T=T, dt=dt)
    s2 = CoupledSystem(p=8, n_cells=32,
                       model=InterfaceModel(single_source=False, mismatch=1e-2))
    r2 = s2.run(s2.smooth_ic(), T=T, dt=dt)
    emit("  coupled single-source: drift=%.3e, leak_max=%.3e" %
         (r1["drift"], r1["leak_max"]))
    emit("  coupled mismatch=1e-2: drift=%.3e, leak_max=%.3e" %
         (r2["drift"], r2["leak_max"]))
    add_check("coupled twin: single-source mass drift at machine precision",
              "%.3e" % abs(r1["drift"]), "< 1e-12", abs(r1["drift"]) < 1e-12)
    add_check("coupled twin: independent mirrors leak (mismatch=1e-2)",
              "%.3e" % r2["leak_max"], "> 1e-3", r2["leak_max"] > 1e-3)
    add_check("coupled twin: single-source leak_max ≈ 0 (vs mismatched)",
              "%.3e ≪ %.3e" % (r1["leak_max"], r2["leak_max"]),
              "ratio < 1e-6", r2["leak_max"] > 0 and
              r1["leak_max"] / r2["leak_max"] < 1e-6)

    CASES["interface"] = dict(
        smooth=r_s.as_dict(), step=r_b.as_dict(),
        weak=r_w.as_dict(), strong_high_threshold=r_t.as_dict(),
        coupled=dict(single_drift=float(r1["drift"]),
                     single_leak=float(r1["leak_max"]),
                     mismatch_leak=float(r2["leak_max"])))
    return r_s, r_b, r_w, r_t


# ==========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CHECKS.clear()
    CASES.clear()

    hr("B1 验收：basis（稀疏度 vs κ(Φ) 税）+ interface（单源合约 + fitting/capturing）")
    res_w, win = case_shear_worst()
    case_shear_structured()
    case_near_normal()
    case_interface()

    hr("VERDICT")
    rows = [(c["name"], str(c["got"]), str(c["expected"]),
             "yes" if c["passed"] else "NO") for c in CHECKS]
    table(["check", "got", "expected", "pass"], rows, [""] * 4)
    n_fail = sum(1 for c in CHECKS if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(CHECKS) - n_fail, len(CHECKS)))
    emit("  shear (worst-case): winner=%s, budget=%.4f "
         "(eigvec Γ=%.3f×κ(Φ)=%.2f=%.2f — selected-against)"
         % (res_w.winner, win.total_budget,
            res_w.candidates["eigenvector"].Gamma,
            res_w.candidates["eigenvector"].cond_phi,
            res_w.candidates["eigenvector"].total_budget))
    emit("  shear (Δ=ε·A): winner=%s, budget=%.4f (chooser not degenerate)"
         % (CASES["shear_structured"]["winner"],
            CASES["shear_structured"]["candidates"]["eigenvector"]["total_budget"]))

    def _jsonable(o):
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return f if np.isfinite(f) else ("inf" if f > 0 else "-inf")
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return _jsonable(o.tolist())
        return o

    with open(os.path.join(OUT, "basis_interface.json"), "w") as fh:
        json.dump(_jsonable(dict(cases=CASES, checks=CHECKS,
                                 summary=dict(n_checks=len(CHECKS),
                                              n_passed=len(CHECKS) - n_fail))),
                  fh, indent=1)
    with open(os.path.join(OUT, "basis_interface_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/basis_interface.json, results/basis_interface_report.txt")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
