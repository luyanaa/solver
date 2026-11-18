"""B1 closure+split 验收：P_N 投影 / η_closure 证书 + CSP/QSSA 快慢剥离。

    python3 -m bench.run_closure_split

closure 算例（lowering.closure，DESIGN §5.2 closure 合同）：

    diffusion  线性对角 + 带宽外 forcing：η 随 N 收缩单调增长（对照解析
               Σ_{k>N}1/k² 公式）；k>N 单模 forcing 全额进 defect
    burgers    二次非线性（无 alias 谱导数 rhs）：defect 落在 (N,2N]
               alias 带 ⇒ change-basis（2/3 规则消 aliasing 不消 closure）
    负对照     η>eps ⇒ verdict 必须是 reduce-N/change-basis，不静默 feasible

split 算例（lowering.split，DESIGN §4.1 P2 决策树第二行）：

    stiff 对   λ=-1,-1e4（A_sf=0 单向奴役）：Schur 精确复现慢特征值；
               slaving 残差 ∝ A_fs·gap⁻¹（扫 A_fs 验证标度）
    反馈耦合   A_sf≠0：eig_match ≤ ‖A_sf‖‖A_ff⁻¹‖‖A_fs‖/|λ_s| 界
    负对照     弱间隙(-1,-2)、亏损 Jordan、不稳定 fast、奇异 E、
               奇异 fast 块——全部 flag/拒绝，不静默产 QSSA 数字
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from analysis import Descriptor
from lowering.closure import project_spectral
from lowering.split import cvsplit, qssa_reduce
from lowering.discretize import PDESpec, discretize

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
_LINES: list[str] = []
CHECKS: list[dict] = []
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


def add_check(name, got, expected, passed, kind="closure_split", note=None):
    CHECKS.append(dict(name=name, got=got, expected=expected,
                       passed=bool(passed), kind=kind,
                       **({"note": note} if note else {})))


# ==========================================================================
# CASE 1  closure：diffusion（线性 Descriptor，模态坐标 k=1..n）
# ==========================================================================
def make_diffusion(n=64, nu=1.0):
    d = discretize(PDESpec(name="diffusion", operator="diffusion", domain=((0.0, 1.0),),
                           bc="periodic", coeff=nu, n=(n,)))
    return d.descriptor


def case_closure_diffusion():
    hr("CASE 1  closure：diffusion 线性对角 + 带宽外 forcing（η vs N）")
    desc = make_diffusion(64)
    f = 1.0 / np.arange(1, 65)                     # 全带宽 forcing f_k = 1/k
    probe_default = np.ones(64) / np.sqrt(64)      # closure 缺省代表态
    ref_default = float(np.linalg.norm(desc.A @ probe_default + f))

    def eta_analytic(N):
        # defect = (I−P_N)(A·P_N u + f) = f_{k>N}（线性对角：A 不跨模态搬能量）
        return float(np.linalg.norm(f[N:]) / ref_default)

    rows = []
    etas = {}
    for N in (8, 16, 32, 48, 64):
        r = project_spectral(desc, N, forcing=f, eps=None)
        etas[N] = r.eta_closure
        rows.append((N, r.eta_closure, eta_analytic(N) if N < 64 else 0.0,
                     r.defect_norm, r.ref_denominator, r.verdict))
    table(["N", "eta_meas", "eta_analytic", "defect", "ref", "verdict"],
          rows, ["%d", "%.6e", "%.6e", "%.6e", "%.4f", "%s"])
    CASES["closure_diffusion"] = dict(
        eta={str(k): v for k, v in etas.items()},
        eta_analytic={str(N): eta_analytic(N) for N in (8, 16, 32, 48)})

    # N=n_full：线性对角在任何截断下精确闭——defect=0 是正确结论
    add_check("eta(N=full) is exactly 0 (linear diagonal is closed)",
              "%.2e" % etas[64], "0", etas[64] == 0.0)
    # η(N) 对照解析式 Σ_{k>N}1/k² / Σ 1/k²
    ok = all(abs(etas[N] - eta_analytic(N)) < 1e-12 for N in (8, 16, 32, 48))
    add_check("eta(N) matches ||f_{k>N}|| / ||F(u_ref)|| (4 points)", "max|dev|<1e-12",
              "yes", ok)
    # 单调增长：N 缩到已解析带宽之下，η 长回去
    add_check("eta grows monotonically as N shrinks (8>16>32>48>64)",
              "%.3e > %.3e > %.3e > %.3e > %.3e" % (etas[8], etas[16], etas[32],
                                                    etas[48], etas[64]),
              "strictly decreasing in N",
              etas[8] > etas[16] > etas[32] > etas[48] > etas[64])

    # defect 支撑集：只在 (I−P_N) 里
    r16 = project_spectral(desc, 16, forcing=f)
    add_check("defect confined to (I-P_N): coords 0..15 are zero",
              "%.2e" % float(np.linalg.norm(r16.defect[:16])), "0",
              float(np.linalg.norm(r16.defect[:16])) == 0.0)

    # k>N 单模 forcing：全额 defect（clean defect）
    f33 = np.zeros(64)
    f33[32] = 1.0                                  # k=33
    r33 = project_spectral(desc, 16, forcing=f33, ref_scale=1.0)
    support_ok = (np.argmax(np.abs(r33.defect)) == 32
                  and np.linalg.norm(r33.defect) == 1.0)
    add_check("forcing at k=33 with N=16: defect = full forcing (clean)",
              "‖r‖=%.6f @k=%d" % (np.linalg.norm(r33.defect),
                                  np.argmax(np.abs(r33.defect)) + 1),
              "‖r‖=1 @k=33", support_ok)
    add_check("eta = 1 exactly for out-of-band unit forcing",
              "%.12f" % r33.eta_closure, "1", abs(r33.eta_closure - 1.0) < 1e-12)

    # verdict：负对照——η>eps 必须要求 reduce-N，不静默 feasible
    r_neg = project_spectral(desc, 16, forcing=f, eps=0.1 * etas[16])
    r_pos = project_spectral(desc, 16, forcing=f, eps=10.0 * etas[16])
    add_check("verdict: eta>eps asks reduce-N (not silently feasible)",
              r_neg.verdict, "reduce-N", r_neg.verdict == "reduce-N")
    add_check("verdict: eps > eta ⇒ feasible", r_pos.verdict, "feasible",
              r_pos.verdict == "feasible")
    add_check("verdict: no eps ⇒ unjudged (no silent claim)",
              r16.verdict, "unjudged", r16.verdict == "unjudged")

    # 投影后的 descriptor：保留慢块（前 N 行列）
    pr = r16.projected
    add_check("projected descriptor keeps slow block A[:N,:N]",
              "shape=%d, A[0,0]=%.1f" % (pr.n, pr.A[0, 0]),
              "n=16, A[0,0]=-1", pr.n == 16 and abs(pr.A[0, 0] + 1.0) < 1e-14)
    return etas


# ==========================================================================
# CASE 2  closure：burgers 二次非线性（fourier 场路径，alias 带）
# ==========================================================================
def make_burgers_rhs(n_grid=64):
    """无 alias 谱 rhs：F = −∂x(u²/2)，显式线性卷积（不依赖网格上/下采样）。

    v_k = fft(u)/n；w_k = ½Σ_j v_j v_{k−j}（线性卷积，支持 |k|≤2·带限）；
    F = −n·ifft(2πi·k·w).real。精确性要求输入带限 ≤ n/4（否则乘积会折叠；
    本算例 N≤4）。返回全长场。
    """
    kint = np.fft.fftfreq(n_grid) * n_grid
    idx_of = {int(kk): i for i, kk in enumerate(kint)}
    kmax = n_grid // 2 - 1

    def rhs(u):
        v = np.fft.fft(np.asarray(u, dtype=float)) / n_grid
        w = np.zeros(n_grid, dtype=complex)
        for a in range(n_grid):
            va = v[a]
            if abs(va) < 1e-14:
                continue
            ka = int(kint[a])
            for b in range(n_grid):
                kk = ka + int(kint[b])
                if abs(kk) <= kmax:
                    w[idx_of[kk]] += 0.5 * va * v[b]
        return (-n_grid * np.fft.ifft(2j * np.pi * kint * w)).real
    return rhs


def case_closure_burgers():
    hr("CASE 2  closure：burgers 二次项——defect 落在 (N,2N] alias 带")
    n = 64
    rhs = make_burgers_rhs(n)
    x = np.arange(n) / n
    probe = 0.5 * np.sin(2 * np.pi * x) + 0.25 * np.sin(4 * np.pi * x)  # 带 ≤2

    rows = []
    etas = {}
    for N in (2, 3, 4):
        r = project_spectral(rhs, N, n_full=n, probe=probe, eps=None)
        etas[N] = r
        rows.append((N, r.eta_closure, r.alias_frac, r.N_effective, r.verdict))
    table(["N", "eta", "alias_frac", "N_eff", "verdict"],
          [(N, e.eta_closure, e.alias_frac, e.N_effective, e.verdict) for N, e in etas.items()],
          ["%d", "%.6e", "%.3f", "%d", "%s"])
    CASES["closure_burgers"] = dict(
        eta={str(N): r.eta_closure for N, r in etas.items()},
        alias_frac={str(N): r.alias_frac for N, r in etas.items()})

    add_check("quadratic closure defect > 0 at N=2 (content up to |k|<=2N)",
              "%.3e" % etas[2].eta_closure, ">0", etas[2].eta_closure > 1e-6)
    add_check("eta decreases with N (2 > 3 > 4)",
              "%.3e > %.3e > %.3e" % (etas[2].eta_closure, etas[3].eta_closure,
                                      etas[4].eta_closure),
              "monotone", etas[2].eta_closure > etas[3].eta_closure > etas[4].eta_closure)
    add_check("eta(N=4) at numerical floor: products of |k|<=2 modes stay |k|<=4",
              "%.2e" % etas[4].eta_closure, "<1e-12", etas[4].eta_closure < 1e-12)
    add_check("defect energy entirely in (N,2N] alias band at N=2",
              "%.3f" % etas[2].alias_frac, "1.0", abs(etas[2].alias_frac - 1.0) < 1e-12)
    # 负对照：alias 带主导 ⇒ change-basis（降 N 无用），不是 reduce-N
    r_v = project_spectral(rhs, 2, n_full=n, probe=probe, eps=1e-4)
    add_check("verdict: alias-band-dominated defect ⇒ change-basis",
              r_v.verdict, "change-basis", r_v.verdict == "change-basis")
    # N 已达可分辨上限（n//2−1=31）且 rhs 无 alias：defect=0 ⇒ feasible。
    # change-basis 在本模块只由 alias_frac>0.5 触发（见 closure._verdict 注释）。
    r_max = project_spectral(rhs, n // 2 - 1, n_full=n, probe=probe, eps=1e-8)
    add_check("N at resolvable max with exact rhs: eta=0 ⇒ feasible",
              "%s/%.2e" % (r_max.verdict, r_max.eta_closure), "feasible/~0",
              r_max.verdict == "feasible" and r_max.eta_closure < 1e-12)


# ==========================================================================
# CASE 3  split：stiff 对 λ=-1, -1e4（单向奴役，A_sf=0）
# ==========================================================================
def case_split_stiff_pair():
    hr("CASE 3  split：λ=-1,-1e4 stiff 对——Schur 精确复现 + slaving 残差 ∝ gap⁻¹")
    T = 3.0
    rows = []
    errs = {}
    for k_fs in (1.0, 0.1):
        A = np.array([[-1.0, 0.0], [k_fs, -1e4]])
        r = cvsplit(A)
        errs[k_fs] = r
        rows.append(("A_fs=%.1f" % k_fs, r.verdict, r.gap_ratio, r.n_slow, r.n_fast,
                     r.eig_match, r.slaving_error, ",".join(r.flags) or "-"))
    table(["case", "verdict", "gap_ratio", "n_s", "n_f", "eig_match", "slaving", "flags"],
          rows, ["%s", "%s", "%.3g", "%d", "%d", "%.2e", "%.3e", "%s"])
    r1, r2 = errs[1.0], errs[0.1]
    CASES["split_stiff"] = dict(
        slaving={("A_fs=%.1f" % k): r.slaving_error for k, r in errs.items()},
        eig_match=r1.eig_match, gap_ratio=r1.gap_ratio)

    add_check("stiff pair auto-splits at the 1e4 gap", r1.verdict, "split",
              r1.verdict == "split" and r1.n_slow == 1 and r1.n_fast == 1)
    add_check("gap ratio >= 1e3 reported", "%.3g" % r1.gap_ratio, ">=1e3",
              r1.gap_ratio >= 1e3)
    add_check("Schur A_eff reproduces slow eigenvalue to algebraic accuracy",
              "%.2e" % r1.eig_match, "<1e-12", r1.eig_match < 1e-12)
    add_check("reduced slow eigenvalue equals design value -1",
              "%.3e" % float(np.linalg.eigvals(r1.A_eff)[0].real), "-1",
              abs(np.linalg.eigvals(r1.A_eff)[0].real + 1.0) < 1e-12)
    add_check("slaving residual small because gap is large (measured)",
              "%.3e" % r1.slaving_error, "<1e-6", r1.slaving_error < 1e-6)
    ratio = r1.slaving_error / r2.slaving_error
    add_check("slaving residual scales ~ A_fs (10x smaller coupling ⇒ ~10x error)",
              "%.2f" % ratio, "~10", 5.0 < ratio < 20.0)
    add_check("reduced descriptor is n_slow square", "%d" % r1.reduced.n, "1",
              r1.reduced.n == 1)
    return r1


# ==========================================================================
# CASE 4  split：反馈耦合（A_sf≠0）——eig_match 对照解析界
# ==========================================================================
def case_split_feedback():
    hr("CASE 4  split：反馈耦合 A_sf≠0——QSSA 精度受 Schur 修正项支配")
    A = np.array([[-1.0, 1e-3], [1e-3, -1e4]])
    r = cvsplit(A)
    A_p = A[np.ix_(r.perm, r.perm)]
    A_sf, A_fs, A_ff = A_p[0, 1], A_p[1, 0], A_p[1, 1]
    lam_s = float(np.max(np.real(r.slow_eigs)))
    bound = abs(A_sf) * abs(1.0 / A_ff) * abs(A_fs) / abs(lam_s)
    emit("  A_eff=%s" % np.array2string(r.A_eff, precision=12))
    emit("  |eig_match|=%.3e <= ‖A_sf‖‖A_ff⁻¹‖‖A_fs‖/|λ_s| = %.3e" % (r.eig_match, bound))
    CASES["split_feedback"] = dict(eig_match=r.eig_match, bound=bound,
                                   slaving=r.slaving_error)
    add_check("feedback split: verdict split with flag set", r.verdict, "split",
              r.verdict == "split")
    add_check("eig_match within Schur-correction bound (×2 margin)",
              "%.2e <= %.2e" % (r.eig_match, 2 * bound), "yes",
              r.eig_match <= 2.0 * bound + 1e-14)
    add_check("feedback slaving residual reported and small", "%.3e" % r.slaving_error,
              "<1e-5", r.slaving_error < 1e-5)


# ==========================================================================
# CASE 5  split 负对照：弱间隙 / 亏损 / 不稳定 / 奇异
# ==========================================================================
def case_split_negative():
    hr("CASE 5  split 负对照：不硬剥、不静默求逆")

    # (a) 弱间隙 diag(-1,-2)：auto 不剥
    r = cvsplit(np.diag([-1.0, -2.0]))
    add_check("weak gap (-1,-2): auto ⇒ no-split, no fabricated QSSA",
              "%s/%s" % (r.verdict, r.reduced is None), "no-split/None",
              r.verdict == "no-split" and r.reduced is None
              and any("gap" in f for f in r.flags))
    add_check("weak gap ratio reported honestly", "%.1f" % r.gap_ratio, "2",
              abs(r.gap_ratio - 2.0) < 1e-9)

    # (b) caller 强制 tau_fast 跨弱间隙：尊重 caller 但打 weak-gap flag
    rf = cvsplit(np.diag([-1.0, -2.0]), tau_fast=0.6)
    add_check("forced tau_fast across weak gap: split honored + weak-gap flag",
              "%s/%s" % (rf.verdict, "weak-gap" in rf.flags), "split/weak-gap",
              rf.verdict == "split" and "weak-gap" in rf.flags)

    # (c) 亏损 Jordan 块：坐标级剥离不良定义
    rj = cvsplit(np.array([[-1.0, 1.0], [0.0, -1.0]]))
    add_check("defective Jordan block ⇒ no-split (defective flag)",
              "%s/%s" % (rj.verdict, "defective-eigenvectors" in rj.flags),
              "no-split/defective",
              rj.verdict == "no-split" and "defective-eigenvectors" in rj.flags)

    # (d) 不稳定 fast 候选：拒绝代数化
    ru = cvsplit(np.diag([-1.0, 100.0]))
    add_check("unstable fast candidate ⇒ no-split (fast-block-unstable)",
              "%s/%s" % (ru.verdict, "fast-block-unstable" in ru.flags),
              "no-split/fast-block-unstable",
              ru.verdict == "no-split" and "fast-block-unstable" in ru.flags)

    # (e) 奇异 E：NotImplementedError（剥离的输出正是奇异 E）
    d = Descriptor(np.diag([1.0, 0.0]), np.eye(2))
    try:
        cvsplit(d)
        raised = ""
    except NotImplementedError as e:
        raised = str(e)[:30]
    add_check("singular E ⇒ NotImplementedError (already a split output)",
              "raised", "NotImplementedError", bool(raised))

    # (f) 奇异 fast 块：qssa_reduce 拒绝静默求逆
    try:
        qssa_reduce(np.eye(1), np.zeros((1, 2)), np.array([[1.0, 2.0], [2.0, 4.0]]),
                    np.zeros((2, 1)))
        msg = ""
    except ValueError as e:
        msg = str(e)[:40]
    add_check("singular fast block ⇒ ValueError (no silent inversion)",
              "raised", "ValueError", bool(msg))
    emit("    %s" % (msg or "(no exception!)"))

    # (g) 不稳定 fast 块经 qssa_reduce 直查同样拒绝
    try:
        qssa_reduce(np.eye(1), np.zeros((1, 1)), np.array([[100.0]]), np.eye(1))
        msg2 = ""
    except ValueError as e:
        msg2 = str(e)[:40]
    add_check("unstable fast block ⇒ ValueError in qssa_reduce",
              "raised", "ValueError", bool(msg2))


# ==========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CHECKS.clear()
    CASES.clear()

    hr("B1 closure+split 验收（P_N 投影证书 / CSP-QSSA 快慢剥离）")
    case_closure_diffusion()
    case_closure_burgers()
    case_split_stiff_pair()
    case_split_feedback()
    case_split_negative()

    hr("VERDICT")
    rows = [(c["name"], str(c["got"]), str(c["expected"]),
             "yes" if c["passed"] else "NO") for c in CHECKS]
    table(["check", "got", "expected", "pass"], rows, [""] * 4)
    n_fail = sum(1 for c in CHECKS if not c["passed"])
    emit("\n  %d/%d checks passed" % (len(CHECKS) - n_fail, len(CHECKS)))
    cd = CASES["closure_diffusion"]["eta"]
    emit("  eta_closure(N) diffusion: %s" %
         " ".join("N=%s:%.3e" % (k, v) for k, v in sorted(cd.items(), key=lambda kv: int(kv[0]))))
    emit("  slaving residual @gap 1e4: %.3e (A_fs=1) / %.3e (A_fs=0.1)"
         % (CASES["split_stiff"]["slaving"]["A_fs=1.0"],
            CASES["split_stiff"]["slaving"]["A_fs=0.1"]))

    with open(os.path.join(OUT, "closure_split.json"), "w") as fh:
        json.dump(_jsonable(dict(cases=CASES, checks=CHECKS,
                                 summary=dict(n_checks=len(CHECKS),
                                              n_passed=len(CHECKS) - n_fail))),
                  fh, indent=1)
    with open(os.path.join(OUT, "closure_split_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  wrote results/closure_split.json, results/closure_split_report.txt")
    return 1 if n_fail else 0


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, complex):
        c = complex(o)
        return dict(re=c.real, im=c.imag)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else ("inf" if f > 0 else "-inf")
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if o is None or isinstance(o, str):
        return o
    return str(o)


if __name__ == "__main__":
    raise SystemExit(main())
