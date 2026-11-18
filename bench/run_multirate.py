"""Phase 4.3 验收：waveform relaxation 收敛率 + replay 底噪 + ZOH 边缘。

    python3 -m bench.run_multirate

算例（DESIGN.md §4.3 合同，review 第三支柱 4）：

    A  spectral radius  两个双岛线性分裂：
       N  正规分裂（两岛 = 同一轻阻尼振子 −0.2·I+5·J）→ G_WR 在峰处
          正规，实测每迭代衰减率 = ρ_sup（合同的公式在此严格成立）；
          弱耦合 ρ_sup<1，强耦合 ρ_sup>1（负控制，合同必须说 NO）。
       NN 非正规分裂（岛 A = 对称 DC 岛，岛 B = 振子）→ σ_max 证书
          保守 ×5：实测率 = 特征值谱半径 sup|λ|，不是 σ_max。
       强耦合 NN(c=1.1)：特征值率本身 >1 → 实测发散（负控制）。
    B  replay floor    每次迭代注入带内重放噪声；平台高度 vs 预测界，
                       观测拐点 k vs 公式 k_max。
    C  ZOH→route       zoh_edge(T_s) 过 route_closure 的 feedback 判定，
                       与手工 PureDelay(T_s/2) 完全同一套机器。

WR 实现：精确 ZOH 离散（expm 步进，交换波形分段常值），**周期域**
求解（每迭代解周期边界条件 x(0)=(I−Φ^P)⁻¹s）。合同 ρ_sup 公式描述
的正是逐频稳态迭代：周期域上迭代算子被 DFT 精确对角化，误差
e^(k) = G_d(e^{iω*dt})·e^(k−1)（离散 G_WR），没有离散化底噪——实测
速率完全归属于 WR 映射本身。另附一个 IVP 窗口诊断：因果 Volterra
结构下实测收敛比 ρ_sup 界**更快**（证书保守方向），如实报告。

诚实声明（实测）：σ_max(G_WR) 的 sup 是 small-gain **充分**证书；
实际每迭代衰减率 = sup_ω |λ_max(G_WR(jω))|（误差幂迭代收敛到主导
特征方向）。正规分裂两者相等；非正规分裂 σ_max 保守（×5，方向：
不会漏报发散，可能错杀可收敛分裂）。强耦合负控制用特征值率>1
的系统——σ_max>1 但特征值率<1 的分裂实测仍收敛，说明合同条件
"ρ_sup<1" 是充分的、不是必要的（不要收紧成充要）。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import block_diag, expm

from twin.multirate import (k_max, wr_error_bound, wr_iteration_matrix,
                            wr_spectral_radius, zoh_edge)
from twin.route import PureDelay, RouteEdge, RouteSpec, route_closure

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
# 双岛测试系统与精确 ZOH 离散周期 WR
# ---------------------------------------------------------------------------
OSC = np.array([[-0.2, 5.0], [-5.0, -0.2]])   # 轻阻尼振子 ω0=5, ζ→σ_max=5 @ ω=5
A_SYM = np.array([[-0.8, 0.3], [0.3, -1.2]])  # 对称 DC 岛（正规但低频）

SYSTEMS = {
    "N":  block_diag(OSC, OSC),        # 正规分裂：R_A=R_B 正规 ⇒ G_WR 正规
    "NN": block_diag(A_SYM, OSC),      # 非正规分裂：σ_max 证书保守
}
E_ID = np.eye(4)
OMEGA_GRID = np.logspace(np.log10(0.03), np.log10(30.0), 241)

W_DRIVE = 5.0                       # 激励在 interior peak 频率
N_PERIODS = 8
NT = 2010
DT = (N_PERIODS * 2.0 * np.pi / W_DRIVE) / NT   # ω*·T = 2π·8 ⇒ 精确 DFT bin


def build_split(sys_name, c12, c21):
    A_diag = SYSTEMS[sys_name]
    A_off = np.zeros((4, 4))
    A_off[:2, 2:4] = c12 * np.eye(2)     # 岛 A 收岛 B
    A_off[2:4, :2] = c21 * np.eye(2)     # 岛 B 收岛 A
    return A_off, A_diag + A_off


def _zoh_step(Ai, dt):
    m = Ai.shape[0]
    M = np.zeros((2 * m, 2 * m))
    M[:m, :m] = Ai
    M[:m, m:] = np.eye(m)
    Em = expm(M * dt)
    return Em[:m, :m], Em[:m, m:]


_ZOH_CACHE: dict = {}


def _phi_gam(sys_name):
    if sys_name not in _ZOH_CACHE:
        blocks = [OSC, OSC] if sys_name == "N" else [A_SYM, OSC]
        ph, ga = zip(*(_zoh_step(Ab, DT) for Ab in blocks))
        _ZOH_CACHE[sys_name] = (block_diag(*ph), block_diag(*ga))
    return _ZOH_CACHE[sys_name]


def _forcing():
    t = np.arange(NT) * DT
    return np.cos(W_DRIVE * t)[:, None] * np.array([1.0, 0.5, 0.0, 0.0])[None, :]


def _periodic_solve(PHI, GAM, u):
    """解 x(n+1)=Φx(n)+Γu(n) 的周期解；返回 (NT,4)。"""
    s = np.zeros(4)
    for i in range(NT):
        s = PHI @ s + GAM @ u[i]
    x0 = np.linalg.solve(np.eye(4) - np.linalg.matrix_power(PHI, NT), s)
    x = np.empty((NT, 4))
    x[0] = x0
    for i in range(NT - 1):
        x[i + 1] = PHI @ x[i] + GAM @ u[i]
    return x


def _run_wr_periodic(sys_name, c, k_iter, noise_std=0.0, seed=0):
    """Jacobi 周期 WR：每次迭代两个岛都用上一次迭代的完整波形。

    noise_std > 0：交换波形叠加带内窄带高斯噪声（每次迭代独立随机
    幅度/相位，逐样本 std = noise_std）——把噪声功率放在 ω* 处，
    即 σ_max(G_WR) 的峰值频率；白逐样本噪声的带内功率更低，界只会
    更保守。
    """
    PHI, GAM = _phi_gam(sys_name)
    A_off, _ = build_split(sys_name, c, c)
    b = _forcing()
    rng = np.random.default_rng(seed)
    W = np.zeros((k_iter + 1, NT, 4))
    for k in range(1, k_iter + 1):
        xp = W[k - 1]
        if noise_std > 0.0:
            t = np.arange(NT) * DT
            zeta = rng.standard_normal(4)
            phi = rng.uniform(0.0, 2.0 * np.pi, 4)
            # 带内窄带高斯：n(t)=ε√2·ζ·cos(ω*t+φ)，逐样本 std=ε（ζ~N(0,1)）
            n = noise_std * np.sqrt(2.0) * (
                np.cos(W_DRIVE * t)[:, None] * (zeta * np.cos(phi))[None, :]
                - np.sin(W_DRIVE * t)[:, None] * (zeta * np.sin(phi))[None, :])
            xp = xp + n
        W[k] = _periodic_solve(PHI, GAM, xp @ A_off.T + b)
    return W


def _run_wr_ivp(sys_name, c, k_iter):
    """IVP（零初值）窗口 WR 诊断：因果 Volterra 结构，实测率 ≤ ρ_sup。"""
    PHI, GAM = _phi_gam(sys_name)
    A_off, _ = build_split(sys_name, c, c)
    b = _forcing()
    W = np.zeros((k_iter + 1, NT, 4))
    for k in range(1, k_iter + 1):
        u = W[k - 1] @ A_off.T + b
        x = np.empty((NT, 4))
        x[0] = 0.0
        for i in range(NT - 1):
            x[i + 1] = PHI @ x[i] + GAM @ u[i]
        W[k] = x
    return W


def _rms(a):
    return float(np.sqrt(np.mean(np.sum(a * a, axis=1))))


def _fit_rate(errs, k_lo, k_hi):
    """每迭代衰减率：err_k ≈ e0·ρ^k ⇒ slope = ln ρ ⇒ rate = exp(slope)。"""
    ks = np.arange(k_lo, k_hi + 1)
    e = np.array([errs[k] for k in ks])
    m = e > 0
    slope = np.polyfit(ks[m], np.log(e[m]), 1)[0]
    return float(np.exp(slope))


# ---------------------------------------------------------------------------
# 算例
# ---------------------------------------------------------------------------
def case_spectral():
    out = {}
    # 频域：ρ_sup（σ_max small-gain）与 rho_eig_sup（实际逐频率）
    for tag, sys_name, c in (("N_weak", "N", 0.15), ("N_strong", "N", 0.30),
                             ("NN_weak", "NN", 0.15), ("NN_strong", "NN", 1.10),
                             ("replay", "N", 0.08)):
        A_off, _ = build_split(sys_name, c, c)
        r = wr_spectral_radius(E_ID, SYSTEMS[sys_name], A_off, OMEGA_GRID)
        r.update(sys=sys_name, c=c)
        out[tag] = r

    # 周期域暴力 WR（合同的逐频率在此精确成立）
    # N_weak：收敛，实测率 vs ρ_sup
    W = _run_wr_periodic("N", 0.15, k_iter=50)
    ref = W[-1]
    errs = [(_rms(W[0] - ref) if k == 0 else _rms(W[k] - ref)) for k in range(51)]
    out["N_weak"]["errs"] = errs
    out["N_weak"]["rate_measured"] = _fit_rate(errs, 2, 20)

    # N_strong：ρ_sup>1 且正规 ⇒ 特征值率>1 ⇒ 必发散（负控制）
    Ws = _run_wr_periodic("N", 0.30, k_iter=12)
    d = [_rms(Ws[k] - Ws[k - 1]) for k in range(1, 13)]
    out["N_strong"]["diverge_ratio"] = d[11] / d[1]

    # NN_weak：非正规——实测率应 ≈ rho_eig_sup，≪ ρ_sup
    Wn = _run_wr_periodic("NN", 0.15, k_iter=40)
    refn = Wn[-1]
    errsn = [_rms(Wn[0] - refn) if k == 0 else _rms(Wn[k] - refn) for k in range(41)]
    out["NN_weak"]["errs"] = errsn
    out["NN_weak"]["rate_measured"] = _fit_rate(errsn, 2, 14)

    # NN_strong (c=1.1)：σ_max≫1 且特征值率>1 ⇒ 发散（负控制）
    Wns = _run_wr_periodic("NN", 1.10, k_iter=12)
    dn = [_rms(Wns[k] - Wns[k - 1]) for k in range(1, 13)]
    out["NN_strong"]["diverge_ratio"] = dn[11] / dn[1]

    # IVP 窗口诊断（正规分裂、零初值）：Volterra 因果结构 → 比界更快
    Wi = _run_wr_ivp("N", 0.15, k_iter=24)
    refi = Wi[-1]
    errsi = [_rms(Wi[k] - refi) for k in range(25)]
    kk = [k for k in range(2, 18) if errsi[k] > 0]
    out["ivp"] = dict(rate=_fit_rate(errsi, 2, min(12, len(kk) - 1 + 2)),
                      err0=errsi[0], err10=errsi[10],
                      monotone=bool(all(errsi[k] <= errsi[k - 1] * (1 + 1e-9)
                                        for k in range(1, 12) if errsi[k] > 0)))
    CASES["spectral"] = out


def case_replay():
    sp = CASES["spectral"]
    rp, e0 = sp["replay"], None
    rho = rp["rho_sup"]
    # e0：初始误差 = 零交换解 vs 收敛点（无噪 run 的首误差）
    W0 = _run_wr_periodic("N", rp["c"], k_iter=45)
    ref = W0[-1]
    e0 = _rms(-ref)  # 初始猜测 = 0 ⇒ e0 = ‖x*‖
    # 目标 k_max ≈ 8：ε = (1−ρ)·e0·ρ^8 / ρ
    eps = (1.0 - rho) * e0 * rho ** 8 / rho
    R_tight = rho
    # 合同原文的保守 resolvent 范数 sup_ω σ_max((jωE−A_diag)^{-1})（诊断报告）
    R_loose = float(max(
        np.linalg.svd(np.linalg.solve(1j * w * E_ID - SYSTEMS["N"],
                                      np.eye(4)), compute_uv=False)[0]
        for w in OMEGA_GRID))
    k_pred = k_max(rho, R_tight, eps, e0)
    plateau_pred = R_tight * eps / (1.0 - rho)
    plateau_loose = R_loose * eps / (1.0 - rho)

    W = _run_wr_periodic("N", rp["c"], k_iter=30, noise_std=eps, seed=7)
    errs = [_rms(W[k] - ref) for k in range(31)]
    plateau_obs = float(np.median(errs[-6:]))
    # 观测拐点：首个 k 使得其后 5 次的中位数 ≤ 1.5×平台
    # （单次迭代的噪声幅度 ζ 随机，尖峰用中位数抑制）
    knee = None
    for k in range(31):
        if k + 5 <= 30 and np.median(errs[k:k + 5]) <= 1.5 * plateau_obs:
            knee = k
            break
    late_rate = _fit_rate(errs, 24, 30)
    CASES["replay"] = dict(eps=eps, rho=rho, e0=e0, R_tight=R_tight,
                           R_loose=R_loose, k_pred=k_pred, knee=knee,
                           plateau_pred=plateau_pred,
                           plateau_loose=plateau_loose,
                           plateau_obs=plateau_obs, errs=errs,
                           late_rate=late_rate)


def case_zoh():
    T_s = 1e-6
    ez = RouteEdge((zoh_edge(T_s),), "zoh_island_swap")
    em = RouteEdge((PureDelay(T_s / 2.0),), "zoh_island_swap")
    spec = RouteSpec(omega_max=2 * np.pi * 10e3, omega_c=2 * np.pi * 10e3,
                     eps_phase_deg=10.0, feedback_critical=True,
                     phi_reserve_deg=5.0)
    rz = route_closure(ez, spec)
    rm = route_closure(em, spec)
    CASES["zoh"] = dict(T_s=T_s, latency=zoh_edge(T_s).latency,
                        closure_zoh=rz, closure_manual=rm)


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------
def acceptance():
    checks = []

    def add(name, got, expected, passed):
        checks.append(dict(name=name, got=got, expected=expected,
                           passed=bool(passed)))

    sp = CASES["spectral"]
    nw, ns = sp["N_weak"], sp["N_strong"]
    nnw, nns = sp["NN_weak"], sp["NN_strong"]

    add("N weak ρ_sup", "%.4f @ ω=%.2f" % (nw["rho_sup"], nw["omega_peak"]),
        "< 1", nw["rho_sup"] < 1.0)
    add("N strong ρ_sup (negative control)",
        "%.4f @ ω=%.2f" % (ns["rho_sup"], ns["omega_peak"]), "> 1 (NO)",
        ns["rho_sup"] > 1.0)
    add("interior peak, not DC",
        "σ_dc=%.3f ρ=%.3f @ω=%.2f" % (nw["sigma_dc"], nw["rho_sup"],
                                      nw["omega_peak"]),
        "interior & σ_dc<0.9ρ_sup",
        nw["interior_peak"] and nw["sigma_dc"] < 0.9 * nw["rho_sup"])
    rel = abs(nw["rate_measured"] - nw["rho_sup"]) / nw["rho_sup"]
    add("N measured rate vs ρ_sup (normal ⇒ exact)",
        "%.4f vs %.4f (Δ%.1f%%)" % (nw["rate_measured"], nw["rho_sup"],
                                    100 * rel), "within ±25%", rel <= 0.25)
    add("N rate bracket [σ_dc, ρ_sup]", "%.4f" % nw["rate_measured"],
        "σ_dc ≤ rate ≤ 1.05·ρ_sup",
        nw["sigma_dc"] * 0.95 <= nw["rate_measured"] <= 1.05 * nw["rho_sup"])
    add("N strong diverges", "×%.2f per-iter diff growth" % ns["diverge_ratio"],
        "> 1", ns["diverge_ratio"] > 1.0)

    gap_rel = abs(nnw["rate_measured"] - nnw["rho_eig_sup"]) / nnw["rho_eig_sup"]
    add("NN measured rate vs ρ_eig_sup (honest relation)",
        "%.4f vs %.4f (Δ%.1f%%)" % (nnw["rate_measured"], nnw["rho_eig_sup"],
                                    100 * gap_rel), "within ±25%", gap_rel <= 0.25)
    add("NN σ_max bound conservative but valid",
        "rate=%.3f ≤ ρ_sup=%.3f (gap %.1f×)"
        % (nnw["rate_measured"], nnw["rho_sup"], nnw["normal_gap"]),
        "rate ≤ ρ_sup", nnw["rate_measured"] <= nnw["rho_sup"])
    add("NN strong diverges (c=1.1)",
        "×%.2f per-iter diff growth" % nns["diverge_ratio"], "> 1",
        nns["diverge_ratio"] > 1.0)

    iv = sp["ivp"]
    add("IVP window rate ≤ ρ_sup (Volterra faster, bound holds)",
        "%.4f vs ρ_sup=%.4f" % (iv["rate"], nw["rho_sup"]),
        "≤ 1.05·ρ_sup", iv["rate"] <= 1.05 * nw["rho_sup"])

    r = CASES["replay"]
    plat_ratio = r["plateau_obs"] / r["plateau_pred"]
    add("replay plateau vs bound", "%.3e vs %.3e (×%.2f)"
        % (r["plateau_obs"], r["plateau_pred"], plat_ratio),
        "within ×3", 1.0 / 3.0 <= plat_ratio <= 3.0)
    knee_rel = (abs(r["knee"] - r["k_pred"]) / max(r["knee"], 1)
                if r["knee"] is not None else 1.0)
    add("k_max formula vs knee", "pred %d vs obs %s" % (r["k_pred"], r["knee"]),
        "within ±20%", knee_rel <= 0.20)
    add("plateau stationary (more iters don't help)",
        "late rate %.3f" % r["late_rate"], "≈1 (>0.9)",
        0.9 < r["late_rate"] < 1.1)

    z = CASES["zoh"]
    add("zoh_edge latency = T_s/2", "%.3e s" % z["latency"],
        "%.3e s" % (z["T_s"] / 2), abs(z["latency"] - z["T_s"] / 2) < 1e-15)
    same = all(z["closure_zoh"][key] == z["closure_manual"][key]
               for key in ("latency", "phase_delay_omega_c", "verdict",
                           "pass_gain", "pass_phase", "pass_noise",
                           "pass_absolute", "noise_rms"))
    add("zoh closure ≡ manual PureDelay closure",
        "verdict=%s, abs=%s" % (z["closure_zoh"]["verdict"],
                                z["closure_zoh"]["pass_absolute"]),
        "identical dicts", same)
    add("zoh passes feedback absolute-delay budget",
        "τ=%.2e ≤ τ_max=%.2e" % (z["closure_zoh"]["phase_delay_omega_c"],
                                 z["closure_zoh"]["tau_max_abs"]),
        "pass_absolute", z["closure_zoh"]["pass_absolute"])
    return checks


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear()
    CASES.clear()

    hr("Phase 4.3 验收：多速率/分裂合法性（WR + replay + ZOH）")

    case_spectral()
    case_replay()
    case_zoh()

    hr("A. 谱半径：ρ_sup（σ_max small-gain）vs ρ_eig_sup（实际逐频率）")
    sp = CASES["spectral"]
    rows = []
    for tag in ("N_weak", "N_strong", "NN_weak", "NN_strong", "replay"):
        r = sp[tag]
        rows.append(("%s (%s, c=%.2f)" % (tag, r["sys"], r["c"]),
                     "%.4f" % r["rho_sup"], "%.4f" % r["rho_eig_sup"],
                     "%.1f×" % r["normal_gap"],
                     "%.2f rad/s" % r["omega_peak"],
                     "%.4f" % r["sigma_dc"],
                     "interior" if r["interior_peak"] else "DC/edge"))
    table(["split", "ρ_sup", "ρ_eig_sup", "gap", "peak ω", "σ_dc", "峰位"], rows)
    nw = sp["N_weak"]
    emit("  σ_max(G_WR) 沿 ω %s；收敛判据是 ρ_sup<1，不是 σ_dc。"
         % ("单调递减" if nw["monotone_decreasing"] else "有 interior peak"))
    emit("  N（正规）：实测周期 WR 衰减率 %.4f/iter vs ρ_sup %.4f —— 合同公式"
         "在 G_WR 正规时严格成立。" % (nw["rate_measured"], nw["rho_sup"]))
    nnw = sp["NN_weak"]
    emit("  NN（非正规）：实测 %.4f vs ρ_eig_sup %.4f vs σ_max %.4f ——"
         "实际率由特征值谱半径决定，σ_max 证书保守 %.1f×（方向：不漏报发散，"
         "可能错杀可收敛分裂）。"
         % (nnw["rate_measured"], nnw["rho_eig_sup"], nnw["rho_sup"],
            nnw["normal_gap"]))
    emit("  负控制：N_strong ρ_sup=%.2f 发散 ×%.2f；NN_strong(c=1.1) "
         "ρ_sup=%.2f、ρ_eig=%.2f 发散 ×%.2f —— 合同对两条都说 NO。"
         % (sp["N_strong"]["rho_sup"], sp["N_strong"]["diverge_ratio"],
            sp["NN_strong"]["rho_sup"], sp["NN_strong"]["rho_eig_sup"],
            sp["NN_strong"]["diverge_ratio"]))
    iv = sp["ivp"]
    emit("  IVP 窗口诊断（零初值，因果 Volterra）：实测率 %.4f ≤ ρ_sup %.4f，"
         "误差 %g→%g 单调下降 —— ρ_sup 界保守方向成立。"
         % (iv["rate"], nw["rho_sup"], iv["err0"], iv["err10"]))

    hr("B. replay 底噪：平台与 k_max")
    r = CASES["replay"]
    table(["quantity", "value"],
          [("ρ_sup", "%.4f" % r["rho"]),
           ("ε_replay", "%.3e" % r["eps"]),
           ("e0", "%.3e" % r["e0"]),
           ("R∞ tight (= ρ_sup)", "%.4f" % r["R_tight"]),
           ("R∞ loose (resolvent)", "%.4f" % r["R_loose"]),
           ("k_max (公式)", str(r["k_pred"])),
           ("knee (观测)", str(r["knee"])),
           ("plateau 预测 (tight)", "%.3e" % r["plateau_pred"]),
           ("plateau 预测 (loose)", "%.3e" % r["plateau_loose"]),
           ("plateau 实测", "%.3e" % r["plateau_obs"]),
           ("late 迭代速率 (≈1 ⇒ 平台)", "%.4f" % r["late_rate"])])
    emit("  超过 k_max 后误差进 ε_replay 平台——更多迭代不再降低误差。")

    hr("C. ZOH → route：同一套判定机器，不建新机制")
    z = CASES["zoh"]
    cz, cm = z["closure_zoh"], z["closure_manual"]
    table(["field", "zoh_edge", "manual PureDelay"],
          [("latency", "%.3e" % cz["latency"], "%.3e" % cm["latency"]),
           ("phase_delay@ωc", "%.3e" % cz["phase_delay_omega_c"],
            "%.3e" % cm["phase_delay_omega_c"]),
           ("noise_rms", "%.1e" % cz["noise_rms"], "%.1e" % cm["noise_rms"]),
           ("verdict", cz["verdict"], cm["verdict"]),
           ("pass_absolute", cz["pass_absolute"], cm["pass_absolute"])])
    emit("  H_ZOH=(1−e^{−sTs})/(sTs)≈e^{−sTs/2}：跨岛 S&H = PureDelay(T_s/2)；"
         "混叠是单独的 B_p noise 端口，不进 route 轴。")

    hr("D. 验收")
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
    with open(os.path.join(OUT, "multirate.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(os.path.join(OUT, "multirate_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/multirate.json")
    emit("  report → results/multirate_report.txt")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
