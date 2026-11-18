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
    C  ZOH→route       zoh_edge(T_s) 返回真 ZOH 块 H(s)=(1−e^{−sTs})/(sTs)
                       = e^{−sTs/2}·sinc(sTs/2)：相位=PureDelay(Ts/2)，幅度带
                       sinc droop（Nyquist 处 2/π≈0.637，−3.92 dB）。droop 进
                       gain 轴；混叠与重构滤波是独立预算。仅当 droop<ε
                       (ωT_s≲0.155) 才允许显式记录地退化成 PureDelay(T_s/2)。

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

from twin.multirate import (DROOP_EPS_DEFAULT, droop_omega_ts, k_max,
                            wr_error_bound, wr_iteration_matrix,
                            wr_spectral_radius, zoh_contract, zoh_edge)
from twin.route import PureDelay, RouteEdge, RouteSpec, ZOH, route_closure

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
    w_nyq = np.pi / T_s
    # 真 ZOH 边缘（不退化）vs 手工 PureDelay：相位同、幅度差 sinc
    ez = RouteEdge((ZOH(T_s),), "zoh_island_swap")
    em = RouteEdge((PureDelay(T_s / 2.0),), "zoh_island_swap")
    spec = RouteSpec(omega_max=2 * np.pi * 10e3, omega_c=2 * np.pi * 10e3,
                     eps_phase_deg=10.0, feedback_critical=True,
                     phi_reserve_deg=5.0)
    rz = route_closure(ez, spec)
    rm = route_closure(em, spec)
    out = dict(T_s=T_s, w_nyq=w_nyq, latency=ZOH(T_s).latency,
               closure_zoh=rz, closure_manual=rm)

    # 幅度可查：Nyquist 处 ZOH=2/π vs PureDelay=1（低带宽 spec 下才同判）
    out["mag_zoh_nyq"] = float(abs(ZOH(T_s).H(1j * w_nyq)))
    out["mag_pd_nyq"] = float(abs(PureDelay(T_s / 2.0).H(1j * w_nyq)))
    out["phase_zoh_nyq"] = float(np.angle(ZOH(T_s).H(1j * w_nyq)))
    out["phase_pd_nyq"] = float(np.angle(PureDelay(T_s / 2.0).H(1j * w_nyq)))
    out["dc_zoh"] = float(np.real(ZOH(T_s).H(0.0)))

    # 高带宽负载：信号 ω 逼近 Nyquist ⇒ droop 惩罚可见（gain 判据翻 NO）
    spec_hi = RouteSpec(omega_max=w_nyq, omega_c=w_nyq, eps_gain=0.02,
                        eps_phase_deg=10.0, feedback_critical=True,
                        phi_reserve_deg=5.0)
    rz_hi = route_closure(ez, spec_hi)
    rm_hi = route_closure(em, spec_hi)
    out["closure_zoh_hi"] = rz_hi
    out["closure_pd_hi"] = rm_hi

    # 低带宽负载：ωT_s≪1 ⇒ droop 可忽略，允许退化（记录在案）
    w_lo = 0.01 * w_nyq                       # ωT_s≈0.031 ⇒ droop≈8e-6
    ez_lo = RouteEdge((ZOH(T_s),), "zoh_lo")
    spec_lo = RouteSpec(omega_max=w_lo, omega_c=w_lo, eps_gain=0.02,
                        eps_phase_deg=10.0, feedback_critical=True,
                        phi_reserve_deg=5.0)
    out["closure_zoh_lo"] = route_closure(ez_lo, spec_lo)
    out["droop_lo"] = 1.0 - float(abs(ZOH(T_s).H(1j * w_lo)))
    dec_lo = zoh_edge(T_s, omega_max=w_lo)
    out["degrade_lo"] = dict(degraded=dec_lo["degraded"],
                             block=str(type(dec_lo["block"]).__name__),
                             droop=dec_lo["droop"])
    # 高带宽下必须**不**退化（拒绝静默延迟近似）
    dec_hi = zoh_edge(T_s, omega_max=w_nyq)
    out["degrade_hi"] = dict(degraded=dec_hi["degraded"],
                             block=str(type(dec_hi["block"]).__name__),
                             droop=dec_hi["droop"])
    # 退化阈值解析值 vs 直接求 droop(wTs)
    out["wts_threshold"] = droop_omega_ts(DROOP_EPS_DEFAULT)
    out["droop_at_threshold"] = 1.0 - float(
        abs(ZOH(1.0).H(1j * out["wts_threshold"])))

    # 三项独立预算：droop / aliasing / reconstruction
    out["contract_hi"] = zoh_contract(T_s, 0.95 * w_nyq)
    out["contract_mid"] = zoh_contract(T_s, 0.7 * w_nyq)
    out["contract_lo"] = zoh_contract(T_s, 0.1 * w_nyq)
    out["contract_recon"] = zoh_contract(T_s, 0.95 * w_nyq,
                                         recon_filter_db=-40.0)

    CASES["zoh"] = out


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
    cz_, cm_ = z["closure_zoh"], z["closure_manual"]
    same = all(
        (abs(cz_[k] - cm_[k]) <= 1e-8 * max(abs(cm_[k]), 1e-300))
        if isinstance(cm_[k], float) else cz_[k] == cm_[k]
        for k in ("latency", "phase_delay_omega_c",
                  "max_phase_err_deg", "noise_rms", "verdict",
                  "pass_gain", "pass_phase", "pass_noise", "pass_absolute"))
    add("zoh closure ≡ manual PureDelay closure (low-ω spec, tol 1e-8)",
        "verdict=%s, gain_err=%.2e vs %.2e"
        % (cz_["verdict"], cz_["max_gain_err"], cm_["max_gain_err"]),
        "identical within numerical tol", same)
    add("zoh passes feedback absolute-delay budget",
        "τ=%.2e ≤ τ_max=%.2e" % (z["closure_zoh"]["phase_delay_omega_c"],
                                 z["closure_zoh"]["tau_max_abs"]),
        "pass_absolute", z["closure_zoh"]["pass_absolute"])

    # --- ZOH sinc droop（新增：幅度项） ---
    mag_2pi = 2.0 / np.pi
    add("ZOH magnitude @ Nyquist = 2/π (droop)",
        "%.6f" % z["mag_zoh_nyq"], "%.6f" % mag_2pi,
        abs(z["mag_zoh_nyq"] - mag_2pi) < 1e-9)
    add("PureDelay magnitude @ Nyquist = 1.0 (the wrong old model)",
        "%.6f" % z["mag_pd_nyq"], "1.000000",
        abs(z["mag_pd_nyq"] - 1.0) < 1e-12)
    add("ZOH phase @ Nyquist = −π/2 (≡ PureDelay phase)",
        "%.6f vs %.6f rad" % (z["phase_zoh_nyq"], z["phase_pd_nyq"]),
        "both = −π/2", abs(z["phase_zoh_nyq"] + np.pi / 2) < 1e-9
        and abs(z["phase_pd_nyq"] + np.pi / 2) < 1e-9)
    add("ZOH dc gain = 1", "%.12f" % z["dc_zoh"], "1.0",
        abs(z["dc_zoh"] - 1.0) < 1e-12)
    add("droop negligible well below Nyquist (ωT_s≈0.031, ≈x²/6)",
        "%.2e" % z["droop_lo"], "< 1e-4", z["droop_lo"] < 1e-4)

    # --- route 判据看到 droop 惩罚（高带宽）与不惩罚（低带宽） ---
    rz_hi, rm_hi = z["closure_zoh_hi"], z["closure_pd_hi"]
    add("high-ω workload: ZOH gain verdict NO (droop penalty)",
        "gain_err=%.3f (≈1−2/π)" % rz_hi["max_gain_err"], "pass_gain=NO",
        not rz_hi["pass_gain"])
    add("high-ω workload: PureDelay wrongly passes gain (old bug)",
        "gain_err=%.2e" % rm_hi["max_gain_err"], "pass_gain=YES",
        rm_hi["pass_gain"])
    add("high-ω workload: ZOH phase err ≡ PureDelay phase err (droop doesn't touch phase)",
        "%.6f° vs %.6f°" % (rz_hi["max_phase_err_deg"],
                            rm_hi["max_phase_err_deg"]),
        "equal (τ_ideal=0 ⇒ both measure absolute phase)",
        abs(rz_hi["max_phase_err_deg"] - rm_hi["max_phase_err_deg"]) < 1e-6)
    add("low-ω workload: ZOH passes full closure (droop < ε)",
        "gain_err=%.2e droop=%.2e"
        % (z["closure_zoh_lo"]["max_gain_err"], z["droop_lo"]),
        "pass_gain=YES", z["closure_zoh_lo"]["pass_gain"])

    # --- 显式退化（记录，不静默） ---
    dlo, dhi = z["degrade_lo"], z["degrade_hi"]
    add("auto-degrade at low ω → PureDelay(T_s/2), recorded",
        "%s (droop %.2e < ε)" % (dlo["block"], dlo["droop"]),
        "degraded=True, block=PureDelay",
        dlo["degraded"] and dlo["block"] == "PureDelay")
    add("no degradation at Nyquist-scale ω (ZOH kept)",
        "%s (droop %.3f)" % (dhi["block"], dhi["droop"]),
        "degraded=False, block=ZOH",
        (not dhi["degraded"]) and dhi["block"] == "ZOH")
    thr, dthr = z["wts_threshold"], z["droop_at_threshold"]
    add("degradation threshold ωT_s* from droop=ε (ε=0.01 ⇒ √(24ε)≈0.49)",
        "ωT_s*=%.5f, droop there=%.5f" % (thr, dthr),
        "droop(ωT_s*) = ε within 5%",
        abs(dthr - DROOP_EPS_DEFAULT) < 0.05 * DROOP_EPS_DEFAULT
        and abs(thr - np.sqrt(24 * DROOP_EPS_DEFAULT)) < 0.05)

    # --- 三项独立预算 ---
    chi, cmid, clo = z["contract_hi"], z["contract_mid"], z["contract_lo"]
    add("aliasing flagged near Nyquist (violation)",
        "ω_sig/ω_N=%.2f → %s" % (chi["bandwidth_fraction"], chi["aliasing"]),
        "violation", chi["aliasing"] == "violation")
    add("aliasing warning at 0.7 Nyquist",
        "ω_sig/ω_N=%.2f → %s" % (cmid["bandwidth_fraction"], cmid["aliasing"]),
        "warning", cmid["aliasing"] == "warning")
    add("aliasing clean well below Nyquist",
        "ω_sig/ω_N=%.2f → %s" % (clo["bandwidth_fraction"], clo["aliasing"]),
        "none", clo["aliasing"] == "none")
    cr = z["contract_recon"]
    # 镜像位置 ω_s−ω_sig = 1.05·ω_N 处 sinc 自身的衰减（独立重算）：
    x_img = 1.05 * np.pi / 2.0
    atten_expected = 20.0 * np.log10(abs(np.sin(x_img) / x_img))
    add("reconstruction filter is a separate line (not folded into droop)",
        "declared=%s budget=%.0f dB | sinc-only image atten=%.2f dB (exp %.2f)"
        % (cr["reconstruction"]["filter_declared"],
           cr["reconstruction"]["recon_budget_db"],
           cr["reconstruction"]["sinc_image_atten_db"], atten_expected),
        "separate keys; sinc-only atten matches |sinc(1.05π/2)|",
        cr["reconstruction"]["filter_declared"]
        and cr["reconstruction"]["recon_budget_db"] == -40.0
        and abs(cr["reconstruction"]["sinc_image_atten_db"]
                - atten_expected) < 0.05)
    add("droop term stays in route gain axis (not in alias/recon)",
        "droop=%.4f, droop_ok=%s" % (chi["droop"], chi["droop_ok"]),
        "droop ≈ 1−|sinc(0.475π)|=0.323, droop_ok=False",
        abs(chi["droop"] - 0.323) < 0.01 and not chi["droop_ok"])
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

    hr("C. ZOH → route：真 ZOH（sinc droop）+ 三项独立预算")
    z = CASES["zoh"]
    cz, cm = z["closure_zoh"], z["closure_manual"]
    rz_hi, rm_hi = z["closure_zoh_hi"], z["closure_pd_hi"]
    table(["field", "zoh_edge", "manual PureDelay"],
          [("latency", "%.3e" % cz["latency"], "%.3e" % cm["latency"]),
           ("phase_delay@ωc", "%.3e" % cz["phase_delay_omega_c"],
            "%.3e" % cm["phase_delay_omega_c"]),
           ("noise_rms", "%.1e" % cz["noise_rms"], "%.1e" % cm["noise_rms"]),
           ("verdict", cz["verdict"], cm["verdict"]),
           ("pass_absolute", cz["pass_absolute"], cm["pass_absolute"])])
    table(["magnitude @ Nyquist", "ZOH (true)", "PureDelay (old)"],
          [("|H(jω_N)|", "%.6f (2/π, −3.92 dB)" % z["mag_zoh_nyq"],
            "%.6f (0 dB, wrong)" % z["mag_pd_nyq"]),
           ("phase", "%.6f rad" % z["phase_zoh_nyq"],
            "%.6f rad" % z["phase_pd_nyq"])])
    table(["high-ω workload (ω_max=ω_N)", "ZOH", "PureDelay"],
          [("max_gain_err", "%.4f" % rz_hi["max_gain_err"],
            "%.2e" % rm_hi["max_gain_err"]),
           ("pass_gain", rz_hi["pass_gain"], rm_hi["pass_gain"]),
           ("max_phase_err_deg", "%.4f" % rz_hi["max_phase_err_deg"],
            "%.4f" % rm_hi["max_phase_err_deg"])])
    dlo, dhi = z["degrade_lo"], z["degrade_hi"]
    table(["degrade decision", "block", "droop@ω_max", "degraded"],
          [("low ω (0.01·ω_N)", dlo["block"], "%.2e" % dlo["droop"],
            str(dlo["degraded"])),
           ("ω_max=ω_N", dhi["block"], "%.4f" % dhi["droop"],
            str(dhi["degraded"]))])
    emit("  退化阈值：droop=ε(%.2f) ⇒ ωT_s*=%.5f（小 u 展开 droop≈(ωT_s)²/24，"
         "√(24ε)=%.5f；此处 droop(ωT_s*)=%.5f）。"
         % (DROOP_EPS_DEFAULT, z["wts_threshold"],
            np.sqrt(24 * DROOP_EPS_DEFAULT), z["droop_at_threshold"]))
    for tag in ("contract_hi", "contract_mid", "contract_lo"):
        c = z[tag]
        emit("  预算[%s] ω_sig/ω_N=%.2f：droop=%.4f (route gain 轴) | "
             "aliasing=%s (noise 轴) | 重构滤波 declared=%s"
             % (tag, c["bandwidth_fraction"], c["droop"], c["aliasing"],
                c["reconstruction"]["filter_declared"]))
    emit("  H_ZOH=(1−e^{−sTs})/(sTs)=e^{−sTs/2}·sinc(sTs/2)：相位=PureDelay(T_s/2)，"
         "幅度带 sinc droop（Nyquist 处 2/π≈0.637）。低 ωT_s 才允许退化成"
         " PureDelay 且记录 degraded=True；混叠与重构滤波是独立预算，"
         "不折进 droop。")

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
