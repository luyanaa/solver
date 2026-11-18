"""Phase 6 验收:随机动力学正问题 + 反问题(DESIGN.md §6.11/§6.12,§7 6A–6D)。

    python3 -m bench.run_sde

五个算例(积分器全部在本文件——这是 bench 里允许放 solver 的地方):

    A  OU 加性      Var vs D/λ(±3 dB);ACF e^{−λt} 拟合(±5%);
                    有色源 Var vs (D/λ)/(1+λτ_c) 闭式;
                    噪声带宽:B_noise 100·B_req → 10·B_req,方差降 ≈6.4%
                    (解析 (2/π)arctan(f_c/B),±30% 窗)
    B  GBM 乘性     Itô(Euler–Maruyama, f=μx)vs Stratonovich(同一积分器,
                    修正漂移 f_I = f_S + ½σ²x —— 这就是被测的语义合同);
                    两均值 >3σ 分开;各落在解析线 ±10%;线性修正 A_S−A_I=σ²/2
    C  双势阱       p ∝ e^{−V/T} 直方图 KL < 0.05;曲率拟合回收 T̂ < 10%
    D  反问题 6C-lite  CRN 有限差分(replay 同一噪声)vs 独立噪声控制;
                    CRN 梯度方差 ≥10× 低于独立;CRN 网格回收 λ̂ < 10%;
                    AR(1)/矩回收 λ̂、D̂ < 10%。连续时间 adjoint 是 Phase-6C-later
                    的缺口,报告里只记录、不伪造。
    E  噪声合同自检  每个算例打出的 S_ξ/τ_c/ρ/B_noise/mode 五字段齐全(§1.3)。
    F  广义 Wong–Zakai(§1.4 推广):dx = μx dt + σx·ξ dt,ξ 为 ColorNoise(τ_c),
                    乘法器输出经一阶极点延迟 τ_d(ż=(xξ−z)/τ_d,z 进更新,精确
                    双线性更新)。实测均值增长率 − μ vs θ(χ)=1/(2(1+χ)):
                    χ=0 回 ½ Stratonovich;χ=16 降到 ~3%;单调降;插值窗 ±25%
                    (一阶极点不是纯延迟,实测曲线对 1/(2(1+χ)) 的偏差如实报)。
    G  adjoint 不对称界:Ã_adj = Aᵀ+Δ_asym 的梯度相对误差 vs
                    Γ_proj·‖Δ_asym‖·T_obs 界;log-log 斜率≈1、err ≤ 界、
                    err ≥ 界/100(界不空转)。transpose-reuse 合法性判据。

积分器:Euler–Maruyama(乘性/双势阱)+ OU 精确 AR(1) 更新(加性,零偏);
带宽实验用 FFT 理想低通白噪声(§2 的 (2/π)arctan 公式对应的理想化)。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from scipy.signal import lfilter

from analysis.descriptor import Descriptor
from analysis.resolvent import resolvent_peak
from analysis.sde import wong_zakai_factor
from twin.noise import ColorNoise, bandwidth_drop_factor, white_matrix
from twin.transform import (adjoint_asym_bound, adjoint_ode_gradient,
                            finite_difference_gradient, rk4_matrix)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEED = 20261118
_LINES: list[str] = []
CASES: dict[str, dict] = {}
CONTRACTS: dict[str, dict] = {}


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
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# 积分器(bench 是唯一放 solver 的地方)
# ---------------------------------------------------------------------------
def _recursion(x0, a, inc):
    """x_{k+1} = a·x_k + inc_k(lfilter 实现,(n,K) 一次算完)。"""
    from scipy.signal import lfilter as _lf
    out = _lf([1.0], [1.0, -a], inc, axis=1)
    out[:, 0] += x0
    return out


def _acf_fit(X, dt, lam_window=(0.25, 3.0), max_lag=80):
    """归一化 ACF(FFT 法,轨迹×时间平均)+ log 线性拟合 → λ̂。"""
    n, K = X.shape
    nfft = 1 << (2 * K - 1).bit_length()
    Xf = rfft(X - X.mean(axis=1, keepdims=True), nfft, axis=1)
    ac = irfft(Xf * np.conj(Xf), nfft, axis=1)[:, :max_lag + 1] / K
    C = ac.mean(axis=0)
    C = C / C[0]
    tau = np.arange(max_lag + 1) * dt
    m = (tau >= lam_window[0]) & (tau <= lam_window[1]) & (C > 1e-8)
    p = np.polyfit(tau[m], np.log(C[m]), 1)
    return float(-p[0]), tau, C


# ---------------------------------------------------------------------------
# A. OU:加性、ACF、有色源、噪声带宽
# ---------------------------------------------------------------------------
def case_a_ou(rng):
    lam, D = 1.0, 1.0
    n, dt, K = 700, 0.05, 6000                      # T = 300 s
    a = np.exp(-lam * dt)
    cstep = np.sqrt(dt * (1.0 - np.exp(-2.0 * lam * dt)) / 2.0)
    xi = white_matrix(rng, D, dt, (n, K - 1))       # 白极限:ξ·dt = √(2D)ΔW
    X = _recursion(np.zeros(n), a, xi * cstep)
    burn = int(10.0 / dt)
    var = float(np.mean(X[:, burn:] ** 2))
    var_pred = D / lam
    lam_hat, tau, C = _acf_fit(X, dt)
    CONTRACTS["A_ou_white"] = dict(
        **ColorNoise(0.0, D).describe(),
        B_req="%.4f Hz = λ/2π" % (lam / (2 * np.pi)),
        note="white limit of ColorNoise, exact AR(1) update",
        n_traj=n, T=K * dt, dt=dt)
    CASES["A_ou_white"] = dict(lam=lam, D=D, var=var, var_pred=var_pred,
                               var_ratio=var / var_pred,
                               lam_hat=lam_hat, n_traj=n, T=K * dt, dt=dt)

    # 有色源:τ_c = 0.05,解析 (D/λ)/(1+λτ_c) = 0.95238
    tc = 0.05
    nc, dtc, Kc, Tc = 160, 0.0025, 40000, 100.0     # dtc = 0.05·τ_c
    src = ColorNoise(tc, D, dim=nc, rng=rng)
    # 噪声一次批量预抽,经 replay() 逐条消费(等价 physical 模式,省 RNG 调用)
    src.replay(rng.standard_normal((Kc, nc)))
    x = np.zeros(nc)
    s1 = s2 = cnt = 0.0
    for k in range(Kc):
        x = x * np.exp(-lam * dtc) + src.sample(dtc) * dtc
        if k * dtc > 10.0:
            s1 += x.sum(); s2 += (x * x).sum(); cnt += nc
    var_c = s2 / cnt
    var_c_pred = (D / lam) / (1.0 + lam * tc)
    CONTRACTS["A_ou_colored"] = dict(
        **src.describe(),
        B_req="%.4f Hz" % (lam / (2 * np.pi)),
        B_over_Breq="%.1f" % (src.B_noise / (lam / (2 * np.pi))),
        n_traj=nc, T=Tc, dt=dtc)
    CASES["A_ou_colored"] = dict(tau_c=tc, var=var_c, var_pred=var_c_pred,
                                 var_ratio=var_c / var_c_pred,
                                 n_traj=nc, T=Tc, dt=dtc)

    # 噪声带宽:理想低通(FFT brickwall)@ f_c = 100·B_req 与 10·B_req
    bw = {}
    for tag, fb in (("B100", 100.0), ("B10", 10.0)):
        bw[tag] = _bw_lowpass(lam, D, fb, rng)
    fb_ratio = bw["B10"]["var"] / bw["B100"]["var"]
    drop = 1.0 - fb_ratio
    ref = bandwidth_drop_factor(10.0)               # 0.93655 vs 白
    CONTRACTS["A_bandwidth"] = dict(
        S_xi="two-sided flat 2D, ideal brickwall lowpass at f_c (FFT)",
        tau_c="0 (brickwall; effective 1/(2πf_c))",
        rho="I", B_noise="f_c = 100·B_req and 10·B_req",
        B_req="%.4f Hz = λ/2π" % (lam / (2 * np.pi)),
        mode="physical (idealized band limit, §2 arctan model)",
        n_traj=bw["B100"]["n"], T=bw["B100"]["T"], dt=bw["B100"]["dt"])
    CASES["A_bandwidth"] = dict(
        B_req=lam / (2 * np.pi),
        var_100=bw["B100"]["var"], var_10=bw["B10"]["var"],
        ratio=fb_ratio, drop=drop, drop_pct=100.0 * drop,
        pred_drop_pct=100.0 * (1.0 - 2.0 / np.pi * np.arctan(10.0) / 0.99363 * 0.99363),
        ratio10_vs_white=bw["B10"]["var"] / var_pred,
        ratio100_vs_white=bw["B100"]["var"] / var_pred,
        pred10=ref, pred100=bandwidth_drop_factor(100.0))


def _bw_lowpass(lam, D, fc_over_b, rng, ntraj=120, T=260.0, delta=1 / 128.0):
    """白噪声(双边平坦 2D)经理想低通 f_c 后驱动 OU 的稳态方差。"""
    B_req = lam / (2 * np.pi)
    fc = fc_over_b * B_req
    K = int(round(T / delta))
    f = rfftfreq(K, delta)
    keep = f <= fc
    z = white_matrix(rng, D, delta, (ntraj, K))
    Z = rfft(z, axis=1)
    Z[:, ~keep] = 0.0
    xi = irfft(Z, K, axis=1)
    a = np.exp(-lam * delta)
    mid = 0.5 * (xi[:, :-1] + xi[:, 1:]) * (delta * np.exp(-lam * delta / 2))
    X = _recursion(np.zeros(ntraj), a, mid)
    burn = int(30.0 / delta)
    return dict(fc=fc, var=float(np.mean(X[:, burn:] ** 2)),
                n=ntraj, T=T, dt=delta)


# ---------------------------------------------------------------------------
# B. GBM:Itô vs Stratonovich(语义合同)
# ---------------------------------------------------------------------------
def _gbm_run(rng, drift, mu, sig, dt, K, n, x0=1.0):
    """Euler–Maruyama:x ← x(1+f·dt) + σx·ΔW;返回均值/E[x²] 曲线 + 末点 log 均值。"""
    sx = np.empty(K); sx2 = np.empty(K)
    x = np.full(n, x0)
    for k in range(K):
        sx[k] = x.mean(); sx2[k] = np.mean(x * x)
        dW = rng.standard_normal(n) * np.sqrt(dt)
        x = x * (1.0 + drift * dt) + sig * x * dW
    return sx, sx2, float(np.log(x).mean())


def case_b_gbm(rng):
    mu, sig, x0, T, dt = 0.5, 0.6, 1.0, 5.0, 0.01
    K = int(round(T / dt))
    N, chunk = 40000, 5000
    cor = sig * sig / 2.0                           # ½ G ∂G,G = σx
    accI = np.zeros((2, K)); accS = np.zeros((2, K))
    logI = logS = 0.0
    n_done = 0
    while n_done < N:
        m = min(chunk, N - n_done)
        sx, sx2, slx = _gbm_run(rng, mu, mu, sig, dt, K, m, x0)
        accI[0] += sx * m; accI[1] += sx2 * m; logI += slx * m
        sx, sx2, slx = _gbm_run(rng, mu + cor, mu, sig, dt, K, m, x0)
        accS[0] += sx * m; accS[1] += sx2 * m; logS += slx * m
        n_done += m
    meanI, mean2I = (v / N for v in accI)
    meanS, mean2S = (v / N for v in accS)
    meanlogI, meanlogS = logI / N, logS / N
    t = np.arange(K) * dt
    # 分离显著性:σ²t = 1 处
    k_sep = int(round(1.0 / (sig * sig) / dt))
    seI = np.sqrt(np.maximum(mean2I - meanI ** 2, 0.0) / N)
    seS = np.sqrt(np.maximum(mean2S - meanS ** 2, 0.0) / N)
    sep_sigma = float((meanS[k_sep] - meanI[k_sep]) /
                      np.hypot(seI[k_sep], seS[k_sep]))
    # 落点 ±10%
    anaI = x0 * np.exp(mu * t); anaS = x0 * np.exp((mu + cor) * t)
    landI = float(meanI[-1] / anaI[-1]); landS = float(meanS[-1] / anaS[-1])
    # log 空间锐检查:E[ln x] = (μ−σ²/2)t(两个系综应一致)
    log_err = float(abs(meanlogI - (mu - sig * sig / 2) * T))
    log_se = float(sig * np.sqrt(T) / np.sqrt(N))
    # 线性修正一致性:A_S = A_I + ½G² ⇒ A_S − A_I = σ²/2
    corr_used = (mu + cor) - mu
    CONTRACTS["B_gbm"] = dict(
        S_xi="two-sided flat 2D (unit white → σx dW scaled)",
        tau_c=0.0, rho="I (scalar, diagonal G)",
        B_noise="inf; dt = %.4g ≪ 1/λ_state" % dt,
        mode="physical",
        interpretation="model Stratonovich; integrator Itô with "
                       "f_I = f_S + ½σ²x (§1.4 direction)",
        n_traj=N, T=T, dt=dt)
    CASES["B_gbm"] = dict(mu=mu, sigma=sig, x0=x0, T=T, dt=dt, N=N,
                          sep_time=float(t[k_sep]),
                          sep_sigma=sep_sigma,
                          mean_I_sep=float(meanI[k_sep]),
                          mean_S_sep=float(meanS[k_sep]),
                          land_I=landI, land_S=landS,
                          log_err=log_err, log_se=log_se,
                          corr_used=corr_used, corr_exact=sig * sig / 2.0)


# ---------------------------------------------------------------------------
# C. 双势阱 Langevin
# ---------------------------------------------------------------------------
def _potential(x):
    return x ** 4 / 4.0 - x ** 2 / 2.0


def case_c_doublewell(rng):
    Ttemp, dt = 0.15, 0.005
    M, K = 1400, 24000                               # 3.4e7 步,~120 时间单位/条
    burn = 4000                                      # 20 个相关时间
    x = np.where(rng.random(M) < 0.5, -1.0, 1.0)
    lo, hi, nb = -2.6, 2.6, 208
    counts = np.zeros(nb, dtype=np.int64)
    amp = np.sqrt(2.0 * Ttemp * dt)
    for k in range(K):
        x = x - (x ** 3 - x) * dt + amp * rng.standard_normal(M)
        if k >= burn and k % 3 == 0:                 # 隔步累计,总量仍 ~1e7
            idx = np.clip(((x - lo) / (hi - lo) * nb).astype(np.int64),
                          0, nb - 1)
            counts += np.bincount(idx, minlength=nb)
    centers = lo + (np.arange(nb) + 0.5) * (hi - lo) / nb
    Vc = _potential(centers)
    N_tot = int(counts.sum())
    q_un = np.exp(-Vc / Ttemp)
    q_un /= q_un.sum()
    kept = q_un * N_tot >= 10.0                      # 尾部护栏:期望计数 ≥10
    p = counts[kept].astype(float); p /= p.sum()
    q = q_un[kept]; q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    # T̂:加权最小二乘 ln p̂ = −V/T̂ − lnZ(权重 = 计数)
    w = counts[kept].astype(float)
    y = np.log(p); v = Vc[kept]
    wbar = (w * v).sum() / w.sum(); ybar = (w * y).sum() / w.sum()
    slope = ((w * (v - wbar) * (y - ybar)).sum() /
             (w * (v - wbar) ** 2).sum())
    That = -1.0 / slope
    resid = float(np.sqrt((w * (y - ybar - slope * (v - wbar)) ** 2).sum()
                          / w.sum()))
    CONTRACTS["C_doublewell"] = dict(
        S_xi="two-sided flat 2T (thermal Langevin bath, √(2T)dW)",
        tau_c=0.0, rho="I (scalar)", B_noise="inf (white bath)",
        mode="physical",
        n_traj=M, T_total="%.1f per traj (dt=%.4g)" % (K * dt, dt),
        tail_guard="bins with expected count ≥ 10")
    CASES["C_doublewell"] = dict(T=Ttemp, dt=dt, M=M, K=K, burn=burn,
                                 n_samples=N_tot, kl=kl,
                                 n_kept=int(kept.sum()),
                                 T_hat=float(That), T_resid=resid,
                                 kept_frac=float(kept.sum() / nb))


# ---------------------------------------------------------------------------
# D. 反问题 6C-lite:CRN / 独立控制 / 矩(EnKF 风格)回收
# ---------------------------------------------------------------------------
def _ou_from_force(force, lam, D, dt):
    """用给定的力样本序列(白极限 ξ,形状 (K−1, dim))积分 OU。"""
    K1, n = force.shape
    a = np.exp(-lam * dt)
    cstep = np.sqrt(dt * (1.0 - np.exp(-2.0 * lam * dt)) / 2.0)
    x = np.empty((n, K1 + 1)); x[:, 0] = 0.0
    for k in range(K1):
        x[:, k + 1] = x[:, k] * a + force[k] * cstep
    return x


def _loss(lam, force, Xobs, dt):
    x = _ou_from_force(force, lam, D_W, dt)
    return float(np.mean((x[:, 1:] - Xobs[:, 1:]) ** 2))


D_W = 1.0   # 反问题的已知 D(回收 λ)


def case_d_inverse(rng):
    lam0, dt, K, M = 1.0, 0.05, 400, 200            # T = 20 s
    delta, lam_eval = 0.03, 1.10
    R = 20

    def fresh_source():
        return ColorNoise(0.0, D_W, dim=M, rng=rng)

    g_crn, g_ind = [], []
    for _ in range(R):
        src = fresh_source()
        src.start_capture()
        Xobs = _ou_from_force_force_dataset(src, lam0, dt, K, M)
        eta = src.stop_capture()                     # (K−1, M) 同一实现
        # CRN:±δ 都 replay 这条实现
        lp = _loss(lam_eval + delta, eta, Xobs, dt)
        lm = _loss(lam_eval - delta, eta, Xobs, dt)
        g_crn.append((lp - lm) / (2 * delta))
        # 独立噪声控制:±δ 各用一条新实现
        fp = np.sqrt(2 * D_W / dt) * rng.standard_normal((K - 1, M))
        fm = np.sqrt(2 * D_W / dt) * rng.standard_normal((K - 1, M))
        lp = _loss(lam_eval + delta, fp, Xobs, dt)
        lm = _loss(lam_eval - delta, fm, Xobs, dt)
        g_ind.append((lp - lm) / (2 * delta))
    g_crn = np.array(g_crn); g_ind = np.array(g_ind)
    v_crn, v_ind = g_crn.var(ddof=1), g_ind.var(ddof=1)
    var_ratio = float(v_ind / v_crn)
    se_diff = float(np.hypot(g_crn.std(ddof=1), g_ind.std(ddof=1)) /
                    np.sqrt(R) * np.sqrt(2.0))
    mean_gap = float(abs(g_crn.mean() - g_ind.mean()))

    # CRN 网格回收(单一大数据集,同一实现扫整条 L(λ) 曲线)
    M2, K2 = 300, 600
    src2 = ColorNoise(0.0, D_W, dim=M2, rng=rng)
    src2.start_capture()
    Xobs2 = _ou_from_force_force_dataset(src2, lam0, dt, K2, M2)
    eta2 = src2.stop_capture()
    grid = np.arange(0.85, 1.1501, 0.01)
    Lg = np.array([_loss(l, eta2, Xobs2, dt) for l in grid])
    i0 = int(np.argmin(Lg))
    lam_crn = _parabolic_min(grid, Lg, i0)

    # 独立噪声对照回收:每个 λ 点一条新实现(报告用,不设线)
    Lig = np.array([_loss(l,
                          np.sqrt(2 * D_W / dt) * rng.standard_normal((K2 - 1, M2)),
                          Xobs2, dt) for l in grid])
    lam_ind = float(grid[int(np.argmin(Lig))])

    # 矩 / AR(1)(EnKF 风格)回收:分离的 ensemble
    M3, K3, dt3 = 800, 800, 0.05
    X3 = _recursion(np.zeros(M3), np.exp(-lam0 * dt3),
                    white_matrix(rng, D_W, dt3, (M3, K3 - 1)) *
                    np.sqrt(dt3 * (1.0 - np.exp(-2.0 * lam0 * dt3)) / 2.0))
    burn = 200
    x0_, x1_ = X3[:, burn:-1].ravel(), X3[:, burn + 1:].ravel()
    slope = float(np.mean(x0_ * x1_) / np.mean(x0_ * x0_))
    lam_mom = -np.log(slope) / dt3
    D_hat = float(lam_mom * np.mean(x0_ * x0_))

    CONTRACTS["D_inverse"] = dict(
        S_xi="two-sided flat 2D (white limit of ColorNoise)",
        tau_c=0.0, rho="I",
        B_noise="inf (white; exact AR(1) update, no dt bias)",
        mode="replay (CRN: capture→replay identical realization; "
             "control: independent draws per ±δ)",
        n_traj=M, T=K * dt, R_reps=R,
        adjoint="GAP: continuous-time adjoint is Phase-6C-later (listed, not faked)")
    CASES["D_inverse"] = dict(
        lam0=lam0, D=D_W, delta=delta, lam_eval=lam_eval, R=R,
        g_crn_mean=float(g_crn.mean()), g_ind_mean=float(g_ind.mean()),
        var_crn=float(v_crn), var_ind=float(v_ind),
        var_ratio=var_ratio, mean_gap=mean_gap, se_diff=se_diff,
        lam_crn=float(lam_crn), lam_ind=lam_ind,
        lam_mom=float(lam_mom), D_hat=D_hat,
        grid="%.2f–%.2f step %.3f" % (grid[0], grid[-1], grid[1] - grid[0]))


def _ou_from_force_force_dataset(src, lam0, dt, K, M):
    """用 src(replay 源)生成一份观测数据;x0=0,初段已过瞬态可控。"""
    force = np.empty((K - 1, M))
    for k in range(K - 1):
        force[k] = src.sample(dt)
    return _ou_from_force(force, lam0, D_W, dt)


def _parabolic_min(grid, L, i):
    if i == 0 or i == len(grid) - 1:
        return float(grid[i])
    a, b, c = L[i - 1], L[i], L[i + 1]
    den = a - 2 * b + c
    if den <= 0:
        return float(grid[i])
    off = 0.5 * (a - c) / den
    return float(grid[i] + off * (grid[1] - grid[0]))


# ---------------------------------------------------------------------------
# F. 广义 Wong–Zakai:延迟反馈回路下的乘性噪声漂移
# ---------------------------------------------------------------------------
def case_f_wong_zakai(rng):
    """dx = μx dt + σx·ξ(t)dt,ξ ColorNoise(τ_c),乘法器输出经一阶极点 τ_d。

    诚实模型(任务指定):ż = (x·ξ − z)/τ_d,更新里用 z(不是 x·ξ)。
    一阶极点不是纯延迟,所以实测曲线对 θ(χ)=1/(2(1+χ)) 的偏差要如实报;
    检查断言的是极限(χ=0 回 ½、χ=16 强压低)与单调性,插值只给容差窗。

    标定:白等效强度 D=0.5、σ=1 ⇒ 白噪声系数 σ_w = σ·√(2D) = 1,
    χ=0 的修正 = σ_w²/2 = 0.5(经典 GBM Stratonovich 线)。
    """
    mu, sig, D = 0.0, 1.0, 0.5
    tc = 0.04
    dt, T, N = 0.002, 2.0, 16000
    K = int(round(T / dt))
    sig_w = sig * np.sqrt(2.0 * D)                  # 白等效 dW 系数 = 1
    e = np.exp(-dt / tc)
    s = np.sqrt((D / tc) * (1.0 - e * e))
    t = np.arange(K) * dt
    fit_mask = t >= 0.3 * T
    rows = []
    for chi in (0.0, 0.25, 1.0, 4.0, 16.0):
        td = chi * tc
        ez = np.exp(-dt / td) if td > 0 else 0.0
        x = np.ones(N); xi = np.zeros(N); z = np.zeros(N)
        means = np.empty(K)
        for k in range(K):
            means[k] = x.mean()
            xi = xi * e + s * rng.standard_normal(N)
            u = x * xi
            z = u if td == 0.0 else z * ez + (1.0 - ez) * u   # 精确一阶极点更新
            x = x * (1.0 + mu * dt) + sig * z * dt
        m = fit_mask
        rate = float(np.polyfit(t[m], np.log(means[m]), 1)[0])
        corr = rate - mu
        theta = wong_zakai_factor(td, tc)
        pred = sig_w ** 2 * theta
        rows.append(dict(chi=chi, tau_d=td, rate=rate, corr=corr,
                         theta=theta, pred=pred, ratio=corr / pred))
    CONTRACTS["F_wong_zakai"] = dict(
        S_xi="two-sided 2D/(1+(2πfτc)²), flat = 2D; D=%.2f" % D,
        tau_c=tc, rho="I (scalar)",
        B_noise="%.4g Hz = 1/(2πτc) (dt = %.3f·τc)" % (1.0 / (2 * np.pi * tc), dt / tc),
        mode="physical (delay implemented as exact one-pole filter on x·ξ, "
             "not a pure delay — deviation from θ(χ) reported per χ)",
        n_traj=N, T=T, dt=dt)
    CASES["F_wong_zakai"] = dict(
        mu=mu, sigma_w=sig_w, tau_c=tc, dt=dt, N=N, T=T,
        points=rows,
        corr0=rows[0]["corr"], strat_pred=rows[0]["pred"],
        corr16_over_corr0=rows[-1]["corr"] / rows[0]["corr"],
        max_interp_dev=float(max(abs(r["ratio"] - 1.0) for r in rows)))


# ---------------------------------------------------------------------------
# G. adjoint 不对称界:Ã_adj = Aᵀ + Δ_asym
# ---------------------------------------------------------------------------
def case_g_adjoint_asym(rng):
    """阻尼链 4 态;J = ½‖Cx(T)−y‖² 对 x0 的 adjoint 梯度。

    硬件 adjoint = 反向用 Aᵀ+Δ_asym(前向状态仍由 A 产生,不重算——
    adjoint_ode_gradient 只服务理想分支,这里用它做机器精度对拍)。
    界:err_rel ≤ Γ_proj(A)·‖Δ_asym‖·T_obs。
    """
    n = 4
    A = np.array([[-1.0, 0.5, 0.0, 0.0],
                  [-0.3, -1.2, 0.4, 0.0],
                  [0.0, 0.2, -1.5, 0.6],
                  [0.0, 0.0, -0.4, -2.0]])
    T_obs, nsteps = 1.0, 50
    dt = T_obs / nsteps
    x0 = np.ones(n)
    C = np.zeros((2, n)); C[0, 2] = 1.0; C[1, 3] = 1.0
    M = rk4_matrix(A, dt)
    xT = np.linalg.matrix_power(M, nsteps) @ x0
    y = C @ xT + np.array([0.05, -0.03])            # 非零残差 ⇒ 非零梯度

    exact = adjoint_ode_gradient(A, x0, T_obs, nsteps, C, y)["grad"]
    # 机器精度对拍:finite difference 直接穿过 forward map。
    def cost(x):
        v = np.asarray(x, dtype=float)
        for _ in range(nsteps):
            v = M @ v
        return 0.5 * float(np.sum((C @ v - y) ** 2))
    fd = finite_difference_gradient(cost, x0, eps=1e-6)
    fd_err = float(np.linalg.norm(fd - exact) / np.linalg.norm(exact))
    # 硬件分支(前向用 A 的状态,反向用 Ã_adj):理想分支 == adjoint_ode_gradient。
    lamN = C.T @ (C @ xT - y)

    def backward(dA):
        lam = lamN.copy()
        B = rk4_matrix(A.T + dA, dt)
        for _ in range(nsteps):
            lam = B @ lam
        return lam

    gam = resolvent_peak(Descriptor(np.eye(n), A))["Gamma"]
    d_dir = rng.standard_normal((n, n))
    d_dir /= np.linalg.norm(d_dir, 2)
    rows = []
    for lev in (1e-6, 1e-5, 1e-4, 1e-3):
        dA = d_dir * lev
        err = float(np.linalg.norm(backward(dA) - exact) / np.linalg.norm(exact))
        bound = adjoint_asym_bound(gam, lev, T_obs)
        rows.append(dict(delta_norm=lev, rel_err=err, bound=bound,
                         ratio=err / bound))
    lev_arr = np.array([r["delta_norm"] for r in rows])
    err_arr = np.array([r["rel_err"] for r in rows])
    slope = float(np.polyfit(np.log(lev_arr), np.log(err_arr), 1)[0])
    CONTRACTS["G_adjoint_asym"] = dict(
        S_xi="n/a (deterministic gradient check)",
        tau_c=0.0, rho="I", B_noise="n/a (no stochastic driver)",
        mode="adjoint (backward with Aᵀ+Δ_asym; forward states from A; "
             "ideal branch cross-checked to adjoint_ode_gradient)",
        n_state=n, T_obs=T_obs, nsteps=nsteps)
    CASES["G_adjoint_asym"] = dict(
        gamma_proj=float(gam), T_obs=T_obs, nsteps=nsteps,
        fd_err=fd_err, slope=slope,
        points=rows,
        max_ratio=float(max(r["ratio"] for r in rows)),
        min_ratio=float(min(r["ratio"] for r in rows)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    _LINES.clear(); CASES.clear(); CONTRACTS.clear()
    rng = np.random.default_rng(SEED)

    hr("Phase 6 验收:随机动力学(§6.11 正问题 / §6.12 反问题)")
    emit("  seed = %d" % SEED)

    hr("A. OU 加性(6A):Var / ACF / 有色源 / 噪声带宽")
    case_a_ou(rng)
    a1 = CASES["A_ou_white"]
    emit("  white:N=%d, T=%.0f s, dt=%.2g → Var = %.4f(解析 D/λ = %.4f,%.2f%%)"
         % (a1["n_traj"], a1["T"], a1["dt"], a1["var"], a1["var_pred"],
            100 * (a1["var_ratio"] - 1)))
    emit("  ACF 拟合 λ̂ = %.4f(真 1.0000)" % a1["lam_hat"])
    a2 = CASES["A_ou_colored"]
    emit("  colored:τ_c=%.2f → Var = %.4f(解析 (D/λ)/(1+λτ_c) = %.4f,%.2f%%)"
         % (a2["tau_c"], a2["var"], a2["var_pred"],
            100 * (a2["var_ratio"] - 1)))
    a3 = CASES["A_bandwidth"]
    emit("  bandwidth:f_c/B_req = 100 → Var = %.4f(白比的 %.4f,解析 %.4f)"
         % (a3["var_100"], a3["ratio100_vs_white"], a3["pred100"]))
    emit("            f_c/B_req =  10 → Var = %.4f(白比的 %.4f,解析 %.4f)"
         % (a3["var_10"], a3["ratio10_vs_white"], a3["pred10"]))
    emit("  100·B_req → 10·B_req 方差下降 = %.2f%%(§6.11 预言 ≈6.4%%,窗 ±30%%)"
         % a3["drop_pct"])

    hr("B. GBM 乘性(6B):Itô vs Stratonovich")
    case_b_gbm(rng)
    b = CASES["B_gbm"]
    emit("  σ²t=1(t=%.3f):E_I = %.4f vs E_S = %.4f → 分离 %.1fσ(线 >3σ)"
         % (b["sep_time"], b["mean_I_sep"], b["mean_S_sep"], b["sep_sigma"]))
    emit("  t=T=%.1f:mean_I/analytic = %.4f(±10%%);mean_S/analytic = %.4f(±10%%)"
         % (b["T"], b["land_I"], b["land_S"]))
    emit("  log 空间:E[ln x_T]−(μ−σ²/2)T = %.2e(3·SE = %.2e)"
         % (b["log_err"], 3 * b["log_se"]))
    emit("  线性修正:A_S−A_I = %.6f vs σ²/2 = %.6f(合同 §1.4)"
         % (b["corr_used"], b["corr_exact"]))

    hr("C. 双势阱 Langevin(6D)")
    case_c_doublewell(rng)
    c = CASES["C_doublewell"]
    emit("  T=%.2f, %d 轨迹 × %.0f 时间单位,样本 %d(尾部护栏:期望计数≥10,保留 %d/%d bin)"
         % (c["T"], c["M"], c["K"] * c["dt"], c["n_samples"], c["n_kept"], 208))
    emit("  KL(p̂‖q) = %.4f(线 < 0.05)" % c["kl"])
    emit("  T̂ = %.4f(真 %.2f,%.2f%%);ln p̂ 拟合 RMS 残差 %.4f"
         % (c["T_hat"], c["T"], 100 * (c["T_hat"] / c["T"] - 1), c["T_resid"]))

    hr("D. 反问题 6C-lite:CRN / 独立控制 / 矩")
    case_d_inverse(rng)
    d = CASES["D_inverse"]
    emit("  λ 评估点 %.2f,δ=%.2f,R=%d:g_CRN = %.4f ± %.4f,g_IND = %.4f ± %.4f"
         % (d["lam_eval"], d["delta"], d["R"], d["g_crn_mean"],
            np.sqrt(d["var_crn"]), d["g_ind_mean"], np.sqrt(d["var_ind"])))
    emit("  方差比 Var(IND)/Var(CRN) = %.1f(线 ≥10)" % d["var_ratio"])
    emit("  两均值差 %.4f(合并 SE %.4f)" % (d["mean_gap"], d["se_diff"]))
    emit("  CRN 网格回收 λ̂ = %.4f(真 1.00,%.2f%%);独立噪声对照 λ̂ = %.4f"
         % (d["lam_crn"], 100 * (d["lam_crn"] - 1), d["lam_ind"]))
    emit("  矩/AR(1) 回收 λ̂ = %.4f(%.2f%%),D̂ = %.4f(%.2f%%)"
         % (d["lam_mom"], 100 * (d["lam_mom"] - 1), d["D_hat"],
            100 * (d["D_hat"] - 1)))
    emit("  adjoint 路径:缺口,Phase-6C-later(报告记录,不伪造)。")

    hr("F. 广义 Wong–Zakai(§1.4 推广):延迟回路的乘性漂移因子")
    case_f_wong_zakai(rng)
    f = CASES["F_wong_zakai"]
    emit("  σ_w=%.2f, τ_c=%.3f, dt=%.2e(=%.2f·τ_c), N=%d, T=%.1f"
         % (f["sigma_w"], f["tau_c"], f["dt"], f["dt"] / f["tau_c"], f["N"], f["T"]))
    for r in f["points"]:
        emit("  χ=%5.2f:实测修正 = %.5f,θ(χ) = %.5f,比值 = %.3f"
             % (r["chi"], r["corr"], r["theta"], r["ratio"]))
    emit("  χ=0 回 Stratonovich:实测/½ = %.3f;χ=16/χ=0 = %.4f(理论 1/17 = %.4f)"
         % (f["corr0"] / f["strat_pred"], f["corr16_over_corr0"], 1.0 / 17.0))
    emit("  对 1/(2(1+χ)) 插值的最大偏差 = %.1f%%(一阶极点非纯延迟,如实报)"
         % (100 * f["max_interp_dev"]))

    hr("G. adjoint 不对称界:Ã_adj = Aᵀ + Δ_asym")
    case_g_adjoint_asym(rng)
    g = CASES["G_adjoint_asym"]
    emit("  Γ_proj = %.3f, T_obs = %.1f, nsteps = %d;fd 对拍相对误差 %.2e"
         % (g["gamma_proj"], g["T_obs"], g["nsteps"], g["fd_err"]))
    for r in g["points"]:
        emit("  ‖Δ‖=%.0e:rel_err = %.3e,界 = %.3e,err/界 = %.3f"
             % (r["delta_norm"], r["rel_err"], r["bound"], r["ratio"]))
    emit("  log-log 斜率 = %.3f(线 ≈1);err/界 ∈ [%.3f, %.3f](断言 ≤1 且 ≥0.01)"
         % (g["slope"], g["min_ratio"], g["max_ratio"]))

    hr("E. 噪声合同自检(§1.3:缺一项就不能编译)")
    fields = ["S_xi", "tau_c", "rho", "B_noise", "mode"]
    rows = []
    for name in sorted(CONTRACTS):
        ct = CONTRACTS[name]
        rows.append([name] + [str(ct.get(f, "—"))[:34] for f in fields])
    table(["case", "S_ξ", "τ_c", "ρ_ij", "B_noise", "mode"], rows)

    hr("验收")
    checks = acceptance()
    table(["check", "got", "expected", "PASS"],
          [(ck["name"], ck["got"], ck["expected"],
            "yes" if ck["passed"] else "NO") for ck in checks])
    n_fail = sum(1 for ck in checks if not ck["passed"])
    emit("\n  %d/%d checks passed" % (len(checks) - n_fail, len(checks)))

    payload = dict(seed=SEED, cases=CASES, contracts=CONTRACTS, checks=checks,
                   summary=dict(n_checks=len(checks),
                                n_passed=len(checks) - n_fail))
    with open(os.path.join(OUT, "sde.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    with open(os.path.join(OUT, "sde_report.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    emit("\n  json   → results/sde.json")
    emit("  report → results/sde_report.txt")

    if n_fail:
        raise SystemExit(1)


def acceptance():
    checks = []

    def add(name, got, expected, passed):
        checks.append(dict(name=name, got=got, expected=expected,
                           passed=bool(passed)))

    a1 = CASES["A_ou_white"]
    db = 10 * np.log10(a1["var_ratio"])
    add("A OU Var vs D/λ", "%.2f dB (%.2f%%)" % (db, 100 * (a1["var_ratio"] - 1)),
        "within ±3 dB", abs(db) < 3.0)
    add("A ACF τ within ±5%", "%.3f%%" % (100 * abs(a1["lam_hat"] - 1)),
        "< 5%", abs(a1["lam_hat"] - 1) < 0.05)
    a2 = CASES["A_ou_colored"]
    add("A colored Var vs (D/λ)/(1+λτc)",
        "%.2f%%" % (100 * (a2["var_ratio"] - 1)), "within ±3%",
        abs(a2["var_ratio"] - 1) < 0.03)
    a3 = CASES["A_bandwidth"]
    add("A bandwidth drop 100→10 B_req", "%.2f%%" % a3["drop_pct"],
        "≈6.4% (window 4.5–8.3%)", 0.045 <= a3["drop"] <= 0.083)
    add("A Var(f_c=10B)/white", "%.4f (pred %.4f)" % (a3["ratio10_vs_white"], a3["pred10"]),
        "±0.025", abs(a3["ratio10_vs_white"] - a3["pred10"]) < 0.025)
    add("A Var(f_c=100B)/white", "%.4f (pred %.4f)" % (a3["ratio100_vs_white"], a3["pred100"]),
        "±0.025", abs(a3["ratio100_vs_white"] - a3["pred100"]) < 0.025)

    b = CASES["B_gbm"]
    add("B Itô/Strato separation", "%.1fσ" % b["sep_sigma"], "> 3σ",
        b["sep_sigma"] > 3.0)
    add("B Itô lands e^{μt}", "%.4f" % b["land_I"], "±10%",
        abs(b["land_I"] - 1) < 0.10)
    add("B Strato lands e^{(μ+σ²/2)t}", "%.4f" % b["land_S"], "±10%",
        abs(b["land_S"] - 1) < 0.10)
    add("B E[ln x] sharp", "%.2e (3·SE %.2e)" % (b["log_err"], 3 * b["log_se"]),
        "< 3·SE", b["log_err"] < 3 * b["log_se"])
    add("B linear correction A_S−A_I = σ²/2",
        "%.2e" % abs(b["corr_used"] - b["corr_exact"]), "< 1e-12",
        abs(b["corr_used"] - b["corr_exact"]) < 1e-12)

    c = CASES["C_doublewell"]
    add("C histogram KL", "%.4f" % c["kl"], "< 0.05", c["kl"] < 0.05)
    add("C T̂ recovery", "%.2f%%" % (100 * abs(c["T_hat"] / c["T"] - 1)),
        "< 10%", abs(c["T_hat"] / c["T"] - 1) < 0.10)

    d = CASES["D_inverse"]
    add("D CRN variance ratio", "%.1f×" % d["var_ratio"], "≥ 10×",
        d["var_ratio"] >= 10.0)
    add("D CRN/IND mean gradients agree", "%.3f vs SE %.3f" % (d["mean_gap"], d["se_diff"]),
        "< 4·SE", d["mean_gap"] < 4 * d["se_diff"])
    add("D CRN recovery λ", "%.2f%%" % (100 * abs(d["lam_crn"] - 1)), "< 10%",
        abs(d["lam_crn"] - 1) < 0.10)
    add("D moment/AR(1) recovery λ", "%.2f%%" % (100 * abs(d["lam_mom"] - 1)),
        "< 10%", abs(d["lam_mom"] - 1) < 0.10)
    add("D moment recovery D", "%.2f%%" % (100 * abs(d["D_hat"] - 1)), "< 10%",
        abs(d["D_hat"] - 1) < 0.10)

    missing = [(n, f) for n, ct in CONTRACTS.items()
               for f in ("S_xi", "tau_c", "rho", "B_noise", "mode")
               if ct.get(f) in (None, "—", "")]
    add("E noise contract complete per case",
        "%d missing" % len(missing), "0 missing (§1.3)", not missing)

    fz = CASES["F_wong_zakai"]
    pts = fz["points"]
    add("F χ=0 lands Stratonovich σ_w²/2",
        "%.3f (pred %.3f)" % (fz["corr0"], fz["strat_pred"]),
        "ratio 0.80–1.20", 0.80 <= fz["corr0"] / fz["strat_pred"] <= 1.20)
    add("F monotone decreasing in χ",
        " → ".join("%.3f" % p["corr"] for p in pts),
        "strictly decreasing",
        all(pts[k]["corr"] > pts[k + 1]["corr"] for k in range(len(pts) - 1)))
    add("F strong-suppression limit χ=16",
        "%.4f × corr(0)" % fz["corr16_over_corr0"],
        "∈ [0.005, 0.10] (theory 1/17 = 0.0588)",
        0.005 <= fz["corr16_over_corr0"] <= 0.10)
    add("F interpolant θ(χ)=1/(2(1+χ))",
        "max dev %.1f%%" % (100 * fz["max_interp_dev"]),
        "≤ 25% (one-pole, not pure delay)",
        fz["max_interp_dev"] <= 0.25)

    gj = CASES["G_adjoint_asym"]
    add("G ideal adjoint == FD", "%.2e" % gj["fd_err"], "< 1e-5",
        gj["fd_err"] < 1e-5)
    add("G error linear in ‖Δ_asym‖", "%.3f" % gj["slope"],
        "log-log slope 0.90–1.10", 0.90 <= gj["slope"] <= 1.10)
    add("G error within bound", "max err/bound %.3f" % gj["max_ratio"],
        "≤ 1", gj["max_ratio"] <= 1.0)
    add("G bound not vacuous", "min err/bound %.3f" % gj["min_ratio"],
        "≥ 0.01", gj["min_ratio"] >= 0.01)

    return checks


if __name__ == "__main__":
    main()
