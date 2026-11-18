"""多速率/分裂合法性合同（DESIGN.md §4.3，review 第三支柱 4）。

三个解析对象，没有求解器：

1. **WR 迭代算子**。线性化分裂 ``E ẋ − A_diag x = A_off x_prev + Bu``
   的频域 Jacobi waveform-relaxation 迭代算子是

       G_WR(jω) = (jωE − A_diag)^{-1} A_off

   收敛条件是 **sup_ω σ_max(G_WR(jω)) = ρ_sup < 1**——不是只看 DC 值。
   ρ_sup 一般不等于 DC 值：A_diag 非正规或含轻阻尼复极点时，resolvent
   范数在带内某 ω* 处有 interior peak（本仓库的测试系统就是这种：
   岛 B 是轻阻尼振子，σ_max 在 ω≈ω0 处比 DC 高一个数量级）。判收敛
   时必须扫频，不能只算 (−A_diag)^{-1}A_off。

   **ρ_sup 是 small-gain 充分证书，不是实际收敛率**。逐频稳态误差
   满足 e^(k) = G_WR(jω)·e^(k−1)，其衰减率由 G_WR 的**特征值**谱半径
   sup_ω |λ_max(G_WR(jω))| 决定；σ_max 是它的范数上界
   （‖e^(k)‖ ≤ ρ_sup^k‖e^(0)‖ 对任何激励方向都成立）。两者当且仅当
   G_WR 在激发频率**正规**时相等；非正规时 σ_max 证书保守（bench
   实测：非正规分裂实测 0.149 vs σ_max 0.738，差 5×，但 σ_max^k 界
   仍然成立且方向保守）。`wr_spectral_radius` 两个值都返回。

2. **replay 底噪界**。硬件波形重放每次叠加 DAC 量化 + 重构滤波误差
   ε_replay。k 次迭代后

       ‖e^(k)‖ ≤ ρ_sup^k·‖e^(0)‖ + (1−ρ_sup^k)/(1−ρ_sup)·R∞·ε_replay

   第二项给出有效迭代次数上限 k_max：几何项降到 replay 平台之后，
   继续迭代不再降误差——k_max 是编译器必须输出的数，不是仿真里
   试出来的。R∞ 是每次迭代的"重放误差 → 状态误差"增益的 sup_ω：
   重放误差从交换端口进入，恰好乘 A_off，所以紧的取值是
   R∞ = sup_ω σ_max((jωE−A_diag)^{-1}A_off) = ρ_sup 本身；合同原文的
   ‖(jωE−A_diag)^{-1}‖∞ 是不挑扰动方向的保守上界（任何 port 扰动
   都成立），两者都在 bench 里报告。

3. **ZOH 等效传递（含 sinc droop）**。跨岛 S&H 的精确等效传递是
   H_ZOH(s) = (1−e^{−sT_s})/(sT_s) = e^{−sT_s/2}·sinc(sT_s/2)：相位
   与 e^{−sT_s/2} 完全一致，但**幅度不是 1**——jω 处
   |H| = |sinc(ωT_s/2)|，DC=1，Nyquist(ω=π/T_s) 处只有 2/π≈0.637
   （−3.92 dB）。早期版本把它并成 PureDelay(T_s/2)：相位正确、幅度
   错误（Nyquist 处报告 0 dB）。`twin/route.py` 现在有 `ZOH` 块，
   `zoh_edge(T_s)` 返回它，droop 直接进 route_closure 的 gain 轴。
   只有当带宽足够低（droop < ε，默认 0.01 ⇒ ωT_s ≲ 0.155）时才允许
   退化成 PureDelay(T_s/2)——由 `zoh_edge(..., degrade=...)` 显式
   记录（`degraded`/`droop` 字段），不允许在高带宽下静默使用延迟
   近似。droop 小 u 展开 ≈ (ωT_s)²/24，ε=0.01 的退化阈值为
   ωT_s* = √(24ε) ≈ 0.49。

   合同里 ZOH 是**三项独立预算**，不许折叠成一个数：
   (a) **sinc droop**（幅度损失 vs ω，−3.92 dB @ Nyquist）——route
       gain 轴，进 η_route/J_H；
   (b) **采样混叠**——f_s/2 以上的信号/噪声功率折叠进带内，混叠
       方差 ≈ 信号在 Nyquist 以上的方差；信号带宽逼近 Nyquist 时
       `zoh_contract` 置 flag。它进 §4.4/P0.5 的 noise 轴，不进
       route 轴（本模块只立 flag，不做方差计算）；
   (c) **重构滤波**——消除镜像的滤波器本身有纹波/延迟/极点预算，
       依赖它必须在别处立单独预算，不能算进 droop 里。
"""

from __future__ import annotations

import numpy as np

from .route import PureDelay, RouteEdge, ZOH

__all__ = ["wr_iteration_matrix", "wr_spectral_radius", "wr_error_bound",
           "resolvent_norm_inf", "k_max", "zoh_edge", "zoh_contract",
           "DROOP_EPS_DEFAULT"]


def wr_iteration_matrix(E, A_diag, A_off, omega):
    """G_WR(jω) = (jωE − A_diag)^{-1} A_off，按 ω 逐点求解。

    omega 标量 → 返回 (n,n)；数组 → 返回 (len(omega), n, n)。
    """
    E = np.asarray(E, dtype=complex)
    A_diag = np.asarray(A_diag, dtype=complex)
    A_off = np.asarray(A_off, dtype=complex)
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.empty((w.size, A_diag.shape[0], A_diag.shape[1]), dtype=complex)
    for i, wi in enumerate(w):
        out[i] = np.linalg.solve(1j * wi * E - A_diag, A_off)
    if np.ndim(omega) == 0:
        return out[0]
    return out


def wr_spectral_radius(E, A_diag, A_off, omega_grid):
    """ρ_sup = sup_ω σ_max(G_WR(jω)) 与逐频特征值谱半径，及 interior-peak 诊断。

    返回 dict：
      rho_sup        sup_ω σ_max(G_WR)——small-gain 充分收敛证书（要求 < 1）；
      rho_eig_sup    sup_ω |λ_max(G_WR)|——逐频稳态 WR 的实际每迭代
                     衰减率（幂迭代收敛到主导特征方向）；ρ_sup = σ_max
                     范数上界，两者当且仅当 G_WR 在该频率正规时相等；
      normal_gap     rho_sup / rho_eig_sup（=1 ⇒ 正规；>1 ⇒ σ_max 保守）；
      omega_peak     σ_max 取到最大值的 ω；
      sigma_dc       ω→0 的 σ_max（在 grid[0]/100 处取）；
      interior_peak  True 若峰值 ω 在 grid 内部且 σ_max(ω*) > σ_dc；
      monotone_decreasing  True 若 σ_max 沿 grid 单调不增；
      sigma_max_grid / lambda_grid / omega_grid  逐点值（诊断用）。

    注意：σ_max(G_WR) 沿 ω **不保证单调**，也**不保证峰值在 DC**。
    A_diag 正规且谱为负实时通常 DC 最大；含复极点或非正规块时
    peak 移到带内。收敛判据用的是 ρ_sup，不是 σ_dc。
    """
    res = wr_iteration_matrix(E, A_diag, A_off, omega_grid)
    smax = np.array([np.linalg.svd(g, compute_uv=False)[0] for g in res])
    lam = np.array([np.max(np.abs(np.linalg.eigvals(g))) for g in res])
    i = int(np.argmax(smax))
    dc = wr_iteration_matrix(E, A_diag, A_off, float(omega_grid[0]) * 1e-2)
    sigma_dc = float(np.linalg.svd(dc, compute_uv=False)[0])
    interior = bool(i > 0 and i < len(omega_grid) - 1 and smax[i] > sigma_dc)
    mono = bool(np.all(np.diff(smax) <= 1e-12))
    rho_eig = float(np.max(lam))
    return dict(
        rho_sup=float(smax[i]), omega_peak=float(omega_grid[i]),
        sigma_dc=sigma_dc, interior_peak=interior,
        monotone_decreasing=mono, rho_eig_sup=rho_eig,
        normal_gap=float(smax[i] / rho_eig) if rho_eig > 0 else float("inf"),
        lambda_grid=lam,
        sigma_max_grid=smax, omega_grid=np.asarray(omega_grid, dtype=float))


def resolvent_norm_inf(E, A_diag, omega_grid):
    """R∞ = sup_ω σ_max((jωE − A_diag)⁻¹)：重放误差 → 状态误差的最坏增益。

    §4.3 合同的保守侧（任意端口扰动方向都成立）；replay 噪声只从 A_off
    交换口进入时紧值就是 ρ_sup 本身（run_multirate.py 的 R_tight）。
    """
    E = np.asarray(E, dtype=complex)
    A_diag = np.asarray(A_diag, dtype=complex)
    w = np.atleast_1d(np.asarray(omega_grid, dtype=float))
    eye = np.eye(A_diag.shape[0])
    peak = 0.0
    peak_w = float(w[0])
    for wi in w:
        s = np.linalg.svd(np.linalg.solve(1j * wi * E - A_diag, eye),
                          compute_uv=False)[0]
        if s > peak:
            peak, peak_w = float(s), float(wi)
    return dict(resolvent_norm_inf=peak, omega_peak=peak_w)


def wr_error_bound(rho_sup, resolvent_norm_inf, eps_replay, k, e0_norm=1.0):
    """k 次 WR 迭代 + 每次重放误差 ε_replay 的误差上界。

        ‖e^(k)‖ ≤ ρ_sup^k·‖e^(0)‖ + (1−ρ_sup^k)/(1−ρ_sup)·R∞·ε_replay

    R∞ = resolvent_norm_inf 是每次迭代的"重放误差→状态误差"增益的
    sup_ω。对从交换端口进入的重放噪声，紧值为 ρ_sup；保守值为
    ‖(jωE−A_diag)^{-1}‖∞（任意端口扰动方向都成立）。
    """
    if not 0.0 <= rho_sup < 1.0:
        return float("inf")
    geo = rho_sup ** k * e0_norm
    floor = (1.0 - rho_sup ** k) / (1.0 - rho_sup) * resolvent_norm_inf * eps_replay
    return float(geo + floor)


def k_max(rho_sup, resolvent_norm_inf, eps_replay, e0_norm):
    """几何项降到 replay 底噪的迭代次数上限（向上取整）。

    解 ρ_sup^k·e0 = R∞·ε_replay/(1−ρ_sup)：

        k_max = ceil( log( R∞·ε_replay / ((1−ρ_sup)·e0) ) / log(ρ_sup) )

    超过 k_max 后 replay 底噪主导——**更多迭代不降低误差**，只烧
    能量。ρ_sup ≥ 1（不收敛）或 e0 ≤ 0 时返回 inf。
    """
    if rho_sup >= 1.0 or rho_sup < 0.0 or e0_norm <= 0.0:
        return float("inf")
    if eps_replay <= 0.0 or resolvent_norm_inf <= 0.0:
        return 0.0
    ratio = resolvent_norm_inf * eps_replay / ((1.0 - rho_sup) * e0_norm)
    if ratio >= 1.0:
        return 0
    return int(np.ceil(np.log(ratio) / np.log(rho_sup)))


#: droop 退化阈值：|sinc| 损失 < DROOP_EPS_DEFAULT（1%，−0.087 dB）
#: 时允许把 ZOH 记成 PureDelay(T_s/2)。droop(u)=1−|sinc(u/2)|，
#: u=ωT_s；小 u 时 droop≈u²/24，故阈值 u*≈√(24ε)≈0.49（ε=0.01）。
DROOP_EPS_DEFAULT = 0.01


def _droop(omega, T_s):
    x = np.asarray(omega, dtype=float) * T_s / 2.0
    return np.abs(np.sinc(x / np.pi))


def droop_omega_ts(eps: float = DROOP_EPS_DEFAULT) -> float:
    """droop 恰好等于 eps 的 ωT_s 值（退化阈值的解析定义，供验收对表）。

    droop(u)=1−|sinc(u/2)| 在 u∈(0,2) 上单调；小 u 近似 droop≈u²/24，
    ε=0.01 ⇒ u*≈√(24ε)=0.49（与二分一致）。droop 在 |sinc| 的零点之外
    回升（u>2 后 |sinc| 再增大），二分只在 (0,2) 内找第一个根——那正是
    退化有意义的区间。
    """
    lo, hi = 1e-12, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - _droop(mid, 1.0) > eps:
            hi = mid
        else:
            lo = mid
    return lo


def zoh_edge(T_s, omega_max: float | None = None,
             eps_droop: float = DROOP_EPS_DEFAULT) -> dict:
    """跨岛 S&H 的 ZOH 等效传递，含 sinc droop，不静默近似。

    返回 dict：
      block      `twin/route.py` 的 `ZOH` 块（H(s)=(1−e^{−sT_s})/(sT_s)），
                 相位 = PureDelay(T_s/2)，幅度带 sinc droop（Nyquist 处
                 2/π≈0.637）。直接进 RouteEdge/route_closure，droop 由
                 gain 轴的 gain_err 判定，无新机制；
      degraded   True ⇒ ω_max 处 droop < eps_droop，允许（且已经）用
                 PureDelay(T_s/2) 代替 ZOH 块（记录在案，不静默）；
      droop      ω_max 处（未给出 ω_max 时取 Nyquist）的幅度损失因子
                 1−|sinc(ωT_s/2)|；
      droop_max_allowed  eps_droop。

    不给 omega_max ⇒ 不做退化判断，永远返回真 ZOH（保守侧）。
    **混叠与重构滤波是单独预算，见 zoh_contract。**
    """
    zoh = ZOH(T_s)
    out = dict(block=zoh, degraded=False, droop=float("nan"),
               droop_max_allowed=eps_droop)
    if omega_max is None:
        return out
    d = 1.0 - float(_droop(omega_max, T_s))
    out["droop"] = d
    if d < eps_droop:
        out["block"] = PureDelay(T_s / 2.0)
        out["degraded"] = True
    return out


def zoh_contract(T_s, omega_sig: float, recon_filter_db: float | None = None,
                 eps_droop: float = DROOP_EPS_DEFAULT) -> dict:
    """ZOH 三项独立预算（不许折叠成一个数）：

      droop        sinc 幅度损失 1−|sinc(ω_sig T_s/2)|——route gain 轴；
      aliasing     flag：信号带宽 ω_sig 是否逼近/越过 Nyquist π/T_s。
                   flag 只立合同：混叠方差进 §4.4/P0.5 noise 轴，不进
                   route 轴。阈值：ω_sig > 0.5·π/T_s 记 warning，
                   > 0.9·π/T_s 记 violation；
      reconstruction  重构滤波器预算，**独立一行**：只消除 f_s 附近的
                   镜像，其纹波/延迟另行立账。recon_filter_db=None ⇒
                   未声明滤波器，镜像残余由 sinc 本身在 ω_s−ω_sig 处
                   的衰减量报告（不能默认为 0）。
    """
    w_nyq = np.pi / T_s
    droop = 1.0 - float(_droop(omega_sig, T_s))
    frac = omega_sig / w_nyq
    alias_flag = ("violation" if frac > 0.9 else
                  "warning" if frac > 0.5 else "none")
    # 镜像位置 ω_s−ω_sig 处 sinc 本身的衰减（dB，负=衰减）：
    img_atten_db = 20.0 * np.log10(max(_droop(2.0 * np.pi / T_s - omega_sig, T_s),
                                       1e-300))
    recon = dict(filter_declared=recon_filter_db is not None,
                 recon_budget_db=recon_filter_db,
                 image_freq=(2.0 * np.pi / T_s - omega_sig),
                 sinc_image_atten_db=img_atten_db)
    return dict(T_s=T_s, omega_nyquist=w_nyq, omega_sig=omega_sig,
                bandwidth_fraction=frac, droop=droop,
                droop_eps=eps_droop, droop_ok=droop < eps_droop,
                aliasing=alias_flag,
                reconstruction=recon)

