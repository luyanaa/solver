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

3. **ZOH 等效延迟**。跨岛 S&H 的等效传递 H_ZOH(s) = (1−e^{−sT_s})/(sT_s)
   ≈ e^{−sT_s/2}：把跨岛 S&H 直接并入 ``twin/route.py`` 的
   ``PureDelay(T_s/2)``，复用同一套 route_closure/slack 判定，**不建
   新机制**。折叠/混叠噪声是单独的 B_p 端口：采样率 f_s 以上的信号
   功率折叠到带内，混叠方差 ≈ 信号在 f_s/2 以上的方差（本模块不
   实现，只立合同——混叠进 P0.5 的 noise 轴，不进 route 轴）。
"""

from __future__ import annotations

import numpy as np

from .route import PureDelay

__all__ = ["wr_iteration_matrix", "wr_spectral_radius", "wr_error_bound",
           "k_max", "zoh_edge"]


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


def zoh_edge(T_s):
    """跨岛 S&H 的 ZOH 等效延迟：PureDelay(T_s/2)。

    精确等效传递是 H_ZOH(s) = (1−e^{−sT_s})/(sT_s) = e^{−sT_s/2}·sinc(sT_s/2)：
    相位与 e^{−sT_s/2} 完全一致，幅度差一个 sinc 因子（ωT_s ≪ 1 时
    可忽略）。因此跨岛采样交换在 route 层就是 PureDelay(T_s/2)，由
    ``route_closure`` 的同一套 gain/phase/noise/absolute-delay 判定
    处理，不需要新机制。

    **混叠是单独的 B_p 端口，不在这里**：采样把 f_s/2 以上的信号与
    噪声功率折叠进带内，混叠方差 ≈ 信号在 f_s/2 以上的方差（加上
    采样孔径/量化项）。它进 §4.4/P0.5 的 noise 轴预算，不进 route
    轴；调用方负责在 noise slack 里加这一项。
    """
    return PureDelay(T_s / 2.0)
