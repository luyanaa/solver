"""Adjoint ε_J 双轨：QoI 导向的误差估计与精度分配（DESIGN.md §6.22 续）。

全局误差合同 ε_global 保证物理场不被严重失真；ε_J 只校正**对目标结果
有影响的误差**。线性系统 ẋ = Ax + r、终端 QoI J = cᵀx(T) 时：

    λ(t) = expm(Aᵀ(T−t))·c          （伴随场，λ(T) = c）
    δJ  = cᵀe(T) = λ(0)ᵀe(0) + ∫₀ᵀ λ(t)ᵀ r(t) dt      —— 恒等式

第一级估计 δJ₁ = Σ_k λ(t_k)ᵀ R_k（窗口注入残差 × 伴随权重）：

  * 对线性系统，窗口足够细时 δJ₁ 收敛到真实 δJ（恒等式，非界）——
    run_error_control 的 case_adjoint_qoi 直接对账实测相对误差；
  * **它不是严格上界**：各窗口 λᵀR_k 可正负抵消。
    cancellation_factor = |Σ_k λᵀR_k| / Σ_k |λᵀR_k| 如实记录；
    → 0 时 δJ 小不代表误差贡献小，只能靠 ε_global 兜底（双轨合同
    存在的理由）。非线性/离散 DWR 情形估计本身还有额外偏差，
    本模块不给那个承诺。

精度分配：w_i = ∫|λ_i(t)|dt 是节点 i 的 QoI 敏感度权重。同样的
bias 注到高 w 节点与低 w 节点，对 J 的影响差 w 倍——校正/校准
预算（更大电容、更频繁 trim）按 w 排序分配，低权重节点不烧预算。
c_i = 0 的节点（QoI 不可观）w=0：其误差完全不进 ε_J，只进 ε_global。

依赖方向：纯线性代数（E=I 口径；descriptor/E≠I 的伴随是 TODO，
不静默声称支持）。估计与权重是 evidence，不是判决——判决在
lowering/budget.py 的聚合层。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "adjoint_field",
    "qoi_error_estimate",
    "dJ_weights",
    "dJ_weights_exact",
    "descriptor_adjoint_field",
    "descriptor_qoi_error_estimate",
    "nonlinear_adjoint_backward",
]


def adjoint_field(A, c, t_grid) -> np.ndarray:
    """λ(t) = expm(Aᵀ(T−t))·c，返回 (n, len(t_grid))。

    t_grid 必须从 0 递增到 T。O(n³·len) 的暴力实现，n≤256 判定器
    规模内可接受；更大规模走矩阵指数的缩放平方法一次算 Φ(T,·) 采样，
    本模块不做那个优化。
    """
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    if c.shape[0] != A.shape[0]:
        raise ValueError("c 维度必须与 A 匹配")
    if t.shape[0] < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t_grid 必须严格递增且至少两点")
    T = t[-1]
    out = np.empty((A.shape[0], t.shape[0]))
    for k, tk in enumerate(t):
        out[:, k] = expm(A.T * (T - tk)) @ c
    return out


def qoi_error_estimate(lam_field, t_grid, window_rs, window_centers) -> dict:
    """第一级 QoI 误差估计 δJ₁ = Σ_k λ(t_k)ᵀ R_k + 抵消指标。

    window_rs: 每窗口的注入残差 R_k（twin.residual 的输出，= ∫r dt）。
    window_centers: 每窗口中点时刻（λ 按中点采样——窗口常数 r 近似，
    误差 O(ΔT²·λ'·r)，调用方保证窗口够细或自行对账）。
    返回 delta_J（估计）、逐窗口贡献、abs_sum 与 cancellation_factor。
    """
    lam = np.asarray(lam_field, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    if lam.shape[1] != t.shape[0]:
        raise ValueError("lam_field 列数必须与 t_grid 一致")
    centers = np.asarray(window_centers, dtype=float)
    lam_at = np.empty((lam.shape[0], centers.shape[0]))
    for k, tc in enumerate(centers):
        lam_at[:, k] = lam[:, int(np.argmin(np.abs(t - tc)))]
    contrib = []
    for k, R in enumerate(window_rs):
        contrib.append(float(lam_at[:, k] @ np.asarray(R, dtype=float)))
    total = float(sum(contrib))
    abs_sum = float(sum(abs(v) for v in contrib))
    denom = max(abs_sum, 1e-300)
    return dict(
        delta_J=total,
        contributions=contrib,
        abs_sum=abs_sum,
        # →0：正负抵消严重，δJ₁ 不可单独信任（双轨合同的触发条件）
        cancellation_factor=abs(total) / denom,
        n_windows=len(contrib),
    )


def descriptor_adjoint_field(E, A, c, t_grid) -> np.ndarray:
    """descriptor 系统 E·ẋ = A·x + r 的伴随场（E≠I 口径）。

    E 可逆时 ẋ = E⁻¹Ax，前向 Φ(T,t)=expm(E⁻¹A·(T−t))，终端 QoI
    J=cᵀx(T) 的伴随 λ(t)=Φ(T,t)ᵀc=expm((E⁻¹A)ᵀ(T−t))c。
    **陷阱**：直接把 Aᵀ 当生成元（expm(Aᵀ(T−t))c）在 E≠I 时是错的
    ——链里 E⁻¹ 同时出现在前向生成元与伴随力项。E 奇异时 raise
    （代数约束须先投影/降维，不静默伪逆）。E=I 时退化为 adjoint_field。
    """
    E = np.asarray(E, dtype=float)
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    if E.shape != A.shape or A.shape[0] != c.shape[0]:
        raise ValueError("E/A 同形且 c 维度匹配")
    if abs(np.linalg.det(E)) < 1e-12 * max(1.0, float(np.linalg.norm(E))):
        raise ValueError("E 奇异：descriptor 伴随须先投影掉代数约束，不静默伪逆")
    if t.shape[0] < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t_grid 必须严格递增且至少两点")
    G = np.linalg.solve(E, A)              # 前向生成元
    T = t[-1]
    out = np.empty((A.shape[0], t.shape[0]))
    for k, tk in enumerate(t):
        out[:, k] = expm(G.T * (T - tk)) @ c
    return out


def descriptor_qoi_error_estimate(E, lam_field, t_grid, window_rs,
                                   window_centers) -> dict:
    """descriptor 端口权重下的 QoI 误差估计 δJ₁=Σ λ(t_k)ᵀE⁻¹R_k。

    R_k=∫r dt 是**源空间**的窗口残差（E·(x₁−x₀)−A∫x dt），λ 是
    状态空间伴随——两者之间必须夹 E⁻¹（源经 E⁻¹ 进状态方程）。
    缺 E⁻¹ 的 naive 估计在 E 病态（mode-C 电容跨 10³）时可偏差
    κ(E) 倍——DESIGN.md §5.8 的 M 度量是同一陷阱的频域版。
    """
    E = np.asarray(E, dtype=float)
    lam = np.asarray(lam_field, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    centers = np.asarray(window_centers, dtype=float)
    Ginv = np.linalg.inv(E)
    contrib = []
    for k, R in enumerate(window_rs):
        j = int(np.argmin(np.abs(t - centers[k])))
        contrib.append(float(lam[:, j] @ Ginv @ np.asarray(R, dtype=float)))
    total = float(sum(contrib))
    abs_sum = float(sum(abs(v) for v in contrib))
    return dict(delta_J=total, contributions=contrib, abs_sum=abs_sum,
                cancellation_factor=abs(total) / max(abs_sum, 1e-300),
                n_windows=len(contrib))

def nonlinear_adjoint_backward(F_jac, x_ref_traj, t_grid, c) -> np.ndarray:
    """沿参考轨迹反向积分伴随：λ̇ = −J_F(x_ref(t))ᵀλ，λ(T)=c。

    F_jac(t, x) -> J_F（n×n，在参考状态处线性化）。x_ref_traj 为
    (n, len(t_grid))。用后向 RK4（h>0 从右往左推进）。非线性系统
    的 J_F 沿轨迹变化，不能写成单个 expm——§6.22 非线性 DWR 基础件。
    """
    xr = np.asarray(x_ref_traj, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    c = np.asarray(c, dtype=float)
    if xr.ndim != 2 or xr.shape[1] != t.shape[0] or c.shape[0] != xr.shape[0]:
        raise ValueError("x_ref_traj 必须是 (n, len(t_grid))，c 维度匹配")
    if t.shape[0] < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t_grid 必须严格递增")
    out = np.empty_like(xr)
    out[:, -1] = c
    for k in range(t.shape[0] - 2, -1, -1):
        h = float(t[k + 1] - t[k])
        x_hi = xr[:, k + 1]
        x_lo = xr[:, k]
        x_mid = 0.5 * (x_lo + x_hi)
        t_mid = 0.5 * (t[k] + t[k + 1])
        J_hi = np.asarray(F_jac(t[k + 1], x_hi), dtype=float)
        J_mid = np.asarray(F_jac(t_mid, x_mid), dtype=float)
        J_lo = np.asarray(F_jac(t[k], x_lo), dtype=float)
        l = out[:, k + 1]
        k1 = -J_hi.T @ l
        k2 = -J_mid.T @ (l - 0.5 * h * k1)
        k3 = -J_mid.T @ (l - 0.5 * h * k2)
        k4 = -J_lo.T @ (l - h * k3)
        out[:, k] = l - h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return out

def dJ_weights_exact(A, c, T: float) -> np.ndarray:
    """对角实负系统的解析 dJ 权重：w_i ∝ c_i·(1−e^{−γ_iT})/γ_i。

    **刚性系统的梯形采样陷阱**（实测 diffusion-1d n=64）：快模态的
    伴随 λ_i(t) = c_i·e^{−γ_i(T−t)} 在 t=T 处是 1/γ_i 尖峰，均匀网格
    （65 点、Δt=1/64）完全分辨不了——dJ_weights 的梯形积分只剩 t=T
    端点样本的贡献，权重退化成 1/n 均匀（端点伪影），而解析权重
    应集中在慢模态（w ∝ 1/γ，比值 ~10³）。γ_max·Δt ≫ 1 的刚性系统
    **必须**走本闭式；非对角系统逐模态 |·| 不可对角化，只能采样，
    调用方自行保证 γ_max·Δt ≪ 1（否则权重不可信，宁缺毋滥）。

    只支持 A 在标准基下近对角且特征值实负（spectral tile 的形态）；
    否则 raise，不静默回退到不可信的采样。
    """
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)
    if A.shape[0] != c.shape[0] or A.shape[0] != A.shape[1]:
        raise ValueError("A 方阵、c 维度匹配")
    off = np.abs(A - np.diag(np.diag(A))).max()
    scale = max(1.0, float(np.abs(np.diag(A)).max()))
    if off > 1e-9 * scale:
        raise ValueError("A 非对角（off=%.3g）：解析权重只支持对角实负"
                         "系统；非对角请用 dJ_weights 采样并保证 "
                         "γ_max·Δt ≪ 1" % off)
    gamma = -np.diag(A)
    if np.any(gamma <= 0):
        raise ValueError("存在非负特征值：不是收缩对角系统，闭式不适用")
    if np.any(c < 0):
        raise ValueError("c 含负分量：|·| 逐模态口径要求 c ≥ 0（QoI 同向）")
    w = c * (1.0 - np.exp(-gamma * T)) / gamma
    s = float(w.sum())
    if s <= 0:
        raise ValueError("权重全零：c=0 或 T 过短")
    return w / s


def dJ_weights(lam_field, t_grid) -> np.ndarray:
    """逐节点 QoI 敏感度权重 w_i = ∫|λ_i(t)|dt（归一到 Σw=1）。

    分配语义：校正预算按 w 排序；w_i=0 的节点误差不进 ε_J（只进
    ε_global）。注意 w 是**积分**敏感度——瞬时 |λ_i| 大但衰减快的
    模态不一定排前，与"驻留时间"加权一致。
    """
    lam = np.asarray(lam_field, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    w = np.array([np.trapezoid(np.abs(lam[i, :]), t) for i in range(lam.shape[0])])
    # 采样分辨率自查（诚实口径）：γ_max·Δt ≫ 1 的快模态伴随在 t=T
    # 处是 1/γ 尖峰，均匀网格必然欠采样——见 dJ_weights_exact 的
    # 陷阱记录。这里不静默修正，调用方按 γ_max·Δt 自查。
    s = float(w.sum())
    if s <= 0:
        raise ValueError("伴随场全零：c=0 或系统在 (0,T) 内完全衰减")
    return w / s
