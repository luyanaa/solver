"""组合误差合同（§6.23）：把一个 $n$ 态系统的误差控制，升到 $m$ 个计算岛。

岛级模型：

    ẋ_i = F_i(x_i) + Σ_{j≠i} F_ij(x_i, x_j)
    D⁺ε_i ≤ μ_i ε_i + Σ_{j≠i} g_ij ε_j + d_i

其中 μ_i 是局部 logarithmic norm（在岛自己的度量 M_i 下），g_ij 是跨岛
耦合的诱导增益。堆成 Metzler 矩阵

    H = [[μ_1, g_12, ...], [g_21, μ_2, ...], ...]

若 H Hurwitz（Metzler + 稳定 ⇒ −H⁻¹ ≥ 0），则

    limsup ε(t) ⪯ −H⁻¹ d.

离散窗口版（采样/保持/周期性校正）：

    ε_{k+1} ⪯ K_k ε_k + η_k,   K_k ≥ 0
    固定 K、ρ(K) < 1 ⇒ ε_∞ ⪯ (I−K)⁻¹ η

**时变拓扑/AMR 的陷阱**：逐窗口 ρ(K_k)<1 不够。K1=[[0,2],[0,0]]、
K2=[[0,0],[2,0]] 各自 ρ=0，乘积 ρ=4。必须检查共同加权范数
（∃w>0: K_k w ≤ (1−δ)w）或周期乘积/矩阵乘积稳定性。

**守恒系统不能用 small-gain 判死**：周期 diffusion Cẋ=−L_Gx 有常数零模，
岛级 μ_i=0、g_ij>0 ⇒ H 有零特征值 ⇒ 不 Hurwitz。但 diffusion 当然可
实现。正确做法是分解

    e = e_mean + e_⊥,   ė_mean = r_leak,   ė_⊥ = A_⊥ e_⊥ + r_⊥

分别维护 ε_conservation（守恒漂移）与 ε_dissipative（投影收缩）。本模块
同时给出 passivity（M A + AᵀM ⪯ 0）与 small-gain 两条对照判据，并把
"small-gain 误拒、守恒+投影通过"作为可复现的验收事实，而不是口号。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Island", "lognorm", "comparison_matrix", "comparison_bound",
    "discrete_bound", "window_contract", "common_metric_contraction",
    "conservation_split", "dissipativity_index", "three_criteria",
]


@dataclass
class Island:
    """一个计算岛：局部动力学 + 度量 + 跨岛耦合块。"""

    name: str
    A: np.ndarray                  # 局部 Jacobian（n_i × n_i）
    M: np.ndarray | None = None    # 度量（默认 I；能量度量 M=diag(C)）
    coupling_out: dict = field(default_factory=dict)   # {j: A_ij}
    d: float = 0.0                 # 局部注入界

    def metric(self) -> np.ndarray:
        n = self.A.shape[0]
        return np.eye(n) if self.M is None else np.asarray(self.M, dtype=float)

    def mu(self) -> float:
        return lognorm(self.A, self.M)


def lognorm(A, M=None) -> float:
    """M-内积下的 logarithmic norm：λmax((B+Bᵀ)/2)，B=M^{1/2}AM^{-1/2}。

    对称正定 M 用 Cholesky；M 退化成 I 时就是 μ₂(A)。这个量是
    ‖e^{At}‖_M ≤ e^{μt} 的生成元上界——岛级误差界的第一块砖。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if M is None:
        B = A
    else:
        M = np.asarray(M, dtype=float)
        L = np.linalg.cholesky(M)
        Linv = np.linalg.inv(L)
        B = L @ A @ Linv
    return float(np.max(np.linalg.eigvalsh((B + B.T) / 2.0)))


def _coupling_gain(A_ij, M_i, M_j) -> float:
    """‖M_i^{1/2} A_ij M_j^{-1/2}‖₂：岛 j 的单位误差在岛 i 度量下的增益。"""
    A = np.asarray(A_ij, dtype=float)
    Li = np.linalg.cholesky(M_i)
    Lj = np.linalg.cholesky(M_j)
    return float(np.linalg.norm(Li @ A @ np.linalg.inv(Lj), 2))


def comparison_matrix(islands: list[Island]) -> np.ndarray:
    """Metzler 比较矩阵 H：对角 μ_i，非对角跨岛增益 g_ij。"""
    m = len(islands)
    H = np.zeros((m, m))
    for i, isl in enumerate(islands):
        H[i, i] = isl.mu()
        for j, Aij in isl.coupling_out.items():
            H[i, int(j)] = _coupling_gain(Aij, isl.metric(),
                                          islands[int(j)].metric())
    return H


def comparison_bound(H, d, eps0=None, T: float = 50.0, n: int = 400) -> dict:
    """H Hurwitz ⇒ limsup ε ⪯ −H⁻¹d；同时数值积分 ε'=Hε+d 做对照。

    返回 dict(hurwitz, bound, simulated_limsup, monotone, spectral_abscissa)。
    `bound` 的每个分量必须 ≥ 模拟终值（保守性检查），且 d=0 时收敛到 0。
    """
    H = np.asarray(H, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    m = H.shape[0]
    if np.any(H - np.diag(np.diag(H)) < -1e-12):
        raise ValueError("H 非 Metzler（非对角为负）：不能作为比较矩阵")
    absc = float(np.max(np.real(np.linalg.eigvals(H))))
    tol = 1e-10 * max(1.0, float(np.max(np.abs(H))))
    hurwitz = bool(absc < -tol)      # 零特征值（守恒零模态）不算 Hurwitz
    if not hurwitz:
        return dict(hurwitz=False, bound=None, simulated_limsup=None,
                    monotone=False, spectral_abscissa=absc)
    bound = -np.linalg.solve(H, d)
    if np.any(bound < -1e-12):
        # Metzler+Hurwitz ⇒ −H⁻¹ ≥ 0；出现负值说明数值/结构有问题
        raise ValueError("−H⁻¹d 有负分量：Metzler/Hurwitz 前提被破坏")
    # 数值积分（RK4）
    e0 = np.zeros(m) if eps0 is None else np.asarray(eps0, dtype=float)
    dt = T / n
    e = e0.copy()
    traj = [e.copy()]
    for _ in range(n):
        k1 = H @ e + d
        k2 = H @ (e + 0.5 * dt * k1) + d
        k3 = H @ (e + 0.5 * dt * k2) + d
        k4 = H @ (e + dt * k3) + d
        e = e + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        traj.append(e.copy())
    traj = np.array(traj)
    tail = traj[-max(1, n // 10):]
    limsup = np.max(tail, axis=0)
    # d ≥ 0 时界轨迹单调不减；d=0 时单调不增。按 d 的符号定方向。
    dmax = float(np.max(np.abs(d)))
    mono = bool(np.all(np.diff(np.max(traj, axis=1)) >= -1e-9)) \
        if dmax > 0 else bool(np.all(np.diff(np.max(traj, axis=1)) <= 1e-9))
    return dict(hurwitz=True, bound=bound.tolist(),
                bound_max=float(bound.max()),
                simulated_limsup=limsup.tolist(),
                overshoot=float(np.max(limsup - bound)),
                monotone=mono, spectral_abscissa=absc,
                traj=traj)


def discrete_bound(K, eta, eps0=None, n: int = 200) -> dict:
    """ρ(K)<1 ⇒ ε_∞ ⪯ (I−K)⁻¹η；同时迭代验证。"""
    K = np.asarray(K, dtype=float)
    eta = np.asarray(eta, dtype=float).ravel()
    if np.any(K < 0):
        raise ValueError("K 必须非负（误差界的比较矩阵）")
    rho = float(np.max(np.abs(np.linalg.eigvals(K))))
    if rho >= 1.0:
        return dict(contracting=False, rho=rho, bound=None)
    bound = np.linalg.solve(np.eye(K.shape[0]) - K, eta)
    e = np.zeros(K.shape[0]) if eps0 is None else np.asarray(eps0, float)
    traj = [e.copy()]
    for _ in range(n):
        e = K @ e + eta
        traj.append(e.copy())
    traj = np.array(traj)
    return dict(contracting=True, rho=rho, bound=bound.tolist(),
                bound_max=float(bound.max()),
                simulated=traj[-1].tolist(),
                overshoot=float(np.max(traj[-1] - bound)),
                traj=traj)


def common_metric_contraction(Ks, delta: float = 1e-6) -> dict:
    """共同加权范数收缩：∃w>0 使 K_k w ≤ (1−δ)w 对所有 k 成立。

    等价于存在对角度量 W 使 ‖W K_k W⁻¹‖_∞ ≤ 1−δ。用 LP 判可行性
    （K_ij ≥ 0 时约束对 w 线性）。这是时变拓扑/AMR 的正确判据：
    逐窗口 ρ(K_k)<1 不足以保证乘积稳定。
    """
    from scipy.optimize import linprog

    Ks = [np.asarray(K, dtype=float) for K in Ks]
    m = Ks[0].shape[0]
    for K in Ks:
        if K.shape != (m, m) or np.any(K < 0):
            raise ValueError("K_k 需同尺寸非负")
    lo, hi = 1e-9, 1.0 - 1e-9
    best = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        A_ub, b_ub = [], []
        for K in Ks:
            A_ub.append(K - (1.0 - mid) * np.eye(m))
            b_ub.append(np.zeros(m))
        # w_i ≥ 1（归一化），目标恒 0：只判可行
        res = linprog(c=np.zeros(m), A_ub=np.vstack(A_ub),
                      b_ub=np.concatenate(b_ub),
                      bounds=[(1.0, None)] * m, method="highs")
        if res.status == 0:
            best = (mid, np.asarray(res.x))
            lo = mid
        else:
            hi = mid
    rho_prod = float(np.max(np.abs(np.linalg.eigvals(
        np.linalg.multi_dot(Ks) if len(Ks) > 1 else Ks[0]))))
    return dict(feasible=best is not None,
                contraction=best[0] if best else None,
                metric=(best[1].tolist() if best else None),
                product_rho=rho_prod,
                individual_rho=[float(np.max(np.abs(np.linalg.eigvals(K))))
                                for K in Ks])


def window_contract(Ks, eta_max, horizon: int = 50) -> dict:
    """周期窗口合同：重复序列 Ks 的乘积谱半径 + 共同度量 + 界。

    固定序列按周期处理；ρ(ΠK) ≥1 ⇒ 无界（不因逐窗口 ρ<1 而放过）。
    """
    prod = np.linalg.multi_dot(Ks) if len(Ks) > 1 else np.asarray(Ks[0])
    rho = float(np.max(np.abs(np.linalg.eigvals(prod))))
    cm = common_metric_contraction(Ks)
    out = dict(cycle_rho=rho, common_metric=cm)
    if rho < 1.0:
        # 周期 Lyapunov：ε_{k+c} ⪯ ΠK ε_k + ηΣ；界 = (I−ΠK)⁻¹ ηΣ
        eta = np.asarray(eta_max, dtype=float)
        S = sum(np.linalg.multi_dot(Ks[i + 1:]) if i + 1 < len(Ks)
                else np.eye(prod.shape[0]) for i in range(len(Ks)))
        eta_sum = S @ eta
        out["bound"] = np.linalg.solve(np.eye(prod.shape[0]) - prod,
                                       eta_sum).tolist()
    else:
        out["bound"] = None
    return out


def conservation_split(A, M=None, null=None) -> dict:
    """守恒/耗散分解：e = e_mean + e_⊥。

    null 是 A 的零模态（默认取 A 的最小 |λ| 特征向量；周期 diffusion 的
    常数向量）。返回 μ_⊥（投影子空间上的 lognorm）、零模态漂移增益
    g_mean（‖P b‖，b 由调用方给；这里只给结构量）、以及 M-正交投影。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    M = np.eye(n) if M is None else np.asarray(M, dtype=float)
    if null is None:
        w, V = np.linalg.eig(A)
        null = np.real(V[:, int(np.argmin(np.abs(w)))])
    v = np.asarray(null, dtype=float).ravel()
    v = v / np.sqrt(float(v @ M @ v))
    P = np.eye(n) - np.outer(v, M @ v)          # M-正交投影（P v = 0）
    # 补空间基 Z：P 是 M-自伴投影，广义特征分解 eigh(P, M) 给出
    # λ≈0（零模态）与 λ≈1（补空间）两组 M-正交向量。
    from scipy.linalg import eigh
    w, U = eigh(P, M)
    Z = U[:, w > 0.5]                            # n−1 维补空间
    A_perp = Z.T @ M @ A @ Z
    mu_perp = lognorm(A_perp)
    mu_full = lognorm(A, M)
    return dict(null=v, projector=P, mu_full=mu_full, mu_perp=mu_perp,
                conservation_dim=1, dissipative_dim=n - 1,
                projected_A=A_perp)


def dissipativity_index(A, M=None) -> float:
    """耗散指数 λmax((MA+AᵀM)/2)；≤0 ⇒ 关于存储 xᵀMx 被动/耗散。"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    M = np.eye(n) if M is None else np.asarray(M, dtype=float)
    S = (M @ A + A.T @ M) / 2.0
    return float(np.max(np.linalg.eigvalsh(S)))


def three_criteria(islands: list[Island], null=None) -> dict:
    """三条组合判据在同一分区上的对照（small-gain / passivity / 守恒）。

    * small-gain：H Hurwitz ⇒ 给出 −H⁻¹d；
    * passivity：Σ 岛耗散指数（M 加权）≤0 且耦合是功率中性 ⇒ 通过；
    * conservation：零模态存在时拆 mean/⊥，分别给漂移率与 μ_⊥。

    返回 dict，其中 `small_gain_reject` 与 `conservation_accept` 是
    可复现的"误拒"证据（周期 diffusion 上 small-gain 必然拒绝）。
    """
    H = comparison_matrix(islands)
    d = np.array([isl.d for isl in islands], dtype=float)
    sg = comparison_bound(H, d)
    diss = [dissipativity_index(isl.A, isl.metric()) for isl in islands]
    # 全局装配
    n = sum(isl.A.shape[0] for isl in islands)
    A_full = np.zeros((n, n))
    M_full = np.zeros((n, n))
    off = 0
    idx = []
    for isl in islands:
        ni = isl.A.shape[0]
        idx.append((off, off + ni))
        off += ni
    for i, isl in enumerate(islands):
        oi, ei = idx[i]
        ni = ei - oi
        A_full[oi:ei, oi:ei] = isl.A
        M_full[oi:ei, oi:ei] = isl.metric()
        for j, Aij in isl.coupling_out.items():
            oj, ej = idx[int(j)]
            Aij = np.asarray(Aij, dtype=float).reshape(ni, ej - oj)
            A_full[oi:ei, oj:ej] = Aij
    cons = conservation_split(A_full, M_full, null=null)
    return dict(H=H, d=d,
                small_gain=dict(hurwitz=sg["hurwitz"],
                                bound=sg["bound"],
                                spectral_abscissa=sg["spectral_abscissa"],
                                reject=not sg["hurwitz"]),
                passivity=dict(indices=diss,
                               total=float(sum(diss)),
                               accept=all(v <= 1e-12 for v in diss)),
                conservation=dict(mu_perp=cons["mu_perp"],
                                  mu_full=cons["mu_full"],
                                  accept=bool(cons["mu_perp"] < 0),
                                  null=cons["null"].tolist()))
