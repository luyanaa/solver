"""广义 Lyapunov → 输出方差（DESIGN.md §0(B)、§1.2）。

    E P Aᵀ + A P Eᵀ + B_p Q B_pᵀ = 0 ,   σ_y² = tr(C P Cᵀ)

E 奇异时 P 在代数子空间上不唯一，噪声注入代数方程也会产生脉冲。做法是
QZ 排序取出稳定有限谱的 deflating subspace，只在那个子空间上解方程；
如果 B 在其余子空间上有分量，H2 直接判为无穷（这是物理事实，不是数值问题）。

奇异 E 的 H2 有限性判据（QZ 路径，Weierstrass 块消去）：

QZ 排序把谱分成三块 Z = [Zs | Zm | Zi]：
  Zs  稳定有限谱（Re λ < 0）——真正的动力学；
  Zm  边缘有限谱（含虚轴/零模态）——被白噪声驱动**且在输出可观**才 σ_y=∞；
  Zi  无穷谱（β≈0）——代数/约束方程。

边缘模态的 driven-vs-observable 细则（H2 是输入→输出性质，不是内部状态性质）：

被驱动的边缘模态若在输出不可观（PBH：rank([λE−A; C]) < n，即存在右特征向量
v 使 (λE−A)v=0 且 Cv=0），它在 H(s) 中的极点残差 C·v·ŵᵀB = 0，被零点对消——
与耦合路径无关（v 取全 pencil 的右特征向量，自动带上它反馈进稳定子空间的
分量）。做法（`_drop_unobservable_marginal`）：收集全部边缘模态的 PBH 零空间
方向（不可观子空间，pencil 不变），正交化后剔除该子空间得到最小实现，在
ω 网格上运行时核对剔除前后 H(s) 逐点一致（对消是恒等式；核对失败——例如
Jordan 链部分可观——就回落到诚实的 ∞），再对降维系统递归。剔除是精确的：
稳定极点处的残差由双正交关系逐项相等，不可观边缘模态的全部贡献（含它经
耦合进入稳定行的通道）随剔除消失。驱动的可观边缘模态仍判 ∞（随机游走），
Bbs_m·u 沿可观边缘模态的馈通也仍判 ∞（那是 D 项，核对会拒绝错误的剔除）。

在 QZ 块上三角形式里（QᵀAZ = AA、QᵀEZ = BB，方程 BB·ż = AA·z + QᵀB·u），
第 I 块行不含 z_s、z_m（下三角块结构为 0），所以代数方程
局部可解出 z_i = g_u·u，g_u = −AA_ii⁻¹·(QᵀB)_i。

注意左投影必须用 Q：状态坐标在 Z 里（x = Z z），但方程行空间在 Q 里。
用 Z1ᵀ 当左投影只在 Q=Z（对称 pencil）时正确——非对称奇异 DAE 下会留
交叉项（P0-G 回归）；P 回全空间用 P = Z1·Pr·Z1ᵀ（协方差在右子空间）。

BB_ii = 0（index-1）时 z_i 纯代数、s 无关。有限行的 BB_rI·ż_i 耦合不是脉冲：
恒等式 s·(sBB−AA)⁻¹ = BB⁻¹ + (sBB−AA)⁻¹·AA·BB⁻¹（右端第二项可分解成
standard form）把该通道变成

    z_s = (s·BB_ss−AA_ss)⁻¹·B_eff − BB_ss⁻¹·BB_sI·g_u
    B_eff = (QᵀB)_s + AA_sI·g_u − AA_ss·BB_ss⁻¹·BB_sI·g_u
    D_total = CZs·(−BB_ss⁻¹·BB_sI·g_u) + CZi·g_u

H2 有限 ⇔ 传递函数严格真且稳定，逐条对应：
  1. BB_ii ≈ 0 且 AA_ii 非奇异：否则无穷特征值带 nilpotent 链（index>1），
     z_i = (s·BB_ii−AA_ii)⁻¹(…) 含 s 的多项式 ⇒ 真脉冲 ⇒ H2 = ∞；
  2. 边缘块不被驱动（直接 QB_m，或经代数/导数通道 AA_mI·g_u / BB_mI·g_u）：
     否则零/虚轴模态共振，白噪声方差发散，σ_y = ∞；
  3. D_total = 0：否则 H(jω) 在高频趋于 D_total，H2 积分发散；
     D_total = 0 时 σ_y² = tr(CZs·P·ZsᵀCᵀ)，P 由 (BB_ss, AA_ss, B_eff) 解出
     （代数变量反馈回动力学的有效输入映射，含 BB_sI 通道）。

其中 (a) 用 ordqz 块消去是主判据（精确、无频带选择问题）；高频扫频 |H|→0
只作为 bench 里的交叉验证，不作判据（数值上脆弱）。

选 (a) 而非 (b) 的理由：判据 1–3 全是有限维代数条件（一次求逆 + 两个范数），
无带宽、无截断误差；而 |H(jω_max)| 阈值依赖网格与条件数，对大 n 会误判。

返回的 residual 是稳定子空间上的相对残差，作用与 coupling.py 的质量账本相同：
它必须 ~1e-12，否则说明求解器本身有隐藏错误。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .descriptor import Descriptor, stable_finite_mask

__all__ = ["LyapunovResult", "solve_generalized_lyapunov"]

_TINY = 1e-300


def _pbh_nullspace(sys: Descriptor, lam: complex, tol_rel: float = 1e-9) -> np.ndarray:
    """PBH 不可观方向：[λE−A; C] 的零空间（复正交基，n×d 列）。

    H2 是输入→输出性质：有限模态 λ 在输出不可观（存在 v 使 (λE−A)v=0 且
    Cv=0）时，它在 H(s) 中的极点残差 C·v·ŵᵀB = 0，被零点对消。v 取自全
    系统（不是边缘块）——右特征向量自动带上该模态反馈进稳定子空间的分量，
    对消判据因此与耦合路径无关。
    """
    M = np.vstack([(lam * sys.E - sys.A).astype(complex),
                   sys.C.astype(complex)])
    _, s, Vh = linalg.svd(M)
    smax = float(s[0]) if s.size else 0.0
    r = int(np.sum(s > tol_rel * max(smax, 1e-300)))
    return Vh[r:].conj().T


def _drop_unobservable_marginal(
    sys: Descriptor, lam_m, Q: np.ndarray, zero_tol: float, depth: int
) -> LyapunovResult | None:
    """被驱动边缘模态不可观 ⇒ 最小实现剔除 + 传递函数核对 + 递归。

    步骤：
      1. 对每个边缘特征值 λ 取 PBH 零空间方向（不可观子空间，pencil 不变）；
         一个方向都找不到 ⇒ 所有边缘模态在输出可观 ⇒ 返回 None（诚实 ∞）。
      2. 方向集实化（复共轭对取 Re/Im）并正交化，得 Vb；正交补 Wb 给出
         [Wb | Vb] 正交基，剔除 Vb 块后得降维系统 (WbᵀEW, WbᵀAW, WbᵀB, CW)。
      3. 运行时核对：在 ω ∈ [1e-4, 1e4] 网格上逐点比较 H_orig 与 H_red——
         对消是恒等式，数值上不成立（如 Jordan 链部分可观）就拒绝剔除。
      4. 对降维系统递归 solve_generalized_lyapunov；每层至少少一维，必终止。

    注意：稳定模态的极点残差由双正交关系在剔除前后逐项相等，且被驱动但
    不可观的边缘模态的全部贡献（含它经耦合进入稳定行的通道）都随剔除消失，
    所以 σ_y 取递归结果并回升 P = Wb·P_r·Wbᵀ 是精确的，不是近似。
    """
    if depth > sys.n:
        return None
    cols = []
    for lm in lam_m:
        N = _pbh_nullspace(sys, lm)
        if N.shape[1]:
            cols.append(N)
    if not cols:
        return None
    Vc = np.hstack(cols)
    real_cols = []
    for j in range(Vc.shape[1]):
        v = Vc[:, j]
        if np.max(np.abs(v.imag)) > 1e-8 * max(np.max(np.abs(v.real)), 1e-300):
            real_cols.extend([v.real, v.imag])   # 复共轭对 ⇒ Re/Im 两个实方向
        else:
            real_cols.append(v.real)
    Vr = np.column_stack(real_cols)
    Ub, s2, _ = linalg.svd(Vr)
    d = int(np.sum(s2 > 1e-9 * max(float(s2[0]) if s2.size else 0.0, 1e-300)))
    if d == 0:
        return None
    Vb = Ub[:, :d]
    Wb = linalg.null_space(Vb.T)                 # 实正交补，n×(n−d)
    red = Descriptor(Wb.T @ sys.E @ Wb, Wb.T @ sys.A @ Wb,
                     Wb.T @ sys.B, sys.C @ Wb, sys.D)
    # 运行时核对：剔除必须不改变传递函数（极点对消是恒等式，不是近似）。
    w_grid = np.logspace(-4, 4, 33)
    H_o = sys.freqresp(w_grid)
    H_r = red.freqresp(w_grid)
    diff = float(np.max(np.abs(H_o - H_r)))
    scale_h = max(1.0, float(np.max(np.abs(H_o))), float(np.max(np.abs(H_r))))
    if not np.isfinite(diff) or diff > 1e-6 * scale_h:
        return None
    if red.n == 0:
        # 全部状态都是不可观边缘模态：严格真、无极点 ⇒ H ≡ 0 ⇒ σ_y = 0。
        return LyapunovResult(np.zeros((sys.n, sys.n)), 0.0, 0.0, 0.0, True,
                              0, sys.n, 0.0,
                              "qz-marginal-unobservable (PBH: dropped %d dim(s), H≡0)" % d,
                              0, 0.0, d)
    res = solve_generalized_lyapunov(red, Q, zero_tol, depth + 1)
    P_full = Wb @ res.P @ Wb.T
    P_full = 0.5 * (P_full + P_full.T)
    method = ("qz-marginal-unobservable (PBH: dropped %d unobservable marginal "
              "dim(s); inner: %s)" % (d, res.method))
    return LyapunovResult(P_full, res.sigma_y2, res.residual, res.rel_residual,
                          res.h2_finite, res.n_stable, sys.n, res.driven_other,
                          method, res.n_infinite, res.d_eff_norm,
                          d + res.n_marginal_dropped)


@dataclass
class LyapunovResult:
    P: np.ndarray
    sigma_y2: float
    residual: float
    rel_residual: float
    h2_finite: bool
    n_stable: int
    n_total: int
    driven_other: float
    method: str
    n_infinite: int = 0
    d_eff_norm: float = 0.0
    n_marginal_dropped: int = 0

    @property
    def sigma_y(self) -> float:
        return float(np.sqrt(max(self.sigma_y2, 0.0)))


def solve_generalized_lyapunov(
    sys: Descriptor, Q: np.ndarray | None = None, zero_tol: float = 1e-12,
    _depth: int = 0,
) -> LyapunovResult:
    """解 E P Aᵀ + A P Eᵀ + B Q Bᵀ = 0。"""
    if Q is None:
        Q = np.eye(sys.m)
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (sys.m, sys.m):
        raise ValueError("Q 形状应为 (m, m)")
    W = sys.B @ Q @ sys.B.T
    n = sys.n

    # sys.D（显式直接馈通）在任何路径上都必须判 H2 = ∞：白噪声经 D 直达输出，
    # H2 积分发散。旧实现只在 QZ 路径通过 D_total 间接看到 C 侧的馈通，
    # sys.D 本身被静默忽略（D=1 的稳定 SISO 错报 σ_y²=0.5）。
    d_sys = float(np.linalg.norm(sys.D))
    tol_sys_d = 1e-12 * max(1.0, float(np.linalg.norm(sys.C)) * float(np.linalg.norm(sys.B)))
    if d_sys > tol_sys_d:
        return LyapunovResult(np.zeros((n, n)), float("inf"), 0.0, 0.0, False,
                              0, n, 0.0,
                              "direct-feedthrough (sys.D≠0, ‖D‖=%.2e)" % d_sys,
                              0, d_sys)

    # 严格稳定且 E 良态：直接降为 (E⁻¹A, E⁻¹B)，精度最好。
    use_direct = sys.is_stable() and np.linalg.cond(sys.E) < 1e10
    if use_direct:
        As = linalg.solve(sys.E, sys.A)
        Bs = linalg.solve(sys.E, sys.B)
        P = linalg.solve_lyapunov(As, -(Bs @ Q @ Bs.T))
        P = 0.5 * (P + P.T)
        R = sys.E @ P @ sys.A.T + sys.A @ P @ sys.E.T + W
        num = np.linalg.norm(R)
        den = np.linalg.norm(sys.E @ P @ sys.A.T) + np.linalg.norm(sys.A @ P @ sys.E.T) + np.linalg.norm(W)
        sigma_y2 = float(np.real(np.trace(sys.C @ P @ sys.C.T)))
        return LyapunovResult(P, sigma_y2, float(num), float(num / max(den, 1e-300)),
                              True, n, n, 0.0, "direct")

    # 奇异/临界：QZ 排序，把谱分成 稳定有限 | 边缘有限 | 无穷(代数) 三块。
    # 左投影必须用 Q：ordqz 给 QᵀAZ = AA、QᵀEZ = BB（块上三角），变换后的方程
    # 是 BB·ż = AA·z + QᵀB·u——状态坐标在 Z 里，方程行空间在 Q 里。用 Z1ᵀ 当
    # 左投影只在 Q=Z（对称 pencil）时正确，非对称 DAE 下会留交叉项（P0-G）。
    w = sys.finite_eigs()
    scale = max(1.0, float(np.max(np.abs(w))) if w.size else 1.0)
    sort = lambda a, b: stable_finite_mask(a, b, zero_tol, scale)
    AA, BB, alpha, beta, Ql, Z = linalg.ordqz(sys.A, sys.E, sort=sort, output="real")
    k = int(np.count_nonzero(sort(alpha, beta)))

    # 第二次 ordqz：在“非稳定”块内部把 边缘有限 排到 无穷 之前（Q、Z 两侧都要复合）。
    if k < n:
        sort2 = lambda a, b: np.abs(b) > 1e-12 * (np.abs(a) + np.abs(b) + _TINY)
        AA2, BB2, al2, be2, Q2, Z2r = linalg.ordqz(AA[k:, k:], BB[k:, k:],
                                                   sort=sort2, output="real")
        km = int(np.count_nonzero(sort2(al2, be2)))
        AA_new = np.zeros_like(AA)
        BB_new = np.zeros_like(BB)
        AA_new[:k, :k] = AA[:k, :k]
        BB_new[:k, :k] = BB[:k, :k]
        AA_new[:k, k:] = AA[:k, k:] @ Z2r
        BB_new[:k, k:] = BB[:k, k:] @ Z2r
        AA_new[k:, k:] = AA2
        BB_new[k:, k:] = BB2
        AA, BB = AA_new, BB_new
        Ql = np.concatenate([Ql[:, :k], Ql[:, k:] @ Q2], axis=1)
        Z = np.concatenate([Z[:, :k], Z[:, k:] @ Z2r], axis=1)
    else:
        km = 0
    ni = n - k - km
    Qs, Qm, Qi = Ql[:, :k], Ql[:, k:k + km], Ql[:, k + km:]
    Zs, Zm, Zi = Z[:, :k], Z[:, k:k + km], Z[:, k + km:]

    scale_AE = max(1.0, float(np.linalg.norm(sys.A)), float(np.linalg.norm(sys.E)))

    # 稳定块上的降阶 pencil（诊断残差与求解共用）。
    def _stable_blocks(B_extra=None):
        Er = Qs.T @ sys.E @ Zs
        Ar = Qs.T @ sys.A @ Zs
        Br = Qs.T @ sys.B if B_extra is None else B_extra
        return Er, Ar, Br

    def _solve_stable(Er, Ar, Br):
        As = linalg.solve(Er, Ar)
        Bs = linalg.solve(Er, Br)
        Wr = Bs @ Q @ Bs.T
        Pr = linalg.solve_lyapunov(As, -Wr)
        Pr = 0.5 * (Pr + Pr.T)
        # E 形残差：AsP + PAsᵀ = −Wr 两边左乘 Er、右乘 Erᵀ 得
        # Ar P Erᵀ + Er P Arᵀ + Er·Wr·Erᵀ = 0（Er≠I 时必须带上 Er·Wr·Erᵀ）。
        Wt = Er @ Wr @ Er.T
        Rr = Er @ Pr @ Ar.T + Ar @ Pr @ Er.T + Wt
        num = np.linalg.norm(Rr)
        den = np.linalg.norm(Er @ Pr @ Ar.T) + np.linalg.norm(Ar @ Pr @ Er.T) + np.linalg.norm(Wt)
        P = Zs @ Pr @ Zs.T
        return 0.5 * (P + P.T), float(num), float(num / max(den, 1e-300))

    def _marginal_driven() -> float:
        if km == 0:
            return 0.0
        Bm = Qm.T @ sys.B
        return float(np.linalg.norm(Bm @ Q @ Bm.T))

    # 无穷块不存在：全部有限谱。边缘块（零/虚轴模态）被驱动**且可观** ⇒ H2 = ∞；
    # 被驱动但不可观 ⇒ 极点对消，剔除不可观子空间后 σ_y 仍有限（最小实现）。
    if ni == 0:
        driven = _marginal_driven()
        if driven > 1e-12 * max(1.0, float(np.linalg.norm(W))):
            AA_mm = Qm.T @ sys.A @ Zm
            BB_mm = Qm.T @ sys.E @ Zm
            lam_m = linalg.eigvals(AA_mm, BB_mm)
            res = _drop_unobservable_marginal(sys, lam_m, Q, zero_tol, _depth)
            if res is not None:
                return res
            return LyapunovResult(np.zeros((n, n)), float("inf"), 0.0, 0.0, False,
                                  k, n, driven,
                                  "qz-marginal-driven (zero/imaginary mode integrates "
                                  "noise, observable at output)", 0, 0.0)
        if k == 0:
            return LyapunovResult(np.zeros((n, n)), 0.0, 0.0, 0.0, True, 0, n, driven,
                                  "qz-empty", 0, 0.0)
        Er, Ar, Br = _stable_blocks()
        P, num, rel = _solve_stable(Er, Ar, Br)
        sigma_y2 = float(np.real(np.trace(sys.C @ P @ sys.C.T)))
        return LyapunovResult(P, sigma_y2, num, rel, True, k, n, driven, "qz", 0, 0.0)

    # ---- 无穷块存在：index-1 假设下做 Weierstrass 块消去 ----
    # 行空间用 Q：第 I 块行（Qiᵀ 方程）不含 z_s、z_m（AA/BB 下三角块为 0），
    # 代数方程局部可解出 z_i = g_u·u，g_u = −AA_ii⁻¹·(QᵀB)_i。
    A_ii = Qi.T @ sys.A @ Zi
    E_ii = Qi.T @ sys.E @ Zi
    B_i = Qi.T @ sys.B
    C_i = sys.C @ Zi
    # QᵀB 的边缘块行：边缘动力学直接吃到的输入。
    Bm_q = Qm.T @ sys.B if km > 0 else None

    g_u: np.ndarray | None = None

    # 1) BB_ii ≈ 0（Q 左基下的 E 无穷块）？不为零 ⇒ BB_ii 是非零 nilpotent
    #    （无穷特征值带 Jordan 链，index>1），z_i = (s·BB_ii−AA_ii)⁻¹(…) 含 s 的
    #    多项式 ⇒ 传递函数非真 ⇒ 脉冲 ⇒ H2 = ∞。
    if float(np.linalg.norm(E_ii)) > 1e-9 * scale_AE:
        driven = _marginal_driven()
        Er, Ar, Br = _stable_blocks()
        P, _, _ = _solve_stable(Er, Ar, Br) if k > 0 else (np.zeros((n, n)), 0.0, 0.0)
        return LyapunovResult(P, float("inf"), 0.0, 0.0, False, k, n, driven,
                              "qz-index>1 (nilpotent infinite block, |BB_ii|=%.2e)" % np.linalg.norm(E_ii),
                              ni, 0.0)
    # 2) AA_ii 非奇异？奇异 ⇒ 约束块不可解出（index>1 / 非正则）。
    try:
        g_u = linalg.solve(A_ii, -B_i)
        scale_i = max(1.0, float(np.linalg.norm(A_ii)), float(np.linalg.norm(B_i)))
        resid = float(np.linalg.norm(A_ii @ g_u + B_i))
        if resid > 1e-8 * scale_i:
            g_u = None
    except linalg.LinAlgError:
        g_u = None
    if g_u is None:
        driven = _marginal_driven()
        Er, Ar, Br = _stable_blocks()
        P, _, _ = _solve_stable(Er, Ar, Br) if k > 0 else (np.zeros((n, n)), 0.0, 0.0)
        return LyapunovResult(P, float("inf"), 0.0, 0.0, False, k, n, driven,
                              "qz-singular-algebraic-block (AA_ii not invertible)", ni, 0.0)
    # 3) BB_ii = 0 且 AA_ii 非奇异 ⇒ z_i = g_u·u 纯代数（s 无关）。
    #    有限行的 BB_rI·ż_i 耦合不是脉冲：s·(sBB_rr−AA_rr)⁻¹ → BB_rr⁻¹ 是有界的
    #    （proper），它只贡献一条带直接馈通的有效通道，进 (5) 的 D_eff。
    # 4) 边缘块被驱动（直接 QB_m 或经代数/导数通道）且可观 ⇒ 随机游走/共振
    #    ⇒ σ_y = ∞；被驱动但不可观 ⇒ 极点对消：剔除不可观边缘子空间
    #   （PBH 零空间 + 传递函数运行时核对），对降维系统递归求 σ_y。
    #    代数偏置 Bbs_m·u 沿可观边缘模态的馈通不是极点对消——剔除核对会把
    #    它当作 H(s) 的不匹配拒绝，回落到诚实的 ∞。
    BB_mm = Qm.T @ sys.E @ Zm if km > 0 else None
    AA_mm = Qm.T @ sys.A @ Zm if km > 0 else None
    if km > 0:
        BB_mI = Qm.T @ sys.E @ Zi
        AA_mI = Qm.T @ sys.A @ Zi
        Bbs_m = linalg.solve(BB_mm, BB_mI @ g_u)          # 边缘坐标的直接馈通
        f_m = Bm_q + AA_mI @ g_u - AA_mm @ Bbs_m          # 边缘动力学的有效输入
        driven = float(max(np.linalg.norm(f_m @ Q @ f_m.T),
                           np.linalg.norm(Bbs_m @ Q @ Bbs_m.T)))
    else:
        Bbs_m = None
        driven = 0.0
    tol_driven = 1e-12 * max(1.0, float(np.linalg.norm(sys.B @ Q @ sys.B.T)))
    n_drop = 0
    if driven > tol_driven and km > 0:
        lam_m = linalg.eigvals(AA_mm, BB_mm)
        res = _drop_unobservable_marginal(sys, lam_m, Q, zero_tol, _depth)
        if res is not None:
            return res
        Er, Ar, Br = _stable_blocks()
        P, _, _ = _solve_stable(Er, Ar, Br) if k > 0 else (np.zeros((n, n)), 0.0, 0.0)
        return LyapunovResult(P, float("inf"), 0.0, 0.0, False, k, n, driven,
                              "qz-marginal-driven (zero/imaginary mode integrates "
                              "noise, observable at output)", ni, 0.0)
    # 5) 直接馈通与有效输入映射。恒等式
    #      s·(sBB−AA)⁻¹ = BB⁻¹ + (sBB−AA)⁻¹·AA·BB⁻¹
    #    把 s·BB_sI·g_u·u̇ 通道分解成 standard form：
    #      z_s = (s·BB_ss−AA_ss)⁻¹·B_eff − BB_ss⁻¹·BB_sI·g_u
    #      B_eff = QB_s + AA_sI·g_u − AA_ss·BB_ss⁻¹·BB_sI·g_u
    #    总馈通 D_total = CZs·(−BB_ss⁻¹·BB_sI·g_u) + CZi·g_u。
    Cm_Bbs = sys.C @ Zm @ Bbs_m if km > 0 else np.zeros((sys.p, sys.m))
    if k > 0:
        BB_ss = Qs.T @ sys.E @ Zs
        AA_ss = Qs.T @ sys.A @ Zs
        BB_sI = Qs.T @ sys.E @ Zi
        AA_sI = Qs.T @ sys.A @ Zi
        Bbs_s = linalg.solve(BB_ss, BB_sI @ g_u)
        B_eff = Qs.T @ sys.B + AA_sI @ g_u - AA_ss @ Bbs_s
        D_total = sys.C @ Zs @ (-Bbs_s) + C_i @ g_u + Cm_Bbs + sys.D
    else:
        B_eff, D_total = None, C_i @ g_u + Cm_Bbs + sys.D
    d_norm = float(np.linalg.norm(D_total))
    tol_d = 1e-9 * max(1.0, float(np.linalg.norm(sys.C)) * float(np.linalg.norm(g_u)))
    if k == 0:
        return LyapunovResult(np.zeros((n, n)), 0.0, 0.0, 0.0, d_norm <= tol_d,
                              0, n, driven, "qz-dae-eliminated (no dynamics)", ni, d_norm,
                              n_drop)
    Er, Ar, _ = _stable_blocks()
    P, num, rel = _solve_stable(Er, Ar, B_eff)
    h2_finite = d_norm <= tol_d
    sigma_y2 = float(np.real(np.trace(sys.C @ P @ sys.C.T)))
    if h2_finite and n_drop:
        method = ("qz-dae-eliminated (PBH: dropped %d unobservable marginal dim(s))"
                  % n_drop)
    else:
        method = "qz-dae-eliminated" if h2_finite else "qz-direct-feedthrough (D_total≠0)"
    return LyapunovResult(P, sigma_y2 if h2_finite else float("inf"),
                          num, rel,
                          h2_finite, k, n, driven, method, ni, d_norm, n_drop)
