"""广义 Lyapunov → 输出方差（DESIGN.md §0(B)、§1.2）。

    E P Aᵀ + A P Eᵀ + B_p Q B_pᵀ = 0 ,   σ_y² = tr(C P Cᵀ)

E 奇异时 P 在代数子空间上不唯一，噪声注入代数方程也会产生脉冲。做法是
QZ 排序取出稳定有限谱的 deflating subspace，只在那个子空间上解方程；
如果 B 在其余子空间上有分量，H2 直接判为无穷（这是物理事实，不是数值问题）。

奇异 E 的 H2 有限性判据（QZ 路径，Weierstrass 块消去）：

QZ 排序把谱分成三块 Z = [Zs | Zm | Zi]：
  Zs  稳定有限谱（Re λ < 0）——真正的动力学；
  Zm  边缘有限谱（含虚轴/零模态）——仍是状态，但被白噪声驱动即 σ_y=∞（随机游走）；
  Zi  无穷谱（β≈0）——代数/约束方程。

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

    @property
    def sigma_y(self) -> float:
        return float(np.sqrt(max(self.sigma_y2, 0.0)))


def solve_generalized_lyapunov(
    sys: Descriptor, Q: np.ndarray | None = None, zero_tol: float = 1e-12
) -> LyapunovResult:
    """解 E P Aᵀ + A P Eᵀ + B Q Bᵀ = 0。"""
    if Q is None:
        Q = np.eye(sys.m)
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (sys.m, sys.m):
        raise ValueError("Q 形状应为 (m, m)")
    W = sys.B @ Q @ sys.B.T
    n = sys.n

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

    # 无穷块不存在：全部有限谱。边缘块（零/虚轴模态）被驱动 ⇒ H2 = ∞。
    if ni == 0:
        if k == 0:
            driven = _marginal_driven()
            return LyapunovResult(np.zeros((n, n)), 0.0 if driven <= 1e-12 else float("inf"),
                                  0.0, 0.0, driven <= 1e-12, 0, n, driven, "qz-empty",
                                  0, 0.0)
        Er, Ar, Br = _stable_blocks()
        P, num, rel = _solve_stable(Er, Ar, Br)
        driven = _marginal_driven()
        h2_finite = driven <= 1e-12 * max(1.0, float(np.linalg.norm(W)))
        sigma_y2 = float(np.real(np.trace(sys.C @ P @ sys.C.T)))
        return LyapunovResult(P, sigma_y2 if h2_finite else float("inf"),
                              num, rel, h2_finite, k, n, driven, "qz", 0, 0.0)

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
    # 4) 边缘块被驱动（直接 QB_m 或经代数/导数通道）⇒ 随机游走/共振 ⇒ σ_y = ∞。
    if km > 0:
        BB_mm = Qm.T @ sys.E @ Zm
        AA_mm = Qm.T @ sys.A @ Zm
        BB_mI = Qm.T @ sys.E @ Zi
        AA_mI = Qm.T @ sys.A @ Zi
        Bbs_m = linalg.solve(BB_mm, BB_mI @ g_u)          # 边缘坐标的直接馈通
        f_m = Bm_q + AA_mI @ g_u - AA_mm @ Bbs_m          # 边缘动力学的有效输入
        driven = float(max(np.linalg.norm(f_m @ Q @ f_m.T),
                           np.linalg.norm(Bbs_m @ Q @ Bbs_m.T)))
    else:
        driven = 0.0
    if driven > 1e-12 * max(1.0, float(np.linalg.norm(sys.B @ Q @ sys.B.T))):
        Er, Ar, Br = _stable_blocks()
        P, _, _ = _solve_stable(Er, Ar, Br) if k > 0 else (np.zeros((n, n)), 0.0, 0.0)
        return LyapunovResult(P, float("inf"), 0.0, 0.0, False, k, n, driven,
                              "qz-marginal-driven (zero/imaginary mode integrates noise)", ni, 0.0)
    # 5) 直接馈通与有效输入映射。恒等式
    #      s·(sBB−AA)⁻¹ = BB⁻¹ + (sBB−AA)⁻¹·AA·BB⁻¹
    #    把 s·BB_sI·g_u·u̇ 通道分解成 standard form：
    #      z_s = (s·BB_ss−AA_ss)⁻¹·B_eff − BB_ss⁻¹·BB_sI·g_u
    #      B_eff = QB_s + AA_sI·g_u − AA_ss·BB_ss⁻¹·BB_sI·g_u
    #    总馈通 D_total = CZs·(−BB_ss⁻¹·BB_sI·g_u) + CZi·g_u。
    if k > 0:
        BB_ss = Qs.T @ sys.E @ Zs
        AA_ss = Qs.T @ sys.A @ Zs
        BB_sI = Qs.T @ sys.E @ Zi
        AA_sI = Qs.T @ sys.A @ Zi
        Bbs_s = linalg.solve(BB_ss, BB_sI @ g_u)
        B_eff = Qs.T @ sys.B + AA_sI @ g_u - AA_ss @ Bbs_s
        D_total = sys.C @ Zs @ (-Bbs_s) + C_i @ g_u
    else:
        B_eff, D_total = None, C_i @ g_u
    d_norm = float(np.linalg.norm(D_total))
    tol_d = 1e-9 * max(1.0, float(np.linalg.norm(sys.C)) * float(np.linalg.norm(g_u)))
    if k == 0:
        return LyapunovResult(np.zeros((n, n)), 0.0, 0.0, 0.0, d_norm <= tol_d,
                              0, n, driven, "qz-dae-eliminated (no dynamics)", ni, d_norm)
    Er, Ar, _ = _stable_blocks()
    P, num, rel = _solve_stable(Er, Ar, B_eff)
    h2_finite = d_norm <= tol_d
    sigma_y2 = float(np.real(np.trace(sys.C @ P @ sys.C.T)))
    method = "qz-dae-eliminated" if h2_finite else "qz-direct-feedthrough (D_total≠0)"
    return LyapunovResult(P, sigma_y2 if h2_finite else float("inf"),
                          num, rel,
                          h2_finite, k, n, driven, method, ni, d_norm)
