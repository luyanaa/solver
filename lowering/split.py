"""CSP/QSSA 快慢模态剥离（DESIGN §4.1 P2 决策树第二行："快模态稳定 && 有谱间隙
&& 被奴役 ⇒ QSSA/CSP 降为代数约束（E 奇异化）"）。

输入：线性 descriptor（E 可逆）或裸 Jacobian（n×n，视为 E=I）。奇异 E ⇒
NotImplementedError——QSSA 剥离的**输出**正是奇异 E 的慢子系统；奇异 E 的
系统已经剥离过了，不重复剥。

分类（诚实，不硬剥）：

* caller 给 tau_fast：fast = {λ : Re λ < −1/tau_fast}，其余 slow。跨切点的
  比值间隙 < GAP_MIN 时照切但打 "weak-gap" flag（caller 显式指定是被尊重的，
  但弱间隙要记录）。
* 否则 auto：按 |Re λ| 降序找最大比值间隙；最大 ratio ≥ GAP_MIN（10）才切，
  否则 verdict="no-split"（连续谱/弱间隙不硬剥，不产 QSSA 数字）。

校验（不静默）：

* fast 块必须全为衰减（Re<0）且良态（cond(A_ff) ≤ COND_MAX），否则 flag 并
  拒绝——不静默求逆。
* 可对角化检查：特征向量矩阵 V 满秩，rank(V)<n ⇒ "defective" ⇒ no-split。
* 坐标归属：每个坐标归 |V[j,i]| 最大的特征值；有特征值分不到坐标、或前两名
  近并列（比值 > TIE_MAX）⇒ "coord-assignment-ambiguous" ⇒ no-split。

QSSA 化简（物理坐标 Schur 补，qssa_reduce）：

    x_f = −A_ff⁻¹ (A_fs x_s + B_f u)
    A_eff = A_ss − A_sf A_ff⁻¹ A_fs
    B_eff = B_s − A_sf A_ff⁻¹ B_f ;  C_eff = C_s − C_f A_ff⁻¹ A_fs
    D_eff = D − C_f A_ff⁻¹ B_f

slaving 残差（本模块的诚实 η）：初值取在代数流形上（x_f = 慢流的代数函数），
RK 积分全系统，量测 fast 坐标对代数流形的偏离（归一到慢态幅度）。量级由
间隙决定（间隙越大越小）——报告实测数字，不宣称解析量级。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp

__all__ = ["SplitResult", "cvsplit", "qssa_reduce", "GAP_MIN", "COND_MAX"]

GAP_MIN = 10.0     # auto 模式：跨间隙 |Re λ| 比值低于此 ⇒ no-split
COND_MAX = 1e13    # fast 块条件数上限；超过 ⇒ 拒绝求逆
TIE_MAX = 0.9      # 坐标归属：第二名/第一名超过此值 ⇒ 近并列 ⇒ 拒绝


@dataclass
class SplitResult:
    """快慢剥离的完整记录；no-split 时 reduced=None（诊断信息照给）。"""

    verdict: str                  # "split" | "no-split"
    n_slow: int
    n_fast: int
    gap_ratio: float | None       # 跨切点的 |Re λ| 比值
    tau_fast: float | None        # caller 指定（None=auto）
    slow_eigs: np.ndarray
    fast_eigs: np.ndarray
    reduced: object | None        # 慢子系统 Descriptor（modal 无；物理坐标 Schur）
    A_eff: np.ndarray | None
    eig_match: float | None       # max|eig(A_eff) − λ_slow|
    slaving_error: float | None   # 轨迹级 slow 坐标最大相对偏差（实测 η）
    flags: list = field(default_factory=list)
    perm: np.ndarray | None = field(default=None)   # 原坐标 → (slow…, fast…) 排列
    meta: dict = field(default_factory=dict)


def qssa_reduce(A_ss, A_sf, A_ff, A_fs, B_s=None, B_f=None, C_s=None, C_f=None, D=None):
    """物理坐标 Schur 补：x_f = −A_ff⁻¹(A_fs x_s + B_f u)。

    fast 块不稳定（任一 Re eig ≥ 0）或 cond > COND_MAX ⇒ ValueError——
    flag 出来，不静默求逆。返回 (A_eff, B_eff, C_eff, D_eff)。
    """
    A_ff = np.asarray(A_ff, dtype=float)
    A_fs = np.asarray(A_fs, dtype=float)
    A_sf = np.asarray(A_sf, dtype=float)
    A_ss = np.asarray(A_ss, dtype=float)
    nf = A_ff.shape[0]
    wA = linalg.eigvals(A_ff)
    scale = max(1.0, float(np.max(np.abs(wA))) if nf else 1.0)
    if np.any(np.real(wA) > -1e-12 * scale):
        raise ValueError(
            "qssa_reduce: fast 块含非衰减模态（Re λ ≥ 0）——QSSA 假设快模态被奴役，"
            "不稳定快模态不能代数化；flag 出来而不是静默求逆")
    cond_ff = float(np.linalg.cond(A_ff))
    if not np.isfinite(cond_ff) or cond_ff > COND_MAX:
        raise ValueError(
            f"qssa_reduce: fast 块病态（cond={cond_ff:.3g} > {COND_MAX:.0g}）——"
            "Schur 补不做静默求逆")
    A_ff_inv = np.linalg.inv(A_ff)
    A_eff = A_ss - A_sf @ A_ff_inv @ A_fs
    if B_s is None:
        return A_eff, None, None, None
    B_s = np.asarray(B_s, dtype=float)
    B_f = np.asarray(B_f, dtype=float)
    B_eff = B_s - A_sf @ A_ff_inv @ B_f
    C_eff = None
    D_eff = None
    if C_s is not None:
        C_s = np.asarray(C_s, dtype=float)
        C_f = np.asarray(C_f, dtype=float)
        C_eff = C_s - C_f @ A_ff_inv @ A_fs
        D_eff = np.zeros((C_eff.shape[0], B_eff.shape[1])) if D is None \
            else np.asarray(D, dtype=float) - C_f @ A_ff_inv @ B_f
    return A_eff, B_eff, C_eff, D_eff


def _extract_A(system):
    """Descriptor（E 可逆）或裸 Jacobian → (A, B, C, D)。"""
    from analysis import Descriptor
    if isinstance(system, np.ndarray):
        A = np.asarray(system, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("裸 Jacobian 必须是方阵")
        n = A.shape[0]
        return A, np.zeros((n, 1)), np.eye(n), None
    if isinstance(system, Descriptor):
        n = system.n
        if np.allclose(system.E, np.eye(n)):
            return system.A, system.B, system.C, system.D
        if abs(np.linalg.det(system.E)) < 1e-12 * max(1.0, np.linalg.norm(system.E)):
            raise NotImplementedError(
                "cvsplit: 奇异 E 的 descriptor 不做快慢剥离——它已经是剥离后的"
                "形式（QSSA/虚地/KCL 的输出）。要检查其快慢结构请用 "
                "analysis.descriptor 的有限特征值诊断")
        A = np.linalg.solve(system.E, system.A)
        return A, system.B, system.C, system.D
    raise NotImplementedError(
        f"cvsplit: system 类型 {type(system).__name__} 不支持"
        f"（支持 analysis.Descriptor 或 n×n Jacobian ndarray）")


def _classify(w: np.ndarray, tau_fast: float | None):
    """→ (fast_mask, slow_mask, gap_ratio, split_reason)。"""
    m = np.abs(np.real(w))                      # 衰减速率量级（非正规不影响分类轴）
    order = np.argsort(-m)                      # 降序：快在前
    if tau_fast is not None:
        thr = 1.0 / float(tau_fast)
        fast = np.real(w) < -thr
        if np.any(fast):
            m_fast = m[fast].min()
            m_slow = m[~fast]
            m_slow_pos = m_slow[m_slow > 0]
            ratio = float(m_fast / m_slow_pos.min()) if m_slow_pos.size else float("inf")
            return fast, ~fast, ratio, "caller-tau_fast"
        return fast, ~fast, None, "no-fast-modes"
    # auto：最大比值间隙。分母 0（真零模/边缘模在下方）⇒ inf 也是合法间隙；
    # 0/0 ⇒ nan（无意义，跳过）。
    m_sorted = m[order]
    best_i, best_r = -1, 0.0
    for i in range(len(m_sorted) - 1):
        if m_sorted[i + 1] == 0.0:
            r = float("inf") if m_sorted[i] > 0.0 else float("nan")
        else:
            r = m_sorted[i] / m_sorted[i + 1]
        if np.isnan(r):
            continue
        if r > best_r:
            best_r, best_i = r, i
    if best_i < 0 or best_r < GAP_MIN:
        return None, None, float(best_r), "no-gap"
    fast = np.zeros(len(w), dtype=bool)
    fast[order[:best_i + 1]] = True
    return fast, ~fast, float(best_r), "auto-gap"


def _no_split(w, flags, gap_ratio, tau_fast, meta_extra=None):
    return SplitResult(
        verdict="no-split", n_slow=0, n_fast=0, gap_ratio=gap_ratio,
        tau_fast=tau_fast, slow_eigs=w, fast_eigs=np.array([]),
        reduced=None, A_eff=None, eig_match=None, slaving_error=None,
        flags=list(flags),
        meta=meta_extra or {})


def cvsplit(system, tau_fast: float | None = None, T: float | None = None,
            x0: np.ndarray | None = None) -> SplitResult:
    """快慢模态剥离：分类 → 物理坐标 Schur QSSA → slaving 残差实测。

    tau_fast caller 指定快/慢分界（fast = Re λ < −1/tau_fast）；None = auto
    谱间隙检测（ratio ≥ GAP_MIN 才切）。T 为 slaving 轨迹时长；None = 
    3/max|Re λ_slow|。x0 慢坐标初值（缺省 e_1；fast 坐标自动放到代数流形上）。
    """
    A, B, C, D = _extract_A(system)
    n = A.shape[0]
    if n < 2:
        raise ValueError("n<2 无快慢可言")
    w, V = linalg.eig(A)
    flags: list[str] = []

    # 可对角化检查（defective ⇒ 坐标级剥离不良定义）
    if np.linalg.matrix_rank(V, tol=1e-8) < n:
        return _no_split(w, ["defective-eigenvectors"], None, tau_fast)

    # 不稳定模态记录（信息级）
    scale = max(1.0, float(np.max(np.abs(w))))
    if np.any(np.real(w) > 1e-10 * scale):
        flags.append("unstable-mode-present")

    fast_mask, slow_mask, gap_ratio, reason = _classify(w, tau_fast)
    if reason == "no-fast-modes":
        flags.append("no-fast-modes")
        return _no_split(w, flags, gap_ratio, tau_fast)
    if reason == "no-gap":
        flags.append("no-gap-below-%.0f" % GAP_MIN)
        return _no_split(w, flags, gap_ratio, tau_fast)

    nf = int(fast_mask.sum())
    ns = n - nf

    # fast 块稳定性预检（qssa_reduce 会再查一次；这里为了给 no-split verdict）
    w_fast = w[fast_mask]
    if np.any(np.real(w_fast) > -1e-12 * scale):
        flags.append("fast-block-unstable")
        return _no_split(w, flags, gap_ratio, tau_fast, meta_extra=dict(
            fast_eigs_re=[float(v) for v in np.real(w_fast)]))

    # 特征向量 → 坐标归属；冲突/近并列 ⇒ 拒绝
    amp = np.abs(V)
    assign = np.argmax(amp, axis=1)           # coord j → eig index
    covered = set(assign.tolist())
    if len(covered) < n:
        flags.append("coord-assignment-ambiguous")
        return _no_split(w, flags, gap_ratio, tau_fast)
    srt = np.sort(amp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tie = np.where(srt[:, -1] > 0, srt[:, -2] / srt[:, -1], 0.0)
    if np.any(tie > TIE_MAX):
        flags.append("coord-assignment-tie")
        return _no_split(w, flags, gap_ratio, tau_fast)

    # perm：慢坐标在前（保持原顺序）
    slow_eig_idx = np.where(slow_mask)[0]
    slow_coords = sorted(j for j in range(n) if assign[j] in set(slow_eig_idx.tolist()))
    fast_coords = [j for j in range(n) if j not in set(slow_coords)]
    perm = np.array(slow_coords + fast_coords, dtype=int)

    A_p = A[np.ix_(perm, perm)]
    B_p = B[perm, :]
    C_p = C[:, perm]
    A_ss, A_sf = A_p[:ns, :ns], A_p[:ns, ns:]
    A_fs, A_ff = A_p[ns:, :ns], A_p[ns:, ns:]
    B_s, B_f = B_p[:ns, :], B_p[ns:, :]
    C_s, C_f = C_p[:, :ns], C_p[:, ns:]

    A_eff, B_eff, C_eff, D_eff = qssa_reduce(A_ss, A_sf, A_ff, A_fs, B_s, B_f, C_s, C_f, D)

    if reason == "caller-tau_fast" and gap_ratio is not None and gap_ratio < GAP_MIN:
        flags.append("weak-gap")

    # eig_match：Schur 补对慢特征值的复现精度
    w_eff = linalg.eigvals(A_eff)
    w_slow = np.sort_complex(w[slow_mask])
    w_eff_s = np.sort_complex(w_eff)
    eig_match = float(np.max(np.abs(w_eff_s - w_slow))) if ns else 0.0

    # slaving 残差：初值放在代数流形上，积分全系统，量测 fast 坐标对代数
    # 流形的偏离（归一到慢态幅度）。量级由间隙决定——报告实测数字。
    if x0 is None:
        x0_s = np.zeros(ns)
        x0_s[0] = 1.0
    else:
        x0_s = np.asarray(x0, dtype=float).ravel()
        if x0_s.size != ns:
            raise ValueError(f"x0 长度必须 = n_slow ({ns})")
    A_ff_inv = np.linalg.inv(A_ff)
    x0_f = -A_ff_inv @ (A_fs @ x0_s)
    x0_p = np.concatenate([x0_s, x0_f])
    re_sl = np.real(w[slow_mask])
    Tmax = float(T) if T is not None else (3.0 / max(abs(re_sl)) if np.any(re_sl != 0) else 3.0)
    ts = np.linspace(0.0, Tmax, 101)
    sol_f = solve_ivp(lambda t, x: A_p @ x, (0.0, Tmax), x0_p,
                      rtol=1e-11, atol=1e-11, dense_output=True)
    X = sol_f.sol(ts)
    x_s, x_f_true = X[:ns, :], X[ns:, :]
    x_f_alg = -(A_ff_inv @ (A_fs @ x_s))
    dev = float(np.max(np.abs(x_f_true - x_f_alg)))
    scale = float(np.max(np.abs(x_s)))
    slaving_error = dev / scale if scale > 0 else (0.0 if dev == 0 else float("inf"))

    from analysis import Descriptor
    reduced = Descriptor(np.eye(ns), A_eff, B_eff, C_eff, D_eff)
    return SplitResult(
        verdict="split", n_slow=ns, n_fast=nf, gap_ratio=gap_ratio, tau_fast=tau_fast,
        slow_eigs=w[slow_mask], fast_eigs=w[fast_mask], reduced=reduced, A_eff=A_eff,
        eig_match=eig_match, slaving_error=slaving_error, flags=flags, perm=perm,
        meta=dict(split_reason=reason, T=Tmax, method="physical-coordinate Schur QSSA",
                  coords_slow=slow_coords, coords_fast=fast_coords))
