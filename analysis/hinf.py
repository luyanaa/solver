"""H∞ 范数：Hamiltonian 二分 + 扫频交叉验证（DESIGN.md §0、§7 Phase 0）。

    γ > ‖H‖∞  ⇔  Hamiltonian 矩阵/束在虚轴上没有特征值。

对 E 奇异或含守恒零模态的系统，虚轴模态可能是不可控的（B 与左零空间正交），
此时 Hamiltonian 判据会误报。实现上先检查 has_imaginary()；有虚轴模态就退回
扫频，并明确标注 method="sweep"。两种方法都保留在返回值里，方便验收时对比。
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .descriptor import Descriptor

__all__ = ["hinf_norm"]


def _sweep(sys: Descriptor, n_grid: int = 1600) -> float:
    # n>256 只做交叉验证，粗网格足够；1024 模态的 O(n) 频响仍要几十秒。
    n_grid = 1600 if sys.n <= 256 else 400
    scale = max(1.0, sys.spectral_radius())
    w = np.logspace(-10.0, 10.0, n_grid) * scale
    H = sys.freqresp(w)
    if H.shape[2] == 1:
        smax = np.linalg.norm(H[:, :, 0], axis=1)
    else:
        smax = np.linalg.svd(H, compute_uv=False)[:, 0]
    smax = np.atleast_1d(smax)
    i = int(np.argmax(smax))
    # 局部加密：在相邻两个网格点之间再取 200 个点
    lo = np.log10(w[max(i - 1, 0)])
    hi = np.log10(w[min(i + 1, w.size - 1)])
    wl = np.logspace(lo, hi, 200)
    Hl = sys.freqresp(wl)
    if Hl.shape[2] == 1:
        sl = np.linalg.norm(Hl[:, :, 0], axis=1)
    else:
        sl = np.linalg.svd(Hl, compute_uv=False)[:, 0]
    sl = np.atleast_1d(sl)
    return float(max(smax[i], sl.max()))


def _ok(sys: Descriptor, gamma: float, tol: float = 1e-9) -> bool:
    """γ 是否严格大于 ‖H‖∞（Hamiltonian 无虚轴特征值）。"""
    if gamma <= 0:
        return False
    g2 = gamma * gamma
    if np.linalg.cond(sys.E) < 1e10:
        As = linalg.solve(sys.E, sys.A)
        Bs = linalg.solve(sys.E, sys.B)
        M = np.block([[As, (Bs @ Bs.T) / g2],
                      [-(sys.C.T @ sys.C), -As.T]])
        ev = np.linalg.eigvals(M)
    else:
        Hn = np.block([[sys.A, (sys.B @ sys.B.T) / g2],
                       [-(sys.C.T @ sys.C), -sys.A.T]])
        Hd = np.block([[sys.E, np.zeros_like(sys.E)],
                       [np.zeros_like(sys.E), sys.E.T]])
        ev = linalg.eigvals(Hn, Hd)
        ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(ev))))
    return bool(np.min(np.abs(np.real(ev))) > tol * scale)


def _hamiltonian(sys: Descriptor, upper: float, tol: float = 1e-6) -> float | None:
    if sys.has_imaginary():
        return None
    hi = max(2.0 * upper, 1e-12)
    for _ in range(30):
        if _ok(sys, hi):
            break
        hi *= 2.0
    else:
        return None
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _ok(sys, mid):
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * hi:
            break
    return float(hi)


def hinf_norm(sys: Descriptor, tol: float = 1e-6, max_hamiltonian_n: int = 256) -> dict:
    """返回 {value, method, sweep, rel_diff}。

    Hamiltonian 是 2n×2n 稠密特征值；n>256 时改用扫频（§7 Phase 0 只承诺
    n≤200 的 ms 级）。两种方法的数值都留在返回值里。
    """
    if np.any(sys.D):
        raise NotImplementedError("hinf_norm 目前只支持 D=0（§7 Phase 0 的所有算例都满足）")
    w = sys.finite_eigs()
    if w.size and np.any(np.real(w) > 1e-10 * max(1.0, float(np.max(np.abs(w))))):
        return dict(value=float("inf"), method="unstable", sweep=float("inf"), rel_diff=0.0)
    sweep = _sweep(sys)
    if sys.n > max_hamiltonian_n or sys.has_imaginary():
        return dict(value=sweep, method="sweep", sweep=sweep, rel_diff=0.0)
    ham = _hamiltonian(sys, sweep, tol=tol)
    if ham is None:
        return dict(value=sweep, method="sweep", sweep=sweep, rel_diff=0.0)
    return dict(value=ham, method="hamiltonian", sweep=sweep,
                rel_diff=float(abs(ham - sweep) / max(abs(sweep), 1e-300)))
