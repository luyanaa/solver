"""Resolvent 峰值 Γ = 1 / min_ω σ_min(jωE − A)（DESIGN.md §0、§5.1）。

Γ 是"最坏情况下一个扰动被放大多少"的线性化度量，也是换基条件数税的来源。
对角系统有闭式解；非正规系统在 log 网格上扫 σ_min，再局部加密。
"""

from __future__ import annotations

import numpy as np

from .descriptor import Descriptor

__all__ = ["resolvent_peak", "peak_with_projection"]


def resolvent_peak(sys: Descriptor, n_grid: int = 1600) -> dict:
    if sys._diag is not None:
        d = np.abs(sys._diag)
        d = d[d > 0]
        if d.size == 0:
            return dict(Gamma=float("inf"), omega=0.0, sigma_min=0.0, method="diag")
        return dict(Gamma=float(1.0 / d.min()), omega=0.0,
                    sigma_min=float(d.min()), method="diag")

    scale = max(1.0, sys.spectral_radius())
    n_grid = 1600 if sys.n <= 64 else 400
    w = np.logspace(-8.0, 8.0, n_grid) * scale
    smin = sys.sigma_min(w)
    i = int(np.argmin(smin))
    lo = np.log10(w[max(i - 1, 0)])
    hi = np.log10(w[min(i + 1, w.size - 1)])
    wl = np.logspace(lo, hi, 80)
    sl = sys.sigma_min(wl)
    j = int(np.argmin(sl))
    if sl[j] <= smin[i]:
        w_best, s_best = float(wl[j]), float(sl[j])
    else:
        w_best, s_best = float(w[i]), float(smin[i])
    if s_best < 1e-14 * scale:
        return dict(Gamma=float("inf"), omega=w_best, sigma_min=s_best, method="sweep")
    return dict(Gamma=float(1.0 / s_best), omega=w_best, sigma_min=s_best, method="sweep")


def peak_with_projection(sys: Descriptor, n_grid: int = 1600) -> dict:
    """全 Γ 作诊断，Γ_proj 去掉零/虚轴模态后进 slack。

    守恒零模态（E=I、A 奇异）让全 Γ 在 s=0 处发散，但它通常是不可观测/
    不可控的；把 B、C 投影到稳定有限子空间后再算 Γ，才是失配鲁棒性的度量。
    没有虚轴模态时两者相同，不付 QZ 的代价。
    """
    out = dict(resolvent_peak(sys, n_grid))
    out.update(Gamma_proj=out["Gamma"], n_proj=sys.n, projected=False)
    if not sys.has_imaginary():
        return out
    red, k = sys.stable_projection()
    if red is None:
        out.update(Gamma_proj=float("nan"), n_proj=0, projected=True)
        return out
    pj = resolvent_peak(red, n_grid)
    out.update(Gamma_proj=pj["Gamma"], omega_proj=pj["omega"],
               sigma_min_proj=pj["sigma_min"], n_proj=k, projected=True)
    return out
