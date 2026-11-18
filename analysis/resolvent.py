"""Resolvent 峰值 Γ = 1 / min_ω σ_min(jωE − A)（DESIGN.md §0、§5.1、§5.8）。

Γ 是"最坏情况下一个扰动被放大多少"的线性化度量，也是换基条件数税的来源。
对角系统有闭式解；非正规系统在 log 网格上扫 σ_min，再局部加密。

度量（§5.8）：缺省的 σ_min 是裸 Euclidean 范数。当物理内积是能量度量
M = diag(C)（per-node 电容，如 mode-C wavelet scaling 里 C_j 跨 1024×）
时，裸 σ_min 把电容尺度混进 Γ。M 度量的 resolvent 范数是

    ‖(jωE−A)⁻¹‖_M = σ_max(M^{1/2} (jωE−A)⁻¹ M^{−1/2})

这与裸定义一致：Γ = max_ω ‖(jωE−A)⁻¹‖ = 1/min_ω σ_min(jωE−A)（σ_min(T) =
1/‖T⁻¹‖）。所以 M 度量版本是 Γ_M = max_ω ‖T⁻¹‖_M，等价于
1/min_ω σ_min^M(T)，其中 σ_min^M(T) = 1/‖T⁻¹‖_M。注意这不是
σ_min(M^{1/2}(jωE−A)M^{−1/2})——被求逆的是整个 resolvent，不是它的
系数矩阵。已知一致性事项：slack 的
robustness_A 轴最终应改用 M 度量的 Γ（per-node C 进 IR 之后），目前
slack.py 仍用裸 Γ，属于 owner scope，本文件不重接。
"""

from __future__ import annotations

import numpy as np

from .descriptor import Descriptor

__all__ = ["resolvent_peak", "peak_with_projection"]


def _metric_matrix(sys: Descriptor, metric) -> np.ndarray:
    """metric → 对称正定矩阵 M。"energy" 是 diag(diag(E)) 的简写。"""
    if isinstance(metric, str):
        if metric != "energy":
            raise ValueError("metric 简写只支持 'energy'")
        d = np.diag(sys.E).astype(float)
        off = float(np.max(np.abs(sys.E - np.diag(d))))
        if off > 1e-9 * max(1.0, float(np.max(np.abs(d)))):
            raise ValueError("'energy' 简写要求 E 对角占优；请显式传 M")
        return np.diag(d)
    M = np.asarray(metric, dtype=float)
    if M.shape != sys.E.shape:
        raise ValueError("metric 形状必须与 E 相同")
    if not np.allclose(M, M.T):
        raise ValueError("metric 必须对称")
    return M


def _sqrtm_pd(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对称正定 M 的 (M^{1/2}, M^{−1/2})，用特征分解。"""
    w, Q = np.linalg.eigh(M)
    if w[0] <= 0.0:
        raise ValueError("metric 必须正定")
    r = np.sqrt(w)
    return (Q * r) @ Q.T, (Q / r) @ Q.T


def _smax_M(T: np.ndarray, Mh: np.ndarray, Mhi: np.ndarray) -> float:
    """‖T⁻¹‖_M = σ_max(M^{1/2} T⁻¹ M^{−1/2})；T = jωE−A。"""
    B = Mh @ np.linalg.solve(T, Mhi)
    e = np.linalg.eigvalsh(B.conj().T @ B)
    return float(np.sqrt(max(e[-1], 0.0)))


def resolvent_peak(sys: Descriptor, n_grid: int = 1600,
                   metric=None) -> dict:
    """Γ；metric=None 保持裸 Euclidean（向后兼容），否则用 M 度量。"""
    if metric is not None:
        M = _metric_matrix(sys, metric)
        Mh, Mhi = _sqrtm_pd(M)
        scale = max(1.0, sys.spectral_radius())
        n_grid = 1600 if sys.n <= 64 else 400
        w = np.logspace(-8.0, 8.0, n_grid) * scale
        smax = np.array([_smax_M(1j * wi * sys.E - sys.A, Mh, Mhi) for wi in w])
        i = int(np.argmax(smax))
        lo = np.log10(w[max(i - 1, 0)])
        hi = np.log10(w[min(i + 1, w.size - 1)])
        wl = np.logspace(lo, hi, 80)
        sl = np.array([_smax_M(1j * wi * sys.E - sys.A, Mh, Mhi) for wi in wl])
        j = int(np.argmax(sl))
        s_best = float(max(sl[j], smax[i]))
        w_best = float(wl[j]) if sl[j] >= smax[i] else float(w[i])
        if s_best < 1e-14 * scale:
            return dict(Gamma=float("inf"), omega=w_best, sigma_min=s_best,
                        method="sweep-M")
        # Γ_M = max_ω ‖(jωE−A)⁻¹‖_M —— s_best 已经是 resolvent 的 σ_max，
        # 不再取倒数（裸分支的 1/σ_min 是因为 σ_min(T)=1/‖T⁻¹‖，
        # 度量路径直接算的是 ‖T⁻¹‖_M，两个方向的"被求逆"不同）。
        return dict(Gamma=float(s_best), omega=w_best,
                    sigma_min=float(1.0 / s_best), method="sweep-M")

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


def peak_with_projection(sys: Descriptor, n_grid: int = 1600,
                         metric=None) -> dict:
    """全 Γ 作诊断，Γ_proj 去掉零/虚轴模态后进 slack。

    守恒零模态（E=I、A 奇异）让全 Γ 在 s=0 处发散，但它通常是不可观测/
    不可控的；把 B、C 投影到稳定有限子空间后再算 Γ，才是失配鲁棒性的度量。
    没有虚轴模态时两者相同，不付 QZ 的代价。metric 透传给两处 resolvent_peak；
    投影路径的 metric 先在全系统上求 M，再用 Z1ᵀ M Z1 投到降阶子空间：
    直接把全维 M 传给降阶系统会形状不匹配（k×k 的 E），把 "energy" 透传则会
    在 k×k 非对角的降阶 E 上触发对角占优检查错误。注意：slack 的
    robustness_A 轴目前仍消费裸 Γ，M 度量的接入是已知
    一致性事项（per-node C 进 IR 之后，owner scope）。
    """
    out = dict(resolvent_peak(sys, n_grid, metric))
    out.update(Gamma_proj=out["Gamma"], n_proj=sys.n, projected=False)
    if not sys.has_imaginary():
        return out
    red, k, Z1 = sys._stable_projection_z1()
    if red is None:
        out.update(Gamma_proj=float("nan"), n_proj=0, projected=True)
        return out
    metric_red = metric
    if metric is not None:
        # 度量随子空间投影：M_red = Z1ᵀ M Z1（Z1 列正交 ⇒ M_red 仍对称正定）。
        M = _metric_matrix(sys, metric)
        metric_red = Z1.T @ M @ Z1
    pj = resolvent_peak(red, n_grid, metric_red)
    out.update(Gamma_proj=pj["Gamma"], omega_proj=pj["omega"],
               sigma_min_proj=pj["sigma_min"], n_proj=k, projected=True)
    return out
