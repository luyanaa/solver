"""谱尾与 shock sensor（DESIGN.md §5.3、§6.6）。

给定一个状态的 Fourier 系数（正 k 部分，k=1..N/2），返回两个数：

    p      = -d log|a_k| / d log k   （谱尾指数；代数衰减 p 小 ⇒ 正则性差）
    E_tail = Σ_{tail}|a_k|² / Σ|a_k|²

Phase 0 只用静态版本做表示轴判定；速率检测（dp/dt 提前切换）属于 Phase 4。
阈值是验收参数，不是自然常数：E_tail_req=1e-4 对应 ~1% L2 表示误差。
"""

from __future__ import annotations

import numpy as np

__all__ = ["tail_exponent", "energy_tail", "fourier_metrics", "shock_sensor"]


def tail_exponent(coeffs, frac: float = 0.5) -> float:
    a = np.abs(np.asarray(coeffs, dtype=float)).ravel()
    n = a.size
    if n < 4:
        return float("nan")
    i0 = max(0, int(n * (1.0 - frac)))
    k = np.arange(i0 + 1, n + 1, dtype=float)
    y = a[i0:]
    if np.all(y <= 0) or y.max() <= 1e-14 * (a.max() + 1e-300):
        # 单音/机器精度尾部：没有可拟合的代数尾，视为已解析
        return float("inf")
    with np.errstate(divide="ignore"):
        p = -np.polyfit(np.log(k), np.log(y + 1e-300), 1)[0]
    return float(p)


def energy_tail(coeffs, frac: float = 0.5) -> float:
    a = np.abs(np.asarray(coeffs, dtype=float)).ravel()
    if a.size == 0:
        return float("nan")
    e = a * a
    tot = float(e.sum())
    if tot <= 0:
        return float("nan")
    i0 = max(0, int(a.size * (1.0 - frac)))
    return float(e[i0:].sum() / tot)


def fourier_metrics(u) -> dict:
    """周期网格上的状态 → (p, E_tail, 正 k 系数)。去掉均值模态。"""
    u = np.asarray(u, dtype=float)
    a = np.fft.rfft(u - u.mean())
    c = np.abs(a[1:])          # k = 1 .. N/2
    return dict(p=tail_exponent(c), E_tail=energy_tail(c), coeffs=c)


def shock_sensor(u, p_thresh: float = 1.5, etail_req: float = 1e-4) -> dict:
    """静态 shock/basis-switch 判定。

    返回 representation = max(E_tail/etail_req, p_thresh/p)；>1 表示当前谱基
    装不下这个状态（要么尾部能量太大，要么代数衰减太慢）。
    """
    m = fourier_metrics(u)
    p, e = m["p"], m["E_tail"]
    rep_e = e / etail_req if np.isfinite(e) else float("nan")
    rep_p = p_thresh / p if np.isfinite(p) and p > 0 else 0.0
    rep = max(rep_e, rep_p) if np.isfinite(rep_e) else rep_p
    m.update(p_thresh=p_thresh, etail_req=etail_req,
             representation=float(rep), shock=bool(rep > 1.0))
    return m
