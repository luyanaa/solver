"""QoI 提取：把"几阶"换成可测量的输运系数（DESIGN.md §6.1）。

本模块只做测量/反演，不含任何求解器 —— 这样它既能用于数字孪生，
也能直接接实测波形（示波器采样 / VCD）。

约定
----
激波：Burgers 型稳态黏性剖面

    u(x) = m + a·tanh(β(x − x_s)),    a = Δu/2,  β = Δu/(4ν)

10–90% 宽度  w = 2·artanh(0.8)/β = 2.1972/β
反演         ν_num = Δu·w / 8.789 = Δu·W_d·Δx / 8.789

锋面：erf 型扩散剖面 c = ½(1+erf(z/√(4κt)))

    w_10_90 = 3.6248·√(κt)   ⟹   κ = w²/(13.139·t)

两个常数都是从解析解推出来的，不是拟合出来的；因此估计器的误差只来自
"真实剖面是否接近这两个族"，这一点在 run_twin_experiments 里单独验证。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "ATANH08", "C_ERF",
    "taylor_shock", "fit_tanh", "width_10_90",
    "nu_from_beta", "nu_from_width", "width_from_nu",
    "front_width_10_90", "kappa_from_width", "kappa_from_width_growth",
    "xs_metrics",
]

# 2·artanh(0.8)：tanh 剖面 10–90% 的跨度系数
ATANH08 = 2.0 * np.arctanh(0.8)          # ≈ 2.19722
# erf 族：w² = C_ERF·κ·t，(2·0.90619)²·4
C_ERF = (2.0 * 0.9061938024368232) ** 2 * 4.0   # ≈ 13.1394


# --------------------------------------------------------------------------
# 激波
# --------------------------------------------------------------------------
def taylor_shock(x, *, xs: float, du: float, nu: float, s: float = 0.0):
    """解析黏性激波剖面（用于生成 ground truth 和标定）。"""
    return s - 0.5 * du * np.tanh(du * (x - xs) / (4.0 * nu))


def width_10_90(beta: float) -> float:
    """tanh(β(x−xs)) 剖面的 10–90% 宽度。"""
    return ATANH08 / beta


def nu_from_beta(beta: float, du: float) -> float:
    return du / (4.0 * beta)


def nu_from_width(w: float, du: float, dx: float | None = None) -> float:
    """宽度 → 等效黏性。dx 给出时 w 按 cell 数解释。"""
    w = w * dx if dx is not None else w
    return du * w / (2.0 * ATANH08)


def width_from_nu(nu: float, du: float, dx: float | None = None):
    """等效黏性 → 宽度。返回 (物理宽度, cell 数或 None)。"""
    w = 2.0 * ATANH08 * nu / du
    return (w, w / dx) if dx is not None else w


def fit_tanh(x, u, *, x_window: float | None = None):
    """在窗口内拟合 u = m + a·tanh(β(x−x_s))，返回 QoI 字典。

    返回键：xs, xs_err, du, beta, nu_num, w_10_90, w_cells, rms, ok
    """
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    if x_window is not None:
        i0 = int(np.searchsorted(x, x.min() + x_window))
        i1 = int(np.searchsorted(x, x.max() - x_window))
        x, u = x[i0:i1], u[i0:i1]
    if x.size < 6:
        raise ValueError("窗口内点数不足，无法拟合")

    m0 = 0.5 * (u.max() + u.min())
    a0 = 0.5 * (u.max() - u.min())
    i_s = int(np.argmax(np.abs(np.gradient(u, x))))
    xs0 = float(x[i_s])
    span = x.max() - x.min()
    b0 = 4.0 / max(span, 1e-12)

    def model(xx, m, a, b, xs):
        return m + a * np.tanh(b * (xx - xs))

    try:
        popt, pcov = curve_fit(
            model, x, u, p0=[m0, a0, b0, xs0],
            bounds=([-np.inf, -np.inf, 1e-9, x.min() - span],
                    [np.inf, np.inf, 1e6, x.max() + span]),
            maxfev=20000,
        )
        ok = True
    except Exception:
        popt = np.array([m0, a0, b0, xs0], float)
        pcov = np.full((4, 4), np.nan)
        ok = False

    m, a, b, xs = popt
    du = 2.0 * abs(a)
    beta = abs(b)
    w = width_10_90(beta)
    rms = float(np.sqrt(np.mean((model(x, *popt) - u) ** 2)))
    return dict(
        xs=float(xs), xs_err=float(np.sqrt(abs(pcov[3, 3]))) if ok else np.nan,
        du=float(du), beta=float(beta),
        nu_num=float(nu_from_beta(beta, du)),
        w_10_90=float(w), w_cells=np.nan, rms=rms, ok=ok,
        n_fit=int(x.size),
    )


# --------------------------------------------------------------------------
# 锋面 / 混合
# --------------------------------------------------------------------------
def front_width_10_90(z, c):
    """任意单调剖面的 10–90% 宽度（线性插值交点）。"""
    z = np.asarray(z, float)
    c = np.asarray(c, float)
    if c[0] > c[-1]:
        z, c = z[::-1], c[::-1]
    lo, hi = c.min(), c.max()
    if hi - lo < 1e-14:
        return np.nan
    cn = (c - lo) / (hi - lo)

    def cross(level):
        i = int(np.argmax(cn >= level))
        if i == 0:
            return float(z[0])
        c0, c1 = cn[i - 1], cn[i]
        if c1 <= c0:
            return float(z[i])
        return float(z[i - 1] + (level - c0) / (c1 - c0) * (z[i] - z[i - 1]))

    return abs(cross(0.9) - cross(0.1))


def kappa_from_width(w: float, t: float) -> float:
    """单时刻：erf 族下 κ = w²/(C_ERF·t)。"""
    return w * w / (C_ERF * t)


def kappa_from_width_growth(w, t) -> float:
    """多时刻：对 w² 线性回归取斜率，κ = slope/C_ERF（对一般增长更稳）。"""
    w = np.asarray(w, float)
    t = np.asarray(t, float)
    A = np.vstack([np.ones_like(t), t]).T
    (_, slope), *_ = np.linalg.lstsq(A, w * w, rcond=None)
    return float(slope / C_ERF)


# --------------------------------------------------------------------------
# 界面轨迹
# --------------------------------------------------------------------------
def xs_metrics(xs, xs_ref, t=None):
    """x_s(t) 与参考轨迹的误差：max / L2 / 漂移率。"""
    xs = np.asarray(xs, float)
    xs_ref = np.asarray(xs_ref, float)
    e = xs - xs_ref
    out = dict(max_abs=float(np.max(np.abs(e))),
               l2=float(np.sqrt(np.mean(e ** 2))))
    if t is not None:
        t = np.asarray(t, float)
        A = np.vstack([np.ones_like(t), t]).T
        (_, rate), *_ = np.linalg.lstsq(A, e, rcond=None)
        out["drift_rate"] = float(rate)
    return out
