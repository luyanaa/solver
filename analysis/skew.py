"""互连延迟的一阶效应：DDE → descriptor with mass matrix（DESIGN.md §6.14）。

对 ẋ = A x，若 x_j 到 i 的路径延迟为 τ_ij：

    ẋ_i(t) = Σ_j A_ij x_j(t − τ_ij)

对 ωτ ≪ 1：

    x_j(t − τ_ij) ≈ x_j(t) − τ_ij ẋ_j(t)

    ẋ_i(t) ≈ Σ_j A_ij x_j(t) − Σ_j A_ij τ_ij ẋ_j(t)

    [I + (A ∘ T)] ẋ = A x

所以 delay 不是“结果晚了几 ps”：它改的是 descriptor 的 E。这个 E 可以直接
喂给 analysis/descriptor.py，算 κ、Γ、H∞；它也可以给出一个一阶稳定边界

    τ_crit = min_i(−Re λ_i / |λ_i|²)

但一阶模型只在 ωτ ≪ 1 时可信。标量 DDE 的精确边界是 |a|τ = π/2，而
E=I+τA 给出 |a|τ=1，乐观 2/π ≈ 0.637 倍。这个差就是 §2.8 里
“ω_max Δτ_route ≪ 1”不能省的原因。
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .descriptor import Descriptor
from .resolvent import peak_with_projection

__all__ = [
    "delay_mass_matrix", "skew_descriptor", "uniform_delay_poles",
    "dde_characteristic_roots", "scalar_dde_root", "uniform_tau_crit",
    "scalar_tau_crit", "phase_budget", "skew_report",
]


def _as_matrix(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")
    return A


def delay_mass_matrix(A, tau) -> np.ndarray:
    """E = I + A ∘ T；tau 可以是标量（uniform）或与 A 同形。"""
    A = _as_matrix(A)
    tau = np.asarray(tau, dtype=float)
    if tau.ndim == 0:
        tau = np.full_like(A, float(tau))
    if tau.shape != A.shape:
        raise ValueError("tau 必须是标量或与 A 同形")
    if np.any(tau < 0):
        raise ValueError("delay 必须非负")
    return np.eye(A.shape[0]) + A * tau


def skew_descriptor(A, tau, B=None, C=None) -> Descriptor:
    A = _as_matrix(A)
    return Descriptor(delay_mass_matrix(A, tau), A, B, C)


def uniform_delay_poles(A, tau0) -> np.ndarray:
    """Uniform τ：E=I+τA 与 A 可交换，广义特征值有闭式 λ_k = λ0/(1+τλ0)。"""
    A = _as_matrix(A)
    w = np.linalg.eigvals(A)
    return w / (1.0 + float(tau0) * w)


def dde_characteristic_roots(A, tau0, guesses=None) -> np.ndarray:
    """解 det(sI − A e^{−sτ}) = 0，取离 guesses 最近的根。

    只用于小维数（n≤4）和 τ 不大的情况；它是验证一阶展开的参照，
    不是通用 DDE 求解器。
    """
    A = _as_matrix(A)
    n = A.shape[0]
    if n > 8:
        raise ValueError("dde_characteristic_roots 只用于小矩阵")
    if guesses is None:
        guesses = uniform_delay_poles(A, tau0)
    out = []
    for g in np.atleast_1d(guesses):
        def f(z):
            s = z[0] + 1j * z[1]
            d = np.linalg.det(s * np.eye(n) - A * np.exp(-s * float(tau0)))
            return np.array([d.real, d.imag])

        sol = optimize.root(f, np.array([g.real, g.imag]), method="hybr")
        out.append(sol.x[0] + 1j * sol.x[1] if sol.success else np.nan + 0j)
    return np.asarray(out)


def scalar_dde_root(a, tau, guess=None) -> complex:
    """解 s − a e^{−sτ}=0（标量 DDE 的特征根）。"""
    a = complex(a)
    tau = float(tau)
    if guess is None:
        guess = a / (1.0 + tau * a)

    def f(z):
        s = z[0] + 1j * z[1]
        d = s - a * np.exp(-s * tau)
        return np.array([d.real, d.imag])

    sol = optimize.root(f, np.array([guess.real, guess.imag]), method="hybr")
    return complex(sol.x[0], sol.x[1]) if sol.success else complex(np.nan, np.nan)


def uniform_tau_crit(A) -> float:
    """一阶 E=I+τA 的均匀延迟稳定边界：min_i(−Re λ_i/|λ_i|²)。

    只在均匀 τ 下成立；非均匀 τ 的 E 不再与 A 可交换，必须直接扫广义特征值。
    """
    w = np.linalg.eigvals(_as_matrix(A))
    vals = [-lam.real / abs(lam) ** 2 for lam in w if abs(lam) > 0]
    return float(min(vals)) if vals else float("inf")


def scalar_tau_crit(a) -> float:
    """标量 DDE ẋ = a x(t−τ) 的精确稳定边界：|a|τ = π/2（a<0）。"""
    a = float(a)
    if a >= 0:
        return 0.0
    return float(np.pi / (2.0 * abs(a)))


def phase_budget(omega_max, phi_budget_deg) -> float:
    """ω_max τ < φ_budget ⇒ τ_max = φ_budget/ω_max（φ 用角度，输出秒）。"""
    return float(np.deg2rad(float(phi_budget_deg)) / float(omega_max))


def skew_report(A, tau, *, n_grid: int = 400) -> dict:
    """比较理想 ẋ=Ax 与一阶延迟 descriptor Eẋ=Ax。"""
    A = _as_matrix(A)
    ideal = Descriptor(np.eye(A.shape[0]), A)
    skew = skew_descriptor(A, tau)
    w0 = ideal.finite_eigs()
    w1 = skew.finite_eigs()

    shifts = []
    for w in w0:
        if w1.size == 0:
            shifts.append(float("inf"))
            continue
        j = int(np.argmin(np.abs(w1 - w)))
        shifts.append(abs(w1[j] - w) / max(abs(w), 1e-300))

    stable0 = ideal.is_stable()
    stable1 = skew.is_stable()
    if stable0 and A.shape[0] <= 64:
        gamma0 = peak_with_projection(ideal, n_grid)["Gamma_proj"]
    else:
        gamma0 = float("inf") if not stable0 else float("nan")
    if stable1 and A.shape[0] <= 64:
        gamma1 = peak_with_projection(skew, n_grid)["Gamma_proj"]
    else:
        gamma1 = float("inf") if not stable1 else float("nan")

    return dict(
        n=A.shape[0],
        tau_max=float(np.max(np.asarray(tau, dtype=float))),
        E=np.asarray(skew.E, dtype=float),
        E_cond=float(np.linalg.cond(skew.E)),
        eig_ideal=w0,
        eig_skew=w1,
        max_rel_shift=float(max(shifts)) if shifts else 0.0,
        kappa_ideal=ideal.kappa(),
        kappa_skew=skew.kappa(),
        gamma_ideal=float(gamma0),
        gamma_skew=float(gamma1),
        stable_ideal=bool(stable0),
        stable_skew=bool(stable1),
        tau_crit_first_order=uniform_tau_crit(A),
    )
