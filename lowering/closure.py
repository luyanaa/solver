"""P_N 谱/空间投影 + η_closure 证书（DESIGN §4.0 P2 投影、§5.2、§6.1 表示层对应物）。

DESIGN 的口径（§5.2，行 822-826）：

    r_closure = (I−P_N) F(u_N),    η_closure = ‖r_closure‖ / ref

ref 是**固定参考分母**：一次测好，不随 u_N 缩水，避免 ‖F‖→0 零除。缺省
ref = ‖F(u_ref)‖（u_ref = 全分辨率代表态 probe）；caller 可用 ref_scale 覆盖。
ref=0 而 defect>0 ⇒ η=inf ⇒ verdict=infeasible——不静默给 0。

两条输入路径（按 system 类型分派，其余组合 NotImplementedError，不静默回退）：

* Descriptor（线性）：模态坐标约定 k=1..n（与 discretize 的 diffusion /
  run_phase0 完全一致），P_N = 前 N 个坐标。注意：**线性对角系统在任何
  截断下都是精确闭的**——defect=0 是正确结论不是 bug；defect 只能来自
  (I−P_N) 里的 forcing（k>N 的强迫项全额进 defect）。E 非奇异才收；
  奇异 E 应先走 lowering.split（QSSA 剥离的输出正是奇异 E）。
* rhs callable + basis="fourier"：状态是周期网格上的实场；P_N 用 FFT 截断
  |k|≤N。rhs 必须在 caller 给定的全分辨率网格上无 alias 地求值（这是 rhs
  的合同，本模块不替它消 alias）。二次非线性把内容搬到 |k|≤2N：2/3 规则
  消的是 aliasing，不消 closure——defect 谱落在 (N,2N]（alias 带）的占比
  作为 alias_frac 报告；alias_frac>0.5 ⇒ verdict=change-basis（降 N 无用，
  要换/扩基）。

verdict（caller 给 eps 才判；不给 ⇒ "unjudged"）：

  feasible       η ≤ eps
  reduce-N       η > eps 且 defect 主要在 alias 带外（>2N）——加大 N 可收
  change-basis   defect 主要在 (N,2N]（alias 带）——降 N 无用，须换/扩基
  infeasible     η 非有限（ref=0 而 defect>0）
  unjudged       caller 未给 eps

（线性路径 N=n 时 (I−P_N)≡0 ⇒ η≡0，"无可降之 N"的情形不会出现；非线性
路径 N < n_full/2 恒成立——所以 change-basis 只由 alias_frac 触发。）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis import Descriptor

__all__ = ["ClosureResult", "project_spectral", "fourier_project_field"]


@dataclass
class ClosureResult:
    """P_N 投影的完整记录：defect 实测值 + verdict + 投影后的系统。"""

    N: int
    n_full: int
    basis: str
    eta_closure: float            # ‖(I−P_N)F(u_N)‖ / ref（实测，非假设）
    defect_norm: float
    ref_denominator: float        # 固定参考分母（一次测好）
    alias_frac: float             # defect 能量在 (N,2N] alias 带的占比
    N_effective: int              # 实际保留的模态数（本路径 = N）
    verdict: str                  # feasible/reduce-N/change-basis/infeasible/unjudged
    projected: object | None      # 线性路径：投影后的 Descriptor
    projected_rhs: object | None  # 非线性路径：u ↦ P_N F(u)（全长场入/出）
    defect: np.ndarray            # 全分辨率 defect 向量（线性=坐标；fourier=场）
    meta: dict = field(default_factory=dict)


def fourier_project_field(u, N: int) -> np.ndarray:
    """周期实场的 Fourier 截断：保留 |k| ≤ N（k = n·fftfreq 约定）。"""
    u = np.asarray(u, dtype=float)
    n = u.size
    U = np.fft.fft(u)
    k = np.fft.fftfreq(n) * n
    U[np.abs(k) > N + 1e-9] = 0.0
    return np.fft.ifft(U).real


def _fft_k(n: int) -> np.ndarray:
    return np.fft.fftfreq(n) * n


def _verdict(eta: float, N: int, n_full: int, alias_frac: float,
             eps: float | None) -> str:
    if eps is None:
        return "unjudged"
    if not np.isfinite(eta):
        return "infeasible"
    if eta <= eps:
        return "feasible"
    if alias_frac > 0.5:
        return "change-basis"     # 2/3 规则消 aliasing 不消 closure
    return "reduce-N"


def project_spectral(system, N: int, *, basis: str = "fourier", eps: float | None = None,
                     probe: np.ndarray | None = None, forcing: np.ndarray | None = None,
                     ref_scale: float | None = None, n_full: int | None = None) -> ClosureResult:
    """P_N 投影 + η_closure 证书。system 是 Descriptor（线性）或 rhs callable。

    probe   代表态（全长）。缺省 = 归一化全 1 向量（线性路径）/ 必填（fourier 场路径）。
    forcing 线性路径的强迫向量 f（全长；F(x) = A·x + f）。缺省 0。
    ref_scale 覆盖固定参考分母；缺省 = ‖F(u_ref)‖（u_ref = probe，一次测好）。
    n_full  fourier 场路径的网格点数（rhs 输出长度），必填。
    """
    if isinstance(system, Descriptor):
        if basis != "fourier":
            raise NotImplementedError(
                f"project_spectral: 线性 Descriptor 路径只实现了模态坐标 "
                f"k=1..n 约定（basis='fourier'），got basis={basis!r}")
        return _project_descriptor(system, int(N), eps, probe, forcing, ref_scale)
    if callable(system):
        if basis != "fourier":
            raise NotImplementedError(
                f"project_spectral: rhs 路径只实现了 basis='fourier'（周期场 FFT 截断），"
                f"got basis={basis!r}；chebyshev/fv/wavelet 基选择是 §10 的待补项")
        if n_full is None:
            raise ValueError("rhs 路径必须给 n_full（网格点数）")
        return _project_rhs(system, int(N), int(n_full), eps, probe, ref_scale)
    raise NotImplementedError(
        f"project_spectral: system 类型 {type(system).__name__} 不支持"
        f"（支持 analysis.Descriptor 或 rhs callable）")


# ---------------------------------------------------------------------------
# 线性 Descriptor 路径：模态坐标 k=1..n，P_N = 前 N 个坐标
# ---------------------------------------------------------------------------
def _project_descriptor(desc, N: int, eps, probe, forcing, ref_scale) -> ClosureResult:
    n = desc.n
    if not 1 <= N <= n:
        raise ValueError(f"N 必须在 [1, n] 内（n={n}），got N={N}")
    E = desc.E
    if not np.allclose(E, np.eye(n)):
        if abs(np.linalg.det(E)) < 1e-12 * max(1.0, np.linalg.norm(E)):
            raise NotImplementedError(
                "project_spectral: 奇异 E 的 descriptor 不做 P_N 投影——"
                "QSSA 剥离（lowering.split.cvsplit）的输出正是奇异 E；"
                "请先剥离再投影")
        A = np.linalg.solve(E, desc.A)
    else:
        A = desc.A
    f = np.zeros(n) if forcing is None else np.asarray(forcing, dtype=float).ravel()
    if f.size != n:
        raise ValueError("forcing 长度必须 = n")
    if probe is None:
        probe = np.ones(n) / np.sqrt(n)
    probe = np.asarray(probe, dtype=float).ravel()
    if probe.size != n:
        raise ValueError("probe 长度必须 = n")

    # 固定参考分母：全分辨率代表态处一次测好
    F_ref = A @ probe + f
    ref = float(ref_scale) if ref_scale is not None else float(np.linalg.norm(F_ref))

    # u_N = P_N probe；defect = (I−P_N)(A u_N + f)：保留坐标 k>N，清零 k≤N
    u_N = probe.copy()
    u_N[N:] = 0.0
    F_uN = A @ u_N + f
    defect = np.zeros(n)
    defect[N:] = F_uN[N:]
    dnorm = float(np.linalg.norm(defect))
    eta = dnorm / ref if ref > 0.0 else (0.0 if dnorm == 0.0 else float("inf"))

    # alias 带诊断只对非线性卷积有意义：线性对角系统没有 aliasing，
    # defect 全部是"_representation 缺失的强迫"——置 0，verdict 走 reduce-N
    alias_frac = 0.0

    A_N = A[:N, :N]
    projected = Descriptor(np.eye(N), A_N, desc.B[:N, :], desc.C[:, :N], desc.D)
    return ClosureResult(
        N=N, n_full=n, basis="fourier-coordinate", eta_closure=eta,
        defect_norm=dnorm, ref_denominator=ref, alias_frac=alias_frac,
        N_effective=N, verdict=_verdict(eta, N, n, alias_frac, eps),
        projected=projected, projected_rhs=None, defect=defect,
        meta=dict(kind="linear-descriptor", convention="modes k=1..n, P_N=first-N",
                  note="线性对角系统在任何截断下精确闭；defect 只来自 (I−P_N) forcing；"
                       "alias_frac≡0（aliasing 是非线性卷积概念）"))


# ---------------------------------------------------------------------------
# 非线性 rhs 路径：周期场 + FFT 截断 |k| ≤ N
# ---------------------------------------------------------------------------
def _project_rhs(rhs, N: int, n_full: int, eps, probe, ref_scale) -> ClosureResult:
    if not 1 <= N < n_full // 2:
        raise ValueError(
            f"N 必须在 [1, n_full//2) 内（n_full={n_full}，可分辨 |k|≤n_full/2−1），got N={N}")
    if probe is None:
        raise ValueError("fourier 场路径必须给 probe（全长场代表态）")
    probe = np.asarray(probe, dtype=float).ravel()
    if probe.size != n_full:
        raise ValueError("probe 长度必须 = n_full")

    F_ref = np.asarray(rhs(probe), dtype=float).ravel()
    if F_ref.size != n_full:
        raise ValueError("rhs 必须返回长度 = n_full 的场")
    ref = float(ref_scale) if ref_scale is not None else float(np.linalg.norm(F_ref))

    u_N = fourier_project_field(probe, N)
    F_uN = np.asarray(rhs(u_N), dtype=float).ravel()

    # defect = (I−P_N) F(u_N)：谱里把 |k|≤N 的部分清零
    UF = np.fft.fft(F_uN)
    k = _fft_k(n_full)
    keep = np.abs(k) <= N + 1e-9
    defect = np.fft.ifft(np.where(keep, 0.0, UF)).real
    dnorm = float(np.linalg.norm(defect))
    eta = dnorm / ref if ref > 0.0 else (0.0 if dnorm == 0.0 else float("inf"))

    # alias 带诊断：|k| ∈ (N, 2N]
    dk = np.abs(k[~keep])
    dspec = np.abs(UF[~keep]) ** 2
    band = dk <= 2 * N + 1e-9
    alias_frac = float(dspec[band].sum() / dspec.sum()) if dnorm > 0 else 0.0

    def projected_rhs(u):
        """u ↦ P_N F(u)：u 是全长场（应已含于 P_N 子空间或任意试验场）。"""
        Fu = np.asarray(rhs(np.asarray(u, dtype=float).ravel()), dtype=float).ravel()
        return fourier_project_field(Fu, N)

    return ClosureResult(
        N=N, n_full=n_full, basis="fourier-field", eta_closure=eta,
        defect_norm=dnorm, ref_denominator=ref, alias_frac=alias_frac,
        N_effective=N, verdict=_verdict(eta, N, n_full // 2, alias_frac, eps),
        projected=None, projected_rhs=projected_rhs, defect=defect,
        meta=dict(kind="nonlinear-rhs", convention="periodic field, keep |k|≤N",
                  rhs_contract="rhs 必须在全分辨率网格上无 alias 求值；"
                               "二次项把内容搬到 |k|≤2N——2/3 规则消 aliasing 不消 closure"))
