"""基选择：稀疏度 vs κ(Φ) 税（DESIGN.md §5.1、§8 shear 行）。

问题（§5.1）：换基 z = T⁻¹x 同时改 Γ 和条件数税。实测先例
（`bench/run_stagea.py` CASE 1 / §8 shear 行）：非正规 shear
A=[[-1,10],[0,-2]] 上 Γ=5.12 来自非对角耦合 10，不是谱；

  - Gramian 对角 scaling（P3 预条件）：κ 不变，本模块 model-time 口径下
    与恒等基打平（stagea 报的 Γ=3.98e-8 是 **物理时间** α-scaled 量，
    时间缩放属于 `scaling.py`，不在换基的账里）；
  - 特征向量基：Γ 降到对角值 1.000，但 κ(Φ)=cond(V)=20.05，
    失配总预算 Γ·κ(Φ)=20.05 比原基 5.12 **更差**——
    "禁止病态特征基、用块正交基" 是定量结论，不是偏好。

总预算口径（本模块的判定量）：

    budget(T) = Γ(T⁻¹AT) · amplification(T)

amplification 是"物理扰动穿过基的放大"：

  - delta_pert=None（最坏情况）：amplification = cond(T)。
    此时有不变式 ‖R_x‖ ≤ cond(T)·‖R_z‖（R = (jωE−A)⁻¹），因此
    **纯相似变换候选的总预算永远 ≥ 恒等基**——特征向量基在最坏
    情况下几乎从不赢；这正是 §8 行的定量内容。
  - delta_pert 给定（矩阵 Δ 或标量 ε⇒Δ=ε·A）：amplification =
    ‖T⁻¹ΔT‖/‖Δ‖。扰动结构已知时换基可以赢（ε·A 型系数漂移在
    模态坐标里被对角化，amp=‖Λ‖/‖A‖<1）——chooser 的判定依赖
    失配模型，这就是 delta_pert 必须显式进 API 的原因。

候选集（缺省）："identity" / "schur" / "gramian" / "pod" / "eigenvector"。
记号约定与 run_stagea 一致：x = T z（T 是模态→物理映射），
A_z = T⁻¹AT，B_z = T⁻¹B，C_z = CT，税 κ(Φ)=cond(T)。

已知限制：schur 候选只在 E≈I 时可用（QZ 广义 Schur 的 Z 不给
相似变换）；pod 候选用 Moore 平衡对角 d_i=(P_ii·Wo_ii)^{1/4}
（Gramian 双对角 = 无快照数据的 POD 代理；数据驱动 POD 需要
snapshot，超出 descriptor API）。退化的对角项（不可控/不可观
态，P_ii≈0）用正项几何均值做 floor——shear 的 x_2 就不可控，
不 floor 会得到 cond=1e300 的伪候选。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from analysis.descriptor import Descriptor
from analysis.lyapunov import solve_generalized_lyapunov
from analysis.resolvent import resolvent_peak

__all__ = ["BasisCandidate", "BasisResult", "select_basis",
           "DEFAULT_CANDIDATES"]

DEFAULT_CANDIDATES = ("identity", "schur", "gramian", "pod", "eigenvector")

_DEGEN_FRAC = 1e-12


# ---------------------------------------------------------------------------
# 记录容器
# ---------------------------------------------------------------------------
@dataclass
class BasisCandidate:
    """单个候选基的记分卡。"""

    name: str
    Gamma: float                     # Γ(T⁻¹AT)（model time，resolvent peak）
    cond_phi: float                  # κ(Φ) = cond(T)
    nnz: int                         # T 的非零元个数（稀疏度）
    tinv_norm: float                 # ‖T⁻¹‖₂
    amplification: float             # 扰动穿过基的放大（cond(T) 或 ‖T⁻¹ΔT‖/‖Δ‖）
    total_budget: float              # Γ · amplification —— 判定量
    selected: bool = False
    T: np.ndarray | None = None
    transformed: Descriptor | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return dict(
            name=self.name, Gamma=self.Gamma, cond_phi=self.cond_phi,
            nnz=int(self.nnz), tinv_norm=self.tinv_norm,
            amplification=self.amplification, total_budget=self.total_budget,
            selected=self.selected, note=self.note,
        )


@dataclass
class BasisResult:
    """select_basis 的输出：全候选记分卡 + 赢家 + 变换后描述符。"""

    winner: str | None
    candidates: dict = field(default_factory=dict)   # name -> BasisCandidate
    transformed: Descriptor | None = None            # 赢家的 T⁻¹(E,A,B,C)T
    T: np.ndarray | None = None                      # 赢家的基矩阵（x = T z）
    delta_pert: str = "worst-case (cond(T))"         # 失配模型记录
    reason: str = ""

    def as_dict(self) -> dict:
        return dict(
            winner=self.winner,
            candidates={k: v.as_dict() for k, v in self.candidates.items()},
            delta_pert=self.delta_pert, reason=self.reason,
        )


# ---------------------------------------------------------------------------
# 内部：变换 / 记分
# ---------------------------------------------------------------------------
def _similarity(sys: Descriptor, T: np.ndarray) -> Descriptor:
    """x = T z（T: 模态→物理），z 描述符 = (T⁻¹ET, T⁻¹AT, T⁻¹B, CT)。"""
    Tinv = np.linalg.inv(T)
    return Descriptor(Tinv @ sys.E @ T, Tinv @ sys.A @ T,
                      Tinv @ sys.B, sys.C @ T, sys.D)


def _gamma_of(sys: Descriptor) -> float:
    r = resolvent_peak(sys)
    g = float(r["Gamma"])
    return g if np.isfinite(g) else float("inf")


def _amplification(T: np.ndarray, delta: np.ndarray | None) -> float:
    Tinv = np.linalg.inv(T)
    if delta is None:
        return float(np.linalg.cond(T))
    return float(np.linalg.norm(Tinv @ delta @ T) / np.linalg.norm(delta))


def _diag_candidate(sys: Descriptor, sqrt_diag: np.ndarray,
                    name: str, delta: np.ndarray | None = None,
                    note: str = "") -> BasisCandidate | None:
    """对角基 D=diag(s)：退化（≈0）项 floor 到正项几何均值。"""
    p = np.clip(sqrt_diag, 0.0, None)
    good = p[p > _DEGEN_FRAC * max(1.0, float(p.max(initial=0.0)))]
    if good.size == 0:
        return None
    floor = float(np.exp(np.mean(np.log(good))))
    s = np.where(p > _DEGEN_FRAC * max(1.0, float(p.max(initial=0.0))),
                 p, floor)
    T = np.diag(float(1.0 / 3.0) / s)     # 与 scaling.py rms3σ 同款 3σ 满摆
    return _score(sys, T, name, delta, note=note)


def _score(sys: Descriptor, T: np.ndarray, name: str,
           delta: np.ndarray | None, note: str = "") -> BasisCandidate:
    d = _similarity(sys, T)
    g = _gamma_of(d)
    amp = _amplification(T, delta)
    return BasisCandidate(
        name=name, Gamma=g, cond_phi=float(np.linalg.cond(T)),
        nnz=int(np.count_nonzero(T)), tinv_norm=float(np.linalg.norm(np.linalg.inv(T), 2)),
        amplification=amp, total_budget=float(g * amp),
        T=T.copy(), transformed=d, note=note,
    )


# ---------------------------------------------------------------------------
# 候选构造
# ---------------------------------------------------------------------------
def _identity(sys: Descriptor, delta) -> BasisCandidate:
    return _score(sys, np.eye(sys.n), "identity", delta,
                  note="物理坐标原样；amplification=1")


def _schur(sys: Descriptor, delta) -> BasisCandidate | None:
    """正交 Schur 基（E≈I 时）：cond=1，Γ 诚实保留（正交不变）。"""
    if not np.allclose(sys.E, np.eye(sys.n), atol=1e-9):
        return None
    _, Z = linalg.schur(sys.A, output="real")
    return _score(sys, Z, "schur", delta,
                  note="块正交基：cond=1，Γ 诚实（不换 tax 买 Γ）")


def _eigenvector(sys: Descriptor, delta) -> BasisCandidate | None:
    """模态基：x = Vz，A_z=V⁻¹AV 对角；复对用实块形式 [Re v, Im v]。"""
    w, V = linalg.eig(sys.A, sys.E)
    fin = np.isfinite(w)
    if not np.all(fin):
        return None                      # 无穷特征值（奇异 E）：模态基不可用
    order = np.argsort(w.real)
    w, V = w[order], V[:, order]
    if np.max(np.abs(V.imag)) > 1e-8 * max(1.0, np.max(np.abs(V.real))):
        # 复共轭对 → 实块形式，保持 T 实（tile 系数必须实）
        Vr = np.zeros_like(V.real)
        k = 0
        i = 0
        while i < V.shape[1]:
            if abs(V[:, i].imag).max() > 1e-8:
                Vr[:, k] = V[:, i].real
                Vr[:, k + 1] = V[:, i].imag
                k += 2
                i += 2
            else:
                Vr[:, k] = V[:, i].real
                k += 1
                i += 1
        V = Vr
    else:
        V = V.real
    try:
        c = _score(sys, V, "eigenvector", delta,
                   note="Γ 降到对角值，但付 κ(Φ)=cond(V) 税（§8 shear 实测 20.05）")
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(c.cond_phi) or c.cond_phi > 1e12:
        c.note += "；cond(V) 病态，候选按不可用处理"
    return c


def _gramian(sys: Descriptor, delta) -> BasisCandidate | None:
    """rms3σ 对角预条件（P3 同款，model time）：√P_ii，退化项 floor。"""
    ly = solve_generalized_lyapunov(sys)
    if not ly.h2_finite:
        return None
    c = _diag_candidate(sys, np.sqrt(np.clip(np.diag(ly.P), 0.0, None)),
                        "gramian",
                        delta,
                        note="可控性 Gramian 对角（P3 预条件的 model-time 版）；"
                             "退化（不可控）态 floor 到正项几何均值")
    return c


def _pod(sys: Descriptor, delta) -> BasisCandidate | None:
    """Moore 平衡对角 d_i=(P_ii·Wo_ii)^{1/4}（无快照 POD 代理）。"""
    from lowering.scaling import _Wo_via_transposed
    ly = solve_generalized_lyapunov(sys)
    if not ly.h2_finite:
        return None
    Wo, _path = _Wo_via_transposed(sys)
    if Wo is None:
        return None
    d = (np.clip(np.diag(ly.P), 0.0, None) * np.clip(np.diag(Wo), 0.0, None)) ** 0.25
    c = _diag_candidate(sys, d, "pod",
                        delta,
                        note="Moore 平衡对角（Gramian-POD 代理；数据驱动 POD 需 snapshot）")
    return c


_BUILDERS = {"identity": _identity, "schur": _schur,
             "gramian": _gramian, "pod": _pod, "eigenvector": _eigenvector}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def select_basis(descriptor: Descriptor,
                 candidates: tuple = DEFAULT_CANDIDATES,
                 delta_pert=None) -> BasisResult:
    """给 (E,A,B,C) 打分选基：最小化失配总预算 Γ·amplification。

    delta_pert: None（最坏情况，amplification=cond(T)）| n×n 矩阵
    （给定物理失配结构 Δ，amplification=‖T⁻¹ΔT‖/‖Δ‖）| 标量
    （ε⇒Δ=ε·A，系数成比例的漂移）。

    排序键：(total_budget, nnz, cond_phi, 候选序)。非有限预算 /
    构造失败的候选记为不可用并保留 reason，不静默丢弃。
    """
    if not isinstance(descriptor, Descriptor):
        raise TypeError("select_basis 需要 analysis.descriptor.Descriptor")
    delta: np.ndarray | None
    note_model: str
    if delta_pert is None:
        delta = None
        note_model = "worst-case (cond(T))"
    elif np.isscalar(delta_pert):
        delta = float(delta_pert) * descriptor.A
        note_model = "delta_pert = eps*A (coefficient-proportional)"
    else:
        delta = np.asarray(delta_pert, dtype=float)
        if delta.shape != descriptor.A.shape:
            raise ValueError("delta_pert 形状必须与 A 相同")
        note_model = "delta_pert = explicit matrix"

    cards: dict[str, BasisCandidate] = {}
    for i, name in enumerate(candidates):
        builder = _BUILDERS.get(name)
        if builder is None:
            cards[name] = BasisCandidate(
                name=name, Gamma=float("inf"), cond_phi=float("inf"),
                nnz=0, tinv_norm=float("inf"), amplification=float("inf"),
                total_budget=float("inf"), note="未知候选名")
            continue
        try:
            card = builder(descriptor, delta)
        except (np.linalg.LinAlgError, ValueError) as exc:
            card = None
            reason = "构造失败: %s" % exc
        else:
            reason = card.note if card is None else ""
        if card is None or not np.isfinite(card.total_budget):
            if card is not None:
                reason = card.note
            cards[name] = BasisCandidate(
                name=name, Gamma=float("inf"), cond_phi=float("inf"),
                nnz=0, tinv_norm=float("inf"), amplification=float("inf"),
                total_budget=float("inf"), note=reason or "不可用")
            continue
        cards[name] = card

    usable = [(float("%.10e" % c.total_budget), c.nnz, c.cond_phi, i, c.name)
              for i, (name, c) in enumerate(cards.items())
              if np.isfinite(c.total_budget)]
    result = BasisResult(winner=None, candidates=cards,
                         delta_pert=note_model)
    if not usable:
        result.reason = "所有候选都不可用"
        return result
    usable.sort()
    win_name = usable[0][4]
    win = cards[win_name]
    win.selected = True
    result.winner = win_name
    result.transformed = win.transformed
    result.T = win.T
    g0 = _gamma_of(descriptor)
    result.reason = ("winner=%s: budget=%.6g (Γ=%.6g × amp=%.6g, nnz=%d)"
                     " vs identity Γ=%.6g" % (win_name, win.total_budget,
                                              win.Gamma, win.amplification,
                                              win.nnz, g0))
    return result
