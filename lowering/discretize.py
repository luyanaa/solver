"""P0 离散化/结构化：PDESpec → Discretized（DESIGN §4）。

DESIGN §4 "P0 做 jet/DAE 与无量纲化…(M,K,f)" 的离散化一侧：把三个 anchor
operator 变成 analysis 栈已经验收过的形式，而不是另起一套求解器：

  * diffusion（1D Fourier，周期）：对角模态算子，直接给出
    analysis.Descriptor(E=I, A=op, B=e1, C=I) —— 与 bench/run_phase0
    build_diffusion 完全一致（B=e1 是规范扰动端口）。
  * poisson（2D FD，Dirichlet / neumann-mixed）：约束网格，即 T10 的
    KCL 松弛基底。descriptor 在此为 None —— 调用方用
    twin.constraint.kcl_relaxation_descriptor(mesh, c_node) 按自己选的
    c_node 建（c_node 是编译器参数，不属于离散化）。
  * burgers（1D FV，周期）：rhs 保持**非线性**。线性化是独立的、显式的
    一步（Jacobian 在一个具名状态处做中心差分，linearize_rhs），从不隐
    藏在离散化里 —— 与 run_phase0 build_burgers 的做法一致。

lazy import 说明：poisson 分支在函数体内 import twin.constraint
（lowering → twin 方向）。这是 P0 的便利做法，先例是 twin/route.py 里
的 `from analysis.skew import ...` lazy import；bench 层（bench/problems）
负责把 mesh 变成 Descriptor。

未实现的 (operator, basis, bc) 组合一律 NotImplementedError，点名组合，
不做静默回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PDESpec:
    name: str
    operator: str            # "diffusion" | "poisson" | "burgers"
    domain: tuple            # ((x0, x1),) 1D；((x0, x1), (y0, y1)) 2D
    bc: str                  # "periodic" | "dirichlet" | "neumann-mixed"
    coeff: float = 1.0       # ν (diffusion) / 电导率 (poisson) / 未用 (burgers)
    n: tuple = (64,)         # 每维节点/模态数
    basis: str = "fourier"   # "fourier" | "fd" | "fv"
    meta: dict = field(default_factory=dict)  # c_node, rhs_func, linearize_at…


@dataclass
class Discretized:
    descriptor: object | None    # analysis.Descriptor（线性）；burgers 为 None
    operator_matrix: np.ndarray  # 空间算子（poisson 为 G；diffusion 为对角阵）
    grid: tuple                  # 每维坐标数组
    states: list
    rhs_func: object | None      # 非线性 rhs callable（burgers）
    meta: dict


def discretize(spec: PDESpec) -> Discretized:
    if spec.operator == "diffusion" and spec.basis == "fourier" \
            and spec.bc == "periodic" and len(spec.domain) == 1:
        return _diffusion_fourier(spec)
    if spec.operator == "poisson" and spec.basis == "fd" \
            and spec.bc in ("dirichlet", "neumann-mixed") and len(spec.domain) == 2:
        return _poisson_fd2d(spec)
    if spec.operator == "burgers" and spec.basis == "fv" \
            and spec.bc == "periodic" and len(spec.domain) == 1:
        return _burgers_fv(spec)
    raise NotImplementedError(
        f"discretize: unimplemented combo operator={spec.operator!r} "
        f"basis={spec.basis!r} bc={spec.bc!r} domain-dims={len(spec.domain)} "
        f"(implemented: diffusion/fourier/periodic-1d, poisson/fd/"
        f"dirichlet|neumann-mixed-2d, burgers/fv/periodic-1d)")


# ---------------------------------------------------------------------------
# diffusion：1D Fourier 模态，周期
# ---------------------------------------------------------------------------
def _diffusion_fourier(spec: PDESpec) -> Discretized:
    """A = diag(−ν k²)，k = 1..n —— 与 run_phase0.build_diffusion 完全一致。

    注意约定：run_phase0 的扩散算子用的是无量纲模态 k = 1..n（即 L 已被
    无量纲化为 1），解析对照 κ = n²、‖H‖∞ = 1/ν、σ_y = sqrt(1/(2ν)) 都是
    在这个约定下成立的。本实现逐字复用该约定（domain 只用来记录/校验
    x1−x0；若 domain 长度 ≠ 1，调用方应先做无量纲化——这是 P0 的职责）。
    """
    (x0, x1), = spec.domain
    n = spec.n[0]
    if x1 - x0 != 1.0:
        raise ValueError(
            f"diffusion/fourier expects a unit domain after nondimensionalization"
            f"(got length {x1 - x0}); run P0 nondimensionalize first")
    nu = spec.coeff
    k = np.arange(1, n + 1, dtype=float)
    op = np.diag(-nu * k * k)
    from analysis import Descriptor
    desc = Descriptor(np.eye(n), op, np.eye(n)[:, :1], np.eye(n))
    meta = dict(
        kind="diffusion", nu=nu, n=n, L=x1 - x0,
        kappa_analytic=float(n * n),
        hinf_analytic=float(1.0 / nu),
        sigma_y_analytic=float(np.sqrt(1.0 / (2.0 * nu))))
    meta.update(spec.meta)
    return Discretized(
        descriptor=desc,
        operator_matrix=op,
        grid=(np.linspace(x0, x1, n, endpoint=False),),
        states=[f"u_k{j + 1}" for j in range(n)],
        rhs_func=None,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# poisson：2D 五点 FD 约束网格
# ---------------------------------------------------------------------------
def _poisson_fd2d(spec: PDESpec) -> Discretized:
    """约束网格 G（T10 基底）；descriptor 留给调用方 + kcl_relaxation_descriptor。

    坐标约定（与 twin.constraint.poisson_2d 的接地环一致）：nx×ny 个自由
    节点取内点 x_i = (i+1)·h、h = 1/(nx+1)；最外圈节点经单位电导接地，
    是"真网格总会漏电"的软 Dirichlet —— 对解析解 u = sin(πx)sin(πy)/(2π²)
    的检验精度是 O(h²)（‖G·u−I‖∞ ≈ h²/2），不是机器精度，见 §3.4 的
    neumann-mixed/leakage 说明。
    """
    (x0, x1), (y0, y1) = spec.domain
    nx, ny = int(spec.n[0]), int(spec.n[1])
    if (x1 - x0, y1 - y0) != (1.0, 1.0):
        raise ValueError("poisson/fd expects the unit square after P0 "
                         "nondimensionalization")
    # twin.constraint 只建单位电导网格；conductance scaling（coeff）属于
    # 硬件侧的 gm 选择，不影响离散解 —— 记录在 meta，不进 G。
    from twin.constraint import ConstraintMesh  # lazy：见模块 docstring
    mesh = ConstraintMesh.poisson_2d(nx, ny, bc=spec.bc)
    h = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)
    x = (np.arange(nx) + 1) * h
    y = (np.arange(ny) + 1) * hy
    states = [f"v_{ix}_{iy}" for iy in range(ny) for ix in range(nx)]
    return Discretized(
        descriptor=None,
        operator_matrix=mesh.G,
        grid=(x, y),
        states=states,
        rhs_func=None,
        meta=dict(
            kind="poisson", nx=nx, ny=ny, h=float(h), hy=float(hy),
            bc=spec.bc, conductance=float(spec.coeff), mesh=mesh,
            **{k_: v for k_, v in spec.meta.items()}),
    )


# ---------------------------------------------------------------------------
# burgers：1D FV，周期，一阶 Roe
# ---------------------------------------------------------------------------
def _burgers_fv(spec: PDESpec) -> Discretized:
    """一阶 Roe 通量的 FV rhs（非线性，保持为 callable）。

    通量逐字取自 bench/run_twin_experiments.py:85-121 的 `_roe` /
    `face_flux(u, "first", alpha, h)`（run_phase0.build_burgers 的 rhs 同一
    scheme）。放在 discretize.py 而不是 import bench，原因：bench 代码是
    "允许含求解器的位置"，lowering 不该反向依赖 bench —— 但 twin/ 没有
    独立的通量模块，故按任务规格把 ~10 行一阶 Roe 内联于此，注明出处。
    """
    (x0, x1), = spec.domain
    n = spec.n[0]
    if x1 - x0 != 1.0:
        raise ValueError("burgers/fv expects a unit domain after P0 "
                         "nondimensionalization")
    dx = 1.0 / n
    alpha = float(spec.meta.get("alpha", 1.6))  # LF 分裂速度；一阶 Roe 实际不用它

    def _roe(uL, uR):
        # Burgers 的 Roe 通量（run_twin_experiments._roe，逐字）：
        # 耗散 ∝ |ã|(uR−uL)，激波处为零（精确迎风）。
        a = 0.5 * (uL + uR)
        return (0.5 * (0.5 * uL * uL + 0.5 * uR * uR)
                - 0.5 * np.maximum(np.abs(a), 1e-6) * (uR - uL))

    def rhs_func(v):
        F = _roe(v, np.roll(v, -1))          # face_flux(v, "first", alpha, dx)
        return -(F - np.roll(F, 1)) / dx

    base = dict(kind="burgers", n=n, dx=dx, alpha=alpha, scheme="roe-first")
    base.update(spec.meta)
    return Discretized(
        descriptor=None,
        operator_matrix=np.empty((0, 0)),    # 非线性：无全局算子矩阵
        grid=(np.linspace(x0, x1, n, endpoint=False),),
        states=[f"u_{j}" for j in range(n)],
        rhs_func=rhs_func,
        meta=base,
    )


def linearize_rhs(rhs_func, u0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心差分 Jacobian：J[:, j] = (rhs(u0+e_j·ε) − rhs(u0−e_j·ε)) / 2ε。

    与 run_phase0.build_burgers 的线性化完全同式。这是**显式的一步**：
    Burgers 的 Descriptor(I, J, B, C) 由调用方在具名状态（如 pre-shock
    exact 解）处构造，离散化本身从不隐藏线性化。
    """
    u0 = np.asarray(u0, dtype=float)
    n = u0.size
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        J[:, j] = (rhs_func(u0 + e) - rhs_func(u0 - e)) / (2.0 * eps)
    return J
