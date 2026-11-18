"""T10 constraint fabric: KCL / Poisson / pressure-projection digital twin.

DESIGN.md §3.4 quote (normative):

    三类 workload 共享的不是"ODE 积分器"，而是一个稀疏代数约束：A(x) y = b(x)
    硬件不是数字求解器，而是快速松弛：
        C_y ẏ = −A y + b,    τ_relax ~ C_y/λ_min(A)
    这个式子只对 SPD/对称部分正定的 A 给出单调松弛；非对称 KCL 要用
    λ_min((A+Aᵀ)/2) 或谱横坐标，且必须检查反馈稳定性——电阻 mesh 的
    M-矩阵性质不能直接搬到有源 gm 网络。
    必须保证 ε_c = τ_constraint / τ_state,min ≪ 1
    否则 MD 出现假 retardation，NS 的 pressure projection 传播不动，
    SPICE 的 KCL 没收敛。

Physical contract: the constraint fabric solves A(x)·y = b(x) not digitally
but as the natural KCL relaxation of a conductance mesh:

    C_node·V̇ + G·V = I_rhs   →   steady state V* = G⁻¹·I_rhs = A⁻¹·b

G is the conductance matrix (graph Laplacian for a resistor mesh — SPD
M-matrix on the free node set), C_node the parasitic nodal capacitance,
I_rhs the injected source currents. The compiler needs four numbers from
this module: τ_relax, κ(G), the mismatch error floor, and the timing ratio
ε_c against the state fabric's fastest time constant.

Sign convention (must match the analysis stack): physical G is positive
semidefinite on the free nodes, dynamics V̇ = −G/C·V + I/C, so the
descriptor handed to `analysis` is E = C_node·I, A = −G (stable when G is
SPD — eigenvalues −λ/C with λ > 0).

Boundary conditions: Dirichlet rows are *eliminated from the free set*, not
given a large fake diagonal. A pinned node (u = u_bc) is removed from G; its
contribution to each neighbor's KCL row, g_ij·u_bc, is folded into I_rhs by
the caller. For u_bc = 0 (the common case, and all bench cases here) the
folded term vanishes, which is exactly a ground conductance on that node —
mathematically identical to eliminating the pinned node. A `neumann-mixed`
mesh leaves ONE boundary node weakly grounded with conductance g_ref: this
models a real mesh that always has leakage and keeps G nonsingular without
pretending the pure-Neumann problem is well posed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Canonical mirror-lambda table (λ = gds/I_D at the vsweep bias point). These
# are LOADED from results/device_extract.json by path, not hardcoded — the
# constants below are only the fallback when the extraction file is absent,
# and they are documented in DESIGN.md §6.9 (HBT 0.0138 @20µA, gf180 0.034,
# sky130 0.081, IHP MOS 0.970 /V).
_LAMBDA_FALLBACK = {
    "sky130": 0.081,
    "gf180mcu": 0.034,
    "ihp-sg13g2": 0.970,
    "ihp-hbt": 0.0138,
}
_LAMBDA_CACHE: dict | None = None


def load_lambda_table() -> dict:
    """Load λ = gds/I_D per PDK from results/device_extract.json (lazy, by path)."""
    global _LAMBDA_CACHE
    if _LAMBDA_CACHE is not None:
        return _LAMBDA_CACHE
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "device_extract.json")
    table = {}
    if os.path.exists(path):
        with open(path) as fh:
            for entry in json.load(fh):
                if "lambda_eff" in entry:
                    table[entry["pdk"]] = float(entry["lambda_eff"])
    for k, v in _LAMBDA_FALLBACK.items():
        table.setdefault(k, v)
    _LAMBDA_CACHE = table
    return table


@dataclass
class ConstraintMesh:
    """A conductance mesh: the T10 substrate for A(x)·y = b(x).

    G : (n,n) conductance matrix on the FREE node set (pinned Dirichlet nodes
        eliminated; ground conductances folded into the diagonal).
    edges : the free-to-free edges (i, j, g_ij) that built G (diagnostic).
    bc : boundary-condition tag, purely informational.
    """
    G: np.ndarray
    edges: list = field(default_factory=list)
    bc: str = "dirichlet"
    name: str = "mesh"

    # ---- constructors -----------------------------------------------------
    @staticmethod
    def from_edges(edges: Iterable[Sequence], n_nodes: int | None = None,
                   bc: str = "dirichlet", name: str = "mesh") -> "ConstraintMesh":
        """edges: (i, j, g). Node -1 means ground → adds g to G_ii (a pinned
        zero-value Dirichlet node, eliminated per the docstring policy)."""
        edges = [(int(i), int(j), float(g)) for i, j, g in edges]
        if n_nodes is None:
            n_nodes = max(max(i, j) for i, j, _ in edges) + 1
            n_nodes = max(n_nodes, 1)
        G = np.zeros((n_nodes, n_nodes))
        free_edges = []
        for i, j, g in edges:
            if g < 0:
                raise ValueError("negative conductance")
            if i == j:
                G[i, i] += g
                continue
            if j == -1:
                G[i, i] += g
            elif i == -1:
                G[j, j] += g
            else:
                G[i, i] += g
                G[j, j] += g
                G[i, j] -= g
                G[j, i] -= g
                free_edges.append((i, j, g))
        return ConstraintMesh(G=G, edges=free_edges, bc=bc, name=name)

    @staticmethod
    def poisson_1d(n: int, g: float = 1.0, bc: str = "dirichlet",
                   g_ref: float = 1e-3) -> "ConstraintMesh":
        """1D Laplacian on n free nodes, unit chain conductance g.

        dirichlet: the two chain ends connect to pinned u=0 nodes → each end
        row carries an extra g (equivalent to eliminating the boundary node).
        neumann-mixed: chain ends free, ONE node (index 0) weakly grounded
        with g_ref — the "real mesh always leaks" mode of §3.4.
        """
        edges = [(i, i + 1, g) for i in range(n - 1)]
        if bc == "dirichlet":
            edges.append((0, -1, g))
            edges.append((n - 1, -1, g))
        elif bc == "neumann-mixed":
            edges.append((0, -1, g_ref))
        else:
            raise ValueError("bc must be 'dirichlet' or 'neumann-mixed'")
        return ConstraintMesh.from_edges(edges, n_nodes=n, bc=bc,
                                        name=f"poisson1d-n{n}-{bc}")

    @staticmethod
    def poisson_2d(nx: int, ny: int, g: float = 1.0, bc: str = "dirichlet",
                   g_ref: float = 1e-3) -> "ConstraintMesh":
        """2D five-point Poisson stencil, nx*ny free nodes, row-major (ix + nx*iy).

        dirichlet: boundary neighbors are pinned u=0 nodes (folded to ground
        conductance g on boundary rows). neumann-mixed: free boundaries,
        one corner node grounded with g_ref.
        """
        n = nx * ny
        idx = lambda ix, iy: ix + nx * iy
        edges = []
        for iy in range(ny):
            for ix in range(nx):
                if ix + 1 < nx:
                    edges.append((idx(ix, iy), idx(ix + 1, iy), g))
                if iy + 1 < ny:
                    edges.append((idx(ix, iy), idx(ix, iy + 1), g))
        if bc == "dirichlet":
            for iy in range(ny):
                for ix in range(nx):
                    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
                        edges.append((idx(ix, iy), -1, g))
        elif bc == "neumann-mixed":
            edges.append((0, -1, g_ref))
        else:
            raise ValueError("bc must be 'dirichlet' or 'neumann-mixed'")
        return ConstraintMesh.from_edges(edges, n_nodes=n, bc=bc,
                                        name=f"poisson2d-{nx}x{ny}-{bc}")

    # ---- properties -------------------------------------------------------
    @property
    def n(self) -> int:
        return self.G.shape[0]

    @property
    def is_symmetric(self) -> bool:
        return bool(np.allclose(self.G, self.G.T, rtol=0, atol=1e-13 * max(1.0, np.abs(self.G).max())))

    @property
    def row_sums(self) -> np.ndarray:
        return self.G.sum(axis=1)


# ---------------------------------------------------------------------------
# spectral machinery
# ---------------------------------------------------------------------------
def relaxation_modes(mesh: ConstraintMesh) -> dict:
    """Spectral verdict for the mesh (§3.4 guard).

    λ_min is taken on the SYMMETRIC part G_sym=(G+Gᵀ)/2 (the §3.4 rule for
    non-symmetric KCL); stability is additionally checked on the FULL
    spectrum of G because an M-matrix property does not survive active-gm
    asymmetry. Sign convention: the dynamics are V̇ = −G/C·V + I/C, so the
    reported spectral abscissa is max Re eig(−G) = −min Re eig(G) — a
    dissipative mesh has all Re eig(G) > 0 and therefore a NEGATIVE
    abscissa. monotone relaxation additionally requires the mesh to be
    symmetric (SPD); a non-symmetric stable mesh reports monotone_spd=False
    and must pass the full-spectrum feedback-stability check instead.
    """
    G = mesh.G
    G_sym = 0.5 * (G + G.T)
    lam = np.linalg.eigvalsh(G_sym)
    lam_max = float(lam[-1])
    lam_min = float(lam[0])
    full = np.linalg.eigvals(-G)          # dynamics matrix −G
    abscissa = float(np.max(full.real))   # decay rates: stable iff < 0
    scale = max(lam_max, 1e-30)
    singular = lam_min <= 1e-10 * scale
    monotone_spd = bool(mesh.is_symmetric and lam_min > 1e-10 * scale)
    return dict(
        lam_min_sym=lam_min,
        lam_max_sym=lam_max,
        kappa_G=(lam_max / lam_min) if not singular else float("inf"),
        spectral_abscissa=abscissa,
        stable_full=bool(abscissa < -1e-12 * scale),
        monotone_spd=monotone_spd,
        symmetric=mesh.is_symmetric,
        singular=singular,
        n=mesh.n,
    )


def relax_time(mesh: ConstraintMesh, c_node: float) -> float:
    """τ_relax = C_node / λ_min(G_sym) (§3.4 formula, symmetric-part rule)."""
    if c_node <= 0:
        raise ValueError("c_node must be positive")
    modes = relaxation_modes(mesh)
    if modes["singular"]:
        raise ValueError("mesh is singular (zero mode): cannot define τ_relax")
    return c_node / modes["lam_min_sym"]


def eps_c(mesh: ConstraintMesh, c_node: float,
          external_state_timescale: float) -> float:
    """Timing ratio ε_c = τ_constraint / τ_external — the only hard timing
    contract between T10 and T1 (§3.4). Must be ≪ 1.

    τ_external is the EXTERNAL constrained-dynamics timescale, declared by
    the caller: the fastest time constant of the dynamic state the constraint
    fabric serves (e.g. NS pressure-projection vs the fluid's τ_fluid,min,
    MD field update vs the particle-dynamics τ). It is NOT derivable from the
    constraint subsystem's own poles — comparing the relaxation to its own
    fastest mode is meaningless (that ratio says nothing about whether the
    constraint converges before the state needs it). A pure steady-state
    problem (pure Poisson, no external dynamic state) has NO external
    timescale: pass None / don't call this, and judge the solver on
    solve-deadline + residual instead (see solve_constraint).
    """
    if external_state_timescale is None:
        raise ValueError("external_state_timescale=None: pure steady-state "
                         "problem has no ε_c to compute — use the "
                         "solve-deadline+residual verdict path instead")
    t_ext = float(external_state_timescale)
    if not (np.isfinite(t_ext) and t_ext > 0):
        raise ValueError("external_state_timescale must be finite and > 0")
    return relax_time(mesh, c_node) / t_ext


def settle(mesh: ConstraintMesh, i_rhs: np.ndarray, t: Sequence[float],
           v0: np.ndarray | None = None) -> np.ndarray:
    """Analytic per-mode relaxation V(t) for C·V̇ + G·V = I.

    Per-mode form (symmetric G): decompose G = Q·diag(λ)·Qᵀ, then
        V(t) = V* − Q·diag(e^{−λ t/C})·Qᵀ·(V* − v0),   V* = G⁻¹ I.
    For a non-symmetric G the relaxation is evaluated on G_sym (the §3.4
    symmetric-part rule) while the asymptote stays G⁻¹·I — documented
    approximation, valid when the asymmetry is a small perturbation.

    Returns array of shape (len(t), n); scalar t gives shape (n,).
    """
    G = mesh.G
    n = mesh.n
    i_rhs = np.asarray(i_rhs, dtype=float)
    if i_rhs.shape != (n,):
        raise ValueError("i_rhs must have shape (n,)")
    v_star = np.linalg.solve(G, i_rhs)
    v0 = np.zeros(n) if v0 is None else np.asarray(v0, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    G_sym = 0.5 * (G + G.T)
    lam, Q = np.linalg.eigh(G_sym)
    if lam[0] <= 1e-10 * max(lam[-1], 1e-30):
        raise ValueError("mesh is singular (zero mode): no steady relaxation")
    a0 = Q.T @ (v_star - v0)
    decay = np.exp(-np.outer(t_arr, lam) / _c_of(mesh))
    out = v_star[None, :] - (decay * a0[None, :]) @ Q.T
    return out[0] if np.isscalar(t) or np.asarray(t).ndim == 0 else out


def _c_of(mesh: ConstraintMesh) -> float:
    """Nodal capacitance used by `settle` (uniform C; taken from attribute)."""
    return float(getattr(mesh, "c_node", 1.0))


def settle_with_c(mesh: ConstraintMesh, c_node: float, i_rhs: np.ndarray,
                  t: Sequence[float], v0: np.ndarray | None = None) -> np.ndarray:
    """`settle` with an explicit uniform C_node (the usual entry point)."""
    mesh.c_node = float(c_node)  # transient attribute consumed by _c_of
    try:
        return settle(mesh, i_rhs, t, v0=v0)
    finally:
        if hasattr(mesh, "c_node"):
            del mesh.c_node


def residual(mesh: ConstraintMesh, v: np.ndarray, i_rhs: np.ndarray) -> float:
    """R(t) = ‖A·V(t) − b‖_2 = ‖G·V − I‖_2 at one time point."""
    return float(np.linalg.norm(mesh.G @ v - np.asarray(i_rhs)))


def mismatch_floor(mesh: ConstraintMesh, i_rhs: np.ndarray, lam: float,
                   eps_dev: float, eps_sys: float | None = None,
                   seed: int | None = None, delta_v: np.ndarray | None = None,
                   n_draws: int = 1) -> dict:
    """Static-error floor of the mesh under mirror (gm-cell) mismatch.

    Model (matched to twin/coupling.py InterfaceModel's error classes):
    the hardware injects I' = I + δI with
        δI_i = lam·ΔV_i·i_scale            (systematic gds/ΔV mirror-load term,
                                            λ = gds/I_D, table from
                                            results/device_extract.json)
           + I_i·eps_dev·ξ_i           (random per-node Pelgrom-style error,
                                            ξ ~ N(0,1), seeded)
           + I_i·eps_sys·η             (common-mode gain error, shared η)
    ΔV_i is the drain-voltage deviation of node i from the mirror reference
    node; pass `delta_v=None` (default) when the mirror devices sit at the
    reference bias so the systematic term vanishes — the λ term only matters
    when ΔV ≠ 0. i_scale = ‖I‖∞ linearizes λ·I_D·ΔV at the operating current.

    §3.4 trade-off made quantitative: δV = G⁻¹·δI amplifies the λ_min
    direction, so the floor scales like ‖δI‖/λ_min — high-κ meshes have a
    high floor. Returns stochastic draws (seeded) plus the deterministic
    bound ‖δV‖∞ ≤ ‖G⁻¹‖∞·‖δI‖∞.
    """
    G = mesh.G
    n = mesh.n
    I = np.asarray(i_rhs, dtype=float)
    if I.shape != (n,):
        raise ValueError("i_rhs must have shape (n,)")
    i_scale = max(float(np.max(np.abs(I))), 1e-30)
    dv = np.zeros(n) if delta_v is None else np.asarray(delta_v, dtype=float)
    if dv.shape != (n,):
        raise ValueError("delta_v must have shape (n,)")
    dv = dv - dv[np.argmax(np.abs(I))] if np.any(dv) else dv  # reference node = largest injector

    rng = np.random.default_rng(seed)
    Ginv = np.linalg.inv(G)  # caller must ensure nonsingular (checked upstream)
    floors, floors_rms = [], []
    for _ in range(max(1, n_draws)):
        dI = lam * dv * i_scale + I * eps_dev * rng.standard_normal(n)
        if eps_sys:
            dI = dI + I * eps_sys * float(rng.standard_normal())
        dV = Ginv @ dI
        floors.append(float(np.max(np.abs(dV))))
        floors_rms.append(float(np.sqrt(np.mean(dV ** 2))))
    # deterministic bound
    dI_bound = lam * np.abs(dv) * i_scale + np.abs(I) * (eps_dev + (eps_sys or 0.0)) * 5.0
    bound = float(np.max(np.abs(Ginv)) * np.max(np.abs(dI_bound)))  # conservative sup-norm product
    bound_inf = float(np.abs(Ginv).sum(axis=1).max() * np.max(np.abs(dI_bound)))
    # dominant mode: projection onto the λ_min eigendirection of G_sym
    G_sym = 0.5 * (G + G.T)
    lam_e, Q = np.linalg.eigh(G_sym)
    dV_ref = Ginv @ (lam * dv * i_scale + I * eps_dev)
    w = np.abs(Q.T @ dV_ref)
    dominant = int(np.argmax(w))
    return dict(
        floor_inf=float(np.mean(floors)),
        floor_rms=float(np.mean(floors_rms)),
        floor_inf_max=float(np.max(floors)),
        bound_inf=bound_inf,
        bound_sup=bound,
        dominant_mode=dominant,
        dominant_lambda=float(lam_e[dominant]),
        delta_v_max=float(np.max(np.abs(dv))),
        i_scale=i_scale,
        n_draws=max(1, n_draws),
    )


@dataclass
class ConstraintResult:
    """Aggregated T10 certificate for one (mesh, c_node, i_rhs) triple."""
    tau_relax: float
    kappa_G: float
    lambda_min_sym: float
    spectral_abscissa: float
    monotone_spd: bool
    singular: bool = False
    stable_full: bool = True
    eps_c: float | None = None
    external_state_timescale: float | None = None
    steady_state: bool = False
    solve_deadline: float | None = None
    residual_tol: float | None = None
    residual_at_deadline: float | None = None
    converged: bool | None = None
    v_star_norm: float = 0.0
    v_star: np.ndarray | None = field(default=None, repr=False)
    mismatch_floor_inf: float | None = None
    mismatch_floor_rms: float | None = None
    dominant_mode: int | None = None
    verdict: str = ""
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(
            tau_relax=self.tau_relax, kappa_G=self.kappa_G,
            lambda_min_sym=self.lambda_min_sym,
            spectral_abscissa=self.spectral_abscissa,
            monotone_spd=self.monotone_spd, singular=self.singular,
            stable_full=self.stable_full, eps_c=self.eps_c,
            external_state_timescale=self.external_state_timescale,
            steady_state=self.steady_state,
            solve_deadline=self.solve_deadline, residual_tol=self.residual_tol,
            residual_at_deadline=self.residual_at_deadline,
            converged=self.converged,
            v_star_norm=self.v_star_norm,
            mismatch_floor_inf=self.mismatch_floor_inf,
            mismatch_floor_rms=self.mismatch_floor_rms,
            dominant_mode=self.dominant_mode, verdict=self.verdict,
            notes=list(self.notes),
        )
        return d


def solve_constraint(mesh: ConstraintMesh, i_rhs: np.ndarray, c_node: float,
                     external_state_timescale: float | None = None,
                     mismatch: dict | None = None,
                     solve_deadline: float | None = None,
                     residual_tol: float | None = None) -> ConstraintResult:
    """One-stop T10 evaluation: modes + τ_relax + ε_c + (optional) mismatch.

    ε_c semantics (P0-finding fix): the timing ratio is only defined against
    an EXTERNAL constrained-dynamics timescale, declared explicitly by the
    caller. When `external_state_timescale is None` the workload is treated
    as pure steady-state (e.g. a standalone Poisson solve): ε_c is NOT
    fabricated from the constraint's own poles — the verdict is instead a
    fact about (solve_deadline met?, residual < tol?):

      * `solve_deadline` given  → residual is evaluated at t=deadline via
        the analytic relaxation; verdict PASS iff res < residual_tol
        (default 1e-6·‖I‖₂), else FAIL ("constraint not converged at
        deadline") — an honest convergence fact, not an infeasibility claim
        about a dynamics the workload doesn't have.
      * no deadline             → verdict "STEADY": reports τ_relax and the
        steady residual as facts; ε_c stays None and no feasibility
        judgement is made.
    """
    modes = relaxation_modes(mesh)
    if modes["singular"]:
        return ConstraintResult(
            tau_relax=float("inf"), kappa_G=float("inf"),
            lambda_min_sym=modes["lam_min_sym"],
            spectral_abscissa=modes["spectral_abscissa"],
            monotone_spd=modes["monotone_spd"], singular=True,
            stable_full=modes["stable_full"], verdict="SINGULAR",
            steady_state=external_state_timescale is None,
            notes=["zero mode: pure-Neumann mesh, G not invertible — flagged, "
                   "not silently inverted"])
    tau = c_node / modes["lam_min_sym"]
    v_star = np.linalg.solve(mesh.G, np.asarray(i_rhs, dtype=float))
    eps = (None if external_state_timescale is None
           else tau / float(external_state_timescale))
    steady = external_state_timescale is None
    res_deadline = None
    converged = None
    if steady:
        if solve_deadline is not None:
            if solve_deadline <= 0:
                raise ValueError("solve_deadline must be positive")
            t_dl = float(solve_deadline)
            tol = (residual_tol if residual_tol is not None
                   else 1e-6 * float(np.linalg.norm(np.asarray(i_rhs, dtype=float))))
            v_dl = settle_with_c(mesh, c_node, i_rhs, t=t_dl)
            res_deadline = residual(mesh, v_dl, i_rhs)
            converged = bool(res_deadline < tol)
            notes_dl = ["pure steady-state (no external dynamic state): ε_c "
                        "undefined — judged on deadline T=%s vs residual "
                        "%s (tol %s)" % (_g(t_dl), _g(res_deadline), _g(tol))]
            ok = modes["monotone_spd"] and modes["stable_full"] and converged
            res = ConstraintResult(
                tau_relax=tau, kappa_G=modes["kappa_G"],
                lambda_min_sym=modes["lam_min_sym"],
                spectral_abscissa=modes["spectral_abscissa"],
                monotone_spd=modes["monotone_spd"],
                stable_full=modes["stable_full"], eps_c=None,
                external_state_timescale=None, steady_state=True,
                solve_deadline=t_dl, residual_tol=tol,
                residual_at_deadline=res_deadline, converged=converged,
                v_star_norm=float(np.linalg.norm(v_star)), v_star=v_star,
                verdict="PASS" if ok else "FAIL", notes=notes_dl,
            )
            if not converged:
                res.notes.append("constraint not converged at deadline: "
                                 "res=%s ≥ tol=%s (τ_relax=%s ≫ T)"
                                 % (_g(res_deadline), _g(tol), _g(tau)))
            if mismatch is not None:
                _attach_mismatch(res, mesh, i_rhs, mismatch)
            return res
        # no deadline: report facts only, no feasibility verdict
        res = ConstraintResult(
            tau_relax=tau, kappa_G=modes["kappa_G"],
            lambda_min_sym=modes["lam_min_sym"],
            spectral_abscissa=modes["spectral_abscissa"],
            monotone_spd=modes["monotone_spd"], stable_full=modes["stable_full"],
            eps_c=None, external_state_timescale=None, steady_state=True,
            v_star_norm=float(np.linalg.norm(v_star)), v_star=v_star,
            verdict="STEADY",
            notes=["pure steady-state (no external dynamic state): no ε_c "
                   "computed — τ_relax=%.3e and steady residual are reported "
                   "as facts, not a feasibility verdict" % tau],
        )
        if mismatch is not None:
            _attach_mismatch(res, mesh, i_rhs, mismatch)
        return res
    res = ConstraintResult(
        tau_relax=tau, kappa_G=modes["kappa_G"],
        lambda_min_sym=modes["lam_min_sym"],
        spectral_abscissa=modes["spectral_abscissa"],
        monotone_spd=modes["monotone_spd"], stable_full=modes["stable_full"],
        eps_c=eps, external_state_timescale=float(external_state_timescale),
        steady_state=False, v_star_norm=float(np.linalg.norm(v_star)),
        v_star=v_star,
    )
    if mismatch is not None:
        _attach_mismatch(res, mesh, i_rhs, mismatch)
    ok = modes["monotone_spd"] and modes["stable_full"] and (eps is None or eps < 0.01)
    res.verdict = "PASS" if ok else "FAIL"
    if eps is not None and eps >= 0.01:
        res.notes.append("ε_c = τ_constraint/τ_external ≥ 0.01: §3.4 timing "
                         "contract violated (constraint too slow for the "
                         "declared external dynamics — fake retardation / "
                         "unconverged KCL)")
    return res


def _g(v) -> str:
    try:
        return "%.3e" % float(v)
    except (TypeError, ValueError):
        return str(v)


def _attach_mismatch(res: ConstraintResult, mesh: ConstraintMesh,
                     i_rhs: np.ndarray, mismatch: dict) -> None:
    mf = mismatch_floor(mesh, i_rhs, **mismatch)
    res.mismatch_floor_inf = mf["floor_inf"]
    res.mismatch_floor_rms = mf["floor_rms"]
    res.dominant_mode = mf["dominant_mode"]
    res.notes.append("mismatch model: lam·ΔV + eps_dev + optional eps_sys "
                     "(coupling.py InterfaceModel error classes)")


def kcl_relaxation_descriptor(mesh: ConstraintMesh, c_node: float):
    """Hand the KCL relaxation to the validated analysis stack as a Descriptor.

        C_node·V̇ = −G·V + I_rhs  ⇔  E·V̇ = A·V + B·u
        E = C_node·I, A = −G, B = I, C = I.

    Sign convention (docstring contract): physical G is positive semidefinite
    on the free nodes and the dynamics are V̇ = −G/C·V + I/C, so descriptor
    A = −G/C_node is stable exactly when G is SPD — eigenvalues −λ/C with
    λ > 0. This reuses the already-validated generalized-eigenvalue and
    output-noise analysis (κ, Γ, Lyapunov σ_y, slack) instead of a parallel
    implementation (复用已验证的广义特征值与输出底噪分析).
    """
    from analysis.descriptor import Descriptor

    n = mesh.n
    return Descriptor(E=c_node * np.eye(n), A=-mesh.G,
                      B=np.eye(n), C=np.eye(n))
