"""§3.7 heterogeneous substrate table — minimal, only what the constraint
(T10) work needs. Not the full Phase-8 substrate module.

DESIGN.md §3.7 quote (normative):

    把 §1 的 descriptor 放宽到
        E_q·q̇ = F_q(q) + Σ_r C_qr·z_r ,  q ∈ { V, I, x_mech, P_fluid, c_chem, a_optical }
    1. κ_hw 变成 per-domain。电子窗约 1e-8–1e-3 s，MEMS 约 1e-6–1e-1 s，
       微流控约 1e-3–1e2 s，光子约 1e-12–1e-9 s（但没有存储）。
    2. transducer 是一等公民。每次跨域都要付 gain、nonlinearity、noise、
       bandwidth、energy，并且它自己就是一个 §1.2 的扰动端口。

    微流控：同一类 graph Laplacian
        C_f·dP_i/dt = Σ_j Q_ji − Σ_j Q_ij ,  Q_ij = (P_i − P_j)/R_ij
        ⇒ C_f·Ṗ = −L_R·P + Q
    与 C·V̇ = −G·V + I 同构：P↔V, Q↔I, R_f↔R。这不是类比，是同一类
    graph Laplacian dynamics。

The mapping table below makes the "fluidic network is the same graph
Laplacian" claim executable: `fluidic_mesh_to_conductance` emits a G matrix
that `twin.constraint.ConstraintMesh` consumes directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# §3.7 per-domain time-constant windows (seconds). Key = substrate domain,
# value = (tau_min, tau_max). Photonic is listed but carries "no storage":
# there is no C_f analog, so nothing accumulates state on that substrate.
KAPPA_HW: dict[str, tuple[float, float]] = {
    "electronic": (1e-8, 1e-3),
    "mems": (1e-6, 1e-1),
    "fluidic": (1e-3, 1e2),
    "photonic": (1e-12, 1e-9),
}

# §3.7 fluidic ↔ electrical correspondence for the linear resistive network.
# P↔V (pressure ↔ node voltage), Q↔I (flow ↔ injected current),
# R_f↔R (flow resistance ↔ electrical resistance, so conductance g=1/R_f),
# C_f↔C (fluid compliance ↔ nodal capacitance). Same graph-Laplacian math:
# C_f·Ṗ = −L_R·P + Q   ⇔   C·V̇ = −G·V + I.
FLUIDIC_TO_ELECTRICAL: dict[str, str] = {
    "P": "V",
    "Q": "I",
    "R_f": "R",
    "C_f": "C",
    "L_R": "G",
}


def fluidic_resistance(l: float, r: float, mu: float) -> float:
    """Hagen–Poiseuille resistance of a circular channel.

        R_f = 8·μ·l / (π·r⁴)

    l: channel length [m], r: channel radius [m], mu: dynamic viscosity
    [Pa·s] (water ≈ 1e-3). Fourth-power in r — this is why channel geometry,
    not device count, sets the fluidic conductance scale.
    """
    if r <= 0:
        raise ValueError("channel radius must be positive")
    return 8.0 * mu * l / (math.pi * r ** 4)


def fluidic_capacitance(volume: float, beta_wall: float | None = None) -> float:
    """Fluid compliance C_f = dV/dP.

    RIGID-CHANNEL LIMIT (the default in this repo): an incompressible liquid
    in a rigid channel has C_f ≈ 0, i.e. pressure is an algebraic variable and
    the fluidic mesh is a pure constraint (exactly the T10 case). A finite
    C_f only appears from (a) gas/compressible content or (b) wall
    elasticity. Pass `beta_wall` [Pa] as an effective bulk modulus for the
    wall/compressibility lump: C_f = V/beta_wall.

    TODO (Phase 8): real PDMS chips are wall-dominated
    (C_f ~ V/E_eff with E_eff ≪ liquid bulk modulus) plus gas bubbles;
    neither is a single number, so we do not fabricate a model here. §3.7
    explicitly warns 不能假设"流体网络 = 理想 RC".
    """
    if beta_wall is None:
        return 0.0
    if beta_wall <= 0:
        raise ValueError("beta_wall must be positive")
    return volume / beta_wall


def fluidic_mesh_to_conductance(edges, n_nodes: int | None = None) -> np.ndarray:
    """Build the graph-Laplacian G = L_R from fluidic edges.

    edges: iterable of (i, j, R_f) with R_f > 0. Node index -1 (or any index
    ≥ n when n_nodes is given) is the reservoir / pressure reference and is
    eliminated: its conductance folds into the diagonal of the free node
    (pressure of the reservoir is the Dirichlet value; a zero-valued
    reference folds to a ground conductance, matching
    `twin.constraint.ConstraintMesh.from_edges`).

    Returns the dense (n_free × n_free) Laplacian: G_ii = Σ 1/R_ij,
    G_ij = −1/R_ij. Feeds directly into `twin.constraint.ConstraintMesh`.
    """
    if n_nodes is None:
        n_nodes = max(max(i, j) for i, j, _ in edges) + 1
        n_nodes = max(n_nodes, 1)
    g_edges = []
    for i, j, r_f in edges:
        if r_f <= 0:
            raise ValueError("fluidic resistance must be positive")
        g = 1.0 / r_f
        g_edges.append((i, j, g))
    # Reuse the constraint-mesh assembler so both substrates share one
    # Laplacian definition (single source of truth for the §3.7 identity).
    from twin.constraint import ConstraintMesh

    mesh = ConstraintMesh.from_edges(g_edges, n_nodes=n_nodes)
    return mesh.G


@dataclass
class TransducerCost:
    """Cross-domain transducer cost entry (§3.7 point 2, contract stub).

    Every cross-domain trip must pay gain, nonlinearity, noise, bandwidth,
    energy, and the transducer itself is a §1.2 perturbation port. The future
    `budget.py` substrate axis consumes these fields as a slack entry. This
    dataclass only fixes the field contract — no model is claimed here; a
    populated instance must come from measurement or a cited model, never
    from a guess.

    Fields:
        gain          : transducer small-signal gain (units per pairing)
        noise_rms     : input-referred RMS noise of the transducer
        bandwidth_hz  : usable bandwidth of the cross-domain link
        energy        : energy per cross-domain transaction [J]
        notes         : provenance / caveats (which measurement, which corner)
    """

    gain: float
    noise_rms: float
    bandwidth_hz: float
    energy: float
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(gain=self.gain, noise_rms=self.noise_rms,
                    bandwidth_hz=self.bandwidth_hz, energy=self.energy,
                    notes=self.notes, extra=dict(self.extra))


# ===========================================================================
# MEMS substrate (§3.7: "MEMS 供给二阶 primitive mẍ+cẋ+kx=F，ω0=√(k/m) 作为
# dynamical reference")
#
# Scope (honest): this is the *constitutive law + calibration observable*
# layer, not a MEMS device model. The resonator below is the ideal lumped
# second-order ODE; everything device-specific (electrostatic drive
# nonlinearity F∝V², ΔC/C readout, pull-in, stiction, Q tempco) is NOT
# modeled and must come from a cited device model or measurement before any
# transducer number is treated as real. §9 risk 33/34/35 apply.
# ===========================================================================


def mems_omega0(k: float, m: float) -> float:
    """Natural frequency ω0 = √(k/m) [rad/s]. Exact definition, no model."""
    if m <= 0 or k <= 0:
        raise ValueError("k and m must be positive")
    return math.sqrt(k / m)


def mems_Q(m: float, omega0: float, c: float) -> float:
    """Quality factor Q = m·ω0/c (≡ 1/(2ζ), ζ = c/(2mω0)). Exact definition."""
    if c <= 0:
        raise ValueError("damping c must be positive for a finite Q")
    return m * omega0 / c


@dataclass
class MemsResonator:
    """Lumped second-order resonator, linear part + Duffing cubic.

        m·ẍ + c·ẋ + k·x + k3·x³ = F

    The linear part (k3=0) is the §3.7 primitive exactly. k3 adds the Duffing
    nonlinearity; it is only used through the perturbative backbone curve —
    no full nonlinear solver is built here (the bench does an RK4 spot check
    at small amplitude instead). Units are SI only if (m, c, k, k3) are SI;
    the math is unit-agnostic.

    The descriptor of the LINEAR part (state [x, ẋ]) is
        ẋ = v,   m·v̇ = F − c·v − k·x
    i.e. E = diag(1, m), A = [[0, 1], [−k, −c]]. The Duffing term is NOT in
    the descriptor — it would make A state-dependent, which is exactly the
    §3.7 point that the nonlinearity lives in F_q, not E_q.
    """

    m: float
    c: float
    k: float
    k3: float = 0.0

    def __post_init__(self):
        if self.m <= 0 or self.k <= 0 or self.c < 0:
            raise ValueError("m, k must be positive, c non-negative")

    @property
    def omega0(self) -> float:
        return mems_omega0(self.k, self.m)

    @property
    def zeta(self) -> float:
        return self.c / (2.0 * self.m * self.omega0)

    @property
    def Q(self) -> float:
        """Q = mω0/c; infinite (math.inf) for the undamped c=0 case."""
        if self.c == 0.0:
            return math.inf
        return mems_Q(self.m, self.omega0, self.c)

    @property
    def tau_settle(self) -> float:
        """Ring-down/settling time 2Q/ω0 = 2m/c (§3.7: τ_settle ~ 2Q/ω0)."""
        return 2.0 * self.m / self.c

    def descriptor(self):
        """Linear-part descriptor matrices (E, A, B) for state [x, ẋ].

        Returns (E, A, B) with E=diag(1, m), A=[[0,1],[−k,−c]],
        B=[[0],[1]] so that F enters as m·v̇ += F. k3 deliberately excluded
        (state-dependent A — see class docstring).
        """
        E = np.diag([1.0, self.m])
        A = np.array([[0.0, 1.0], [-self.k, -self.c]])
        B = np.array([[0.0], [1.0]])
        return E, A, B

    def duffing_backbone(self, amplitude: float) -> float:
        """Perturbative backbone curve ω(A) ≈ ω0·(1 + 3·k3·A²/(8·k)).

        Standard first-order Duffing frequency shift for the undamped
        steady response amplitude A (e.g. Nayfeh & Mook, perturbation
        analysis; classical Lindstedt–Poincaré result). Valid for
        |3·k3·A²/(8·k)| ≪ 1 and weak damping; the bench checks it against a
        small-amplitude RK4 measurement and reports the honest error.

        Hardening (k3 > 0) pushes ω(A) UP, softening (k3 < 0) down — the
        sign of k3 is directly observable from the detuning direction.
        """
        return self.omega0 * (1.0 + 3.0 * self.k3 * amplitude ** 2 / (8.0 * self.k))

    def rhs(self, state, t: float, F=0.0):
        """Full nonlinear RHS (including Duffing), state = [x, v]."""
        x, v = state
        return np.array([v,
                         (F - self.c * v - self.k * x - self.k3 * x ** 3) / self.m])


def gm_c_calibrate(omega_ref: float, C_target: float, drift_ppm: float,
                   meas_noise_ppm: float, loop_gain: float = 1e4) -> float:
    """Residual fractional error of a gm/C pole locked to a MEMS ω_ref.

    §C4 / §3.7 motivation: an electronic gm/C pole drifts with temperature
    (drift_ppm = fractional tempco of g_m/C over the observed swing). A MEMS
    resonator supplies ω0 = √(k/m) as a stable mechanical reference. The
    calibration loop compares the electronic pole ω_pole = g_m/C against
    ω_ref through a phase/frequency measurement and nulls the RATIO error;
    after locking, the electronics' own tempco is removed up to

        (a) finite loop gain:  raw drift / (1 + T)   (T = loop_gain)
        (b) the measurement residual floor meas_noise_ppm
            (phase-detector noise, reference readout noise, averaging)

    The residual is therefore set by (b) — the transducer/measurement noise
    floor — NOT by drift_ppm, provided loop_gain is large. That is the whole
    point of referencing to a mechanical ω0.

    ASSUMPTION (documented, not modeled): ω_ref itself is stable over the
    calibration interval. §9 risk 33 says the MEMS ω0 also drifts with
    temperature/aging and ratio accuracy does NOT replace absolute
    characterization — so this function is the *locked electronics*
    residual, valid only relative to a characterized reference. Drift of the
    reference must be budgeted separately; it does not enter this number.

    Returns the residual fractional error of the locked pole.
    """
    if drift_ppm < 0 or meas_noise_ppm < 0 or loop_gain <= 0:
        raise ValueError("drift/meas_noise must be >= 0, loop_gain > 0")
    # nominal lock target: g_m nominal = ω_ref·C_target; electronics presents
    # g_m·(1+drift); the loop trims g_m until the measured pole ratio error
    # nulls. Residual = trimmed-out drift remainder + measurement floor.
    raw = drift_ppm                                  # fractional error unlocked
    trimmed = raw / (1.0 + loop_gain)                # finite loop gain remainder
    return trimmed + meas_noise_ppm                  # + measurement noise floor


def tau_in_window(domain: str, tau: float) -> bool:
    """Is a characteristic time constant inside the domain's §3.7 κ_hw window?

    This is the per-domain guard: a primitive whose τ falls outside its
    domain window is flagged, not silently accepted (KAPPA_HW is a real
    check, not decoration). Note the window is on TIME CONSTANTS (storage
    elements), and photonic has no storage — see PHOTONIC_NO_STORAGE.
    """
    lo, hi = KAPPA_HW[domain]
    return lo <= tau <= hi


# ===========================================================================
# Photonic substrate (§3.7: "光子只买 global dense 与 linear transform；
# 不做非线性与存储")
#
# Scope (honest): this models the FIDELITY LIMITS of an optical 4×4 (or
# n×n) matrix-vector product — shot noise on the detected output and thermal
# drift of the phase shifters — NOT the physics of any specific mesh
# (MZI/Ring/diffractive). No transfer-matrix cascade, no WDM crosstalk
# structure, no detector model beyond the shot scaling. The two error
# mechanisms are the ones §3.7/§9 risk 37 name; anything beyond them needs a
# device model with provenance.
# ===========================================================================

# §3.7 lists a photonic time-constant window in KAPPA_HW, but explicitly
# flags 光子 "没有 storage": there is no C_f/C analog on the photonic
# substrate, so nothing integrates state and NO time constant can be
# extracted from a stored quantity. The window describes link latency
# (propagation), not a dynamical τ. This module therefore refuses to return
# a photonic τ and asserts the no-storage property textually instead.
PHOTONIC_NO_STORAGE = (
    "photonic substrate has NO storage element (no C_f/C analog): y=Wx is "
    "memoryless, nothing accumulates state, and no dynamical time constant "
    "exists on this domain. The KAPPA_HW['photonic'] window is link "
    "latency, not a storage τ (§3.7)."
)


def photonic_mvm(M, x, n_photons=None, drift_resid=0.0, seed=None):
    """Optical-domain matrix-vector product with fidelity-limit noise.

    Model (both mechanisms labeled, neither a device model):
      (a) SHOT NOISE: each output element k detects a photon number
          N_k = n_photons·p_k, p_k = y_k²/Σy² (intensity detection, energy
          conservation in a passive linear mesh). The relative standard
          deviation of element k is 1/√N_k. Contribution to the RMS output
          relative error: Σ_k p_k·(1/N_k) = n_out/n_photons, i.e.
          RMS rel. error = √(n_out/n_photons) — scales as 1/√n_photons
          exactly, independent of M and x.
      (b) THERMAL DRIFT RESIDUAL: after foreground calibration the static
          drift is removed up to a residual matrix error with relative
          Frobenius magnitude drift_resid (set by the DAC/feedback
          resolution; a background/track residual additionally grows ∝
          drift rate × update interval — that growth is the caller's
          responsibility to model, this function takes the residual as
          given). δM = drift_resid · M ⊙ η with η elementwise unit-RMS
          (seeded), so ‖δM·x‖/‖M·x‖ ≈ drift_resid incoherently.

    Zeroed inputs (n_photons=None, drift_resid=0) give the exact product
    y = M@x to machine precision — the ideal the twin compares against.
    """
    M = np.asarray(M, dtype=float)
    x = np.asarray(x, dtype=float)
    y_ideal = M @ x
    if n_photons is None and drift_resid == 0.0:
        return y_ideal
    rng = np.random.default_rng(seed)
    if drift_resid != 0.0:
        eta = rng.standard_normal(M.shape)
        M_eff = M * (1.0 + drift_resid * eta)
    else:
        M_eff = M
    y = M_eff @ x
    if n_photons is not None:
        if n_photons <= 0:
            raise ValueError("n_photons must be positive")
        p = y ** 2 / np.sum(y ** 2)          # detected energy share
        n_det = n_photons * p
        # shot noise on the counted photons of element k
        y = y * (1.0 + rng.standard_normal(y.shape) / np.sqrt(n_det))
    return y


def mvm_relative_error(M, x, n_photons=None, drift_resid=0.0) -> float:
    """EXPECTED output relative error of the photonic MVM (no randomness).

    Closed form of the two mechanisms in `photonic_mvm`:

        shot term  = √(n_out / n_photons)      (1/√N scaling, energy-share
                   detection makes it M,x-independent — see photonic_mvm)
        drift term = drift_resid               (‖δM·x‖/‖M·x‖, incoherent)

    Returns √(shot² + drift²): independent error mechanisms add in
    quadrature. This is the number the bench checks the Monte-Carlo
    `photonic_mvm` against (in expectation), and the scaling laws it
    verifies directly (halving N → error/√2; drift floor independent of N).
    """
    M = np.asarray(M, dtype=float)
    x = np.asarray(x, dtype=float)
    n_out = M.shape[0]
    shot = math.sqrt(n_out / n_photons) if n_photons else 0.0
    return math.hypot(shot, drift_resid)


def photonic_transducer_cost() -> TransducerCost:
    """Electronic↔photonic link cost (E/O modulator + O/E photodiode).

    HONEST PROVENANCE: the values below are ORDER-OF-MAGNITUDE TARGETS for
    a silicon-photonics link, clearly labeled as targets — NOT measured and
    NOT from a cited device model. Before any §1.2 slack entry consumes
    them, they must be replaced by measured S_n / linearity / energy data
    (§8B acceptance: PD linearity <1%, ring drift <1 pm after calibration,
    4×4 MVM error <5%). They exist so the contract fields are populated and
    serializable, not as claims.
    """
    return TransducerCost(
        gain=1.0,                 # TARGET: unity small-signal E/O→O/E link (V/V), loss-compensated
        noise_rms=1e-9,           # TARGET: input-referred, shot-dominated @ µW-class optical power
        bandwidth_hz=1e10,        # TARGET: 10 GHz-class modulator/PD link (the reason photonics buys bandwidth)
        energy=1e-15,             # TARGET: ~fJ per element-operation, µW·ns-class link budget
        notes="TARGET values (order-of-magnitude, NOT measured): E/O + O/E "
              "link. Must be replaced by measured data per §8B acceptance "
              "before entering a slack axis.",
        extra={"provenance": "target-only", "link": "E/O + O/E"},
    )
