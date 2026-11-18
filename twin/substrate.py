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
