"""Two-stage trim DAC, closed-loop trim, and held-out verify (P0-E, §6.29).

Why this module exists: §6.26's ``full`` mode assumes a *continuous* trim that
leaves only the declared readout floor (100 ppm). A realizable trim carries
three extra terms, each modeled here as a measured or declared number:

  (1) **finite codes.** A coarse+fine two-stage DAC has step
      ``2R/2**(Nc+Nf)`` and clamps at +-R: a chip whose deviation exceeds R
      cannot be trimmed at all (this must be *reported*, never silently
      absorbed). The fine stage must cover at least one coarse LSB, otherwise
      the code set has gaps and the worst residual grows to ``(-e_f)*lsb/2``
      even though the fine step is finer.
  (2) **a noisy reading.** The trim decision uses ``est = delta + n``; the
      resulting residual is ``-n - quantization``, so the residual law is
      ``sqrt((step/sqrt(12))**2 + sigma**2)`` -- quantization-limited below
      ``sigma ~ step``, measurement-limited above it. Dither does NOT bound the
      residual by step/2 once the read noise exceeds the step.
  (3) **actuator gain error.** The realized correction is ``(1+e)*q``; a
      one-shot trim therefore leaves ``e*|delta|`` and a second iteration
      leaves ``e**2*|delta|``. That is why ``twin/calib.py``'s retry path is
      load-bearing rather than decorative.

The other load-bearing fact is **how many independent knobs exist**. §6.26
measured the self (pole) and gain axes to be 96 % collinear at 3.3 V and to
separate at 6 V. Collinearity is enough for coarse tracking but not for a 1 %
trim: one joint knob can only cancel the *mean* of two deviations, leaving
``|delta_self - delta_gain|/2``, which the measured PVT table shows is 6.12 %
at 3v3/ff/125 C (9 of the 16 3v3 points exceed a 1 % workload) and 16.02 % at
6v0/ff/125 C. So the knob count, not the DAC bit count, is the binding
constraint -- and this module lets the caller enumerate both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TwoStageDAC:
    """A coarse+fine trim DAC over +-``range_rel`` (a relative deviation)."""

    range_rel: float = 0.35
    coarse_bits: int = 4
    fine_bits: int = 6
    overlap: float = 0.25          # fine span = (1+overlap) * one coarse LSB

    @property
    def lsb_coarse(self) -> float:
        return 2.0 * self.range_rel / 2 ** self.coarse_bits

    @property
    def step(self) -> float:
        """Nominal fine step == the achievable grid spacing when e_f = 0."""
        return 2.0 * self.range_rel / 2 ** (self.coarse_bits + self.fine_bits)

    @property
    def n_codes(self) -> int:
        return 2 ** (self.coarse_bits + self.fine_bits)

    def quantize(self, command: float, fine_gain: float = 0.0) -> float:
        """Realized correction nearest to ``command`` (clipped to +-range).

        ``fine_gain`` is the fine stage's *gain* error: its realized span is
        ``(1+fine_gain) * lsb_coarse`` instead of one LSB. A negative
        ``fine_gain`` below ``-overlap`` opens a gap at every coarse boundary.
        """
        q = min(max(float(command), -self.range_rel), self.range_rel)
        if abs(q) >= self.range_rel:                      # the rail is exact
            return q

        lsb = self.lsb_coarse
        fine = (1.0 + fine_gain) * lsb / 2 ** self.fine_bits
        n = int(math.floor((q + self.range_rel) / lsb))
        r = q + self.range_rel - n * lsb                  # in [0, lsb)
        k = int(round(r / fine)) if fine > 0 else 0
        k = max(0, min(k, 2 ** self.fine_bits))

        cands = []
        for kk in (k - 1, k, k + 1):
            if 0 <= kk <= 2 ** self.fine_bits:
                cands.append(-self.range_rel + n * lsb + kk * fine)
        if n > 0:                                          # previous cell top
            cands.append(-self.range_rel + (n - 1) * lsb + 2 ** self.fine_bits * fine)
        if n < 2 ** self.coarse_bits - 1:                  # next cell bottom
            cands.append(-self.range_rel + (n + 1) * lsb)
        cands.append(-self.range_rel)
        cands.append(self.range_rel)
        return min((c for c in cands if -self.range_rel <= c <= self.range_rel),
                   key=lambda c: abs(c - q))

    def max_residual(self, fine_gain: float = 0.0) -> float:
        """Worst |realized - command| over the reachable range (analytic).

        Covered region inside a cell has spacing ``fine``; an uncovered gap of
        width ``max(0, -fine_gain)*lsb`` opens at the top of every cell.
        """
        lsb = self.lsb_coarse
        fine = (1.0 + fine_gain) * lsb / 2 ** self.fine_bits
        gap = max(0.0, -fine_gain) * lsb
        return max(fine / 2.0, gap / 2.0)


@dataclass
class TrimSpec:
    """How the loop is allowed to operate."""

    knobs: Tuple[Tuple[str, ...], ...] = (("self",), ("gain",), ("mirror",))
    iters: int = 2
    read_sigma: float = 100e-6      # declared floor; add ADC quantization if any
    actuator_err: float = 0.03      # realized = (1+e) * commanded (T9 LSB/INL)
    tol_verify: float = 0.01        # held-out VERIFY acceptance threshold
    fine_gain: float = 0.0
    tol_axis: Dict[str, float] | None = None   # per-axis override (QoI-weighted)


def close_loop(deltas: Dict[str, float],
               dacs: Dict[str, TwoStageDAC],
               spec: TrimSpec,
               rng: np.random.Generator) -> dict:
    """Run the trim loop on one chip and then a fresh held-out verify.

    ``deltas`` are the true deviations (self/gain/mirror). Each knob group is
    moved by one Dac; a group with >1 axis is the single-knob (collinear)
    topology, whose command is the group mean (the minimax solution).

    Returns the *true* residuals (what the chip actually does), plus the
    verify decision made from a FRESH noisy reading of each axis.
    """
    state = {k: float(v) for k, v in deltas.items()}

    def measure():
        return {a: state[a] + (rng.normal(0.0, spec.read_sigma)
                               if spec.read_sigma > 0 else 0.0) for a in state}

    def accepted(meas):
        if spec.tol_axis:
            return all(abs(meas[a]) <= spec.tol_axis.get(a, spec.tol_verify)
                       for a in meas)
        return all(abs(meas[a]) <= spec.tol_verify for a in meas)

    clamped = False
    used = 0
    dec = measure()                 # the reading that commands pass 1
    for _ in range(max(1, spec.iters)):
        for grp in spec.knobs:
            members = [a for a in grp if a in state]
            if not members:
                continue
            dac = dacs[members[0]]
            cmd = -float(np.mean([dec[a] for a in members]))
            if abs(cmd) >= dac.range_rel:
                clamped = True
            realized = (1.0 + spec.actuator_err) * dac.quantize(
                cmd, spec.fine_gain)
            for a in members:
                state[a] += realized
        used += 1
        dec = measure()             # a fresh reading decides on another pass
        if accepted(dec):
            break

    # held-out VERIFY: a *different* reading of the committed state
    est = measure()
    verified = accepted(est)
    return dict(residual={a: float(state[a]) for a in state},
                est=est, clamped=bool(clamped), verified=bool(verified),
                iterations=int(used))


def residual_law(spec: TrimSpec, dac: TwoStageDAC, passes: int = 1) -> float:
    """RMS residual after ``passes`` passes: quantization + injected noise.

    Each pass measures once, so the injected noise accumulates as
    ``sqrt(passes) * sigma`` while the quantization term is bounded by
    ``step/sqrt(12)`` per pass; a settled loop (actuator error e) additionally
    carries ``e**passes * |delta|``.
    """
    return math.sqrt((dac.step ** 2) / 12.0 + passes * spec.read_sigma ** 2)


def one_shot_law(delta: float, spec: TrimSpec) -> float:
    """|residual| of a single iteration with actuator error e (no noise)."""
    return abs(spec.actuator_err * delta)


def iterated_law(delta: float, spec: TrimSpec) -> float:
    """|residual| after k iterations with actuator error e (no noise)."""
    return abs(spec.actuator_err) ** max(1, spec.iters) * abs(delta)


def verify_accept_prob(r: float, spec: TrimSpec) -> float:
    """Analytic P(held-out verify accepts | true residual r), Gaussian read.

    Accept iff |r + n| <= tol with n ~ N(0, sigma), i.e.
    Phi((tol-r)/sigma) - Phi((-tol-r)/sigma).
    """
    if spec.read_sigma <= 0:
        return 1.0 if abs(r) <= spec.tol_verify else 0.0
    z1 = (spec.tol_verify - r) / spec.read_sigma
    z2 = (-spec.tol_verify - r) / spec.read_sigma
    return float(0.5 * (math.erf(z1 / math.sqrt(2.0))
                        - math.erf(z2 / math.sqrt(2.0))))


def required_range(deltas: Iterable[float]) -> float:
    """Smallest symmetric range that trims every deviation in ``deltas``."""
    return float(max((abs(float(d)) for d in deltas), default=0.0))
