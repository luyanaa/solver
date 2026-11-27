"""B2 primitive macromodels (twin/macromodel.py) — small analytic objects
that carry the nonidealities SPICE measured on IHP SG13G2 primitives
(results/b2_extract.json, twin/spice/extract_b2.py).

DESIGN.md §7 B0 froze the primitive set (T1 gm/C integrator, T2 HBT
interface, current mirror, transmission gate); §6.19/B2 measured their
nonidealities.  This module turns those measurements into behavioral
parameters:

  * IntegratorStage  — finite DC gain A0 = gm/g_out_total (output
    conductance adds to the load: g_out_total = 1/ro + g_load), pole
    omega_p = g_out_total/C (equivalently GBW/A0), output swing
    saturation (tanh) and, for diff-pair drive, the Ebers-Moll
    sech^2 gm compression.  NOT a solver — a closed-form 1-pole object.
  * CurrentMirror    — measured ratio + Pelgrom sigma; sample() draws a
    mismatch instance.
  * TGSwitch         — measured Ron (+ sigma), parasitic C to substrate,
    charge-injection fraction.
  * NoiseModel       — white input-referred density; HBT is .noise
    MEASURED (6.38 nV/sqrt(Hz) white, excess white factor 1.897x vs the
    gamma=1 derivation); MOS is DERIVED 4kT*gamma/gm and labeled
    "derived-not-measured" (PSP103 OSDI device noise returns zero
    spectra in ngspice-44.2 — no number is fabricated).

Key measured fact this module encodes (and bench/run_b2_behavioral.py
reproduces): the MOS "integrator" self-coefficient is NOT gm/C but
(gm + g_out)/C.  From the b2 AC sweep (gm-cell B, Rl=65k, C=1pF) the
derived device output conductance is 15.9 uS at the TT bias (ro=62.7k,
lambda_fit=1.147/V at Id=13.9uA) — the leak adds ~10.6% to the pole that
the linear gm/C convention assigns.  HBT (lambda=0.0138/V, ro=3.6 MOhm)
is ~400x cleaner: leak/gmC = 0.036%.

Validity limits (each enforced/noted in run_b2_behavioral):
  * 1-pole small-signal model: valid where the measured AC magnitude is
    1-pole (log-mag fit residual < 1e-3); no second pole modeled.
  * ro / lambda_fit are corner- and bias-dependent (lambda_fit spans
    0.66-2.32 /V across ss/tt/ff here); the §6.9 ledger value
    lambda=0.970/V is a point measurement at the diode bias, not a
    constant.  ro is NOT rescaled with per-instance gm (bias unknown).
  * tanh diff-pair law ignores VBIC high-injection/Early effects — the
    measured sweep slope differs by ~3.8% from the ideal-tanh
    prediction (checked, reported as validity residual).
  * charge injection / clock feedthrough are NOT .measured; q_inj_fc is
    a declared model fraction (default 0.5, classic half-channel
    charge), clearly not a SPICE-extracted number.

Dependency direction: pure numpy/math + the b2_extract.json *reader*;
imports nothing from lowering/ or analysis/.  lowering/netlist.py
consumes the attach map produced here, not this module.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["IntegratorStage", "CurrentMirror", "TGSwitch", "NoiseModel",
           "load_b2_extract", "hbt_ro", "mos_corner_fits",
           "b2_nonideal_map", "VT", "FOURKT", "LAMBDA_HBT",
           "RL_AC_OHM", "C_BANK_F",
           # gf180 batch-1 (G2)
           "load_gf180_extract", "gf180_corner_fits", "gf180_nonideal_map",
           "gf180_pvt_factors", "gf180_noise_model",
           "GF180_C_BANK_F", "GF180_I_BIAS", "GF180_RL_OHM"]

# ---------------------------------------------------------------------------
# physical constants (300 K)
# ---------------------------------------------------------------------------
K_B = 1.380649e-23
Q_E = 1.602176634e-19
T_K = 300.0
VT = K_B * T_K / Q_E              # 25.85 mV thermal voltage
FOURKT = 4.0 * K_B * T_K          # 1.657e-20 J

# measured/ledger anchors (DESIGN.md §6.9/§6.19, results/b2_extract.json)
LAMBDA_HBT = 0.0138               # 1/V — HBT Early slope at the 20 uA bias
RL_AC_OHM = 65e3                  # extract_b2 gm-cell-B load resistor
C_BANK_F = 1e-12                  # B0 T1 C_bank = 1 pF


# ---------------------------------------------------------------------------
# results loader
# ---------------------------------------------------------------------------
def load_b2_extract(path: str | None = None) -> dict:
    """results/b2_extract.json → dict.  Defensive: lanes may ADD keys but
    the schema used here (corners/mc/mc_global/noise/notes) is stable."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(here), "results",
                            "b2_extract.json")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# IntegratorStage
# ---------------------------------------------------------------------------
@dataclass
class IntegratorStage:
    """gm/C integrator macromodel with the measured nonidealities.

    State equation (the ONLY dynamics this object claims):

        dV/dt = (gm * f_in(vin) - g_out_total * V) / C

    with f_in(vin) = vin for a single-device (MOS gate) drive, or the
    diff-pair compression for `drive="diffpair"`:
        f_in(vin) = v_sat * tanh(vin / v_sat),  v_sat = tanh_in * VT / 2
    so tanh_in=4 (default) reproduces the Ebers-Moll law tanh(vin/2VT).
    The output additionally saturates at +/- v_swing (tanh on the output
    voltage).

    Linearized (vin -> 0): dV/dt = (gm*vin - g_out_total*V)/C, giving
        A0     = gm / g_out_total           (finite DC gain)
        omega_p = g_out_total / C = (gm/C) / A0   (= GBW / A0, rad/s)
    The pole derivation is exact for this topology: the device output
    conductance 1/ro adds to any external load conductance at the state
    node; H(s) = A0/(1 + s/omega_p).
    """

    gm: float                     # S, small-signal transconductance
    C: float                      # F, state capacitance
    ro: float                     # ohm, device output resistance at bias
    v_swing: float                # V, output swing half-range
    g_load: float = 0.0           # S, external load conductance at the node
    tanh_in: float = 4.0          # diff-pair softness (4 -> tanh(vin/2VT))
    drive: str = "gate"           # "gate" | "diffpair"
    name: str = "stage"

    # ---- linear properties (closed form) ----
    @property
    def g_out_total(self) -> float:
        """Total conductance at the state node: device ro + external load."""
        return 1.0 / self.ro + self.g_load

    @property
    def A0(self) -> float:
        """Finite DC gain gm/g_out_total (infinite iff ro -> inf, g_load=0)."""
        return self.gm / self.g_out_total

    @property
    def pole_rad(self) -> float:
        """omega_p = g_out_total/C (rad/s).  GBW/A0 identity: (gm/C)/A0."""
        return self.g_out_total / self.C

    @property
    def leak_rad(self) -> float:
        """Integrator leak WITHOUT external load: 1/(ro*C) — the amount the
        self-coefficient grows beyond the ideal gm/C convention."""
        return 1.0 / (self.ro * self.C)

    @property
    def v_sat(self) -> float:
        """Diff-pair input saturation voltage (tanh_in * VT / 2; 2*VT at
        the default tanh_in=4)."""
        return self.tanh_in * VT / 2.0

    def freq_response(self, s):
        """H(s) = A0/(1 + s/omega_p); accepts complex s or jw array."""
        s = np.asarray(s)
        return self.A0 / (1.0 + s / self.pole_rad)

    def step_response(self, t, u=1.0):
        """Analytic step response of the linearized stage (V)."""
        t = np.asarray(t, float)
        return (self.gm * u / self.g_out_total) * (1.0 - np.exp(-t * self.pole_rad))

    # ---- nonlinearities ----
    def drive_gain_factor(self, vin):
        """Large-signal drive compression factor f_in(vin)/vin (1 at 0)."""
        vin = np.asarray(vin, float)
        if self.drive == "diffpair":
            vs = self.v_sat
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(vin == 0.0, 1.0, vs * np.tanh(vin / vs) / vin)
        return np.ones_like(vin)

    def drive_gm_factor(self, vin):
        """LOCAL small-signal gm compression at a bias point vin (the
        derivative of the drive law): sech^2(vin/v_sat) for a diff pair,
        1.0 for gate drive.  This is what an offset does to the
        transconductance AT the operating point (the averaged
        drive_gain_factor is the sweep-slope quantity, not this)."""
        if self.drive == "diffpair":
            return float(1.0 / np.cosh(float(vin) / self.v_sat) ** 2)
        return 1.0

    def respond(self, vin):
        """Large-signal steady-state output voltage (V): drive compression,
        then output swing saturation.  d respond/dvin|_0 == A0 exactly."""
        vin = np.asarray(vin, float)
        if self.drive == "diffpair":
            i_over_g = self.v_sat * np.tanh(vin / self.v_sat)
        else:
            i_over_g = vin
        out_lin = self.gm * i_over_g / self.g_out_total
        return self.v_swing * np.tanh(out_lin / self.v_swing)

    def as_dict(self) -> dict:
        return dict(name=self.name, gm=float(self.gm), C=float(self.C),
                    ro=float(self.ro), v_swing=float(self.v_swing),
                    g_load=float(self.g_load), tanh_in=float(self.tanh_in),
                    drive=self.drive, A0=float(self.A0),
                    pole_rad=float(self.pole_rad), leak_rad=float(self.leak_rad),
                    v_sat_V=float(self.v_sat))


# ---------------------------------------------------------------------------
# CurrentMirror
# ---------------------------------------------------------------------------
@dataclass
class CurrentMirror:
    """HBT current mirror: nominal ratio + LOCAL Pelgrom mismatch sigma.

    The center is the TT corner ratio (measured, extract_b2); the sigma
    is the per-instance mismatch sigma from the 24-sample ngspice MC
    (mismatch library, mm_ok=1) — the same distribution run_b2.py feeds
    to CalibratedTwin.vos_hbt_v.  Global-process variation is NOT here
    (it shifts every device together; extract_b2 keeps the two MCs
    separate and b2.json reports both).
    """

    ratio: float                      # TT corner mirror ratio
    sigma_ratio: float                # per-instance Pelgrom sigma
    name: str = "mirror"
    rng: np.random.Generator = field(default_factory=lambda:
                                     np.random.default_rng(0))

    def sample(self, n: int | None = None):
        """Draw mismatch instance(s): ratio*(1 + sigma*N(0,1))."""
        if n is None:
            return float(self.ratio * (1.0 + self.sigma_ratio * self.rng.standard_normal()))
        return self.ratio * (1.0 + self.sigma_ratio * self.rng.standard_normal(n))

    def copy_current(self, i_ref: float, ratio: float | None = None) -> float:
        return i_ref * (self.ratio if ratio is None else ratio)

    def as_dict(self) -> dict:
        return dict(name=self.name, ratio=float(self.ratio),
                    sigma_ratio=float(self.sigma_ratio))


# ---------------------------------------------------------------------------
# TGSwitch
# ---------------------------------------------------------------------------
@dataclass
class TGSwitch:
    """Transmission gate (T5/T7 sampling path): measured Ron + sigma.

    ron/sigma_ron come from the measured MC (tt 451.7 Ohm, sigma 6.94,
    corners 331.3 (ff) / 599.3 (ss)).  c_par and q_inj_fc are MODEL
    parameters, not SPICE-extracted quantities: c_par (F) is the drain/
    substrate parasitic assumed 0 unless declared, q_inj_fc is the
    fraction of channel charge injected on turn-off (default 0.5, the
    classic half-charge figure — a declared modeling choice, not data).
    """

    ron: float                        # ohm (measured)
    sigma_ron: float = 0.0            # ohm (measured MC sigma)
    c_par: float = 0.0                # F (declared, not measured)
    q_inj_fc: float = 0.5             # fraction (declared, not measured)
    name: str = "tg"
    rng: np.random.Generator = field(default_factory=lambda:
                                     np.random.default_rng(0))

    def sample(self, n: int | None = None):
        if n is None:
            return float(self.ron * (1.0 + (self.sigma_ron / self.ron)
                                     * self.rng.standard_normal()))
        return self.ron * (1.0 + (self.sigma_ron / self.ron)
                           * self.rng.standard_normal(n))

    def series_rc_pole_hz(self, c_load: float) -> float:
        """Pole the Ron sets with a load cap (Hz): 1/(2*pi*ron*c_load)."""
        return 1.0 / (2.0 * math.pi * self.ron * c_load)

    def tau(self, r_series: float = 0.0) -> float:
        """Open-time constant with optional series resistance (s)."""
        return (self.ron + r_series) * self.c_par

    def charge_injection_v(self, dv: float, c_load: float) -> float:
        """Voltage error from charge injection for a dV swing (V)."""
        if c_load <= 0:
            raise ValueError("charge_injection_v needs c_load > 0")
        return self.q_inj_fc * self.c_par * dv / c_load

    def as_dict(self) -> dict:
        return dict(name=self.name, ron=float(self.ron),
                    sigma_ron=float(self.sigma_ron), c_par=float(self.c_par),
                    q_inj_fc=float(self.q_inj_fc))


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------
@dataclass
class NoiseModel:
    """White input-referred voltage-noise density + provenance label.

    source is ALWAYS carried: "measured-hbt" (ngspice .noise on the
    VBIC HBT — real data) or "derived-not-measured" (MOS/TG path: the
    4kT*gamma/gm derivation; PSP103 OSDI device noise is unimplemented
    in ngspice-44.2, so nothing there is measurable with this
    toolchain).  `factor` is the excess variance factor vs the gamma=1
    derivation (HBT measured 1.897 — i.e. gamma_eff = 1.897/2·... the
    §6.19 white factor; SNR optimism 10*log10(factor) = 2.78 dB).
    """

    density_v_rthz: float             # V/sqrt(Hz), flat white
    source: str                       # "measured-hbt" | "derived-not-measured"
    factor: float = 1.0               # variance ratio vs gamma=1 derivation
    note: str = ""

    def variance(self, f_lo: float, f_hi: float) -> float:
        """Integrated variance over a one-sided band [f_lo, f_hi] (V^2)."""
        if f_hi < f_lo:
            raise ValueError("band reversed")
        return self.density_v_rthz ** 2 * (f_hi - f_lo)

    @classmethod
    def from_b2_hbt(cls, ext: dict, corner: str = "tt") -> "NoiseModel":
        """HBT interface noise from the measured record.  Density comes
        from var_inoise_white/bw (the white band normalized density),
        factor from the recorded gamma_eff white factor."""
        h = ext["noise"]["hbt"][corner]
        bw = ext["noise"]["hbt"]["gamma_eff"]["bw"]
        dens = math.sqrt(h["var_inoise_white"] / bw)
        fac = ext["noise"]["hbt"]["gamma_eff"]["white_factor"]
        return cls(density_v_rthz=dens, source="measured-hbt", factor=fac,
                   note="ngspice .noise VBIC HBT diff pair, white band "
                        "normalized; no in-band 1/f measured over 1k-1GHz")

    @classmethod
    def derived_mos(cls, gm: float, gamma: float = 0.5) -> "NoiseModel":
        """Derived 4kT*gamma/gm density — HONESTLY labeled: not a
        measurement (PSP103 OSDI device noise returns zero spectra in
        ngspice-44.2).  gamma=0.5 is the long-channel ideal; the real
        gamma_eff on this device is UNKNOWN until C1 or a toolchain fix."""
        return cls(density_v_rthz=math.sqrt(FOURKT * gamma / gm),
                   source="derived-not-measured", factor=1.0,
                   note="4kT*gamma/gm derivation, gamma=0.5; NOT measured "
                        "(PSP103 OSDI device noise unimplemented in "
                        "ngspice-44.2)")

    def as_dict(self) -> dict:
        return dict(density_v_rthz=float(self.density_v_rthz),
                    source=self.source, factor=float(self.factor),
                    note=self.note)


# ---------------------------------------------------------------------------
# corner fits from the measured AC record
# ---------------------------------------------------------------------------
def mos_corner_fits(ext: dict, rl_ohm: float = RL_AC_OHM,
                    c_f: float = C_BANK_F) -> dict:
    """Per-corner macromodel parameters derived from the b2 AC record.

    Uses the GBW/A0 identity: with the measured low-frequency gain A0
    (AC data, DC bin) and the measured op-point device gm (gm_mos2_param,
    the device actually on the AC node — cell B at Vgs=0.3837), the
    total node conductance is g_out_total = gm/A0 and the pole is
    g_out_total/C.  The device's own share is 1/ro = g_out_total - 1/Rl
    (Rl = 65k load), and lambda_fit = 1/(ro*Id) at the measured corner
    Id (id_mos_pole).  Returns {tag: {...}} for every corner that has
    gm_mos2_param and id_mos_pole; corners without an AC record get
    A0=None and are skipped by consumers.
    """
    out = {}
    for tag, c in ext["corners"].items():
        gm2 = c.get("gm_mos2_param")
        ident = c.get("id_mos_pole")
        if gm2 is None or ident is None:
            continue
        a0 = c.get("_a0_ac")            # filled by the bench from the .dat
        if a0 is None:
            continue
        g_total = gm2 / a0
        g_dev = g_total - 1.0 / rl_ohm
        ro = 1.0 / g_dev if g_dev > 0 else float("inf")
        lam = 1.0 / (ro * ident) if math.isfinite(ro) else float("nan")
        out[tag] = dict(gm2=gm2, A0=a0, g_out_total=g_total,
                        ro_ohm=ro, id_a=ident, lambda_fit=lam,
                        pole_pred_hz=g_total / c_f / (2.0 * math.pi),
                        pole_meas_hz=c.get("pole_hz"))
    return out


def hbt_ro(ext: dict, lam: float = LAMBDA_HBT) -> float:
    """HBT output resistance from the measured corner Ic and the ledger
    Early slope: ro = 1/(lambda*Ic).  lambda=0.0138/V is the §6.9
    four-PDK measured value at the 20 uA bias."""
    ic = ext["corners"]["tt"]["ic_hbt"]
    return 1.0 / (lam * ic)


# ---------------------------------------------------------------------------
# nonideality attachment map for BehavioralNetlist
# ---------------------------------------------------------------------------
def b2_nonideal_map(netlist, ext: dict, device: str = "auto",
                    v_swing: float = 0.5) -> dict:
    """Per-integrator nonideality map (name -> instance dict) from the
    measured record, for lowering.netlist.attach_nonidealities.

    device: "mos" (default for auto — B0 T1 state integrators are
    MOS/MIM per the §7 freeze; HBT stages appear on interface EDGES,
    which are not self-coefficients), "hbt", or "auto" (== "mos").

    Each entry carries: device tag, gm implied by the element's own
    a_ii (|pole| * C_bank[pF] * 1e-12), ro from the measured corner fit
    (MOS: TT-fit 62.7k; HBT: 3.62M from lambda*Ic), leak_rad_s = 1/(ro*C),
    A0, v_swing (DESIGN headroom estimate, NOT measured), diff-pair
    v_sat (HBT), noise density + provenance, and the interface vos/TG
    sigmas that cal must absorb.  When a config is compiled WITHOUT an
    extraction (extraction=None path), attach is simply not called and
    the netlist stays ideal — the ideal default lives in
    BehavioralNetlist.nonideal == {}.
    """
    tt = ext["corners"]["tt"]
    mc = ext.get("mc") or {}
    fits = mos_corner_fits(ext)
    fit_tt = fits.get("tt")
    ro_mos = fit_tt["ro_ohm"] if fit_tt else float("inf")
    lam_tt = fit_tt["lambda_fit"] if fit_tt else float("nan")
    ro_hbt = hbt_ro(ext)
    n_hbt = NoiseModel.from_b2_hbt(ext)
    n_mos = NoiseModel.derived_mos(tt["gm_mos_fd"])
    vos_sigma = (mc.get("vd0_offset_raw", {}).get("sigma", 0.0))
    map_out = {}
    for itg in netlist.integrators:
        c_f = float(itg.C_bank) * 1e-12
        gm_needed = abs(float(itg.pole)) * c_f
        if device == "hbt":
            dev, ro, nmodel = "hbt", ro_hbt, n_hbt
            v_sat = 2.0 * VT
        else:
            dev, ro, nmodel = "mos", ro_mos, n_mos
            v_sat = float("nan")        # gate drive: no diff-pair tanh
        leak = 1.0 / (ro * c_f) if math.isfinite(ro) and ro > 0 else 0.0
        map_out[itg.name] = dict(
            device=dev, gm_S=gm_needed, C_F=c_f, ro_ohm=ro,
            A0=gm_needed * ro, leak_rad_s=leak,
            lambda_fit_mos=lam_tt,
            v_swing_V=v_swing,
            v_sat_V=v_sat,
            noise_density_v_rthz=nmodel.density_v_rthz,
            noise_source=nmodel.source,
            noise_factor=nmodel.factor,
            vos_sigma_V=vos_sigma,
            tg_ron_ohm=float(tt["ron_tg"]),
            validity="ro at the measured TT reference bias; NOT rescaled "
                     "with per-instance gm (bias unknown); v_swing is a "
                     "headroom estimate, not a measurement")
    return map_out


# ---------------------------------------------------------------------------
# GF180MCU batch-1 (G2): the same objects fed from results/gf180_extract.json
# ---------------------------------------------------------------------------
GF180_C_BANK_F = 1e-12            # B0 T1 C_bank = 1 pF (same value, same role)
GF180_I_BIAS = 10e-6              # extraction bias: lambda <-> ro is pinned
GF180_RL_OHM = 100e3              # 3v3 gm-cell resistive load (extract_gf180)


def load_gf180_extract(path: str | None = None) -> dict:
    """results/gf180_extract.json -> dict (G1's measured primitive record)."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(here), "results",
                            "gf180_extract.json")
    with open(path) as fh:
        return json.load(fh)


def gf180_corner_fits(ext: dict, c_f: float = GF180_C_BANK_F) -> dict:
    """Per-corner/per-temperature macromodel parameters from the G1 record.

    The gf180 pole cell is an IDEAL current source into 1 pF, so the pole
    is set by the device's own r_o only: omega_p = 1/(r_o*C), i.e.
    r_o = 1/(2*pi*f_pole*C) and lambda = 1/(r_o*I_BIAS).  gm comes from the
    resistive-load DC gate sweep (gm = dId/dVg, the corner netlist's cell),
    and A0 = gm*r_o is the measured intrinsic gain — the same identity the
    b2 lane used, but here g_out_total == 1/r_o (no external load in the
    pole cell), which is why the self-coefficient moves with Id and not
    with gm.

    Returns {tag: dict} for tt/ss/ff corners, plus every PVT and joint row
    that carries the probe (gm_fd + pole_hz).
    """
    out = {}
    for tag, c in ext["corners"].items():
        pole = c.get("pole_hz")
        if pole is None or pole <= 0:
            continue
        ro = 1.0 / (2.0 * math.pi * pole * c_f)
        out[tag] = dict(fam=tag.split("_")[0], temp_c=c.get("temp_c", 27.0),
                        proc=tag.split("_")[1], source="corner",
                        gm=c.get("gm_fd"), pole_hz=pole, ro_ohm=ro,
                        lambda_ohm_inv=1.0 / (ro * GF180_I_BIAS),
                        id_a=c.get("id_at_gmprobe_a"),
                        a0=c.get("a0_gm_ro"),
                        mirror_ratio=c.get("ratio_vds_mismatch"),
                        a_intrinsic_fd=c.get("a_intrinsic_fd"))
    for lane in ("pvt", "joint"):
        for tag, r in ((ext.get(lane) or {}).get("rows") or {}).items():
            pole = r.get("pole_hz")
            gm = r.get("gm_fd")
            if pole is None or pole <= 0 or gm is None:
                continue
            ro = 1.0 / (2.0 * math.pi * pole * c_f)
            out[tag] = dict(fam=r["fam"], temp_c=r["temp_c"],
                            proc=r.get("section", "tt"), source=lane,
                            gm=gm, pole_hz=pole, ro_ohm=ro,
                            lambda_ohm_inv=1.0 / (ro * GF180_I_BIAS),
                            id_a=r.get("id_a"), a0=gm * ro,
                            mirror_ratio=r["mirror_ratio"],
                            a_intrinsic_fd=r.get("a_intrinsic_fd"),
                            bias_ok=r.get("bias_ok", True))
    return out


def gf180_noise_model(ext: dict, fam: str = "3v3", corner: str = "tt",
                 kind: str = "nmos") -> NoiseModel:
    """Measured MOS input-referred white density from the G1 record.

    Unlike IHP (PSP103 OSDI device noise unimplemented -> derived-only),
    gf180's `.noise` on the resistive-load cell is real: the record carries
    sv_white (output plateau), A0 = gm*(Rl||ro) and the derived input
    density.  Source label is "measured-mos" with the measured white
    factor vs the gamma=1 ideal kept in `factor`.
    """
    ent = ext["noise"][kind]["%s_%s" % (fam, corner)]
    return NoiseModel(density_v_rthz=float(ent["density_in_nv_rthz"]) * 1e-9,
                      source="measured-mos-gf180",
                      factor=float(ent["gamma_eff"]["white_factor"]),
                      note="ngspice .noise on the resistive-load nmos cell, "
                           "band plateau + 1/f two-band fit; 1/f corner "
                           "%.1f MHz" % ((ent.get("flicker_corner_hz") or 0)
                                         / 1e6))


def gf180_nonideal_map(netlist, ext: dict, fam: str = "3v3",
                       v_swing: float = 1.5) -> dict:
    """Per-integrator nonideality map for gf180, for
    lowering.netlist.attach_nonidealities.

    Same schema as b2_nonideal_map (so BehavioralNetlist does not care
    which process it is looking at), but every number is G1-measured:
    gm implied by the element's own a_ii, r_o/lambda from the 1 pF pole
    cell, noise density + white factor from `.noise`, vos sigma
    input-referred (Pelgrom), TG Ron + sigma, mirror sigma.

    v_swing default 1.5 V is the 3.3 V family's usable half-range (supply
    3.3 V, 0.3 V headroom both sides) — a declared headroom estimate, not a
    measurement, exactly as in the b2 map.
    """
    tt = ext["corners"]["%s_tt" % fam]
    mc = (ext.get("mc") or {}).get(fam) or {}
    fits = gf180_corner_fits(ext)
    fit_tt = fits.get("%s_tt" % fam, {})
    ro = fit_tt.get("ro_ohm", float("inf"))
    lam = fit_tt.get("lambda_ohm_inv", float("nan"))
    nmodel = gf180_noise_model(ext, fam=fam)
    # vos: the record's vd0 is the DIFFERENTIAL OUTPUT offset; the
    # input-referred sigma is that over 2*|A_dm->se| (see run_gf180 J12)
    vos_out = (mc.get("vd0_offset") or {}).get("sigma", 0.0)
    a_se = abs((mc.get("gain_diff") or {}).get("mean", 0.0)) or 1.0
    vos_in = vos_out / (2.0 * a_se)
    ron = float(tt["ron_tg"])
    ron_sig = (mc.get("ron_tg") or {}).get("sigma", 0.0)
    map_out = {}
    for itg in netlist.integrators:
        c_f = float(itg.C_bank) * 1e-12
        gm_needed = abs(float(itg.pole)) * c_f
        leak = 1.0 / (ro * c_f) if math.isfinite(ro) and ro > 0 else 0.0
        map_out[itg.name] = dict(
            device="mos", process="gf180mcu-%s" % fam,
            gm_S=gm_needed, C_F=c_f, ro_ohm=ro,
            A0=gm_needed * ro, leak_rad_s=leak,
            lambda_fit_mos=lam, id_bias_a=GF180_I_BIAS,
            v_swing_V=v_swing, v_sat_V=float("nan"),   # gate drive: no tanh
            noise_density_v_rthz=nmodel.density_v_rthz,
            noise_source=nmodel.source, noise_factor=nmodel.factor,
            vos_sigma_V=vos_in,            # input-referred (Pelgrom, measured)
            vos_sigma_out_V=vos_out,
            mirror_sigma_rel=(mc.get("ratio_mis") or {}).get("sigma", 0.0),
            tg_ron_ohm=ron, tg_ron_sigma_ohm=ron_sig,
            validity="r_o/lambda/gm at the measured 10 uA, %s TT bias; NOT "
                     "rescaled per instance (bias unknown); v_swing is a "
                     "supply-headroom estimate, not a measurement" % fam)
    return map_out


def gf180_pvt_factors(ext: dict, fam: str = "3v3") -> dict:
    """Measured per-point relative factors for the joint lane.

    Returns {tag: dict(k_pole, k_gm, k_mirror, temp_c, proc, source,
    bias_ok)} where every k is relative to **that family's** TT/27 C point
    (the compiler's A_v target is process-agnostic; the realized
    coefficient is what moves).  k_pole is the self-coefficient axis
    (1/r_o*C = lambda*Id/C), k_gm the input-gain axis, k_mirror the edge
    axis — three axes, measured separately, never one global scale.
    """
    fits = gf180_corner_fits(ext)
    ref = fits["%s_tt" % fam]
    out = {}
    for tag, f in fits.items():
        if f["fam"] != fam:
            continue
        out[tag] = dict(k_pole=f["pole_hz"] / ref["pole_hz"],
                        k_gm=f["gm"] / ref["gm"],
                        k_mirror=f["mirror_ratio"] / ref["mirror_ratio"],
                        temp_c=f["temp_c"], proc=f["proc"],
                        source=f["source"], bias_ok=f.get("bias_ok", True),
                        pole_hz=f["pole_hz"], gm=f["gm"], ro_ohm=f["ro_ohm"],
                        mirror_ratio=f["mirror_ratio"], id_a=f.get("id_a"))
    return out
