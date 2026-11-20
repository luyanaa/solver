# Chapter 2. Layered Fabric Architecture

> **Scope.** This chapter is the English rendition of §3 of `DESIGN.md`. It turns the four design conclusions and seven scaling laws of Chapter 1 into a concrete hardware organization: the tile inventory and its interface contracts, the Computed routing/switch matrix, the stochastic tiles (T8/T9), the sparse constraint fabric (T10), device/voltage classes, calibration, and the generalization to heterogeneous substrates. Section references §1, §4–§10 point to `DESIGN.md`; mathematical preliminaries are in Chapter 1 of this report.

---

## 2.1 The layered fabric: 10 tiles + 1 supervisor

We do **not** build "many op-amps plus one huge crossbar." The reason is not aesthetics but §2.5 (of DESIGN.md): crossbar parasitics grow linearly with size and tolerance is amplified by Q; their product is uncontrollable.

| # | Tile | Semantics | Error model | Hard constraint |
|---|---|---|---|---|
| T1 | **State** | C·V̇ = I, high-impedance storage in the V domain | α kT/C + leakage | a high-impedance node connects **only** to its local C |
| T2 | **Algebra** | current-domain SUM/MUL/NL (Gilbert, tanh) | g_m mismatch, finite swing | only at low-impedance nodes |
| T3 | **Operator** | programmable MVM (T, T⁻¹, dense A) | weight mismatch ∝ 1/√(WL) | size ≤ 256; split above that |
| T4 | **Spectral/Stiff** | diagonal pole bank ż_k = −λ_k z_k + (·)_k | independent kT/C per branch | τ covers 1e-8 … 1e-3 |
| T5 | **Flux** | cell C + nonlinear flux + limiter | loss of monotonicity, g_m mismatch | KCL is inherently conservative |
| T6 | **Event** | comparator + event latch + reset + S&H | jitter ẋ·δt, metastability | hysteresis prevents Zeno |
| T7 | **Route** | local current grid + buffered trunk + switch matrix | Y_c(s) = G_c + sC_c | only current crosses islands |
| T8 | **Noise** | physical thermal noise / replayable PRNG+DAC; programmable PSD, τ_c, ρ | 1/f, bandwidth, quantization, correlation | B_noise ≥ 100·B_req |
| T9 | **Statistics** | on-chip mean/covariance/histogram/likelihood accumulation | multiplier mismatch, integration leakage | read out only O(p²) statistics |
| T10 | **Constraint** | sparse algebraic constraints A(x)y=b(x): a physical relaxation mesh for Poisson/KCL/pressure projection | relaxation speed, conditioning, residual | shared field/constraint resource |
| S | **Supervisor** | digital: configuration, basis change, calibration, timing | — | not in the analog signal path |

---

## 2.2 Two inviolable interface contracts

1. **Impedance-domain annotation**: every network must declare whether it is *current* (low impedance, long distance, summable) or *voltage* (high impedance, local C only). Crossing islands is always current.
2. **Every tile must supply (E,A,B,C) together with the two numbers ‖H‖₂ and ‖H‖∞**, otherwise it cannot enter layout. This binds "analog design" and "the compiler" to the same mathematics.

---

## 2.3 Computing the switch matrix (answer to problem 2)

For each routing configuration c:

1. PEX → Y_c(s) = G_c + sC_c (add s⁻¹L⁻¹ or a frequency-dependent macromodel if needed).
2. Assemble E_c·ẋ = A_c·x + Bu ⇒ H_c(s) = C(sE_c − A_c)⁻¹B.
3. Compare with the ideal: η_route = sup_ω ‖H_real − H_ideal‖₂ / ‖H_ideal‖₂, plus group-delay error τ_g and crosstalk ‖H_ij‖∞/‖H_ii‖∞.
4. Closed-loop criterion: ‖G_loop·Δ_route‖∞ < 1, with Δ_route a **structured bound** (per-switch R_on and C intervals) and μ rather than a norm, otherwise the check is overly conservative.
5. Keep the φ / X_talk terms in the cost function, but **derive their weights from sensitivities ∂σ_y/∂p**, not by hand tuning.

As one expression:

    C_route = w_R R + w_C C + w_φ Δφ + w_X X_talk + w_N N_noise

The weights come from the sensitivities ∂σ_y/∂p of the current dynamical graph and from Γ; wire length is just one term among these. The R_on / parasitic-C conflict (wider reduces R_on but raises C) appears precisely through G_c and C_c both entering Y_c — what is optimized is ‖H_c‖∞, not R_on.

**A configuration switch performs only quasi-static configuration; a switch being simulated must be a computational primitive.** One must not toggle a routing transmission gate every time the emulated switch acts — that would mix R_on, C_off, charge injection and clock feedthrough into the model. An ideal switch uses a finite-slope G(V_c); a real MOS uses the compact model's finite bandwidth; topological events use T6.

**Signals are divided into four levels, and fast states may use only the first two.** local / neighbor (same or adjacent island) travel as direct current; regional uses a buffered differential/current-mode trunk; global carries only slow control, parameters, references and constraint results. Configuration travels by digital distribution plus local DAC — never a global analog parameter voltage, since noise on one global reference would become chip-wide correlated noise that ensemble averaging cannot remove.

**A buffer is not memory — it is impedance isolation plus energy restoration.** In a continuous-time machine all states exist simultaneously; there is no need to "store and then send." A buffer reduces the total load seen by the state capacitor from C_s+ΣC_in+C_wire to C_s+C_in,buf, with the larger downstream load driven by the buffer. The cost is that every buffer adds G(s), offset, THD and noise; the depth of a fanout tree is limited by N_b σ_b² and cannot be extended without bound like a digital clock tree.

**For long lines, reduce contributions at the source rather than broadcasting the state.** A conventional crossbar broadcasts x_j to every sink; a better approach computes I_ij = g_ij V_j near the source, sums by region before routing, and feeds the destination summing node directly. The object of communication changes from a global state to a locally reduced contribution, and the fanout tree naturally becomes shallower.

**Per-edge closure is not "length < L" — it is, over 0 < ω < ω_dyn,max,**

    H_ij(s) = H_wire(s) · ∏_k H_buf,k(s) ≈ g_ij e^{−sτ_ij}

simultaneously satisfying |g_ij − g_ij^ideal| < ε_g, |φ_ij − φ_ij^ideal| < ε_φ and S_n,route < S_n,budget. This is the per-edge version of the second condition in §1.3.8; the φ term of C_route comes from exactly this condition.

`twin/route.py` makes this computable: the wire defaults to a distributed 1/cosh√(sRC) (not a single pole), the buffer to g/(1+s/ω_p), and the pad to an all-pass (1−sτ/2)/(1+sτ/2); `route_closure` returns the four passes (gain/phase/noise/feedback-absolute), η_route, J_H, J_τ and a latency class in one call. §6.15 gives its 38 acceptance checks.

Every edge also carries a `transposeable` flag: the forward direction uses `I_j += g·V_i`, the adjoint uses `I_i += g·V_j`, and `A ↔ A^T` merely reconfigures the switch matrix (§5.8 of DESIGN.md). Non-transposeable elements — comparators, events, reset — must be marked explicitly; the adjoint switches strategy at segment boundaries, and one must not pretend the whole path is reversible.

---

## 2.4 T8/T9: noise contract and inverse-problem readout

T8 is not "a noise source" but a **bank**:

- **Physical mode**: independent resistor / OTA thermal noise; PSD set by temperature, R, gain; bandwidth set by parasitics / amplifier, up to GHz.
- **Replay mode**: PRNG → DAC → reconstruction filter; seed, bit count, update rate and filter all enter the configuration.
- **Correlation matrix**: T3's mixing matrix synthesizes ρ_ij from m independent sources; mixing-matrix mismatch enters Δ and is not assumed ideal.
- **Calibration**: all three quantities PSD, τ_c and ρ are tuned on chip; the calibration residual enters the P6 stochastic certificate directly.

T9 is the key to inverse problems: without it, the raw waveforms of N trajectories would overload the ADC/link.

- Accumulation: time average, covariance, cross-correlation, histogram (comparator bank), Gaussian log-likelihood.
- Readout: only O(p²) numbers are returned per ensemble; EnKF/SMC/optimization runs on the digital side.
- Constraints: T9's multiplier mismatch and integration leakage enter Δ; long integrations need reset/chopping, otherwise 1/f and leakage corrupt the statistics.

Noise routing has a contract of its own: T7's parasitic capacitance/resistance changes the noise PSD, and shared supply/ground creates ρ_ij. Noise crossing islands travels as a differential current pair, not a single-ended voltage.

The whole-machine hierarchy (not a new architecture — the §3 tiles re-annotated with SDE semantics):

    digital host
      ├─ equation compiler (P0–P7)
      ├─ placer / router
      └─ optimizer / calibration / seed
              │
    ─────────────────────────────────────
    Analog dynamical fabric
      State T1/T4 → Drift T2/T3 → Σ → ∫ → State
                        ↑                    │
      Diffusion T2/T3 ← Noise T8            │
      Constraint T10 (Poisson/KCL) ↔ State/Flux
      Event T6 (comparator/reset) ──────────┘
      Observation → T9 → ADC
      Route T7 (configuration, parasitics, correlation)

---

## 2.5 Constraint fabric: Poisson / KCL / pressure projection

Three workload classes share not an "ODE integrator" but a sparse algebraic constraint:

    A(x) y = b(x)

- **Coulomb in MD**: ∇·(σ∇φ)=ρ; the node voltages of a resistor/g_m mesh are the discrete Poisson solution.
- **Pressure projection in NS**: ∇²p = −ρ∇·[(u·∇)u]; the same mesh, with the source term coming from advection.
- **KCL in SPICE**: Σ I=0; nodes without independent dynamic states are algebraic variables.

The hardware is not a digital solver but fast relaxation:

    C_y ẏ = −A y + b,    τ_relax ~ C_y/λ_min(A)

This gives monotone relaxation only for SPD / positive-definite symmetric-part A; for asymmetric KCL one must use λ_min((A+Aᵀ)/2) or the spectral abscissa and check feedback stability — the M-matrix property of a resistor mesh cannot simply be carried over to an active g_m network. The steady state is the constraint solution; residual, conditioning and relaxation speed enter P6. Dynamic states live in T1/T4, algebraic nodes in T10; the two couple through a current/voltage interface. T10 is a shareable resource: MD's field, NS's pressure and SPICE's KCL use the same programmable sparse mesh, differing only in source terms and boundary conditions.

**Timing condition of the constraint fabric.** The true model requires A y=b to be instantaneous; hardware gives

    τ_c ẏ + A y = b

so one must guarantee

    ε_c = τ_constraint / τ_state,min ≪ 1

otherwise MD shows spurious retardation, NS pressure projection fails to propagate, and SPICE KCL does not converge. τ_constraint is set by λ_min((A+Aᵀ)/2) and C_y; this is the third condition of §1.3.8 and the only hard timing contract between T10 and T1.

**The denominator of ε_c is the external timescale, not the constraint's own pole (P0 finding fix, `results/constraint_timing.json`).** τ_state,min is the fastest timescale of the **external constrained dynamics** (NS's τ_fluid,min, MD's particle timescale) and must be declared explicitly by the workload (`eps_c(mesh, c_node, external_state_timescale)` / `solve_constraint(..., external_state_timescale=)`). Deriving `τ_state_min = 1/max|λ|` from the constraint subsystem's own poles is meaningless — a constraint network's ratio to its own fastest mode does not answer "does the constraint converge before the state needs it?" (on poisson-2d that ratio measured 1.17e8, a spurious FAIL). A pure Poisson steady state has no external dynamics ⇒ **no ε_c to compute**: the verdict becomes a matter of fact — (solve_deadline reached? residual < tol?) giving PASS/FAIL on convergence, or `"STEADY"` with only τ_relax + residual reported when there is no deadline — without fabricating a ratio or flagging infeasibility. An ε_c FAIL must enter the P6 certificate (`Certificate.constraint` field, `evaluate_sys(..., constraint=)`): FAIL ⇒ `status="rejected"`; a pure steady-state fact ⇒ the certificate keeps "no violation on evaluated axes" and records the deadline/residual facts in the constraint entry.

| workload | hardest part | principal fabric |
|---|---|---|
| MD | moving particles + long-range force | local particle T1/T2 + T10 Poisson |
| NS | multiscale cascade + incompressibility | local conservation T5 + multiresolution + T10 |
| SPICE | DAE + extreme stiffness + events | T1/T4 + T10 KCL + T6 event |

**Conservation laws are built into the topology, not matched**: MD pair forces are generated in pairs; CFD flux leaving cell A enters cell B in equal amount; SPICE node currents sum automatically. Conservation is therefore not broken to first order by transistor mismatch.

**Correction to the top-level architecture**: the core is not an "analog integrator array" but a **continuous state fabric plus a sparse constraint fabric**; integrators are only half of it.

---

## 2.6 Devices and voltage classes: do not ask one MOS to solve speed, noise and swing at once

| class | typical devices | use | cost |
|---|---|---|---|
| A0 ±0.2–0.4 V | LV RF MOS / HBT | high speed, low-noise g_m, MVM, buffer | small swing; SNR relies on C |
| A1 ±1–2 V | 3.3/5 V MOS + MIM | state, integrators, regional drivers | moderate speed |
| A2 5–20 V | HV MOS/LDMOS | large swing, global drivers, special constraints/interfaces | large C_gd/C_oss/R_on, low f_T |

Three rules:

- **HV devices do not sit on RF nodes**: a 500 fF C_off is 159 Ω at 2 GHz — no longer an "off" switch.
- The **RF/fast domain** sees only LV/MV MOS, HBT, Schottky, MIM and low-C switches; the **HV domain** sees only LDMOS, thick-oxide devices and large diodes; the two are joined by buffers/transconductors/level converters.
- **HV's advantage is swing**: raising V_FS from 0.5 V to 3 V improves kT/C SNR by 15.6 dB, or reduces C by 36× at the same SNR. The cost is area and parasitics; hence HV is used only for "large-scale" global/constraint/actuator functions, not dense state.

**Process choice**: neither a single RF CMOS nor a single BCD process is optimal; the most interesting option is an RF-capable BCD/HVCMOS/SOI mixed-signal process (of the XT011/XH035 class), because it lets speed, noise, swing, isolation and global drivers be assigned to different devices. The value of SOI/DTI is not merely voltage tolerance but separating the noisy high-swing domain from low-noise analog dynamics. HBT's g_m/I_C = 1/V_T ≈ 38.7 V⁻¹ suits line drivers/buffers, not state storage; MOS/MIM store state; CMOS digital does configuration/calibration.

---

## 2.7 Calibration: calibrate the equations, not the transistors

The goal is not g_m = 1.000000 mS but

    the dimensionless dynamics implemented by the current hardware ≈ the target dimensionless dynamics.

The error is decomposed into four terms: **common-mode scale** (global g_m/C drift, equivalent to changing A_t), **low-rank systematic** (spatial gradients / temperature), **local mismatch** (Pelgrom) and **random noise** (not calibratable, only characterized). The first two need only a scaling update; the third is calibrated locally; the fourth enters the P6 stochastic certificate.

**Primitive behavioral model**: every INT/MUL/route/NL stores its measured H(s), offset, THD and noise PSD rather than nominal transistor parameters. The compiler targets the *calibrated digital twin of this particular chip*, not TT_25C.lib.

**Three timescales**: foreground (sweep gain/offset/pole/NL/noise at power-up), background (pilot + lock-in, with ω_p placed where workload energy is low), and run-time residual (replica + physical residual). Do not inject a pilot into a running computational node; use a replica tile (one per 32–64 tiles) or a shadow block.

**The physical residual is a free reference**: MD's ΔH, NS's ‖∇·u‖, SPICE's ΣI and KVL's ΣV are all R(x)=0. Minimizing ‖R(x)‖²+λL_pilot online is self-supervised calibration. Ratio accuracy matters more than absolute accuracy: unit-element arrays and ratiometric design.

**Noise cannot be calibrated away**: estimate S_n(f), Q and ρ_ij; SDE injection is Q_inj = Q_target − Q_hw ≥ 0. If Q_hw > Q_target, the only options are adding C, reducing bandwidth, adding bias, changing the device or changing placement.

**Placement can also adapt to hardware**: under a temperature gradient, migrate high-SNR states to low-noise tiles and put insensitive intermediates on hot tiles — cheaper than forcing the hardware back into nominal behavior.

---

## 2.8 Heterogeneous substrates: state as a generalized physical coordinate

T1–T10 assume by default that the state is a voltage. That is an implementation choice, not a mathematical requirement. Relax the descriptor of §1 to

    E_q·q̇ = F_q(q) + Σ_r C_qr·z_r ,    q ∈ { V, I, x_mech, P_fluid, c_chem, a_optical }

where q is the generalized coordinate, E_q the capacity of that domain (C, m, C_f, …), F_q the domain's constitutive law (I=g(V), −kx−cẋ, −L_R P, S·v(c), …), and C_qr the cross-domain transducer. Three things change:

1. **κ_hw becomes per-domain**. Electronic window ≈ 1e-8–1e-3 s, MEMS ≈ 1e-6–1e-1 s, microfluidic ≈ 1e-3–1e2 s, photonic ≈ 1e-12–1e-9 s (but with no storage). One system may use several windows at once, provided the cross-domain boundary is explicitly sampled or phase-locked; this is not "directly connecting two analog blocks of different speeds" (an explicit non-goal in §9) but a contractual multirate boundary.
2. **The transducer is a first-class citizen**. Every crossing costs gain, nonlinearity, noise, bandwidth and energy, and the transducer is itself a perturbation port of the kind in §1.2.2.
3. **The compiler's operator set changes from {+,×,∫} to {E_CMOS, E_MEMS, E_fluid, E_optical}**. The criterion is still a slack vector over (error, energy, bandwidth, area, communication), with each axis recomputed per substrate.

Status (A7, `results/substrate.json`, 40/40): MEMS `ω0=√(k/m)` and `Q=mω0/c` exact formulas plus the small-amplitude damped analytic response matched by RK4 (4.9e-11); the Duffing first-order-perturbation skeleton `ω(A)≈ω0(1+3k3A²/8k)` within <1% of RK4 frequency measurement at small amplitude (A² scaling ratio 3.99≈4, sign of k3 distinguishable both ways); after locking gm/C to ω_ref the residual is at the measurement noise floor (10 ppm) rather than thermal drift (1000 ppm), and is insensitive to a 100× change in drift — drift of the reference itself (risk 33 of §9) must be accounted separately; photonic 4×4 MVM: machine precision with zero noise, the shot term strictly ∝1/√N (energy-share detection makes it independent of M and x), and a drift residual floor independent of photon number (Monte Carlo agrees with analytic quadrature to ±20%). The fluidic diffusion case (1D fluidic diffusion twin) and real transducer measurements remain Phase 8 TODOs.

| substrate | most natural primitive | mathematical form | weak at |
|---|---|---|---|
| CMOS integrator | first-order ODE / integration | C·V̇ = I | high Q, long delay, second-order mechanical modes |
| CMOS nonlinear | multiply / saturate / compare | I=g(V), tanh | exact linearity, large dynamic range |
| MEMS resonator | second-order ODE / resonance | mẍ+cẋ+kx=F | speed, programmability, pull-in |
| resistor / g_m mesh | graph Laplacian / constraint | GV=I | dynamics, asymmetric coupling |
| microfluidics | transport / diffusion / conservation | C_f·Ṗ = −L_R P + Q | speed (ms–min) |
| chemical reaction network | polynomial nonlinear ODE | ċ = S·v(c) | controllability, observability, lifetime |
| photonics | linear transform / convolution / transmission | y=Wx, e^{−jβL} | nonlinearity, storage, short local links |

One PDE can be split by operator into four paths:

    ∂_t u = L[u] + N[u] + C[u] + η
    L → RC/MEMS/photonic physical operators; N → CMOS nonlinearity;
    C → elliptic/constraint fabric; η → calibrated physical noise

This is not a term-by-term translation of a digital algorithm into circuits — it places each operator on a medium that already obeys the same class of laws. The advantage comes from "what the medium naturally does," not from "trading precision for power."

### MEMS: second-order ODEs are free

    mẍ + cẋ + kx = F(t),    ω0 = √(k/m),    Q = mω0/c

A single mode is already a second-order dynamical primitive. No OTA+C is needed to "simulate" a second-order system — the mechanical structure solves the equation itself. When coupled,

    MẌ + CẊ + KX = Bu

with M, K, C as encoding matrices; the generalized eigenproblem `Kφ=ω²Mφ` is directly the modal reference. Duffing `+αx³`, parametric excitation, mode coupling and bistability are free physical nonlinearities.

Three engineering constraints:

- **High Q is double-edged**. Q = 10³–10⁵ gives narrow bandwidth, low phase noise and long decay, but also τ_settle ~ 2Q/ω0 and drift of Q with temperature/stress/package. A high-Q pole implemented in gm-C would be limited by noise and dynamic range; MEMS suits it naturally — provided it can be calibrated.
- **The interface is nonlinear**. Electrostatic drive `F ∝ V²`, capacitive readout `ΔC/C`. `V↔x` is not an ideal g_m: either differential linearization or inclusion in the calibrated behavioral model.
- **Poor programmability**. M and K are fixed; the compiler can only choose which resonator in a bank, and tune coupling/gain/feedback. MEMS is a fixed-function accelerator, not a general-purpose tile.

**Time reference**: `ω0=√(k/m)` can serve as an on-chip dynamical reference. It is more direct than measuring temperature plus a LUT, because it measures a physical time constant of the same kind. The usage converts absolute accuracy into ratio accuracy:

    g_m/C = [ (g_m/C) / ω_ref ] · ω_ref

the bracket is a ratio of two frequency counts, and the thermal drift of `ω_ref` is characterized and entered into the common-mode scale of §2.7. It is not an absolute clock and does not eliminate drift; it merely replaces "measure temperature then look up a table" with "measure another time constant."

### Microfluidics: the same class of graph Laplacian

A chamber + channel network gives

    C_f·dP_i/dt = Σ_j Q_ji − Σ_j Q_ij ,    Q_ij = (P_i − P_j)/R_ij
    ⇒  C_f·Ṗ = −L_R·P + Q

isomorphic to `C·V̇ = −G·V + I` under `P↔V, Q↔I, R_f↔R`. This is not an analogy but the same class of graph Laplacian dynamics; diffusion, transport and reaction-diffusion are natural fits.

Adding chemistry (mass action `v = k∏c^ν`):

    ċ = S·v(c)

with S the stoichiometric matrix and v the nonlinear term — this directly gives multiplication, i.e. a wetware version of the GPAC primitive (§1.5 of DESIGN.md).

But the timescale is ms–min, 6–9 orders of magnitude slower than electronics. The role is therefore not a fast solver but

    fast electronic solver  ↔  slow physical/chemical system

The electronics estimates `θ̂` in real time, controls valves/heaters/electrodes/pumps, and reads concentration/fluorescence/impedance/electrochemical signals. The real system is `ċ=f(c,θ)`, the on-chip model `ċ̂=f(ĉ,θ̂)`, and the error `e=c−ĉ` updates `θ̂` online. The whole chip becomes a closed loop of experiment + model + parameter inference — more distinctive than "how much faster than a GPU is analog."

Costs: evaporation, bubbles, clogging, surface adsorption, temperature/pH drift and biofouling. These are slow drifts that belong to the background calibration of §2.7 and the slack of §1.2.2 — but one cannot assume "a fluidic network equals an ideal RC."

### Photonics: buy only global dense connectivity and linear transforms

    V_i → modulator → λ_i → waveguide → photodiode → I_j

`I_PD ∝ P_opt`, injected directly into `C_j·V̇_j`. This is consistent with the "voltage state → current communication" rule of §2.3, and the photodiode output is naturally a current.

WDM lets one waveguide carry N wavelengths:

    number of physical wires < number of logical channels

ring/MZI weighting of the coefficients gives

    I_j = Σ_i w_ji·P_i    ⇒    y = W x

and the MVM is exactly the most expensive part of `ẋ = Ax + N(x)`.

It is not a free lunch, though: laser power, shot noise (σ²∝P), photodiode noise, modulator linearity, ring thermal drift (on the order of 0.1 nm/K), wavelength calibration, optical loss and EO-conversion overhead. For a 100 µm local link, optics make essentially no sense. The rule is

    local → electrical
    regional → electrical / current-mode
    global dense → optical / WDM

consistent with the latency hierarchy of §1.3.8. Photonics buys "bandwidth that does not grow with L²," not "a faster transistor."

### Priorities and the benchmark gradient

Research-value ranking:

1. **CMOS FPAA + MEMS**: the most natural interface (g_m drive, C readout); MEMS is itself a dynamical element. Start with a resonator bank plus CMOS programmable coupling/readout.
2. **CMOS FPAA + photonics**: addresses large-scale connectivity / MVM / transforms; if the state count really scales, this may ultimately matter more than MEMS.
3. **CMOS + microfluidics**: not a general high-speed solver, but a closed-loop platform for a real physical system + analog model + inverse solver.

Benchmark gradient (do not start with NS DNS):

| generation | benchmark | what it validates | principal substrate |
|---|---|---|---|
| 1 | Poisson/Laplace, diffusion, reaction-diffusion | the physical operator itself is doing the solving | resistor/g_m mesh + RC + CMOS NL |
| 2 | stiff ODE, DAE/SPICE, SDE/Langevin | multirate, constraint, noise-as-computation | CMOS + T10 + T8 |
| 3 | MD, Navier–Stokes, wave/Maxwell | heterogeneous combinations | CMOS + MEMS/photonic mesh |

Which problems benefit from changing substrate:

| problem | digital difficulty | physical-solver potential | best substrate |
|---|---|---|---|
| Poisson/Laplace | large sparse linear solve | ★★★★★ | resistor/g_m mesh |
| diffusion/heat | large grid, long integration | ★★★★★ | RC / MEMS / fluidic mesh |
| reaction-diffusion | nonlinearity + spatial coupling | ★★★★★ | CMOS + fluidic/chemical |
| stiff ODE | extreme timescales | ★★★★ | CMOS + multirate |
| DAE/SPICE | Newton per step + sparse solve | ★★★★★ | CMOS constraint network |
| wave | second-order PDE | ★★★★ | MEMS lattice |
| Maxwell | 3D mesh + CFL | ★★★ | photonic structure |
| incompressible NS | global pressure solve | ★★★★ | CMOS + physical Poisson mesh |
| MD long-range | N-body | ★★★★ | CMOS local + field solve |
| SDE / optimization | large ensembles | ★★★★★ | noisy analog |

**What will not be rescued** (consistent with §9 of DESIGN.md):

- The state count does not drop: 1024³ is still a billion states; analog does not reduce degrees of freedom.
- The precision–speed conservation law still holds, with each domain using its own noise mechanism (MEMS thermo-mechanical `S_x=4kT·c`, photonic shot noise).
- High-Re DNS, 6D Boltzmann, quantum many-body and long-horizon chaotic trajectories are out of scope; for chaos only statistics are promised (Lyapunov spectrum / attractor statistics / correlation / spectrum), consistent with Level 2 of §1.2.5.

**What does not change is the decision machinery**: `κ_hw` becomes per-domain, Γ is computed from the transducer coupling matrix, and routing becomes loss/thermal/calibration or fluidic tolerance instead of RC. The slack vector of §1.2.2 still applies, with one extra domain index per axis; the per-edge closure of §2.3 and §6.15 still applies, with `H_ij(s)` replaced by that domain's transfer function.
