# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## What this repository is

A design study plus digital twin, not a product codebase. The artifact that matters is `DESIGN.md`: it specifies a compiler and a layered continuous-time analog/hybrid fabric for solving ODE/PDE, and it carries the normative claims, scaling laws, acceptance criteria and experiment results. The Python under `twin/`, `analysis/` and `bench/` exists to falsify or confirm statements in `DESIGN.md`.

Read `DESIGN.md` before touching code. It is ~1720 lines and organized so each section states a claim, then the number that makes it checkable.

## Commands

Plain Python 3 (stdlib + numpy + scipy + matplotlib). No build step, no packaging, no linter config, no test suite. Run modules from the repo root.

    python3 -m bench.run_twin_experiments      # all five twin experiments, ~3 min
    python3 -m bench.run_phase0                # Phase 0 verdicts, ~2 min, 38/38 checks
    python3 -m bench.run_reduce                # Phase 1 front-end, ~10 s, 41/41 checks
    python3 -m bench.run_skew                  # Phase 1.5 interconnect delay, ~2 s, 15/15 checks
    python3 -m bench.run_route                 # Phase 1.6 route closure, ~2 s, 73/73 checks
    python3 -m bench.run_transform             # Phase 4.5 representation/transform, ~6 s, 56/56 checks
    python3 -m twin.spice.extract              # ngspice extraction, every PDK in the dict
    python3 -m twin.spice.extract sky130       # single PDK (~70 s: model loading dominates)
    python3 -m twin.spice.extract ihp-hbt
    python3 -m bench.run_sde                   # Phase 6 SDE 正/反问题, ~55 s, 27/27 checks
    python3 -m bench.run_multirate             # Phase 4.3 WR/S&H 合法性, ~50 s, 16/16 checks
    python3 -m bench.run_compiler              # P6 证书聚合层, ~3 s, 22/22 checks
    python3 -m bench.run_constraint            # T10 KCL 松弛孪生, ~4 s, 44/44 checks
    python3 -m bench.run_pipeline              # 3-anchor e2e (spec→cert→twin), ~5 min on this box, 30/30 checks
    python3 -m bench.run_substrate             # A7 MEMS/photonic substrate, ~1.5 min CPU on this box (RK4-heavy), 40/40 checks
    python3 -m bench.run_stagea                # A4/A8/A9 closeout (shear basis change / VdP / Lorentzian viscoelastic + fallback rate + ledger), ~11 s, 26/26 checks
    python3 -m bench.run_closure_split         # B1 closure+split (P_N projection η_closure certificate / QSSA fast-slow split), ~2 s, 34/34 checks
    python3 -m bench.run_basis_interface       # B1 basis+interface (basis selection Γ·κ(Φ) tax / single-flux-source + fitting/capturing), ~4 s, 31/31 checks
    python3 -m bench.run_calib                 # B1 calib FSM + P7 config/calibration plan, ~5 s, 43/43 checks

In the Trae agent shell, prefix ngspice with `env -u LD_LIBRARY_PATH`. The agent
injects a libstdc++ that ngspice cannot load (`GLIBCXX_3.4.32` missing). A normal
login shell does not need this.

Outputs land in `results/`: `report.txt`, `twin_experiments.json`, `twin_experiments.png`, `device_extract.json`, `phase0.json`, `phase0_report.txt`, `phase0.png`, `reduce.json`, `reduce_report.txt`, `skew.json`, `skew_report.txt`, `route.json`, `route_report.txt`, `transform.json`, `transform_report.txt`, plus per-PDK `ext_*.sp` / `ext_*.log` / `*sweep_*.dat`. The JSON files are the machine-readable record that `DESIGN.md §6.8/§6.9/§6.10/§6.13/§6.14/§6.15/§6.16` quotes.

To run one experiment in isolation, import and call it (each `exp_*` is self-contained):

    python3 -c "from bench.run_twin_experiments import exp_d; exp_d()"

## Implementation phases

Three stages: **A theory validation** (software only), **B chip and software design**, **C silicon validation**. Each stage ends with a gate; each gate names the number that passes and the observation that falsifies it. A gate that cannot fail is not a gate, and a claim without a `results/*.json` entry is still a target.

### Stage A — theory validation (no silicon)

Purpose: decide whether the architecture deserves silicon. The digital twin is the only oracle; there is no "trust the hardware" escape hatch at this stage.

**A0 — feasibility core (implemented, 38/38).** `analysis/` computes `(E,A,B,C) → κ, Γ, ‖H‖₂, ‖H‖∞` and the six-axis slack. Diffusion N=64/1024, biharmonic N=128, Burgers and Robertson match analytic/asymptotic values within ±3 dB / ±0.5 decade; singular-DAE H2 (index-1 eliminated σ_y²=0.5, feedthrough/index-2 ⇒ ∞) and energy-metric Γ are now covered too. Two invariants to keep: κ removes only numerical zero modes; the representation axis is evaluated on the physical state, not on a capturing scheme's output.

**A1 — front-end reducibility (implemented, 41/41).** `lowering/reduce.py` classifies Lorenz/Duffing as polynomial, HH as rational (six `exp` augmentations), Bessel J0 as rational with `1/t` (rejected at `t=0`), Gamma as rejected. The reduction is exact only on the invariant manifold: the `s²+c²` drift grows from 1e-14 to 1e-3 when the augmentation is perturbed by 1e-3. Do not widen a claim beyond that. Optional Baumgarte stabilization (`manifold_gamma`) saturates manifold deviation at E[g²]≈a/γ+b/γ² (measured fit, residual 9e-4); `reduce_with_primitives` compares augmented vs native exp/tanh tile costs (HH: 10 states/65 monomials/degree 6 vs 4/29/5).

**A2 — interconnect and timing (implemented, 15/15 + 73/73).** `bench/run_skew.py` verifies `E=I+A∘T`, `τ_crit` and the `2/π` optimism of the first-order mass matrix. `bench/run_route.py` verifies distributed vs lumped RC (0.3791 vs 0.3795 RC t50), the phase-before-noise fanout limit (11.45° at 4 levels), all-pass vs RC padding (108× `J_H`), deskew vs absolute feedback delay, the crosstalk floor (incl. `crosstalk_hf_victim` for asymmetric nets), the `1/|J_ij|` / `2Q` sensitivity ordering, loop-level closure (`loop_margins`: the 2/π optimism reproduces at loop level, 0.6467), `PhysicalScaling` (t_phys=alpha·t_model verdict flip), and phase-delay-at-ωc (AllPass pads diverge latency vs phase delay; naive `np.angle` wraps to a wrong-small value). End-to-end closure exists as a necessary-condition probe, not exact DDE root-finding.

**A3 — representation and transform (Phase 4.5, implemented, 56/56).** WENO as `S → T_σ → F_σ`, continuous `σ`, conservation on a shared face flux; lifting wavelet with `(W, W_inv, W_star)`; the per-scale `C_j` vs `g_m,j` trade-off (mode C keeps `g_m` range small but noise grows as `4^j`; mode g_m keeps noise flat but needs a `4^L` transconductance range); `G_T = T̃^*T̃` calibration. Three measured traps are now in the bench: (a) the sensor's `ε` must be a ~1% field-range floor, or smooth extrema give `0/0` and read as shocks (sine σ 1.0 → 0.1265); (b) a symmetric σ tracker forms a representation-feedback limit cycle (8 threshold crossings per cell), while attack/release 400/10 gives 2; (c) `G_T=T̃^*T̃` requires right multiplication `T_cal=T̃G_T^{-1/2}`; the naive left `K T̃` leaves isometry at 0.032 (0.342 at 2% perturbation) instead of 3.46e-15.

**A4 — benchmark corpus (§8, partial).** Every system in §8 must produce a verdict, a binding slack axis, and a predicted QoI error. Coverage is now the five Phase-0 cases + 3 anchor problems (`bench/problems.py`) + shear/Van-der-Pol/Lorentzian-viscoelastic (`bench/run_stagea.py`, 26/26) — all with analytic or high-resolution references and a predicted failure where the theory says failure. The gap is the remaining §8 rows (airfoil, bore, front, hh, robertson variants), not coverage of the three anchor classes.

**A5 — compiler end-to-end (partial).** P0–P6 runs the 3 anchors through spec→`discretize`→`check`→`gramian_scaling`→`evaluate_sys` certificate→`run_twin` reconciliation (`bench/run_pipeline.py`, 30/30), including the P3→P6 physical-time hand-off (`evaluate_sys(scaling=)`, α⁻¹·S·A·S⁻¹). P7 config + calibration-plan emission is now implemented (`lowering/p7.py` + `twin/calib.py`, `bench/run_calib.py` 43/43): foreground cal residual = measurement-noise floor (contract, never below it); background residual = drift_rate·interval/(1+loop_gain) + floor (A7 `gm_c_calibrate` semantics; longer interval ⇒ linearly larger residual); `residual_budget = floor/noise_axis`, and a certificate marked infeasible ⇒ elements flagged, not silently planned. The two representation passes are also implemented (`lowering/basis.py`: candidate scored on total mismatch budget Γ·amplification-through-Φ — eigenvector selected-against on the shear (Γ=1.0 × κ(Φ)=20.05=20.05 vs identity 5.12), degenerate-Gramian floor, structured-Δ wins documented; `lowering/interface.py`: capture/fit selector + single-flux-source contract reproducing coupling.py's dM/dt; `bench/run_basis_interface.py` 31/31, `results/basis_interface.json`). The two descriptor-reduction passes are implemented (`lowering/closure.py`: P_N projection + measured η_closure certificate — out-of-band forcing gives full defect, quadratic defect lands in the (N,2N] alias band ⇒ change-basis; `lowering/split.py`: gap/τ_fast classification + physical-coordinate Schur QSSA + measured slaving residual ∝ gap⁻¹; weak-gap/defective/unstable-fast/singular-fast-block flagged, never silently inverted; `bench/run_closure_split.py` 34/34, `results/closure_split.json`). Still missing: per-edge route-contract emission from P7 (element configs emitted; `twin/route.py` closure verdicts not yet wired into the plan). Fallback rate measured: **6/11 = 54.5% raw ⇒ accelerator-not-solver (§9 risk 4); 28.6% excluding deliberate infeasibility probes** — both numbers in `results/stagea.json`, positioning is a continuous-time hybrid accelerator/coprocessor, not a general all-analog solver.

**A6 — stochastic dynamics and inverse problems (implemented, 27/27).** `analysis/sde.py` + `bench/run_sde.py`: OU Var/ACF, GBM Itô/Stratonovich split (51.9σ), double-well KL 5e-4, CRN/IND gradient variance ratio 2.3e5, Wong–Zakai θ(χ)=1/(2(1+χ)) within 15%, adjoint Δ_asym bound err/bound≈0.71. Falsification path (noise-source → contract; adjoint vs FD with W_inv≠W* kept distinct) exercised.

**A7 — heterogeneous substrates (implemented, 40/40).** `twin/substrate.py` + `bench/run_substrate.py`: MEMS `ω0=√(k/m)`, `Q=mω0/c`, Duffing backbone ω(A)≈ω0(1+3k3A²/8k) vs RK4 <1%, gm/C locked residual ≈10ppm meas-floor not 1000ppm drift; photonic 4×4 MVM shot-noise ∝1/√N + drift floor N-independent; fluidic Laplacian mapping. Falsification (transducer cost not in slack; state blowup) is the honest Phase-8 TODO — transducer numbers are TARGET-labeled, not measured.

**A8 — negative controls (implemented).** biharmonic N=128 (`κ=N⁴`), non-normal shear (`Γ=5.12` — and the basis-change quantitative trap: eigvec basis Γ→1.000 but pays κ(Φ)=20.05, Gramian scaling Γ→3.98e-8 with κ invariant), Robertson without QSSA, Burgers representation violation, Van der Pol unstable-equilibrium fallback, Lorentzian viscoelastic finite-Markovian-embedding. Stage A passes because these are rejected/downgraded for the *predicted* reason, recorded in `results/stagea.json`.

**A9 — exit review.** The review reads the assumption ledger and the failure map, not the demos.

What Stage A must validate:

| Category | What must be validated | Pass number | Falsified by |
|---|---|---|---|
| Model algebra | descriptor, generalized eigenproblem, Lyapunov, H∞, Γ, κ | ±3 dB; κ within ±0.5 decade | two independent methods disagree |
| Reduction | polynomial/rational/rejected; invariant manifolds; DAE index | exact at 1e-12; drift grows with mismatch | a "polynomial" case needs non-polynomial terms; index>1 slips through |
| Perturbation budget | `B_p` ports; H2/H∞; noise, mismatch, route, jitter in one resolvent | σ_y within ±3 dB; binding axis correct | a perturbation cannot be represented as a port |
| Timing | `H_ij(s)`, group delay, crosstalk, absolute feedback | phase <3°; `η_route` <20%; crosstalk floor | per-edge passes but the loop fails |
| Representation | basis switch, WENO triple, transform contract | conservation <1e-4; isometry <0.5%; PR <0.2%; adjoint <1e-3 | σ blend on states; inverse used as adjoint |
| Compiler | P0–P7, slack, placement/route contract, calibration plan | QoI ±10%; fallback rate reported | feasible certificate, divergent twin |
| Stochastics | Itô/Stratonovich, noise contract, CRN/EnKF/adjoint | ±10%; KL<0.05; θ<10% | interpretation missing; gradient bias |
| Heterogeneous | MEMS/fluidic/photonic constitutive laws + transducer | 5–10% on toy models | cost not in slack; state blowup |
| Failure map | negative controls | predicted rejection | a known-hard case "passes" without evidence |

Stage A exit criteria (all must hold):

1. Every §8 benchmark has a verdict, a binding axis, and a reference (analytic or high-resolution).
2. Prediction errors are inside the table's intervals; no benchmark is "feasible" in the certificate and divergent in the twin.
3. At least three workloads run end-to-end through P0–P7 and the twin: 1D diffusion (spectral), 2D Poisson (constraint mesh), 1D Burgers (flux + `σ`).
4. Negative controls fail for the predicted reason, not by accident.
5. The digital fallback rate is measured; if >50%, `DESIGN.md` says "accelerator", not "solver".
6. Every quantitative claim in `DESIGN.md` maps to a `results/*.json` or is explicitly marked as a target.
7. An assumption ledger exists: `κ_hw`, α, noise PSD, mismatch, parasitics, calibration residual, each with a sensitivity number.

### Stage B — chip and software design

**B0 — architecture freeze (frozen, DESIGN.md §7 Phase 7.5).** Process = **IHP SG13G2 SiGe BiCMOS**: HBT (λ=0.0138/V, r_o=3.6 MΩ — 70× better than IHP MOS λ=0.970/V, mirror error ~590× lower) for fast current interface/buffer/driver; MOS/MIM for state integration (C is the resource); CMOS for supervisor/calibration FSM. Tile counts frozen for a characterization vehicle (first chip = B3, not a processor): T1 4–8 integrators (C_bank=1pF gradient), T3 one ≤64 MVM (cap 256), T4 pole array τ 1e-8…1e-3, T5 ~64-cell 1D flux chain with HBT single-flux-source mirror, T7 programmable length/fanout route array fed by `route_contract.py` four-verdict contracts, T8 noise bank, T9 statistics, T10 ~16×16 sparse gm constraint mesh. Per-tile `(E,A,B,C)`/`‖H‖₂`/`‖H‖∞`/power/area spec rows frozen structurally; power/area TARGET-labeled (§2.1 — not the bottleneck; κ_hw, route parasitics, cal drift are). V_BE −2 mV/K ⇒ interface bias into background-cal loop. κ_hw=1e6 is nominal (ledger sensitivity 4.86%) — must be measured in C1, not trusted from PDK.

**B1 — compiler and software (near-complete).** Passes landed: `closure.py` (P_N + η_closure, 34/34), `split.py` (CSP/QSSA Schur, gap→slaving ∝ gap⁻¹), `basis.py` (Γ·κ(Φ) tax — eigenbasis selected-against on shear), `interface.py` (capture/fit + single-flux-source), `p7.py` + `twin/calib.py` (config + cal plan + FG/BG FSM, 43/43). Sole remaining gap: **per-edge route-contract emission** (`route_contract.py` wiring `route_closure` four-verdict into P7 — in flight). Exit: the compiler reproduces the Stage A verdicts on the same workloads, with the same slack axes.

**B2 — pre-silicon verification.** SPICE → PEX/EM/substrate/thermal → behavioral model; corner and Monte Carlo separated; calibration-in-the-loop; post-layout slack re-run. Exit: `P(workload passes)` is reported; no block ships without a noise and mismatch model.

**B3 — characterization vehicle.** Primitive bank, route testers (length/fanout/mode), C bank, device bank, replica/temperature/noise/loopback, and 32 copies of the same cell at different locations. Exit: every quantity measured in Stage C has a dedicated structure.

**B4 — design review.** Exit: all blocks close on slack; the test plan maps 1:1 to Stage C; DFM/ESD/IR-drop checked.

### Stage C — silicon validation

- **C0 bring-up**: power, clock, supervisor, DAC/ADC, reference, leakage.
- **C1 primitive characterization**: `g_m/C`, noise PSD, offset, THD, charge injection, C mismatch.
- **C2 route and array**: delay, phase, crosstalk, all-pass, feedback absolute delay, latency class.
- **C3 calibration**: foreground/background, residual distribution, drift vs temperature, replica tracking.
- **C4 workloads**: diffusion N=64, 32×32 Poisson, 1D Burgers, OU/GBM, Robertson with QSSA, then larger cases.
- **C5 heterogeneous (optional)**: MEMS resonator, photonic link, fluidic cell.
- **C6 accounting and go/no-go**: QoI vs digital reference, energy/area/bandwidth per solve, measured failure boundary.

Stage C exit: measured/predicted errors are inside the Stage A intervals; at least one workload runs end-to-end with no digital fallback; the next chip's targets are derived from measured, not nominal, numbers.

## Architecture

Three layers, separated so measurement code never contains a solver and solver code never contains a claim.

**`twin/qoi.py` — estimators only.** Turns a measured profile into a transport coefficient. `fit_tanh` / `nu_from_width` implement the shock-width to equivalent-viscosity inversion (nu = du * Wd * dx / 8.789); `front_width_10_90` / `kappa_from_width_growth` implement front thickening to mixing coefficient (kappa = (1/13.139) * d(w^2)/dt for the erf family). Both constants are derived analytically, not fitted. A new estimator must be validated against a known analytic case, and its constant must be derivable.

**`twin/coupling.py` — the coupled spectral/local system.** The load-bearing file. One high-order nodal-DG element (`SpectralElement`, LGL lumped mass) and a finite-volume region (`FVRegion`, MUSCL + Roe) joined at two periodic interfaces. The invariant everything else tests is the mass budget

    dM/dt = (F_A_fv - F_A_spec) - (F_B_fv - F_B_spec)

`CoupledSystem.run()` integrates the state with RK4 and accumulates that same scalar with the same Butcher weights, so measured drift and predicted drift must agree to ~1e-12. If they diverge, the code has a hidden leak. That check has already caught real errors and is the main correctness signal in the repo.

`InterfaceModel` is where the interface physics lives. `single_source=True` shares one flux value with both sides; `mode="gm"` substitutes a transistor-level model driven by values extracted from real PDKs (`lam`, `eps_dev`, `eps_sys`, `shared_ref`). Do not "fix" a conservation result by changing the flux scheme — the scheme is intentionally minimal so the interface effect stays isolated. `interface_fluxes(u)` is a pure function of (u, _hist): `stale=k` means exactly k accepted timesteps of sample/hold delay, with the history advanced once per `step()` — the delay history must never be mutated inside the RHS (an RK4 step calls it 4×).

**`twin/spice/extract.py` — PDK extraction.** Builds a diode-connected current-mirror netlist per process, runs ngspice in batch, parses the sweep `.dat` files into lambda = gds/ID, gm/ID, ro and the mirror ratio. PDKs are external (`~/.ciel/ciel/...`, `/grid/...`) and must not be vendored or modified.

**`twin/route.py` — the route closure model.** One edge is `H_ij(s) = H_wire(s)·∏ H_buf,k(s)·e^{−sτ_ij}`, built from `Wire` (distributed `1/cosh√(sRC)` by default, `lumped` as the one-pole strawman), `Buffer` (single pole + input-referred noise), `AllPass` (first-order Padé delay trim) and `PureDelay`. `route_closure` returns the four per-edge verdicts — gain, phase, noise, feedback absolute delay — plus η_route, J_H, J_τ and the latency class. Load-bearing policies: (a) distributed vs lumped share the DC slope RC/2 but differ in t50 (0.3795 vs ln2/2) and f_3dB; (b) `feedback_critical` is judged on `phase_delay_at(edge, ωc)` = −∠H(jωc)/ωc with marching unwrap from DC, not the DC `latency` — after AllPass padding they diverge; (c) weighted costs `j_h/j_tau` integrate `W(ω)·y(ω)` in the numerator (a workload spectrum is part of the integrand, not a normalization); (d) `CoupledNets` cross terms put the *input* resistance in the denominator (`H_01=−m12/(det·R2)`); for asymmetric nets the victim floor is `crosstalk_hf_victim`. `edge_sensitivity` computes `S_τ,ij = ω|J_ij|·‖(jωI−J)⁻¹‖` and `τ_tol = ε_φ/S`; near a resonance the whole band's budget shrinks by ~2Q, while the ordering stays `1/|J_ij|`. Loop-level closures: `loop_gain_matrix`/`loop_margins` evaluate `σ_min(jωI−J∘H(jω))` and `S=(I+L)⁻¹` over a band — a necessary-condition probe, not exact DDE root-finding (scalar_dde_root is the 1-DOF exact analog); `PhysicalScaling(S_x, alpha, v_swing)` makes the model-time→physical-time and state→voltage maps explicit IR fields — t_phys = alpha·t_model, never compare raw model bandwidth to B_hw without it. Known simplifications (documented in RouteEdge docstring): `H` multiplies open-circuit transfer functions (no inter-segment loading; ABCD cascade not implemented), and `noise_var` uses DC gain² (no ENBW compression).

**`twin/noise.py` — colored noise source (Phase 6).** `ColorNoise(tau_c, intensity, dim, rho)` is an OU-filtered source with the exact AR(1) update `ξ ← ξ·e^{−dt/τc} + √((D/τc)(1−e^{−2dt/τc}))·N`; PSD convention (two-sided flat 2D, B_noise=1/(2πτc)) is pinned at module top because a one/two-sided mix-up is a 3 dB error (§1.3). `start_capture/replay` replays identical realizations — this is what makes the CRN finite-difference path work (shared-realization gradient noise floor cancels ~1/δ²). `rho_ij` via validated Cholesky. `bandwidth_drop_factor(fc/B)` gives the (2/π)arctan brickwall deficit.

**`twin/transform.py` — representation primitives (Phase 4.5).** `smoothness_indicator` (normalized second difference with a ~1% field-range `ε` floor), `SigmaTracker` (hysteresis + asymmetric attack/release), the face fluxes `weno5/central/rusanov/muscl`, `blend_face_flux` (the blend must happen on one face, never on states), `LiftingWavelet` (orthonormal Haar and biorthogonal CDF53, with explicit `W`, `W_inv`, `W_star` for contract tests), `stationary_haar` (undecimated, translation-invariant), `gram_calibrate` (right Gram `T_cal=T̃G_T^{-1/2}`; left Gram uses `T̃T̃^*`), `TransposeableEdge`/`edge_matrix`, `adjoint_ode_gradient` (transpose of the discrete RK4 map), and `scale_poles` (mode C vs mode g_m). Two traps are load-bearing: a symmetric σ tracker oscillates through a representation feedback loop (8 crossings per cell vs 2 with 400/10 attack/release), and `K T̃` with the right Gram does not orthogonalize (isometry 0.032 vs 3.46e-15).

**`ref/README.md` — annotated bibliography and reading status.** Maps the FPAA physical-synthesis line (Baskaya 2006/2009, Hasler–Twigg 2007, Wang–Vrudhula 2002, George 2016), the waveform-relaxation/multirate line (Lelarasmee 1982, RELAX, Hayashi–Asai 1992), the delay-in-dynamics demo (Jiménez 2022), and the GPAC/differential-algebra/complexity line (Shannon 1941; Bournez–Graça–Pouly 2016/2017; Rubel 1981) to DESIGN.md sections. Open-access PDFs live in `ref/pdfs/`; paywalled items are DOI links only. The stated gap is dynamical-sensitivity-aware physical synthesis: extend `S_τ,ij` to `S_g,ij`/`S_n,ij`, feed the contracts into a P&R cost, and fall back to waveform relaxation when the instantaneous contract cannot be met. §G and §H are different in kind: they list MEMS/fluidic/photonic and WENO/wavelet/adjoint reading *targets* that have not been verified, and say so explicitly. Do not upgrade those to citations without checking the metadata.

**`DESIGN.md §3.7` — heterogeneous substrates.** The descriptor is relaxed to `E_q q̇ = F_q(q) + Σ C_qr z_r` with `q ∈ {V, I, x_mech, P_fluid, c_chem, a_optical}`, so state is a generalized physical coordinate rather than a voltage. MEMS supplies the second-order primitive `mẍ+cẋ+kx=F` (and `ω0=√(k/m)` as a dynamical reference), the fluidic network is the same graph Laplacian as `C V̇ = −GV+I` (`P↔V, Q↔I, R_f↔R`), and photonics is limited to global dense communication and linear transforms. The judgment machinery does not change: `κ_hw` becomes per-domain, `Γ` uses the transducer coupling, and every cross-domain trip pays gain/noise/bandwidth/energy that must enter the §1.2 slack. Stated non-goals: analog does not reduce state count, and high-Re DNS / 6D Boltzmann / quantum many-body / long-term chaos are not promised.

**`DESIGN.md §5.7–§5.9` — adaptive representation and the transform contract.** WENO is not treated as a stencil to be copied into hardware but as three operators `S → T_σ → F_σ` (smoothness detector, state-dependent representation, conservative face flux); `σ` is a continuous coordinate, and the smooth/shock blend must happen on the same face flux or telescoping conservation is lost. The representation contract is `(T, T⁻¹, T^*)`, not `(T, T⁻¹)` — biorthogonal wavelets have `T⁻¹ ≠ T^*` (measured 0.684 for CDF53), and confusing them silently corrupts adjoint gradients. Isometry is measured in the energy metric `M=diag(C)`, not bare voltage. Lifting wavelets with transposeable gm/current edges let forward and adjoint share hardware; undecimated coefficients avoid hidden S/H. Per-scale `C_j` vs `g_m,j` is a trade-off, not a κ reduction: the pole spread stays `4^L` in both modes, mode C puts it in noise (`g_m` spread 1), mode g_m puts it in transconductance (noise spread 1). `G_T=T̃^*T̃` is the calibration observable, but the right Gram requires right multiplication `T_cal=T̃G_T^{-1/2}`; the naive left form leaves isometry at 0.032. Measured numbers are in §6.16, including the relative-ε sensor floor (sine σ 1.0 → 0.1265) and the asymmetric σ tracker (8 → 2 crossings).

**`analysis/` — the feasibility analyzer (Phase 0).** Pure linear algebra, no solver and no hardware model. `descriptor.py` owns (E,A,B,C) and the spectral metrics — `stable_projection` uses the *left* deflating subspace `Q1ᵀ` for projection (`E_r=Q1ᵀEZ1`), because `ordqz` gives `QᵀAZ` triangular only with Q on the left; `Z1ᵀ` is right only when the pencil is symmetric. `lyapunov.py` solves the generalized Lyapunov equation on the stable finite deflating subspace and, for singular E, eliminates the index-1 algebraic block before judging H2 finiteness (`qz-dae-eliminated`); a direct feedthrough or index>1 nilpotent block still gives ∞. `hinf.py` bisects the Hamiltonian pencil and cross-checks against a frequency sweep; `resolvent.py` computes Γ (optionally in an energy metric); `regularity.py` fits the spectral tail; `sde.py` adds second-moment Sylvester covariance, mean-square stability, Itô↔Stratonovich corrections, and `noise_realizability` (flat-band vs Lorentzian deficit). `slack.py` builds the six-axis vector — Γ_proj‖ΔA‖ is legal only for *conservative* ΔA (column-sum-preserving); leakage ΔA that excites the conservation zero mode must be judged by drift, not Γ_proj. `skew.py` turns path delay into a descriptor: `E = I + A∘T`, with the uniform-delay closed form `λ/(1+τλ)`. Two policies are load-bearing: κ drops only *numerical* zero modes (|λ| < 1e-12·max|λ|), so physical slow modes stay in the verdict; and the representation axis must be evaluated on the physical state, never on a capturing scheme's output (WENO smears the tail with ν_num and would pass a test the state fails).

**`lowering/` — the front-end classifier (Phase 1).** `reduce.py` is a tiny symbolic IR (`Expr` + `+ × ^ sin cos exp tanh besselj gamma`), a first-order/jet reducer, a state augmenter that records invariant manifolds (optionally Baumgarte-stabilized via `manifold_gamma`), and a sparse rational canonicalizer (`_poly_frac` merges same-denominator addends before cross-multiplying — a/x+b/x no longer squares the denominator). `reduce_with_primitives` is the breakeven comparator: exp/tanh stay as atomic T2-tile native primitives rather than augmented states. It answers one question: is the right-hand side polynomial, rational, or neither? `bench/run_reduce.py` is the acceptance (41/41): Lorenz/Duffing → polynomial; standard Hodgkin–Huxley → rational (six `exp` states plus three denominators, because α/β contain 1/(1−exp)); Bessel J0 → rational with 1/t, rejected at t=0; Gamma → rejected. Do not describe HH as polynomial — the earlier draft did, and the data says otherwise. The reduction is only equivalent on the invariant manifold: a 1e-3 mismatch in the sin/cos augmentation grows the s²+c² drift from 1e-14 to 1e-3.

**`bench/run_phase0.py` — the Phase 0 acceptance.** Builds five systems (diffusion N=64/1024, biharmonic N=128, Burgers linearized at the pre-shock state, Robertson at t=1e4), runs the analyzer, checks against analytic κ/Γ/σ_y/H∞, and writes the verdict table. The Burgers exact solution is via characteristics; the Jacobian is the first-order Roe operator (differentiable), while the WENO solution is only used to show how capturing hides the spectral tail.

**`bench/run_skew.py` — the Phase 1.5 acceptance.** Converts path delay to the descriptor `E = I + A∘T` and checks three things: the uniform-delay closed form `λ/(1+τλ)` (2.3e-16), the first-order `−τλ²` pole shift, and the scalar DDE stability boundary. The scalar DDE is the warning: its exact boundary is `|a|τ = π/2`, while `E = I + τA` gives `|a|τ = 1`, optimistic by `2/π`. Phase budget at 100 MHz is 27.8 ps for 1°. Do not treat the first-order mass matrix as a substitute for the `ω_max τ` check.

**`bench/run_route.py` — the Phase 1.6 acceptance.** Seven cases and 38 checks. A distributed RC line's t50 is measured with a small implicit-Euler diffusion fixture (`_diffusion_t50`), not assumed: 0.3791 RC vs the analytic 0.3795. The buffer fanout case shows the depth limit is phase, not gain or noise (4 levels at 2 GHz/100 MHz give 11.45° against a 3° budget). The all-pass case quantifies why an RC pole is the wrong pad (same DC delay, 15.3% gain error and 108× the J_H at 500 MHz). The deskew case is the negative control: skew goes to 3e-7 ps while the common 59.8 ps still violates a 1° reserve. The crosstalk case checks the HF floor `C_m/(C_2+C_m)` and the 1 mV/1 V C_m budget. The sensitivity case checks the `1/|J_ij|` ordering and the `2Q` resonance factor.

**`bench/run_transform.py` — the Phase 4.5 acceptance.** Six cases and 50 checks. The WENO case puts a static step and a smooth sine through the sensor (σ 1.0000 vs 0.1265 with a 1% `ε` floor), then runs Burgers to t=0.5 with central/MUSCL blending on a shared face flux: conservation drift 2.22e-16, shock position equal to WENO5 (x=0.7480), L2 4.59e-3, and at most 2 σ threshold crossings per cell (a symmetric tracker gives 8). The P0-F additions: a per-step entropy-inequality check on the moving shock (d/dt∫u²/2 ≤ 0 — "ΣV_i² conserved" is the wrong acceptance once a shock forms), and `case_riemann` — an exact extended-domain Riemann reference (u_L=1,u_R=0, shock speed 1/2, entropy production −(u_L−u_R)³/12) run at N=128/256/512 for both WENO5 and the adaptive blend, reporting L1/L2/shock-position/mass/entropy-rate/ν_num separately. Measured: L1 rate ~1.0 (shock-dominated, not 5th order — honest order), entropy production within ~1% of analytic, adaptive ≈1.34× WENO5's L1 at every N. The lifting case verifies Haar isometry/PR/dot product at ~1e-16 and shows CDF53 has `W_inv−W*` = 0.684 while PR stays at 2.17e-16. The adjoint case checks `A↔Aᵀ` (2.7e-16), quadratic and ODE gradients against finite differences (~1.5e-10), and reports checkpoint bytes. The scale case shows the pole spread stays 1024 for both mode C and mode g_m while the span moves between noise and `g_m`; no normalization gives 1.0 model-time error. The Gram case shows right-Gram calibration 2.27e-2 → 3.46e-15 and the wrong left multiplication stuck at 0.032.

**`bench/run_twin_experiments.py` — the experiments.** Contains the deliberately small Burgers/advection solvers that produce data to be measured (WENO5, minmod/van Leer MUSCL, Roe flux). This is the only place allowed to contain a solver.

## Non-obvious constraints

**ngspice must run with `cwd` set to the PDK model directory.** PDK libraries use relative `.include` (IHP's `sg13_lv_nmos` does `.include sg13g2_moslv_parm.lib`), so a netlist invoked from `results/` fails with "Cannot compute substitute" or "could not find a valid modelname". `run()` derives cwd from the include path for this reason; the netlist writes to `results/` via absolute paths.

**Do not set `PDK_ROOT` for sky130.** With it set, ngspice resolves OSDI library paths to a directory lacking the `.osdi` files and aborts before any analysis. Unset, it degrades gracefully and the BSIM4 models still work. This is sky130-specific: IHP MOS *requires* `PDK_ROOT`/`PDK` so its PSP103 OSDI files load.

**gf180 needs `.param fnoicor=0`.** The model file's `noia`/`noib` expressions reference it; `design.ngspice` gives the default, but `extract.py` includes the model file directly.

**IHP MOS needs `cornerMOSlv.lib`, not the parm file.** The corner lib defines the `sg13_lv_nmos` subcircuit and includes the parm file; including only the parm file yields "could not find a valid modelname". PSP103 is loaded through OSDI, so the subprocess must set `PDK_ROOT` to the version directory and `PDK=ihp-sg13g2`.

**W/L units are PDK-specific.** sky130 subcircuits take microns (`w=10 l=1`); IHP and gf180 take SI with suffixes (`w=10u l=0.13u`). Encoded per-PDK in the `PDKS` dict.

**Set `OPENBLAS_NUM_THREADS=1` for the analysis code.** The default 64-thread OpenBLAS on this machine makes a 128×128 complex `solve`/`svd` take ~1 s (vs ~14 ms single-threaded), which turns the frequency sweeps into minutes. `run_phase0.py` sets it before importing numpy; if you import `analysis/` from another script, do the same. `sigma_min` also uses Gram + `eigvalsh` instead of `svdvals` for the same reason (~40×).

**Report order as measured, never as nominal.** `DESIGN.md §6.8` records that a 2nd-order limiter can be wider than a first-order scheme at a shock, and that the prototype's WENO5 measures ~2nd order (fixed eps=1e-6, not scaled with h). Those caveats are deliberate; do not tidy them into a claim the data does not support.

## Where the numbers live

`DESIGN.md §6.8` (twin experiments), `§6.9` (PDK extraction), `§6.10` (Phase 0 analyzer), `§6.13` (Phase 1 front-end), `§6.14` (interconnect delay), `§6.15` (route closure) and `§6.16` (adaptive representation / transform contract) hold the quantitative record: ideal-mirror drift ~1e-15 (roundoff) versus 1.9e-5 at transistor level, the four-PDK lambda table (HBT 0.0138 @20µA, gf180 0.034, sky130 0.081, IHP MOS 0.970 /V — λ = gds/I_D at the actual vsweep bias), the drift-vs-lambda ratios (~2.4 / 2.38 / 11.97) that confirm the leak mechanism, the Phase 0 verdicts (κ=N², κ=N⁴, Burgers → representation, Robertson → stiffness, singular-DAE H2 via `qz-dae-eliminated`, energy-metric Γ; 38/38 checks), the front-end classifications (Lorenz/Duffing polynomial, HH rational, Bessel rational with a domain, Gamma rejected; manifold-γ saturation fit, native-primitive breakeven; 41/41 checks), the delay results (uniform closed form 2.3e-16, scalar DDE τ_crit ratio 2/π, diffusion τ_crit 0.2578, phase budget 27.8 ps @100 MHz for 1°; 15/15 checks), the route results (0.3791 RC t50, 4-level fanout 11.45°, all-pass J_H 108× better than RC, deskew 39.8 ps → 3e-7 ps but 2.15° absolute, crosstalk floor 9.09% + victim floor, τ_tol ratio 200 and 2Q=20.05, loop margins with τ_fo/τ_ex=0.6467, PhysicalScaling verdict flip, phase-delay@ωc unwrap; 73/73 checks), the transform results (sensor σ 1.0000/0.1265, conservation 2.22e-16, shock x 0.7480, L2 4.59e-3, crossings 8→2, Riemann exact reference with entropy production ±1% and honest L1 rate ~1.0, σ tracker rate bounds r_release≤46/s verified by sweep, Haar isometry 3.5e-16, CDF53 `W_inv−W*` 0.684, adjoint gradients ~1.5e-10, pole spread 1024, Gram 2.27e-2→3.46e-15; 56/56 checks), and the SDE results (OU Var/ACF, bandwidth deficit 6.69% vs 6.4% flat-band, GBM Itô/Strat separation 51.9σ, double-well KL 5e-4, CRN/IND gradient variance ratio 2.3e5, Wong–Zakai θ(χ)=1/(2(1+χ)) within 15% at all χ, adjoint Δ_asym bound err/bound≈0.71; 27/27 checks). Noise values in §6.9 are derived from `4kTγ/g_m`, not `.noise` output — do not describe them as measured. Cite these rather than re-deriving, and update them when a rerun changes a number.
