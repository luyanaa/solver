# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## What this repository is

A design study plus digital twin, not a product codebase. The artifact that matters is `DESIGN.md`: it specifies a compiler and a layered continuous-time analog/hybrid fabric for solving ODE/PDE, and it carries the normative claims, scaling laws, acceptance criteria and experiment results. The Python under `twin/`, `analysis/` and `bench/` exists to falsify or confirm statements in `DESIGN.md`.

Read `DESIGN.md` before touching code. It is ~620 lines and organized so each section states a claim, then the number that makes it checkable.

## Commands

Plain Python 3 (stdlib + numpy + scipy + matplotlib). No build step, no packaging, no linter config, no test suite. Run modules from the repo root.

    python3 -m bench.run_twin_experiments      # all five twin experiments, ~3 min
    python3 -m bench.run_phase0                # Phase 0 verdicts, ~2 min, 23/23 checks
    python3 -m twin.spice.extract              # ngspice extraction, every PDK in the dict
    python3 -m twin.spice.extract sky130       # single PDK (~70 s: model loading dominates)
    python3 -m twin.spice.extract ihp-hbt

In the Trae agent shell, prefix ngspice with `env -u LD_LIBRARY_PATH`. The agent
injects a libstdc++ that ngspice cannot load (`GLIBCXX_3.4.32` missing). A normal
login shell does not need this.

Outputs land in `results/`: `report.txt`, `twin_experiments.json`, `twin_experiments.png`, `device_extract.json`, `phase0.json`, `phase0_report.txt`, `phase0.png`, plus per-PDK `ext_*.sp` / `ext_*.log` / `*sweep_*.dat`. The JSON files are the machine-readable record that `DESIGN.md §6.8/§6.9/§6.10` quotes.

To run one experiment in isolation, import and call it (each `exp_*` is self-contained):

    python3 -c "from bench.run_twin_experiments import exp_d; exp_d()"

## Architecture

Three layers, separated so measurement code never contains a solver and solver code never contains a claim.

**`twin/qoi.py` — estimators only.** Turns a measured profile into a transport coefficient. `fit_tanh` / `nu_from_width` implement the shock-width to equivalent-viscosity inversion (nu = du * Wd * dx / 8.789); `front_width_10_90` / `kappa_from_width_growth` implement front thickening to mixing coefficient (kappa = (1/13.139) * d(w^2)/dt for the erf family). Both constants are derived analytically, not fitted. A new estimator must be validated against a known analytic case, and its constant must be derivable.

**`twin/coupling.py` — the coupled spectral/local system.** The load-bearing file. One high-order nodal-DG element (`SpectralElement`, LGL lumped mass) and a finite-volume region (`FVRegion`, MUSCL + Roe) joined at two periodic interfaces. The invariant everything else tests is the mass budget

    dM/dt = (F_A_fv - F_A_spec) - (F_B_fv - F_B_spec)

`CoupledSystem.run()` integrates the state with RK4 and accumulates that same scalar with the same Butcher weights, so measured drift and predicted drift must agree to ~1e-12. If they diverge, the code has a hidden leak. That check has already caught real errors and is the main correctness signal in the repo.

`InterfaceModel` is where the interface physics lives. `single_source=True` shares one flux value with both sides; `mode="gm"` substitutes a transistor-level model driven by values extracted from real PDKs (`lam`, `eps_dev`, `eps_sys`, `shared_ref`). Do not "fix" a conservation result by changing the flux scheme — the scheme is intentionally minimal so the interface effect stays isolated.

**`twin/spice/extract.py` — PDK extraction.** Builds a diode-connected current-mirror netlist per process, runs ngspice in batch, parses the sweep `.dat` files into lambda = gds/ID, gm/ID, ro and the mirror ratio. PDKs are external (`~/.ciel/ciel/...`, `/grid/...`) and must not be vendored or modified.

**`analysis/` — the feasibility analyzer (Phase 0).** Pure linear algebra, no solver and no hardware model. `descriptor.py` owns (E,A,B,C) and the spectral metrics; `lyapunov.py` solves the generalized Lyapunov equation by QZ on the stable finite deflating subspace; `hinf.py` bisects the Hamiltonian pencil and cross-checks against a frequency sweep; `resolvent.py` computes Γ; `regularity.py` fits the spectral tail; `slack.py` builds the six-axis vector. Two policies are load-bearing: κ drops only *numerical* zero modes (|λ| < 1e-12·max|λ|), so physical slow modes stay in the verdict; and the representation axis must be evaluated on the physical state, never on a capturing scheme's output (WENO smears the tail with ν_num and would pass a test the state fails).

**`bench/run_phase0.py` — the Phase 0 acceptance.** Builds five systems (diffusion N=64/1024, biharmonic N=128, Burgers linearized at the pre-shock state, Robertson at t=1e4), runs the analyzer, checks against analytic κ/Γ/σ_y/H∞, and writes the verdict table. The Burgers exact solution is via characteristics; the Jacobian is the first-order Roe operator (differentiable), while the WENO solution is only used to show how capturing hides the spectral tail.

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

`DESIGN.md §6.8` (twin experiments), `§6.9` (PDK extraction) and `§6.10` (Phase 0 analyzer) hold the quantitative record: ideal-mirror drift ~1e-15 (roundoff) versus 1.9e-5 at transistor level, the four-PDK lambda table (HBT 0.028, gf180 0.034, sky130 0.081, IHP MOS 0.970 /V), the drift-vs-lambda ratios (1.23 / 2.38 / 11.97) that confirm the leak mechanism, and the five Phase 0 verdicts (κ=N², κ=N⁴, Burgers → representation, Robertson → stiffness, 23/23 checks). Noise values in §6.9 are derived from `4kTγ/g_m`, not `.noise` output — do not describe them as measured. Cite these rather than re-deriving, and update them when a rerun changes a number.
