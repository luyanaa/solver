# Chapter 1. Theoretical Framework

> **Scope.** This chapter is the English rendition of the foundational theory in `DESIGN.md` (§0–§2). It establishes *why* feasibility is a computable decision rather than a slogan: the four conclusions that pin down the design, the descriptor-system formulation that unifies every layer, and the scaling laws that determine the architecture. Section references of the form §3–§10 point to `DESIGN.md`, which documents the architecture, compiler passes, implementation status and experimental results.

---

## 1.1 Four conclusions that pin down the design

**(A) Stiffness cannot be removed by scaling or a basis change.**
Time scaling t→at multiplies every λ by the same factor, and a similarity transform TJT⁻¹ leaves the spectrum unchanged, hence

\[
\kappa_t = \frac{\max|\mathrm{Re}\,\lambda|}{\min|\mathrm{Re}\,\lambda|}
\]

is invariant under any pair (T, A_t). The hardware frequency-window width κ_hw = ω_max/ω_min is a process constant (τ_min ~ parasitic limit ≈ ns, τ_max ~ leakage / observation window ≈ ms):

\[
\kappa_{\mathrm{hw}} \approx 10^5\text{–}10^6 \quad (100\text{–}120\ \mathrm{dB}).
\]

Feasibility is therefore a single *"the width must fit into the window"* criterion: κ_t ≤ κ_hw. Time scaling only translates the window center:

\[
A_t = \sqrt{\frac{\omega_{\mathrm{hw,max}}\,\omega_{\mathrm{hw,min}}}{\omega_{\max}\,\omega_{\min}}}.
\]

The only escape when the width is exceeded is **mode deletion** (QSSA/CSP/ILDM) or **multirate integration** (paying a synchronization error) — *not* a "better integrator."

**(B) Single-node precision–speed conservation law.**

\[
\sigma^2 = \alpha\,\frac{kT}{C}, \qquad \tau = \frac{C}{g_m} \quad\Longrightarrow\quad \sigma^2\tau = \frac{\alpha\,kT}{g_m}.
\]

α ≈ 0.4–2 (an ideal single-pole OTA gives (2/π)γ). **C only redistributes precision between σ and τ; the product is set by g_m, i.e. by power.** In terms of dynamic range (V_swing = 1 V):

\[
\mathrm{DR} \approx 85\ \mathrm{dB} + 10\log_{10}\!\left(\frac{C}{\mathrm{pF}}\right),
\]

i.e. "10× the capacitance buys 10 dB." This single relation replaces any per-stage SNR budget.

**(C) All perturbations pass through one resolvent.**
Thermal noise, mismatch, switch charge injection, routing parasitics, event jitter (δx = ẋ·δt) and quantization are all, to first order, equivalent to injecting B_p·p(t) on (E,A); in every case their transfer to the observable is

\[
H(s) = C\,(sE - A)^{-1} B_p .
\]

Consequently three functionals are the complete set of criteria:

| Functional | Mathematics | What it governs |
|---|---|---|
| ‖H‖₂² = tr(C P Cᵀ) | generalized Lyapunov: E P Aᵀ + A P Eᵀ + B_p Q B_pᵀ = 0 | variance / SNR |
| ‖H‖∞ | Hamiltonian bisection | worst case, non-normal amplification, closed-loop small-gain |
| κ_t / spectrum | eigenvalues | frequency window |

**SNR, the switch matrix and stiffness are not three problems — they are three norms of the same (E,A).**

Note: the above is exact for **additive** perturbations. Multiplicative noise G(x)dW has state-dependent ports; the resolvent viewpoint holds only for additive or linear-multiplicative noise (boundary discussed in §1.2.3).

**(D) Noise is a programmable resource — but it shares C with precision.**
The diffusion coefficient of an SDE is not free. Injecting current noise (one-sided PSD S_I) onto a capacitor gives

\[
D = \frac{S_I}{2C^2} \qquad [\mathrm{V^2/s},\ \text{using the definition } D=\sigma^2/2],
\]

while the same C sets σ² = αkT/C from (B). To tune D independently one needs a dedicated noise injector plus gain/filtering, but the gain error multiplies D directly and must be calibrated. The white-noise approximation additionally requires the noise bandwidth to cover the state bandwidth: a first-order system needs B_noise ≥ 100·B_req to keep the variance error below 1%. **"Solving an SDE in analog" is not merely attaching a noise source to a capacitor: source, capacitance, bandwidth and calibration form a coupled set of constraints.**

---

## 1.2 Unified form: descriptor systems in hardware coordinates

Every layer — integrator, MVM, transform, stiff bank, flux, event — normalizes to

\[
E(\theta)\,\dot{x} = A(\theta)\,x + B(\theta)\,u + B_p(\theta)\,p(t), \qquad y = C x,
\]
\[
\theta = (\,T,\; A_t,\; \sigma,\; c,\; w\,) \quad = \quad (\text{basis},\ \text{time-scale},\ \text{splitting},\ \text{routing},\ \text{sizing}).
\]

A singular E is a feature, not a defect: algebraic constraints (the fast manifold after QSSA, the virtual ground of an ideal op-amp, KCL summing nodes) are naturally singular E — this is precisely the formalization of "current summation at a low-impedance node."

### 1.2.1 Perturbation ports B_p (the compiler input specification)

| Source | Injection point | Strength | Remarks |
|---|---|---|---|
| Thermal noise | every integrator state | input-referred 4kTγ/g_m; output σ² = αkT/C_i | diagonal Q |
| 1/f | same, **augmented states** | Lorentzian fit by Σ | stays inside the Lyapunov framework after augmentation |
| Mismatch | ΔA static | Pelgrom σ ∝ 1/√(WL) | structured, not a norm |
| Switch charge injection | at each switching | δQ/C, inverse in C | event-related, non-stationary |
| Routing parasitics | ΔE, ΔA (per config) | PEX gives Y_c(s) = G_c + sC_c | see §3 of DESIGN.md |
| Event jitter | ẋ·δt | δt ~ comparator noise / delay | automatically amplified by the resolvent |
| Sampling | kT/C + aperture jitter | 2πf·σ_j·A | only where S&H exists |

> Because 1/f noise and jitter can both be written as augmented LTI ports, **"non-white noise" does not break the H2 framework** — this is the premise that lets the whole design use one body of linear algebra for budgeting.

### 1.2.2 Feasibility is a slack vector, not a Boolean

The compiler emits **six dimensionless margins** (not a one-dimensional "difficulty score"), with the robustness axis split into mismatch and interface contributions:

\[
\text{slack} = \bigl(\underbrace{\sigma_y/\sigma_{\mathrm{req}}}_{\text{noise}},\;
\underbrace{B_{\mathrm{req}}/B_{\mathrm{hw}}}_{\text{bandwidth}},\;
\underbrace{\kappa_t/\kappa_{\mathrm{hw}}}_{\text{stiffness}},\;
\underbrace{\Gamma_{\mathrm{proj}\,\lVert\Delta A\rVert}}_{\text{robust (mismatch)}},\;
\underbrace{\lVert H\rVert_\infty\lVert\Delta u\rVert}_{\text{robust (interface)}},\;
\underbrace{\eta_{\mathrm{route}}}_{\text{route}},\;
\underbrace{E_{\mathrm{tail}}/\mathrm{tol}_{L2}^2}_{\text{representation}}\bigr).
\]

- **Γ_proj** is the resolvent peak after a QZ projection removes the conservative zero mode. The full Γ has a zero-mode pole at s=0 and is diagnostic only — it must not enter the slack, otherwise conservative systems such as Burgers would be misjudged as "infinitely fragile." **This requires ΔA to be conservative**: ΔA_conservative preserves the conservative invariant subspace (flux column structure), so the zero mode remains within machine precision. Once ΔA is *leaky* (unequal currents injected on the two sides, column sum ≠ 0 — e.g. current-mirror mismatch mapped onto a single flux edge), the zero mode is excited into a slow drift / instability of order 6e-3, which Γ_proj cannot see at all (a measured counterexample is in the docstring of `analysis/slack.py`). Leaky ΔA must instead be accounted for by a separate mass/charge drift rate (the dM/dt ledger in `coupling.py`) or a finite-time gain; never by Γ_proj‖ΔA‖. Verification: check that `sum(dA, axis=0)` (column sum) is ≈ 0.
- **‖H‖∞‖Δu‖** is the small-gain bound for interface disturbances; it is a different object from the mismatch axis and cannot be replaced by a single Γ.
- **E_tail_req = tol_L2²**: a 1% L2 representation error corresponds to a 1e-4 fraction of spectral tail energy.

Any axis exceeding 1 means infeasible — and tells you *which lever to pull*: κ over → delete modes; Γ_proj‖ΔA‖ over → lower Q / calibrate / change basis; ‖H‖∞‖Δu‖ over → attenuate interface disturbance or lower the loop gain; E_tail over → switch to a local representation; σ_y over → add C, but only at nodes with large ‖H_i‖₂.

Note: the stochastic realizability of an SDE (noise bandwidth, τ_c, D range, readout) is not packed into these six axes; it is an independent **stochastic certificate** at P6. These are not tradeable margins but compile-time preconditions (§1.2.4).

**The noise axis is not single-node SNR.** For ẋ = f(x) + B_n n(t), linearized around a trajectory:

\[
\delta\dot{x} = J(t)\,\delta x + B_n n(t), \quad J = \partial f/\partial x,
\qquad
\dot{P} = J P + P J^{\mathsf{T}} + B_n Q B_n^{\mathsf{T}},
\qquad
P_y = C P C^{\mathsf{T}}.
\]

What must be optimized is the observable error tr(WP) (which equals tr(CPCᵀ) when W = CᵀC), not "this OTA has 60 dB." For LTI systems solve the algebraic Lyapunov equation; for time-varying systems integrate along the trajectory, or use a frozen-time approximation with a validity check. This ties SNR, stiffness and non-normal amplification to the same P: the slow/non-normal modes of J determine how far noise is amplified, and bandwidth enters through Q and B_n.

### 1.2.3 SDE extension of the descriptor system: two graphs, drift and diffusion

**Yes, it can be done.** Additive SDEs are the same descriptor system as the ODE in §1.2, with just one extra constant B_p; multiplicative SDEs add a state-dependent diffusion graph. What is genuinely new are the noise contract, the stochastic semantics and the stochastic certificate.

Extend the descriptor of §1.2 to

\[
E\,dx = F(x,u,\theta)\,dt + \sum_k G_k(x,u,\theta)\,dW_k, \qquad y = H(x,u,\theta).
\]

Additive noise is the special case where G_k is independent of x; linear multiplicative noise is G_k(x) = G_k x; general multiplicative noise is G_k(x). On the hardware side it is still

\[
C_i\,\frac{dV_i}{dt} = I_{f,i} + I_{\mathrm{noise},i}.
\]

Hence **an SDE need not first be discretized to Euler–Maruyama**: the capacitor is already a continuous-time integrator. But "not discretizing" does not mean "no error": finite noise bandwidth, τ_c, the correlation matrix and readout sampling are new error sources. The three real advantages of analog here are: continuous-time integration has no step-size error; N trajectories can run in parallel on one array; and the noise is physical rather than pseudorandom. The route itself is not new — programmable analog computers / FPAs have done ODE/PDE/SDE before; the distinction of this design is that noise, stiffness and representation are placed in one computable decision framework (slack + stochastic certificate) instead of relying on hand configuration.

**Where reconfigurability lives**: the hardware does not change with the equation; what changes is the configuration — the weight matrices (A, B, C) of T3, the switches of T7, the PSD/τ_c of T8, the thresholds of T6, the capacitor arrays of T1/T4. The compiler maps five graphs onto this configuration set; so "many kinds of equations" is a compiler problem, not a tile problem.

The compiler input is five graphs rather than one formula string:

\[
\begin{aligned}
\text{STATE} &\quad x_0: C_0, \mathrm{range};\ x_1: C_1, \mathrm{range}; \dots\\
\text{DRIFT} &\quad dx_0 = \mathrm{add}(\mathrm{mul}(k_0,x_0), \tanh(x_1)); \dots\\
\text{DIFFUSION} &\quad g_0 = \mathrm{mul}(\sigma_0, noise_0);\ g_1 = \mathrm{mul}(\sigma_1, x_1, noise_1); \dots\\
\text{NOISE} &\quad noise_0: \mathrm{PSD}, \tau_c, \rho_{ij}, \mathrm{mode}=\mathrm{physical}|\mathrm{replay}\\
\text{OBSERVATION} &\quad y_0 = x_0; \dots
\end{aligned}
\]

An ordinary ODE is just empty DIFFUSION; Langevin is F = −∇V; a neural SDE replaces F/G by MVM→NL→MVM. All five graphs ultimately reduce to the same (E,A,B,C) plus B_p ports; for additive/linearized systems the slack of §1.2.2 still applies; the nonlinear multiplicative part is handled by the analysis boundary below and the P6 stochastic certificate.

**Noise contract** (missing any item ⇒ cannot compile):

| Field | Meaning | Why it is mandatory |
|---|---|---|
| S_ξ(f) | one-sided vs two-sided PSD must be stated | a factor of 2 ⇒ 3 dB error outright |
| τ_c | correlation time | criterion for the white-noise approximation and for Itô/Stratonovich |
| ρ_ij | noise-source correlation matrix | crosstalk from shared supply / ground |
| B_noise | flat PSD bandwidth | first-order system needs ≥100·B_req |
| mode | physical / replay | sampling vs CRN |
| quantization / filtering | replay DAC bit count, reconstruction filter | sets the effective τ_c of replay |
| calibration | residual gain, PSD, ρ | D carries a multiplicative error |

**Analysis boundary** (to be written into `analysis/`, without over-promising):

- **Additive** (G independent of x): existing Lyapunov analysis is exact; colored noise uses augmented states and remains LTI.
- **Linear multiplicative** G_k(x) = G_k x: the second moment is exact,

      dP/dt = A P + P Aᵀ + Σ_k G_k P G_kᵀ + Q,

  which after vectorization is a Sylvester equation of size n²; solvable for n ≤ 32. **Mean-square stability ≠ sample-path stability**: the second moment certifies variance, not almost-sure stability.
- **Nonlinear multiplicative**: the second moment does not close; only Monte Carlo / statistical linearization are available, and H∞ cannot be claimed.

**One extra constraint for PDEs**: the covariance of Q-Wiener noise must be trace-class. Spatial white noise in 1D usually is; in 2D/3D Σσ_k² = ∞, so a correlation length or spectrum must be specified. Upon discretization the noise amplitude must scale with the cell volume, otherwise finer grids make the noise stronger. This enters P0.5.

### 1.2.4 Itô / Stratonovich is a compile-time contract, not an implementation detail

For dx = f dt + G dW, the two interpretations differ in the drift by one term:

\[
f_{S,i} = f_{I,i} - \tfrac12 \sum_{j,k} G_{jk}\,\partial G_{ik}/\partial x_j .
\]

The compiler must therefore record two explicit fields:

\[
\text{interpretation} \in \{\text{ito}, \text{stratonovich}\}\quad(\text{user model}),
\qquad
\text{hardware\_limit} \in \{\text{stratonovich}, \text{ito}\}\quad(\text{actual circuit limit}).
\]

When they differ, P0.5 adds or subtracts the correction drift. The direction is fixed:

- model Itô, hardware Stratonovich ⇒ hardware drift is f_S = f_I − ½ Σ G ∂G;
- model Stratonovich, hardware Itô ⇒ hardware drift is f_I = f_S + ½ Σ G ∂G.

For additive noise (∂G/∂x = 0) the two coincide; for multiplicative noise they differ. For linear multiplicative noise G_k(x) = G_k x the correction reduces to

\[
A_S = A - \tfrac12 \sum_k G_k^2 .
\]

**One must not assume "analog = Stratonovich" by default.** Wong–Zakai is about the τ_c → 0 limit of smooth-noise approximations; real circuits have finite τ_c and delay, so the bias is O(τ_c) and can even exhibit a Stratonovich→Itô transition. The contract records τ_c and the acceptance test measures the bias.

A falsifiable scalar test: dx = μx dt + σx dW.

\[
\text{Itô:}\ \mathbb{E}[x(t)] = x_0 e^{\mu t}, \qquad
\text{Stratonovich:}\ \mathbb{E}[x(t)] = x_0 e^{(\mu+\sigma^2/2)t}.
\]

The two curves separate visibly at σ²t ~ 1; this is the GBM acceptance test of §6 of DESIGN.md. The first version commits only to scalar/diagonal, pairwise-commuting G; non-commuting G (Lévy area) and multi-dimensional corrections come later.

### 1.2.5 Reducibility: higher-order equations → canonical form

**First remove the order, then talk about function spaces.** An n-dimensional r-th order equation x^(r) = F(t, x, …, x^(r−1)) is reduced by setting z = (x, x′, …, x^(r−1)):

\[
\dot{z} = [\,z_2, z_3, \dots, F(t,z)\,].
\]

Order only multiplies the state count (n·r integrators); it adds no primitive type. For an implicit G(t, x, …, x^(r)) = 0: when det ∂G/∂x^(r) ≠ 0 the implicit-function theorem solves locally for the explicit form; otherwise it is a DAE and enters the singular E of the descriptor. **DAEs of index > 1 must first undergo index reduction**, otherwise the Lyapunov/H∞ analysis would need extra index-1/regularity conditions and cannot be applied directly.

**"Reducible" is not "approximable by a polynomial" — there are three levels:**

| Level | Definition | Error measure | Compiler behavior |
|---|---|---|---|
| Level 0, exact finite | a finite state augmentation exists whose right-hand side is built from B={+,×,INT,φ_i} and is an identity on the invariant manifold | 0 (symbolic / high-precision verification) | accept, emit an exact configuration |
| Level 1, ε-reducible | ∃ E~ ∈ Closure(B), Cost(E~) ≤ N_max, with d_O < ε on (K,T,Ω,O) | QoI error, not pointwise state | accept, attach an error certificate |
| Level 2, statistically reducible | pointwise trajectories incomparable (chaos / long horizon), but the invariant measure, spectrum or correlation function agrees within ε | statistical distance | commit only to statistics, not to trajectories |

The Level-1 contract is written as

\[
R = (B, K, T, \Omega, O, \varepsilon, N_{\max}, d_O),
\]
\[
E\ \text{reducible} \iff \exists\tilde{E}\in\mathrm{Closure}(B):\ \mathrm{Cost}(\tilde{E})\le N_{\max}\ \text{and}\ d_O(E,\tilde{E};K,T,\Omega)<\varepsilon .
\]

"Reducible" without O and d_O has no engineering meaning; hardware is correct only on K×[0,T]×Ω.

**The differential-algebraic dictionary of B:**

| Primitive | Augmentation | Relation | Cost / caveat |
|---|---|---|---|
| sin/cos | 2 states | ṡ = c·v, ċ = −s·v, s²+c² = 1 | quadratic; the invariant manifold must be maintained |
| exp | 1 state | ẏ = y·ẋ | linear ×; y>0 needs a limiter |
| tanh | 1 state | ẏ = (1−y²)ẋ | quadratic; \|y\|<1 needs a limiter |
| Bessel | 2 states (+ derivative of x) | linear ODE with coefficients containing 1/x; in the t-domain also needs ẋ, ẍ | rational, not polynomial; x must stay away from 0 |
| Gamma | — | Hölder: no finite-order algebraic differential equation | no finite augmentation; native block or bounded approximation only |

So "sin/cos/exp/tanh can be eliminated" is true, but **only on the invariant manifold**: if the initial condition violates s²+c²=1, or hardware drift leaves the manifold, the result is a different system. The compiler must choose between a native NL block and augmented states plus projection/feedback. The criterion is

\[
\mathrm{Cost}(\text{augmentation}) + \mathrm{Cost}(\text{manifold maintenance}) < \mathrm{Cost}(\text{native block}),
\]

with acceptable conditioning. Example: the pendulum θ̈+γθ̇+sinθ=0 uses 4 states [θ,ω,s,c], ω̇=−γω−s, s²+c²=1 when augmented, but only 2 states with a native sin block. Both are "correct" at different cost.

**Canonical form (structured, not a full polynomial expansion):**

\[
F(x) = \underbrace{A x}_{\text{linear}} + \underbrace{B(x\otimes x)}_{\text{low-rank quadratic}} + \underbrace{\sum_r U_r \varphi_r(V_r x)}_{\text{few special functions}} + \underbrace{R(x)}_{\text{sparse residual}} .
\]

For a scalar quadratic xᵀBx with B = Σ λ_r u_r u_rᵀ, the cost drops from O(n²) to O(Rn). A vector-valued quadratic is a 3-tensor; a low-rank decomposition must be done component-wise, not pretended to be one matrix.

**The error is not ‖F−F~‖∞.** For a vector field with Lipschitz constant L, Grönwall gives

\[
\lVert x(t)-\tilde{x}(t)\rVert \lesssim e^{Lt}\lVert x_0-\tilde{x}_0\rVert + \frac{\varepsilon_f}{L}\bigl(e^{Lt}-1\bigr).
\]

Thus ε_f ≪ 1 is meaningless on its own; what matters is (ε_f, L, T). When LT ≫ 1 or the system is chaotic, switch to measuring the attractor / stationary distribution / spectrum / correlation / Lyapunov spectrum (Level 2). This is the function-space version of the QoI metric in §6 of DESIGN.md.

---

## 1.3 Seven scaling laws (they determine the architecture, not aesthetics)

### 1.3.1 Precision, speed, power

Single node, B = 10 MHz, V_DD = 1 V, g_m/I_D = 15:

| σ_rms | C | g_m | node power | 1000 nodes |
|---|---|---|---|---|
| 1 mV | 2.8 fF | 0.17 µS | 12 nW | 12 µW |
| 100 µV | 0.28 pF | 17 µS | 1.2 µW | 1.2 mW |
| 10 µV | 28 pF | 1.7 mS | 115 µW | 115 mW |

**Conclusion: neither power nor area is the bottleneck** (1000 nodes at 10 µV and 10 MHz ⇒ ~0.1 W, ~3 mm² of capacitance). What is scarce is κ_hw, routing parasitics and calibration drift. Resources should be spent on **topology and calibration**, not on a "more power-efficient OTA."

**A process envelope is not a single f_T number.** Each process provides

\[
P_{\mathrm{tech}} = (V_{\mathrm{swing}},\ B_{\mathrm{accurate}},\ S_n(f),\ \mathrm{DR},\ g_m/I_D,\ f_T,\ f_{\max},\ R_{\mathrm{sw}},\ C_{\mathrm{sw}}).
\]

Three frequency tiers must be kept apart: f_signal ≪ f_block ≪ f_T, f_max. What guarantees analog accuracy is f_signal, not the frequency at which "the transistor still works." The compiler does not ask "can f_T=10 GHz make a 1 GHz solver"; it asks "how much B/SNR/P does this mapping need at the required error."

There is no free term in precision–bandwidth–energy: σ²τ = αkT/g_m, σ² ∝ B, σ² = kT/C. Raising SNR means adding C; keeping ω_c = g_m/C means adding g_m; adding g_m means adding power.

### 1.3.2 The cost of acceleration

B_hw ≈ A_t·B_model, and under white noise σ² ∝ B ⇒ **accelerating by A_t multiplies the noise power by A_t**. From §1.3.1, σ²τ is fixed ⇒ acceleration does not change σ²τ, only where τ lands in the window. The real constraint is the window width of conclusion (A).

Do not import the communication-capacity formula BT log(1+SNR): the constraint of an analog ODE is the state covariance / observable error, not a bit rate.

### 1.3.3 Similarity invariance of κ (a correction to intuition)

κ_t is invariant under (T, A_t), hence:

- **"Implementing it with a dedicated stiff linear block" does not reduce κ.** What it reduces is *implementation cost* (one cheap first-order branch per mode instead of forcing a general integrator to span eight decades), but the chip must still produce τ ∈ [10 ns, 1 ms].
- Only two things change κ: **mode deletion** (a fast mode is slaved and stable ⇒ g(x,y)=0 ⇒ it disappears from the spectrum) and **multirate** (paying a synchronization error).

In practice κ_hw must be replaced by the *usable* bandwidth ratio: f_max,usable = min(GBW_OTA, parasitic pole), f_min,usable = max(leakage, 1/f corner, drift). The gap between the mathematical κ_t and κ_t,usable is risk 1 of §9; the compiler decides using the latter, not the former for self-comfort.

Also, κ is similarity-invariant but **Γ / transients are not**: the V=Sx preconditioning of §4 of DESIGN.md lowers Γ without altering the spectrum.

### 1.3.4 λ_k = −νk²: spectral methods matter more for analog than for digital

\[
\kappa_{\text{diffusion}} = N^2, \qquad \kappa_{\text{biharmonic}} = N^4 .
\]

N=64: 4×10³ (easy); N=1024: 10⁶ (at the edge); biharmonic N=128: 2.7×10⁸ (**necessarily out of window**).

First hard rule: **prefer spectral for diffusion-type problems; biharmonic/higher-order operators need dimensionality reduction or a hybrid already at moderate resolution.** Conversely, in a spectral basis each mode is an independent first-order branch with g_m/C = νk² — exactly what analog hardware is best at — provided the span of k² fits into the window.

### 1.3.5 Routing tolerance ∝ 1/Q

The pole sensitivity of a second-order section is ∂λ/∂p ~ Q. A 2% relative error in switch/R_on/C_par ⇒

\[
\delta\omega_0/\omega_0 \sim Q\cdot 2\% \quad\Longrightarrow\quad \text{error on the order of } 1/Q.
\]

**Rule: "calibration-free" is allowed only for Q ≲ 50; higher Q requires trimming/calibration or decomposition into first-order sections.** This is the quantitative version of "a 2% switch network causes a large pole shift in a high-Q loop", and the engineering form of Γ_proj‖ΔA‖ < 1 in §1.2.2.

### 1.3.6 Stochastic precision, speed, power

Combining the precision law (B) with the SDE diffusion coefficient:

\[
\sigma^2 = \alpha kT/C, \qquad D = S_I/(2C^2) \quad(S_I: \text{one-sided current PSD},\ D=\sigma^2/2).
\]

The same C sets both "state precision" and "diffusion strength." A dedicated noise injector decouples them, but:

- gain error ε_g multiplies D by (1+ε_g)²; **the calibration accuracy of D is the accuracy of the distribution**;
- intrinsic thermal noise gives a floor D_int; the effective D = D_inj + D_int, so injection must be significantly above the floor;
- the white-noise approximation requires B_noise ≥ 100·B_req: the variance error of a first-order system is (2/π)arctan(f_c/B), already 6.4% at f_c = 10B;
- a replay DAC's update rate f_s limits it to noise within f_s/2; a 10 MHz system needs a 1 GHz DAC, which is not cheap in power/area. **Physical thermal noise does sampling; replay does CRN/gradients** — not redundancy.

Orders of magnitude (300 K, R = 1 kΩ):

| C | σ_th = √(kT/C) | D = S_I/(2C²) |
|---|---|---|
| 1 pF | 64 µV | 8.3 V²/s |
| 1 fF | 2.0 mV | 8.3e6 V²/s |

The steady-state variance of an OU process is D/λ and must simultaneously "fit inside the state swing" and "stay above the kT/C floor." These two inequalities bound the usable range of D; outside it, change C, change injector, or reduce bandwidth.

### 1.3.7 Representation resource law: monomial count and low-rank quadratics

A dense polynomial in n states of total degree p has

\[
N_{\mathrm{mono}} = \binom{n+p}{p}
\]

terms; n=100, p=4 gives 4.6e6 — already beyond any crossbar. Stone–Weierstrass only says "it can be approximated", not "it can be implemented efficiently." The canonical form must therefore be structured:

- linear term Ax: T3 MVM, O(n²) weights (or sparse / low-rank);
- scalar quadratic xᵀBx with low rank R: Σ λ_r (u_rᵀx)², O(Rn) multipliers;
- special functions φ_r(V_r x): one T2 NL block each;
- sparse residual R(x): only the terms actually needed.

**State augmentation is not free**: every augmented state is one integrator plus one share of noise / offset / drift / calibration. Price augmentation using the σ²τ law of §1.3.1 and the resource law of §6 of DESIGN.md; prefer a native NL block when possible, and low rank over dense.

**Carleman truncation is a closure, not a lossless representation.** For polynomial systems, monomial lifting is exact in infinite dimensions; truncation can change stability, and the monomial count explodes as N_mono. Like Fourier projection and QSSA, it must carry a closure error and cannot be advertised as an "exact linearization."

### 1.3.8 Analog timing closure: four conditions, not one

Digital STA asks only t_comb < T_clk; this machine must ask at least four:

| Condition | Mathematics | Physics if violated |
|---|---|---|
| device bandwidth | f_dyn/f_device ≪ 1 | the OTA's GBW / parasitic poles enter the dynamics |
| analog skew | ω_max Δτ_route ≪ 1 | the wire becomes a DDE rather than an interconnect; first-order equivalent E = I + A∘T (§6 of DESIGN.md) |
| constraint timing | τ_constraint ≪ τ_state,min | Poisson/KCL relaxation cannot keep up; elliptic turns parabolic |
| noise | P_hw ≺ P_error,budget | correlated noise cannot be removed by ensemble averaging |

The first gives f_signal ≪ f_block ≪ f_T,f_max (§1.3.1). The second ties A_t to die geometry: larger A_t means larger ω_max and a smaller allowable τ; a 1° phase budget at 100 MHz is only 27.8 ps. The third is the timing of T10: ε_c = τ_constraint/τ_state,min; a steady-state constraint is relaxed by hardware into C_y ẏ = −Ay+b, and ε_c must be small enough. The fourth is not "add C": the covariance of correlated noise is low rank and cannot be averaged as 1/√N; noise on a global reference becomes chip-wide correlated noise.
