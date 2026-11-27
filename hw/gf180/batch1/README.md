# gf180mcu batch-1 characterization vehicle — testbench netlists

Netlist-first: **the `.sp` files here are the record**; xschem `.sch` drawings are
documentation to be added later. Where the two disagree, the netlist wins.
Every number this vehicle claims is produced by running these netlists and
reconciling them three ways (SPICE / primitive-bank composition / design closed
form) in `bench/run_gf180_batch1.py`; the prose record is `DESIGN.md §6.27`.

    python3 -m bench.run_gf180_batch1        # 18/18 checks, ~2 s, writes results/gf180_batch1.json

It needs `results/gf180_extract.json` (G1) for its macro column:

    python3 -m twin.spice.extract_gf180      # ~80 s, 10 fail-closed blocks

## Tiles

| file | tile | structure | headline numbers (§6.27) |
|---|---|---|---|
| `t1_integrator.sp` | T1 state | replica-diode-biased nMOS gm cell (10 µA, 10u/1u) + one `cap_mim_2f0_m4m5_noshield` 22.36×22.36 µm (1.0163 pF) + ideal 10 µA sink load | a0 = 331.12, f_3dB = 63.26 kHz; macro 333.59 / 63.23 kHz |
| `t5_flux_chain16.sp` | T5 flux | 16 × 1 pF nodes, 15 `nfet_03v3` 10u/1u pass elements (gate at vdd), source-held, plus a 2-node calibration cell and a 0.1 V step | DC gain 1.0000, R_chord 1189 Ω, g_ss 1.43 mS, τ₁ 187.8 ns vs 73.1 ns composition (tax ×2.57) |
| `t7_route_tester.sp` | T7 route | two rm3 wires (1000u×2u = 500□, 100u×2u = 50□) four-wire probed, then the 1 mm wire as a route into a 1 pF MIM load driven by a gm cell | 0.0906 Ω/□ (rsh_rm3 0.090), 1 mm route invisible (63.3 vs 63.3 kHz), wire t50 1.38 ps = 5.0 % of the 27.8 ps 1° budget |
| `t8_noise_bank.sp` | T8 noise | (a) replica of the extractor's cell (100 kΩ load), (b) m = 1/2/4/8 bank (10 kΩ loads), (c) kT/C anchor (1 kΩ + 1 pF MIM), (d) W=100u clean cell | kT/C 63.52 µV vs 63.83 µV (−0.5 %); bank 16.15→10.42 nV/√Hz (×8 area → ×1.55, **not** √8) |
| `t9_dac5.sp` | T9 statistics | 5-bit binary current DAC, `m=1/2/4/8/16` mirror legs, ideal `S` switches, 1 kΩ summing load, one process per code | LSB 11.100 µA (λ-corrected 11.612), FS 336.55 µA = 30.32 LSB, INL(FS) −2.19 % vs λ −1.19 % |
| `t10_comparator.sp` | T10 constraint/event | nMOS input pair + cross-coupled regeneration pair, 100 kΩ loads, 20 µA tail, `{REGEN_BLOCK}` OFF/ON variants, `{MM}`/`{SEED}` mismatch tokens | offset −0.000 mV (nominal), 4-draw σ 1.84 mV vs Pelgrom 2.29 mV, τ_reg 0.38 ns, 1 V delay 7.07 ns |

## Runner conventions (all of these have bitten at least once)

- Tokens: `{NGS}` PDK ngspice dir, `{OUT}` = `results/`, `{TAG}` per run, `{MM}` 0/1,
  `{SEED}` integer, `{REGEN_BLOCK}` comparator variant, `{B0}..{B4}` DAC bit, `{vdd}`.
  `twin.spice.extract_gf180.run(tag, text)` fills `{NGS}`/`{OUT}` and enforces the
  three ngspice conditions (PDK_ROOT/PDK, cwd = model dir, `env -u LD_LIBRARY_PATH`).
- Every netlist must set `.param fnoicor=0` and both `sw_stat_global` /
  `sw_stat_mismatch`. MIM needs `mimcap_typical`; **metal resistors `rm1..rm4`
  live in `res_typical`, not `typical`**.
- Port order of all four MOS subcircuits is **d g s b**. P-bulk to vdd, n-bulk to 0.
- ngspice switch model parameters are **`Vt`/`Vh`**; `Vth` is silently ignored
  (defaults to 0 → the switch is always on, and every DAC code reads the same).
- A source cannot be both `dc`-swept and carry a PWL waveform. Split them and sum
  with two **series** `E` sources (parallel sources on one node fight).
  In a DC sweep a PWL source takes its `dc` field, **not** its t=0 point.
- `wrdata` writes one (scale, …) group *per vector*: one complex vector = 3
  columns, two real vectors = 4 columns, two complex vectors = 6. Parse by group.
- A −3 dB frequency must be interpolated in log f; taking the first grid point
  below −3 dB overstates the pole by the decade step (5.7 % at `dec 40`).

## Open items handed to C1 (do not quietly close them with simulation)

1. **T5 per-node parasitic tax ×2.57.** The composition has no junction/wire
   capacitance per node; the 1 mm wire's 80 fF explains only a fraction. Needs a
   per-node capacitance model (post-layout) or a silicon measurement.
2. **T8 white noise vs area.** The PDK's MOS white noise does not scale as
   4kTγ/g_m with W×m (`m=8` ≡ `w=80u`, so it is not a netlist artifact). The bank
   must be measured; the replica's flicker corner sits on its pole, so §6.25(c)'s
   γ_eff 0.978 is a fit value with an unquantified band-model error.
3. **T9 m-scaling DNL.** INL(FS) −2.19 % vs the λ mechanism's −1.19 %; the extra
   ~1 % is m=16-finger vs 16-single-finger layout difference. Re-measure after layout.

## Two process-data sources that disagree

PDK `res_typical` says `rsh_rm1..rm4 = 0.09 Ω/□`; magic's `gf180mcuA.tech` prints
`metal3 10` (a different unit convention). SPICE wins for the resistance. The
ngspice kit has **no wire capacitance model at all**, so wire C is taken from the
tech file's "Nominal capacitances" table (foundry doc PDS_035_03): M3 areacap
10.094 aF/µm², perimeter 30.021 aF/µm. That is a layout-document number, not a
SPICE number, and is labelled as such in `DESIGN.md §6.27`.
