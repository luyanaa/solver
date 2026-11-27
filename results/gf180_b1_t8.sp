* T8 noise bank -- gf180mcu batch-1 characterization vehicle
* Three probes on one netlist:
*  (a) replica of the extractor's single noise cell (100 kOhm resistive load)
*      -> absolute input-referred density must reproduce 10.97 nV/rtHz
*  (b) programmability: m = 1/2/4/8 unit cells in parallel, each at its own
*      10 uA, noiseless conductance load -> input-referred density ~ 1/sqrt(m)
*  (c) kT/C anchor: 1 kOhm + 1 pF MIM, total integrated noise must be sqrt(kT/C)
* The (b) load is a 10 k resistor (its 4.07 nV/rtHz is 3 nV input-referred,
* below the MOS term; an ideal VCCS load was tried first and the OP collapsed
* to v(d)=0 because the MOS channel and the VCCS have equal conductance there).
* Tokens: /home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice = PDK ngspice dir, /home/yanlu/solver/results = results dir.
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
.param vdd=3.3
Vdd vdd 0 3.3
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
Eb gb 0 cdr 0 1
* --- (a) extractor replica: resistive load 100 k, m=1
Vga gb ga 0 dc 0 ac 1
XMa da ga 0 0 nfet_03v3 w=10u l=1u
Rla vdd da 100k
* --- (b) programmable bank: 10 k load, m = 1/2/4/8
Vg1 gb g1 0 dc 0 ac 1
XM1 d1 g1 0 0 nfet_03v3 w=10u l=1u m=1
Rl1 vdd d1 10k
Vg2 gb g2 0 dc 0 ac 1
XM2 d2 g2 0 0 nfet_03v3 w=10u l=1u m=2
Rl2 vdd d2 10k
Vg4 gb g4 0 dc 0 ac 1
XM4 d4 g4 0 0 nfet_03v3 w=10u l=1u m=4
Rl4 vdd d4 10k
Vg8 gb g8 0 dc 0 ac 1
XM8 d8 g8 0 0 nfet_03v3 w=10u l=1u m=8
Rl8 vdd d8 10k
* --- (d) well-conditioned white-noise cell: W=100u at 100 uA into 10 k.
*     Flicker corner falls ~1/(W*L) (~0.3 MHz) while the load-gain pole
*     (~3.2 MHz) stays above it, so 0.5..3 MHz is a genuine white window --
*     the (a) cell's corner (2.9 MHz) sits ON its pole, which is why its
*     A+B/f fit is ill-conditioned. AC on the same node gives gm for gamma.
Vgd gb gd 0 dc 0 ac 1
XMd dd gd 0 0 nfet_03v3 w=100u l=1u
Rld vdd dd 10k
* --- (c) kT/C anchor: 1 kOhm + 1 pF MIM
Vn nk 0 dc 0 ac 1
Rk nk ck 1k
XCk ck 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
.control
op
echo T8OP $&v(da) $&v(d1) $&v(d8) $&v(ck)
* (a) the extractor's own bands: wide 1 MHz..1 GHz, low 1..10 MHz, so the
*     A+B/f plateau method reproduces its white number exactly
*     (the extractor's own netlist runs 1..100 MHz + a 1..10 MHz low band)
noise v(da) Vga dec 20 1e6 1e8
let q_a = onoise_total
let r_a = inoise_total
echo NROW a $&q_a $&r_a
* one-decade low band for the plateau fit
noise v(da) Vga dec 20 1e6 1e7
let q_al = onoise_total
let r_al = inoise_total
echo NROW a_low $&q_al $&r_al
* (b) same band, four bank sizes
noise v(d1) Vg1 dec 20 1e6 1e8
let q_1 = onoise_total
let r_1 = inoise_total
echo NROW m1 $&q_1 $&r_1
noise v(d2) Vg2 dec 20 1e6 1e8
let q_2 = onoise_total
let r_2 = inoise_total
echo NROW m2 $&q_2 $&r_2
noise v(d4) Vg4 dec 20 1e6 1e8
let q_4 = onoise_total
let r_4 = inoise_total
echo NROW m4 $&q_4 $&r_4
noise v(d8) Vg8 dec 20 1e6 1e8
let q_8 = onoise_total
let r_8 = inoise_total
echo NROW m8 $&q_8 $&r_8
* (d) gain (for gm) and the clean white band 0.5..3 MHz
ac dec 20 1e3 1e7
wrdata /home/yanlu/solver/results/b1_t8_dbig_ac.dat v(dd)
noise v(dd) Vgd dec 20 5e5 3e6
let q_d = onoise_total
let r_d = inoise_total
echo NROW dbig $&q_d $&r_d
* (c) full band for the kT/C integral
noise v(ck) Vn dec 20 1e3 1e10
let q_k = onoise_total
let r_k = inoise_total
echo NROW kt $&q_k $&r_k
.endc
.end
