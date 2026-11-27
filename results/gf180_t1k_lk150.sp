* T1 four-knob integrator cell -- gf180mcu batch-1 characterization vehicle
* (P0-B, DESIGN 6.30)
*
* The P7 coefficient graph asks one integrator for four things, and this cell
* is the topology that delivers them:
*   a_ii      a binary-weighted NMOS gm bank (4 legs, m=1/2/4/8) off the
*             replica rail -> 15 codes, a_ii = k * gm_unit / C
*   leakage   a triode NMOS whose gate is a continuous voltage; its drain
*             conductance g_leak = k_n*(W/L)*(Vgs-Vt) is the leak knob (a
*             *continuously* settable resistor, not a code)
*   b_ports   two input gm ports, fixed relative weights (m=1 and m=2), each
*             with an enable; port gain = gm_port/C, crosstalk measured
*   STATE_WR  a write/hold path in THREE variants so the charge injection can
*             be compared, not assumed: (1) an ideal S switch (zero injection,
*             the reference), (2) a CMOS TG (n 2u/0.28u + p 4u/0.28u),
*             (3) an NMOS-only switch
*
* One run = one configuration. The runner substitutes the bit/enable/analog
* tokens and one of three analysis blocks (AC integrator / DC leak / transient
* write-hold), so the deck stays a single checked-in netlist.
*
* Conventions that bite (see README): ngspice switch params are Vt/Vh (Vth is
* silently ignored); the pull-up Ipu is matched to the bank code by the runner
* ({IPU} = k*10u), because a fixed 10 uA pull-up only balances code 1.
*
* Tokens (brace-substituted by the runner): NGS = PDK ngspice dir, OUT =
* results dir, CTRL = the analysis block; plus the bit / enable / analog
* parameters listed in the header.
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
.param vdd=3.3
Vdd vdd 0 {vdd}
.model SWM SW(Ron=1 Roff=1e12 Vt=0.5 Vh=0.1)

* --- replica bias: a 30 uA NMOS diode (unit leg) and a 30 uA PMOS diode.
* At 10 uA the unit sits at the threshold and the op point is indeterminate;
* 30 uA puts Vov ~ 100 mV so gm and ro are both well defined.
Iref vdd cdr dc 30u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
Eb gb 0 cdr 0 1
Irefp cdrp 0 dc 30u
XMp0 cdrp cdrp vdd vdd pfet_03v3 w=10u l=1u
Ebp gpb 0 cdrp 0 1

* --- a_ii: binary gm bank. The AC test rides on gin; gin = gb + 0 AC.
Vac gb gin 0 dc 0 ac 0
Vb0 b0 0 dc 0
Vb1 b1 0 dc 0
Vb2 b2 0 dc 0
Vb3 b3 0 dc 0
S0 gin g0 b0 0 SWM
S1 gin g1 b1 0 SWM
S2 gin g2 b2 0 SWM
S3 gin g3 b3 0 SWM
* AC bypass: the SW element is fine in op/tran but its AC linearization sits at
* Roff, which would leave the gate floating in an AC run. These resistors carry
* the code (0.01 ohm if the bit is on, 1e12 if off) and the SW stays for the
* nominal switching transient.
Rby0 g0 gin 1000000000000.0
Rby1 g1 gin 1000000000000.0
Rby2 g2 gin 1000000000000.0
Rby3 g3 gin 1000000000000.0
Rg0 g0 0 1e9
Rg1 g1 0 1e9
Rg2 g2 0 1e9
Rg3 g3 0 1e9
XM1 out g0 0 0 nfet_03v3 w=10u l=1u m=1
XM2 out g1 0 0 nfet_03v3 w=10u l=1u m=2
XM3 out g2 0 0 nfet_03v3 w=10u l=1u m=4
XM4 out g3 0 0 nfet_03v3 w=10u l=1u m=8

* --- leakage: triode NMOS, gate voltage 1.5 (continuous knob)
Vgl vgl 0 dc 1.5
* The leak device's *source* sits on 0: 0 V makes it a plain triode resistor,
* but at a 1.2 V state a ground-source device is in saturation for VGL < 1.81 V
* (it becomes a current sink, g_ds = lambda*I). 0=1.0 V level-shifts it so
* Vds = V_state-1.0 stays in triode at the 1.2 V state; the body is at 0, so Vt
* picks up the body effect (measured, not assumed).
Vsb vsb 0 dc 0
XMlk out vgl vsb 0 nfet_03v3 w=2u l=2u

* --- b ports: weights 1 and 2, individually enabled. Their gate DC is the
* replica rail (so a port unit has the same gm as a bank unit) and each gets
* its own AC drive at that bias.
Vp1 gb p1 0 dc 0 ac 0
Vp2 gb p2 0 dc 0 ac 0
Ve1 e1 0 dc 0
Ve2 e2 0 dc 0
Sp1 p1 q1 e1 0 SWM
Sp2 p2 q2 e2 0 SWM
Rbyp1 p1 q1 1000000000000.0
Rbyp2 p2 q2 1000000000000.0
Rp1 q1 0 1e9
Rp2 q2 0 1e9
XMp1 out q1 0 0 nfet_03v3 w=10u l=1u m=1
XMp2 out q2 0 0 nfet_03v3 w=10u l=1u m=2

* --- the integrating capacitor: the real MIM device, 500 um^2 = 1.0163 pF
XC out 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u

* --- the current-source load: a PMOS leg of the same code (1 = k), so
* the node's DC is set by the NMOS/PMOS ro mismatch (a real integrator, not an
* ideal current source with infinite ro). 0=0 disables it (source at 0 V),
* which is how the hold and leak measurements remove it without m=0. The bulk
* stays at vdd: a bulk at the pulled-down source would forward-bias the
* drain-bulk diode at a 0.1 V drain and add a 37 uS floor (measured).
XMpl out gpb vp vdd pfet_03v3 w=10u l=1u m=1
Vp vp 0 dc 0

* --- CMFB fixture: a weak DC pin to a 1.2 V reference. 1e12=100Meg is the
* normal fixture (10 nS, a 1.4 % load on a 1.3 MOhm node); the hold test sets
* 1e12=1e12 so the pin does not set the droop (it would add 10 nS = a 113 us
* time constant on 1.13 pF and mask the leak knob).
Rcm out cm 1e12
Vcm cm 0 dc 1.2

* --- STATE_WR variant 1: ideal S switch (reference, zero injection)
Vw1 w1 0 dc 1.2
Sw1 w1 out ew1 0 SWM
Vew1 ew1 0 dc 0

* --- STATE_WR variant 2: CMOS TG (write when ENWN=3.3, ENWP=0). Port order is
* d g s b, so the gate comes second and the input node w2 is the source.
Vw2 w2 0 dc 1.2
XWn2 out enwn w2 0 nfet_03v3 w=2u l=0.28u
XWp2 out enwp w2 vdd pfet_03v3 w=4u l=0.28u
Venw enwp 0 dc 3.3
Venwn enwn 0 dc 0

* --- STATE_WR variant 3: NMOS-only switch
Vw3 w3 0 dc 1.2
XWn3 out ewn3 w3 0 nfet_03v3 w=2u l=0.28u
Vewn3 ewn3 0 dc 0

* --- measurement pin: an ideal Vds behind a switch (transient) and a resistor
* (AC/DC). The SW element is fine in op/tran but ngspice's AC analysis
* linearizes it at Roff, so AC uses 0.01=0.01 with the switch off.
Vds dsn 0 dc 0 ac 0
Sds dsn out eds 0 SWM
Veds eds 0 dc 0
Rds dsn out 0.01

* --- AC current injection for the capacitance probe (0=0 otherwise)
Iac 0 out ac 0

* --- the off-state floor (also the DC path for the AC mode): 1e12 = 1 pS
Rdc out 0 1e12

.control
dc Vds 0.098000 0.102000 0.002
wrdata /home/yanlu/solver/results/t1k_dc.dat i(vds)
.endc
.end
