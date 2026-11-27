* T10 regenerative comparator core -- gf180mcu batch-1 characterization vehicle
* NMOS input pair (W/L=10u/1u) + cross-coupled NMOS regeneration pair, 100 k
* loads, 20 uA ideal tail. Unclocked core: the clocked latch shell is a B3
* wrap, this file measures the three numbers the compiler needs (offset,
* regeneration time constant, delay).
*
* Two measurements from one netlist:
*   DC   sweep Vid -50..50 mV -> the trip point = input offset
*   TRAN 4 mV step at t = 5 ns -> time to |v(n1)-v(n2)| = 1 V
* Run twice: mismatch off (offset must be ~0, symmetry check) and on (one
* PDK draw, order-of-magnitude check against the Pelgrom sigma).
* Tokens: /home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice, /home/yanlu/solver/results, mm1, 1 (0 or 1).
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=1
.param vdd=3.3
Vdd vdd 0 {vdd}
Itail tail 0 dc 20u
XM1 n1 inp tail 0 nfet_03v3 w=10u l=1u
XM2 n2 inn tail 0 nfet_03v3 w=10u l=1u
RL1 vdd n1 100k
RL2 vdd n2 100k
XM3 n1 n2 tail 0 nfet_03v3 w=10u l=1u
XM4 n2 n1 tail 0 nfet_03v3 w=10u l=1u
Vcm vcm 0 1.65
* DC probe: slow ramp, and the transient step on the same source
Vid inp vcm dc 0 pwl(0 -0.002 5n -0.002 5.2n 0.002 60n 0.002)
Vinn inn vcm dc 0
.control
op
echo CMPOP $&v(n1) $&v(n2) $&v(tail)
dc Vid -0.05 0.05 0.0005
wrdata /home/yanlu/solver/results/b1_cmp_dc_mm1.dat v(n1) v(n2)
tran 0.05n 60n
wrdata /home/yanlu/solver/results/b1_cmp_tr_mm1.dat v(n1) v(n2)
.endc
.end
