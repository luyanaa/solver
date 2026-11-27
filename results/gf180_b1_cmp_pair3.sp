* T10 regenerative comparator core -- gf180mcu batch-1 characterization vehicle
* NMOS input pair (W/L=10u/1u), cross-coupled NMOS regeneration pair, 100 k
* loads, 20 uA ideal tail. Unclocked core; the clocked shell is a B3 wrap.
*
* Two variants selected by the runner through the REGEN_BLOCK line:
*   REGEN_OFF  gates tied to 0 -> a plain diff pair: the DC sweep gives the
*              input offset (the Pelgrom number, mismatch on/off)
*   REGEN_ON   cross-coupled  -> a latch: the DC sweep is meaningless
*              (loop gain 13 gives a hysteresis wider than +-50 mV), so only
*              the transient is used, fitted for the regeneration time
*              constant in the linear window.
* Tokens: NGS, OUT, TAG, MM, SEED, and the REGEN_BLOCK line.
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=1
.option seed=910003
.param vdd=3.3
Vdd vdd 0 3.3
Itail tail 0 dc 20u
XM1 n1 inp tail 0 nfet_03v3 w=10u l=1u
XM2 n2 inn tail 0 nfet_03v3 w=10u l=1u
RL1 vdd n1 100k
RL2 vdd n2 100k
Vgz gz 0 dc 0
XM3 n1 gz tail 0 nfet_03v3 w=10u l=1u
XM4 n2 gz tail 0 nfet_03v3 w=10u l=1u
Vcm vcm 0 1.65
* sweep and step are separate sources, summed into inp (a PWL source cannot
* also be swept: the sweep silently follows the waveform). The step's DC
* value is its t=0 point (-2 mV), so offset = trip(Vidn) + 2 mV. Two E
* sources in a chain, never in parallel on one node.
Vidn vidn 0 dc 0
Vstep stp 0 dc 0 pwl(0 -0.002 5n -0.002 5.2n 0.002 60n 0.002)
Emid mid vcm vidn 0 1
Eid inp mid stp 0 1
Vinn inn vcm dc 0
.control
op
echo CMPOP $&v(n1) $&v(n2) $&v(tail)
dc Vidn -0.02 0.02 0.00002
wrdata /home/yanlu/solver/results/b1_cmp_dc_pair3.dat v(n1) v(n2)
tran 0.02n 20n
wrdata /home/yanlu/solver/results/b1_cmp_tr_pair3.dat v(n1) v(n2)
.endc
.end
