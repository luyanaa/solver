* T9 5-bit binary current DAC -- gf180mcu batch-1 characterization vehicle
* Unit mirror leg = nfet_03v3 W/L=10u/1u with the replica gate rail; bit weights
* 1:2:4:8:16 by m=; gate switching with an ideal S element (Ron=1) so the leg
* vds is set by the summing node, not by the switch. Summing load 1 kOhm, so
* the output moves 0.33 V over the full code and lambda*DVds is live (that is
* the INL mechanism the macromodel predicts from the extractor's lambda).
*
* The runner substitutes 1..0 (0 or 1) and/or {CODE}; one process per code
* because a single ngspice DC sweep cannot drive 5 independent bit sources.
* Tokens: /home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice = PDK ngspice dir, /home/yanlu/solver/results = results dir, 1 = code label.
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
.param vdd=3.3
Vdd vdd 0 3.3
.model SWM SW(Ron=1 Roff=1e12 Vt=0.5 Vh=0.1)
* --- replica bias
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
* --- bit sources and switches (gate pulled to 0 when the switch is off)
Vb0 b0 0 dc 1
Vb1 b1 0 dc 0
Vb2 b2 0 dc 0
Vb3 b3 0 dc 0
Vb4 b4 0 dc 0
S0 cdr g0 b0 0 SWM
S1 cdr g1 b1 0 SWM
S2 cdr g2 b2 0 SWM
S3 cdr g3 b3 0 SWM
S4 cdr g4 b4 0 SWM
Rg0 g0 0 1e9
Rg1 g1 0 1e9
Rg2 g2 0 1e9
Rg3 g3 0 1e9
Rg4 g4 0 1e9
* --- binary-weighted mirror legs
XMD0 dac g0 0 0 nfet_03v3 w=10u l=1u m=1
XMD1 dac g1 0 0 nfet_03v3 w=10u l=1u m=2
XMD2 dac g2 0 0 nfet_03v3 w=10u l=1u m=4
XMD3 dac g3 0 0 nfet_03v3 w=10u l=1u m=8
XMD4 dac g4 0 0 nfet_03v3 w=10u l=1u m=16
Rdac vdd dac 1k
.control
op
let idac = (3.3-v(dac))/1e3
echo DCODE 1 $&v(dac) $&idac
.endc
.end
