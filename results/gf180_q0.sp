* gf180 pre-PEX parasitic probe: added C = 0 fF at d2b
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
Vdd vdd 0 3.3
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
* bias follow: gb tracks the diode, gb2 = gb - Vgb
Eb gb 0 cdr 0 1
Vgb gb gb2 0 dc 0 ac 1
XMn2 d2b gb2 0 0 nfet_03v3 w=10u l=1u
Ila2 vdd d2b dc 10u
C1 d2b 0 1p

.control
ac dec 40 0.1 1e8
wrdata /home/yanlu/solver/results/gf_ac_q0.dat v(d2b)
.endc
.end
