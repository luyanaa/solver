* gf180 mismatch seed probe
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=1
Vdd vdd 0 3.3
Itail tail 0 dc 20u
XQ1 dc1 db tail 0 nfet_03v3 w=10u l=1u
XQ2 dc2 dcm tail 0 nfet_03v3 w=10u l=1u
Rc1 vdd dc1 100k
Rc2 vdd dc2 100k
Vcm dcm 0 1.2
Vd db dcm dc 0 ac 1
.option seed=11
.control
op
let vd0 = v(dc1)-v(dc2)
echo OPOFF $&vd0
.endc
.end
