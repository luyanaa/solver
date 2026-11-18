* g_m interface extraction: gf180mcu
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
Iin 0 d1 dc 10u
MM1 d1 d1 0 0 nfet_03v3 w=10u l=1u
MM2 d2 d1 0 0 nfet_03v3 w=10u l=1u
Vout d2 0 dc 1.650
.control
op
let io = -i(vout)
let vgs = v(d1)
print io vgs
dc Iin 1u 40u 1u
wrdata /grid/solver/results/isweep_gf180mcu.dat i(vout) v(d1)
alter Iin dc = 10u
dc Vout 0.20 2.40 0.01
wrdata /grid/solver/results/vsweep_gf180mcu.dat i(vout)
.endc
.end
