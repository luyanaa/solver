* g_m interface extraction: sky130
.lib "/home/yanlu/.ciel/ciel/sky130/versions/792ef67037d74c8c51811eab3ce23cf793491296/sky130A/libs.tech/ngspice/sky130.lib.spice" tt
Iin 0 d1 dc 10u
XM1 d1 d1 0 0 sky130_fd_pr__nfet_01v8 w=10 l=1
XM2 d2 d1 0 0 sky130_fd_pr__nfet_01v8 w=10 l=1
Vout d2 0 dc 0.900
.control
op
let io = -i(vout)
let vgs = v(d1)
print io vgs
dc Iin 1u 40u 1u
wrdata /grid/solver/results/isweep_sky130.dat i(vout) v(d1)
alter Iin dc = 10u
dc Vout 0.15 1.20 0.01
wrdata /grid/solver/results/vsweep_sky130.dat i(vout)
.endc
.end
