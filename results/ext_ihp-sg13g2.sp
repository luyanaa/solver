* g_m interface extraction: ihp-sg13g2
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt
Iin 0 d1 dc 10u
XM1 d1 d1 0 0 sg13_lv_nmos w=10u l=0.13u
XM2 d2 d1 0 0 sg13_lv_nmos w=10u l=0.13u
Vout d2 0 dc 0.750
.control
op
let io = -i(vout)
let vgs = v(d1)
print io vgs
dc Iin 1u 40u 1u
wrdata /grid/solver/results/isweep_ihp-sg13g2.dat i(vout) v(d1)
alter Iin dc = 10u
dc Vout 0.15 1.00 0.01
wrdata /grid/solver/results/vsweep_ihp-sg13g2.dat i(vout)
.endc
.end
