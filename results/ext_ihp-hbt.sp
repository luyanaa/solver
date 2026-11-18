* g_m interface extraction: ihp-hbt
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib" hbt_typ
Iin 0 d1 dc 20u
XM1 d1 d1 0 0 npn13G2 
XM2 d2 d1 0 0 npn13G2 
Vout d2 0 dc 0.900
.control
op
let io = -i(vout)
let vgs = v(d1)
print io vgs
dc Iin 1u 60u 1u
wrdata /grid/solver/results/isweep_ihp-hbt.dat i(vout) v(d1)
alter Iin dc = 20u
dc Vout 0.30 1.60 0.01
wrdata /grid/solver/results/vsweep_ihp-hbt.dat i(vout)
.endc
.end
