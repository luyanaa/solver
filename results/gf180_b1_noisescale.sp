* noise scaling probe: m=8 vs w=80u vs 8 parallel instances
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
Vdd vdd 0 3.3
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
Eb gb 0 cdr 0 1
* (i) m=8
Vga gb ga 0 dc 0 ac 1
XMa da ga 0 0 nfet_03v3 w=10u l=1u m=8
Rla vdd da 10k
* (ii) w=80u
Vgb2 gb g2 0 dc 0 ac 1
XMb db g2 0 0 nfet_03v3 w=80u l=1u
Rlb vdd db 10k
* (iii) eight explicit instances
Vgc gb g3 0 dc 0 ac 1
XM1 dc1 g3 0 0 nfet_03v3 w=10u l=1u
XM2 dc2 g3 0 0 nfet_03v3 w=10u l=1u
XM3 dc3 g3 0 0 nfet_03v3 w=10u l=1u
XM4 dc4 g3 0 0 nfet_03v3 w=10u l=1u
XM5 dc5 g3 0 0 nfet_03v3 w=10u l=1u
XM6 dc6 g3 0 0 nfet_03v3 w=10u l=1u
XM7 dc7 g3 0 0 nfet_03v3 w=10u l=1u
XM8 dc8 g3 0 0 nfet_03v3 w=10u l=1u
Rl1 vdd dc1 80k
Rl2 vdd dc2 80k
Rl3 vdd dc3 80k
Rl4 vdd dc4 80k
Rl5 vdd dc5 80k
Rl6 vdd dc6 80k
Rl7 vdd dc7 80k
Rl8 vdd dc8 80k
.control
op
echo OPX $&v(da) $&v(db) $&v(dc1)
noise v(da) Vga dec 20 1e7 1e8
let oa = onoise_total
let ia = inoise_total
echo NROW mm8 $&oa $&ia
noise v(db) Vgb2 dec 20 1e7 1e8
let ob = onoise_total
let ib = inoise_total
echo NROW w80 $&ob $&ib
noise v(dc1) Vgc dec 20 1e7 1e8
let oc = onoise_total
let ic = inoise_total
echo NROW par8 $&oc $&ic
.endc
.end
