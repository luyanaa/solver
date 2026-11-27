* gf180 mismatch MC 6v0_s32 (seed 910032)
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=1
.option seed=910032
Vdd vdd 0 6
* --- diff pair (offset)
Itail tail 0 dc 20u
XQ1 dc1 db tail 0 nfet_06v0 w=10u l=1u
XQ2 dc2 dcm tail 0 nfet_06v0 w=10u l=1u
Rc1 vdd dc1 200k
Rc2 vdd dc2 200k
Vcm dcm 0 3
Vd db dcm dc 0 ac 1
* --- shared-reference mirror 1:1 + 1:4 (ratio mismatch)
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_06v0 w=10u l=1u
XM1 cmout cdr 0 0 nfet_06v0 w=10u l=1u
Eh cmout 0 cdr 0 1
XM4 cm4 cdr 0 0 nfet_06v0 w=10u l=1u m=4
Eh4 cm4 0 cdr 0 1
* --- gm cell (resistive load, Vgb sweep for gm_fd)
Vgb gb cdr dc 0
XMn2 d2b gb 0 0 nfet_06v0 w=10u l=1u
Rl2 vdd d2b 200k
* --- TG
Vtg a 0 dc 3 ac 1
XMtn b gtn a 0 nfet_06v0 w=2u l=0.7u
XMtp b gtp a vdd pfet_06v0 w=4u l=0.7u
Vgn gtn 0 6
Vgp gtp 0 0
Itg b 0 dc 2u
.control
op
let vd0 = v(dc1)-v(dc2)
let ratio = -i(eh)/10u
let ratio4 = -i(eh4)/40u
let vdiode = v(cdr)
let vb = v(b)
echo MCOP $&vd0 $&ratio $&ratio4 $&vdiode $&vb
dc Vd -0.005 0.005 0.001
let gain = (v(dc1)[10]-v(dc1)[0])/0.01
echo MCGN $&gain
dc Vgb -0.05 0.05 0.01
let i0 = (6-v(d2b)[0])/200k
let i1 = (6-v(d2b)[5])/200k
let i2 = (6-v(d2b)[10])/200k
let gmc = abs((i2-i0)/0.1)
let idop = i1
echo MCGM $&gmc $&idop
.endc
.end
