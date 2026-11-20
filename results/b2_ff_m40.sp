* B2 PVT probe: temp=-40 C, vdd=1.5 V (mos_ff/hbt_typ)
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib" mos_ff
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib" hbt_typ
.temp -40
Vdd vdd 0 1.5
Itail tail 0 dc 40u
XQ3 dc1 db tail 0 npn13G2
XQ4 dc2 dcm tail 0 npn13G2
Rc1 vdd dc1 10k
Rc2 vdd dc2 10k
Vcm dcm 0 0.8
Vd db dcm dc 0 ac 1
Iref vdd cmin dc 20u
XQ1 cmin cmin 0 0 npn13G2
XQ2 cmout cmin 0 0 npn13G2
Vh cmout 0 dc 0.9
Vg g 0 dc 0.39 ac 1
XMn da g 0 0 sg13_lv_nmos w=10u l=0.13u
Ila vdd da 0 dc 10u
Vg2 g2 0 dc 0.3837 ac 1
XMn2 d2b g2 0 0 sg13_lv_nmos w=10u l=0.13u
Rl2 vdd d2b 65k
C1 d2b 0 1p
Vtg a 0 dc 0.75 ac 1
XMtn b gtn a 0 sg13_lv_nmos w=2u l=0.13u
XMtp b gtp a 0 sg13_lv_pmos w=4u l=0.13u
Vgn gtn 0 1.5
Vgp gtp 0 0
Itg b 0 dc 10u
.control
op
let ratio = -i(vh)/20u
let gmos = @n.xmn.nsg13_lv_nmos[gm]
let gmos2 = @n.xmn2.nsg13_lv_nmos[gm]
let vd0 = v(dc1)-v(dc2)
let vdc1 = v(dc1)
let vb = v(b)
let idb = (1.5-v(d2b))/65k
echo OPROW $&ratio $&gmos $&gmos2 $&vd0 $&vdc1 $&vb $&idb
dc Vg2 0.3 0.7 0.005
wrdata /grid/solver/results/b2_gmsweep_ff_m40.dat v(d2b)
ac dec 20 1e3 1e9
wrdata /grid/solver/results/b2_ac_ff_m40.dat v(d2b)
.endc
.end
