* B2 primitives: HBT diff/mirror, LV MOS gm-cell x2, transmission gate
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt_mismatch
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib" hbt_typ
* --- HBT diff pair: 20uA/side, 10k loads (the interface gm-stage)
Vdd vdd 0 1.5
Itail tail 0 dc 40u
XQ3 dc1 db tail 0 npn13G2 mm_ok=1
XQ4 dc2 dcm tail 0 npn13G2 mm_ok=1
Rc1 vdd dc1 10k
Rc2 vdd dc2 10k
Vcm dcm 0 0.8
Vd db dcm dc 0 ac 1
* --- HBT current mirror: 20uA, out at 0.9V
Iref vdd cmin dc 20u
XQ1 cmin cmin 0 0 npn13G2 mm_ok=1
XQ2 cmout cmin 0 0 npn13G2 mm_ok=1
Vh cmout 0 dc 0.9
* --- LV MOS gm-cell A (noise): 10u/0.13u, 10uA current-source load
Vg g 0 dc 0.39 ac 1
XMn da g 0 0 sg13_lv_nmos w=10u l=0.13u mm_ok=1
Ila vdd da 0 dc 10u
* --- LV MOS gm-cell B (pole): Rl=65k~ro load + 1pF, Vgs=0.3837 = the
*     10uA diode point (saturation; verified vs isolated diode sweep)
Vg2 g2 0 dc 0.3837 ac 1
XMn2 d2b g2 0 0 sg13_lv_nmos w=10u l=0.13u mm_ok=1
Rl2 vdd d2b 65k
C1 d2b 0 1p
* --- transmission gate: 2u nmos // 4u pmos, 10uA probe
Vtg a 0 dc 0.75 ac 1
XMtn b gtn a 0 sg13_lv_nmos w=2u l=0.13u mm_ok=1
XMtp b gtp a 0 sg13_lv_pmos w=4u l=0.13u mm_ok=1
Vgn gtn 0 1.5
Vgp gtp 0 0
Itg b 0 dc 10u
.control
set seed = 1234567
op
let ratio = -i(vh)/20u
let gmos = @n.xmn.nsg13_lv_nmos[gm]
let gmos2 = @n.xmn2.nsg13_lv_nmos[gm]
let vd0 = v(dc1)-v(dc2)
let vdc1 = v(dc1)
let vb = v(b)
let idb = (1.5-v(d2b))/65k
echo OPROW $&ratio $&gmos $&gmos2 $&vd0 $&vdc1 $&vb $&idb
let i = 0
dowhile i < 48
  let seed = i + 1
  setseed $&seed
  mc_source
  op
  let ratio = -i(vh)/20u
  let vd0 = v(dc1)-v(dc2)
  let vdc1 = v(dc1)
  let vb = v(b)
  echo MCOP $&ratio $&vd0 $&vdc1 $&vb
  dc Vd -0.005 0.005 0.001
  let gain = (v(dc1)[10]-v(dc1)[0])/0.01
  echo MCGN $&gain
  dc Vg2 0.3737 0.3937 0.01
  let i0 = (1.5-v(d2b)[0])/65k
  let i1 = (1.5-v(d2b)[1])/65k
  let i2 = (1.5-v(d2b)[2])/65k
  let gmc = (i2-i0)/0.02
  echo MCGM $&gmc $&i1
  let i = i + 1
end
.endc
.end
