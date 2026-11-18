* absolute Vgs per reset (shared-draw check)
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt_mismatch
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib" hbt_typ
I1 vdd m1 0 dc 10u
XM1 m1 m1 0 0 sg13_lv_nmos w=10u l=0.13u mm_ok=1
I2 vdd m2 0 dc 10u
XM2 m2 m2 0 0 sg13_lv_nmos w=10u l=0.13u mm_ok=1
.control
let k = 0
dowhile k < 6
  let seed = k + 1
  setseed $&seed
  mc_source
  op
  let v1 = v(m1)
  let v2 = v(m2)
  echo V1 $&v1 $&v2
  let k = k+1
end
.endc
.end
