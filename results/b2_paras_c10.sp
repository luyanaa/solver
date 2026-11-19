* B2 pre-PEX parasitic probe: added C = 10 fF at d2b
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib" hbt_typ
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
Cpar d2b 0 10f
Vtg a 0 dc 0.75 ac 1
XMtn b gtn a 0 sg13_lv_nmos w=2u l=0.13u
XMtp b gtp a 0 sg13_lv_pmos w=4u l=0.13u
Vgn gtn 0 1.5
Vgp gtp 0 0
Itg b 0 dc 10u
.control
op
ac lin 6001 1e5 2e7
wrdata /grid/solver/results/b2_ac_paras_c10.dat v(d2b)
.endc
.end
