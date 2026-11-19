* B2 transient: step + sine on HBT diff pair and MOS gm-cell
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt
.lib "/home/yanlu/.ciel/ciel/ihp-sg13/versions/89d42ee6d2fd9b50b86a1aada6fee6673c3e6008/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib" hbt_typ
* --- step copy (suffix _s): HBT diff pair, 10 mV input step
Vdd_s vdd_s 0 1.5
Itail_s tail_s 0 dc 40u
XQ3s dc1_s db_s tail_s 0 npn13G2
XQ4s dc2_s dcm_s tail_s 0 npn13G2
Rc1_s vdd_s dc1_s 10k
Rc2_s vdd_s dc2_s 10k
Vcm_s dcm_s 0 0.8
Vd_s db_s dcm_s PULSE(0 0.01 2e-08 0.5n 0.5n 3.8e-07 8e-07)
* --- step copy: MOS gm-cell B, 1 mV input step, 1 pF load
Vg2_s g2_s 0 PULSE(0.3837 0.3847 2e-08 0.5n 0.5n 3.8e-07 8e-07)
XMn2_s d2b_s g2_s 0 0 sg13_lv_nmos w=10u l=0.13u
Rl2_s vdd_s d2b_s 65k
C1_s d2b_s 0 1p
* --- sine copy (suffix _n): HBT diff pair, 10 mV @ 10 MHz
Vdd_n vdd_n 0 1.5
Itail_n tail_n 0 dc 40u
XQ3n dc1_n db_n tail_n 0 npn13G2
XQ4n dc2_n dcm_n tail_n 0 npn13G2
Rc1_n vdd_n dc1_n 10k
Rc2_n vdd_n dc2_n 10k
Vcm_n dcm_n 0 0.8
Vd_n db_n dcm_n SIN(0 0.01 1e+07)
* --- sine copy: MOS gm-cell B, 1 mV @ 10 MHz
Vg2_n g2_n 0 SIN(0.3837 0.001 1e+07)
XMn2_n d2b_n g2_n 0 0 sg13_lv_nmos w=10u l=0.13u
Rl2_n vdd_n d2b_n 65k
C1_n d2b_n 0 1p
.control
tran 0.2n 400n 0 1n
linearize
wrdata /grid/solver/results/b2_tran_step.dat v(dc1_s) v(dc2_s) v(d2b_s)
tran 0.2n 1000n 0 1n
linearize
wrdata /grid/solver/results/b2_tran_sine.dat v(dc1_n) v(dc2_n) v(d2b_n)
.endc
.end
