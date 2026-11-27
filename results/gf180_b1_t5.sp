* T5 1D flux chain -- gf180mcu batch-1 characterization vehicle
* 16 state nodes, each a 1 pF MIM C_bank, joined by 15 pass-transistor flux
* elements (nfet_03v3 W/L=10u/1u, gate at vdd = always on, bulk at 0).
* The pass device is one physical current shared by both neighbours, so the
* element is conservative by construction (the single-flux-source contract of
* twin/coupling.py, realised with a triode channel instead of a gm mirror).
*
* Measured: DC link resistance R_link (triode), the ladder's dominant pole
* f_3dB of v(n16)/v(n1), and the DC gain (must be 1: a resistive ladder).
* Tokens: /home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice = PDK ngspice dir, /home/yanlu/solver/results = results dir.
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
.param vdd=3.3
Vdd vdd 0 3.3
* gate rail: the flux devices are always on
Vg ng 0 3.3
* --- link resistance probe: same device, 0.2 V across, ammeter from Va
Va a 0 1.65
Vb b 0 1.45
XLp a ng b 0 nfet_03v3 w=10u l=1u
* --- element calibration: two nodes, one link, two 1 pF MIM caps. Its single
*     pole gives the small-signal link conductance at vds ~ 0, which is the
*     number the ladder's mode-1 eigenvalue needs; the DC probe above is a
*     CHORD conductance at 0.2 V and reads ~40 % lower (triode i = vov*vds - vds^2/2).
Vcal c1 0 dc 1.65 ac 1
XLC c1 ng c2 0 nfet_03v3 w=10u l=1u
XCC1 c1 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XCC2 c2 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
* --- the chain, driven by a stiff source at node 1 (DC gain must be 1)
* Vin drives the chain: AC 1 V for the frequency response, and a 0.1 V
* step at t = 1 us whose late tail is the mode-1 eigenvalue
Vin n1 0 dc 1.65 ac 1 pwl(0 1.65 10n 1.65 10.2n 1.75)
XC1 n1 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC2 n2 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC3 n3 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC4 n4 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC5 n5 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC6 n6 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC7 n7 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC8 n8 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC9 n9 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC10 n10 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC11 n11 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC12 n12 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC13 n13 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC14 n14 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC15 n15 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC16 n16 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XL1 n1 ng n2 0 nfet_03v3 w=10u l=1u
XL2 n2 ng n3 0 nfet_03v3 w=10u l=1u
XL3 n3 ng n4 0 nfet_03v3 w=10u l=1u
XL4 n4 ng n5 0 nfet_03v3 w=10u l=1u
XL5 n5 ng n6 0 nfet_03v3 w=10u l=1u
XL6 n6 ng n7 0 nfet_03v3 w=10u l=1u
XL7 n7 ng n8 0 nfet_03v3 w=10u l=1u
XL8 n8 ng n9 0 nfet_03v3 w=10u l=1u
XL9 n9 ng n10 0 nfet_03v3 w=10u l=1u
XL10 n10 ng n11 0 nfet_03v3 w=10u l=1u
XL11 n11 ng n12 0 nfet_03v3 w=10u l=1u
XL12 n12 ng n13 0 nfet_03v3 w=10u l=1u
XL13 n13 ng n14 0 nfet_03v3 w=10u l=1u
XL14 n14 ng n15 0 nfet_03v3 w=10u l=1u
XL15 n15 ng n16 0 nfet_03v3 w=10u l=1u
.control
op
let rl = 0.2/-i(va)
echo T5OP $&v(n1) $&v(n8) $&v(n16) $&rl
ac dec 40 1 1e7
wrdata /home/yanlu/solver/results/b1_t5_ac.dat v(n16)
ac dec 20 1e2 1e9
wrdata /home/yanlu/solver/results/b1_t5_cal_ac.dat v(c2)
tran 0.2n 300n
wrdata /home/yanlu/solver/results/b1_t5_tr.dat v(n16)
.endc
.end
