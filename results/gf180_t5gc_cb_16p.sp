* T5 ladder, step 0.01 V
* N=16 nodes, node 1 held by an ideal source, pass devices 10u/1u
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
Vdd vdd 0 3.3
Vg ng 0 3.3
Vin n1 0 dc 1.65 pwl(0 1.65 20n 1.65 25n 1.66)
XC1 n1 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC2 n2 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC3 n3 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC4 n4 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC5 n5 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC6 n6 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC7 n7 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC8 n8 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC9 n9 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC10 n10 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC11 n11 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC12 n12 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC13 n13 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC14 n14 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC15 n15 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
XC16 n16 0 cap_mim_2f0_m4m5_noshield c_length=89.44u c_width=89.44u
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
tran 1e-08 4e-05
wrdata /home/yanlu/solver/results/b1_cb_16p.dat v(n16) i(vin)
.endc
.end