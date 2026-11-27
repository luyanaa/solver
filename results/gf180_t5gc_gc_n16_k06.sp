* T5 ladder, G/C extraction, inject node 6
* N=16 nodes, node 1 held by an ideal source, pass devices 10u/1u
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
Vdd vdd 0 3.3
Vg ng 0 3.3
Vb n1 0 dc 1.65 ac 0
Iinj n6 0 dc 0 ac 1e-06
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
ac dec 1 1k 1G
wrdata /home/yanlu/solver/results/b1_gc_n16_k06.dat v(n2) v(n3) v(n4) v(n5) v(n6) v(n7) v(n8) v(n9) v(n10) v(n11) v(n12) v(n13) v(n14) v(n15) v(n16)
.endc
.end