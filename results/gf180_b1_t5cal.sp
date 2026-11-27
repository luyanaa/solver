* two-node link calibration
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
Vdd vdd 0 3.3
Vg ng 0 3.3
Vin n1 0 dc 1.65 ac 1
XL n1 ng n2 0 nfet_03v3 w=10u l=1u
XC1 n1 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
XC2 n2 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
.control
op
echo OP $&v(n1) $&v(n2)
ac dec 40 1e2 1e9
wrdata /home/yanlu/solver/results/b1_t5cal.dat v(n2)
.endc
.end
