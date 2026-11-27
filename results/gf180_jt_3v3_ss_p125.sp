* gf180 PVT jt_3v3_ss_p125 (3v3, ss, 125 C, 3.3 V)
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" ss
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
.temp 125
Vdd vdd 0 3.3
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
XM1 cmout cdr 0 0 nfet_03v3 w=10u l=1u
* INDEPENDENT mirror: drain held at vhold (lambda*dVds error shows)
Vh cmout 0 dc 1.65
* gb follows the diode: without this the AC cell is unbiassed
Eb gb 0 cdr 0 1
Vgb gb gb2 0 dc 0 ac 1
XMn2 d2b gb2 0 0 nfet_03v3 w=10u l=1u
Ila2 vdd d2b dc 10u
C1 d2b 0 1p
Vgac cdr gn1 0 dc 0 ac 1
XMn da gn1 0 0 nfet_03v3 w=10u l=1u
Rla vdd da 100k
* gm probe: resistive load + DC gate sweep, the corner netlist's cell
Vgg ggm cdr dc 0
XMgg dgg ggm 0 0 nfet_03v3 w=10u l=1u
Rlg vdd dgg 100k
.control
op
let ratio = -i(vh)/10u
let vdiode = v(cdr)
let ida = (3.3-v(da))/100k
let vd2b = v(d2b)
echo OPROW $&ratio $&vdiode $&ida $&vd2b
dc Vgb -0.08 0.08 0.002
wrdata /home/yanlu/solver/results/gf_gmsweep_jt_3v3_ss_p125.dat v(d2b)
dc Vgg -0.03 0.03 0.002
wrdata /home/yanlu/solver/results/gf_pvtgm_jt_3v3_ss_p125.dat v(dgg)
ac dec 40 0.1 1e8
wrdata /home/yanlu/solver/results/gf_ac_jt_3v3_ss_p125.dat v(d2b)
.endc
.end
