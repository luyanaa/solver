* gf180 corner probe smoke (3v3, typical)
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
* --- NMOS diode reference (replica bias) + 1:1 mirror + 1:4 / 1:16
Vdd vdd 0 3.3
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
XM1 cmout cdr 0 0 nfet_03v3 w=10u l=1u
Vh cmout 0 dc 1.65
* a second 1:1 copy with Vds forced to V(cdr): the shared-bias intent
XM1s cmouts cdr 0 0 nfet_03v3 w=10u l=1u
Es cmouts 0 cdr 0 1
* arrays: a VCVS holds Vds = V(cdr) -- the shared-reference intent
XM4 cm4 cdr 0 0 nfet_03v3 w=10u l=1u m=4
Eh4 cm4 0 cdr 0 1
XM16 cm16 cdr 0 0 nfet_03v3 w=10u l=1u m=16
Eh16 cm16 0 cdr 0 1
* --- gm probe: resistive load + DC gate sweep (gm = dId/dVg)
Vgg ggm cdr dc 0
XMgg dgg ggm 0 0 nfet_03v3 w=10u l=1u
Rlg vdd dgg 100k
* --- diode-node buffers (VCVS, zero output Z) so the series AC
*     source really applies 1 V across Vgs instead of dividing
*     against the diode's own 1/gm (measured 13 vs 340 on pass 1)
Eb gb 0 cdr 0 1
Ebp gbp 0 cdp 0 1
* --- noise cell (nMOS): resistive load + series AC noise reference
Vgac gb gn1 0 dc 0 ac 1
XMn da gn1 0 0 nfet_03v3 w=10u l=1u
Rla vdd da 100k
* --- pole cell: ideal current-source load + 1 pF C_bank -> ro
Vgb gb gb2 0 dc 0 ac 1
XMn2 d2b gb2 0 0 nfet_03v3 w=10u l=1u
Ila2 vdd d2b dc 10u
C1 d2b 0 1p
* --- pFET diode + noise cell (p-bulk on vdd = the n-well)
Irefp cdp 0 dc 10u
XP0 cdp cdp vdd vdd pfet_03v3 w=10u l=1u
Vgacp gbp gp1 0 dc 0 ac 1
XPna dap gp1 vdd vdd pfet_03v3 w=10u l=1u
Rlap 0 dap 100k
* --- transmission gate 2u n // 4u p @ 0.5u, 2 uA so both legs are
*     in triode (n-bulk=0, p-bulk=vdd is how the TG is really built)
Vtg a 0 dc 1.65 ac 1
XMtn b gtn a 0 nfet_03v3 w=2u l=0.5u
XMtp b gtp a vdd pfet_03v3 w=4u l=0.5u
Vgn gtn 0 3.3
Vgp gtp 0 0
Itg b 0 dc 2u
* --- MIM cap probe: 22.36u x 22.36u = 500 um2 @ 2 fF/um2 = 1 pF nominal
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuA/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
Vcb cbn 0 dc 1.5 ac 1
Xcb cbn 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
.control
op
* 1:1 mirror with the output Vds held at vhold (lambda*DVds error)
let ratio_mis = -i(vh)/10u
let ratio_shared = -i(es)/10u
let ratio4 = -i(eh4)/10u
let ratio16 = -i(eh16)/10u
let vdiode = v(cdr)
let vdiodep = v(cdp)
let ida = (3.3-v(da))/100k
let idap = v(dap)/100k
let vd2b = v(d2b)
let vb = v(b)
echo OPROW $&ratio_mis $&ratio4 $&ratio16 $&ratio_shared $&vdiode $&vdiodep $&ida $&idap $&vd2b $&vb
dc Vh 0.2 3 0.01
wrdata /home/yanlu/solver/results/gf_mirror_smoke.dat i(vh)
dc Vgg -0.03 0.03 0.002
wrdata /home/yanlu/solver/results/gf_gm_smoke.dat v(dgg)
ac dec 10 0.1 1e8
wrdata /home/yanlu/solver/results/gf_ac_smoke.dat v(d2b)
ac lin 2 1e+06 1e+06
wrdata /home/yanlu/solver/results/gf_cap_smoke.dat i(vcb)
* ngspice-44 leaves the CURRENT plot on the INTEGRATED noise plot
* after each analysis (spectral densities live in the sibling plot
* and their plot number is not stable across analysis mixes), so
* the band-integrated onoise_total/inoise_total are echoed directly
* -- S_v = onoise_total^2/(f1-f0) over the flat band.
noise v(da) Vgac dec 20 1e6 1e8
let n_o = onoise_total
let n_i = inoise_total
echo NROW nmos $&n_o $&n_i
* decade-narrow band: residual 1/f shows up as a higher density
noise v(da) Vgac dec 20 1e6 1e7
let n1_o = onoise_total
let n1_i = inoise_total
echo NROW nmos_low $&n1_o $&n1_i
noise v(b) Vtg dec 20 1e6 1e9
let t_o = onoise_total
let t_i = inoise_total
echo NROW tg $&t_o $&t_i
noise v(dap) Vgacp dec 20 1e6 1e8
let p_o = onoise_total
let p_i = inoise_total
echo NROW pmos $&p_o $&p_i
noise v(dap) Vgacp dec 20 1e6 1e7
let p1_o = onoise_total
let p1_i = inoise_total
echo NROW pmos_low $&p1_o $&p1_i
.endc
.end
