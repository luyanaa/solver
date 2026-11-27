* T7 route tester -- gf180mcu batch-1 characterization vehicle
* Two metal-3 wires (rm3, the PDK's own metal resistor model) measured four-wire
* for sheet resistance, then one of them used as the route between a gm-cell
* driver and a 1 pF MIM load to see whether the wire, not the buffer, sets the
* pole at batch-1 lengths.
*
* Wire geometry: W = 2 um. 1000 um length = 500 squares, 100 um = 50 squares.
* rsh_rm3 = 0.09 ohm/sq in res_typical (see DESIGN 6.27 for the magic-tech
* conflict: magic's tech file prints 10 for metal3).
* Tokens: {NGS} = PDK ngspice dir, {OUT} = results dir.
.lib "{NGS}/sm141064.ngspice" typical
.lib "{NGS}/sm141064.ngspice" mimcap_typical
.lib "{NGS}/sm141064.ngspice" res_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
.param vdd=3.3
Vdd vdd 0 {vdd}
* --- four-wire sheet-resistance probes (0.1 V forced, virtual-ground ammeter)
Va1 a1 0 dc 0.1
XW1 a1 b1 rm3 r_length=1000u r_width=2u
Vm1 b1 0 dc 0
Va2 a2 0 dc 0.1
XW2 a2 b2 rm3 r_length=100u r_width=2u
Vm2 b2 0 dc 0
* --- driver: gm cell (10 uA, W/L=10u/1u) + ideal current sink, wire + 1 pF
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
Eb gb 0 cdr 0 1
Ila vdd w dc 10u
XMi w gin 0 0 nfet_03v3 w=10u l=1u
Vgac gb gin 0 dc 0 ac 1
XW3 w z rm3 r_length=1000u r_width=2u
XCz z 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
* --- second path: same driver cell, but the 1 mm wire is a 10-section
*     distributed pi-RC ladder. The PDK's ngspice kit models rm3 as a SERIES
*     resistor only -- there is no wire-capacitance model anywhere in the kit --
*     so each shunt C here is the magic-tech nominal for metal3
*     (10.094 aF/um^2 x 2000 um^2 + 30.021 aF/um x 4004 um = 80.4 fF),
*     split 4.02 / 8.04 ... 8.04 / 4.02 so the total is exact and the two end
*     caps are half sections. R total = 45.3 ohm in 10 pieces of 4.53 ohm.
Ila2 vdd wb dc 10u
XMib wb gin 0 0 nfet_03v3 w=10u l=1u
Rw0 wb s1 4.53
Rw1 s1 s2 4.53
Rw2 s2 s3 4.53
Rw3 s3 s4 4.53
Rw4 s4 s5 4.53
Rw5 s5 s6 4.53
Rw6 s6 s7 4.53
Rw7 s7 s8 4.53
Rw8 s8 s9 4.53
Rw9 s9 zb 4.53
CwE wb 0 4.02f
Cw1 s1 0 8.04f
Cw2 s2 0 8.04f
Cw3 s3 0 8.04f
Cw4 s4 0 8.04f
Cw5 s5 0 8.04f
Cw6 s6 0 8.04f
Cw7 s7 0 8.04f
Cw8 s8 0 8.04f
Cw9 s9 0 8.04f
CwL zb 0 4.02f
XCzb zb 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
.control
op
let r1 = 0.1/i(vm1)
let r2 = 0.1/i(vm2)
echo T7OP $&r1 $&r2 $&v(w) $&v(z) $&v(wb) $&v(zb)
ac dec 40 1 1e8
wrdata {OUT}/b1_t7_ac.dat v(z) v(w) v(zb) v(wb)
.endc
.end
