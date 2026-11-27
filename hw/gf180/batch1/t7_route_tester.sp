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
.control
op
let r1 = 0.1/i(vm1)
let r2 = 0.1/i(vm2)
echo T7OP $&r1 $&r2 $&v(w) $&v(z)
ac dec 40 1 1e8
wrdata {OUT}/b1_t7_ac.dat v(z) v(w)
.endc
.end
