* T1 integrator tile -- gf180mcu batch-1 characterization vehicle
* One integration cell: NMOS gm cell (10 uA, W/L=10u/1u) + one 1 pF MIM C_bank,
* biased from a replica diode, integrated node loaded by an ideal current sink
* (the P7 coefficient graph's a_ii = -gm/C; a0 = gm*ro at the diode bias,
* the same bias as the corner probe's pole cell so the numbers chain).
*
* Measured by bench/run_gf180_batch1.py: DC gain a0, -3 dB pole, vds.
* Tokens: {NGS} = PDK ngspice dir, {OUT} = results dir. Filled by the runner.
.lib "{NGS}/sm141064.ngspice" typical
.lib "{NGS}/sm141064.ngspice" mimcap_typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
.param vdd=3.3
Vdd vdd 0 {vdd}
* --- replica bias: diode NMOS at 10 uA, buffered to a stiff gate rail
Iref vdd cdr dc 10u
XM0 cdr cdr 0 0 nfet_03v3 w=10u l=1u
Eb gb 0 cdr 0 1
* --- integrating node
Ila vdd out dc 10u
XMi out gin 0 0 nfet_03v3 w=10u l=1u
Vgac gb gin 0 dc 0 ac 1
* --- C_bank: the real MIM device, 22.36u x 22.36u = 500 um2 (2 fF/um2)
XC out 0 cap_mim_2f0_m4m5_noshield c_length=22.36u c_width=22.36u
.control
op
echo T1OP $&v(out) $&v(gin) $&-i(ila)
ac dec 40 1 1e8
wrdata {OUT}/b1_t1_ac.dat v(out)
.endc
.end
