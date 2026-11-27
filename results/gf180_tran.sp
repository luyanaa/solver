* gf180 transient: fast gm-cell (step+sine THD) + slow pole cell
.lib "/home/yanlu/.ciel/ciel/gf180mcu/versions/792ef67037d74c8c51811eab3ce23cf793491296/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.param fnoicor=0
.param sw_stat_global=0
.param sw_stat_mismatch=0
* --- fast step copy: gm cell A, 1 mV step @ t0=20ns, Rl=100k
Vdd_s vdd_s 0 3.3
Vg_s g_s 0 PULSE(0.765 0.766 2e-08 0.5n 0.5n 3.8e-07 8e-07)
XMn_s da_s g_s 0 0 nfet_03v3 w=10u l=1u
Rla_s vdd_s da_s 100k
* --- fast sine copy (THD): 1 mV @ 10 MHz
Vdd_n vdd_n 0 3.3
Vg_n g_n 0 SIN(0.765 0.001 1e+07)
XMn_n da_n g_n 0 0 nfet_03v3 w=10u l=1u
Rla_n vdd_n da_n 100k
* --- slow copy: pole cell (ideal load + 1 pF), 1 mV gate step
Vdd_q vdd_q 0 3.3
Vg_q g_q 0 PULSE(0.765 0.766 1e-05 1u 1u 9e-05 0.0002)
XMn_q d2b_q g_q 0 0 nfet_03v3 w=10u l=1u
Ila_q vdd_q d2b_q dc 10u
C1_q d2b_q 0 1p
.control
tran 0.2n 400n 0 1n
linearize
wrdata /home/yanlu/solver/results/gf_tran_fast_step.dat v(da_s)
tran 0.2n 1000n 0 1n
linearize
wrdata /home/yanlu/solver/results/gf_tran_fast_sine.dat v(da_n)
tran 0.2u 100u 0 0.2u
linearize
wrdata /home/yanlu/solver/results/gf_tran_slow_step.dat v(d2b_q)
.endc
.end
