* sky130 nmos current mirror
.lib "/home/yanlu/.ciel/ciel/sky130/versions/792ef67037d74c8c51811eab3ce23cf793491296/sky130A/libs.tech/ngspice/sky130.lib.spice" tt
Iin 0 d1 10u
XM1 d1 d1 0 0 sky130_fd_pr__nfet_01v8 W=10 L=1
XM2 d2 d1 0 0 sky130_fd_pr__nfet_01v8 W=10 L=1
Vout d2 0 0.5
.op
.control
run
print @m.xm2.msky130_fd_pr__nfet_01v8[gm] @m.xm2.msky130_fd_pr__nfet_01v8[gds] @m.xm2.msky130_fd_pr__nfet_01v8[vth] @m.xm2.msky130_fd_pr__nfet_01v8[id]
print i(Vout)
.endc
.end
