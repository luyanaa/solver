// -----------------------------------------------------------------------------
// tb_calib_fsm.v -- protocol testbench for calib_fsm.v (the B3 supervisor).
//
// This is NOT a self-checking testbench in the usual sense: it drives the far
// end of the protocol (a host that stimulates, converts, applies the
// correction and returns a held-out verdict) and prints one machine-readable
// line per scenario.  bench/run_gf180_fsm.py owns the expectations, because
// the same RTL is built with HIST=4 (the chosen history depth) and HIST=1 (the
// rejected variant) and only the python side knows which expectation applies.
//
// Build:  iverilog -DHIST=4 -o sim calib_fsm.v tb_calib_fsm.v
// Run:    vvp sim
//
// Line format:  SCN <name> st=<state> stim=<n> meas=<n> corr=<n> cons=<n>
//                        retry=<n> maxretry=<n> stimc=<n> measc=<n> corrc=<n>
//                        consc=<n> ver=<n> ord=<0|1>
//   <n>    = number of RISING edges of that trigger (pulses)
//   <n>c   = number of cycles that trigger was high (pulses==cycles => 1-cycle)
//   ver    = number of VERIFY entries
//   ord    = the last round's four timestamps are strictly increasing
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_calib_fsm;

    reg clk = 0;
    reg rst_n = 0;
    reg start_req = 0;

    wire [2:0] state;
    wire stim_trig, meas_trig, corr_trig, consume, done, accepted, failed;
    wire [1:0] retry_cnt;

    // ---- host-side scenario state -----------------------------------------
    reg [7:0] dg [0:3];          // digest presented for attempt 1..4
    reg       vo [0:3];          // held-out verdict for attempt 1..4
    reg       mv [0:3];          // a fresh reading arrived for attempt 1..4

    integer att;                 // attempts consumed so far (= measurements)
    integer mwait;
    integer stim_n, meas_n, corr_n, cons_n, ver_n;
    integer stimc, measc, corrc, consc;
    integer cyc;
    integer ts, tm, tc, tv, t0, tdone;
    reg     p_done;
    integer p_stim, p_meas, p_corr, p_cons;
    reg [2:0] pstate;

    assign meas_valid = mv[att];
    wire [7:0] digest_in;
    assign digest_in = dg[att];
    assign verify_ok = (state == 3'd4) && (att > 0) && vo[att-1];

    calib_fsm #(.KMAX(2), .HIST(`HIST), .DW(8)) dut (
        .clk(clk), .rst_n(rst_n), .start(start_req),
        .cmp_done(cmp_done), .meas_valid(meas_valid), .verify_ok(verify_ok),
        .digest_in(digest_in), .state(state),
        .stim_trig(stim_trig), .meas_trig(meas_trig), .corr_trig(corr_trig),
        .consume(consume), .done(done), .accepted(accepted), .failed(failed),
        .retry_cnt(retry_cnt)
    );

    reg cmp_done = 0;

    // ---- clock -------------------------------------------------------------
    always #5 clk = ~clk;

    // ---- host: conversion, pulses, counters --------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            cmp_done <= 0; mwait <= 0; att <= 0;
            stim_n <= 0; meas_n <= 0; corr_n <= 0; cons_n <= 0; ver_n <= 0;
            stimc <= 0; measc <= 0; corrc <= 0; consc <= 0;
            cyc <= 0; ts <= 0; tm <= 0; tc <= 0; tv <= 0; t0 <= 0; tdone <= 0;
            p_done <= 0;
            p_stim <= 0; p_meas <= 0; p_corr <= 0; p_cons <= 0;
            pstate <= 3'd0;
        end else begin
            cyc <= cyc + 1;
            pstate <= state;

            // entering MEASURE: arm the conversion, latch this attempt's inputs
            if (state == 3'd2 && pstate != 3'd2) begin
                mwait <= 3;
            end else if (state == 3'd2 && mwait > 0) begin
                mwait <= mwait - 1;
                if (mwait == 1) cmp_done <= 1;
            end else if (state != 3'd2) begin
                cmp_done <= 0;
            end

            // leave MEASURE with the reading consumed: advance the attempt index
            if (pstate == 3'd2 && state == 3'd3) att <= att + 1;

            // pulse counters (rising edges vs cycles-high)
            if (stim_trig) begin
                stimc <= stimc + 1;
                if (!p_stim) begin stim_n <= stim_n + 1; ts <= cyc; end
            end
            if (meas_trig) begin
                measc <= measc + 1;
                if (!p_meas) begin meas_n <= meas_n + 1; tm <= cyc; end
            end
            if (corr_trig) begin
                corrc <= corrc + 1;
                if (!p_corr) begin corr_n <= corr_n + 1; tc <= cyc; end
            end
            if (consume) begin
                consc <= consc + 1;
                if (!p_cons) cons_n <= cons_n + 1;
            end
            if (state == 3'd4 && pstate != 3'd4) begin
                ver_n <= ver_n + 1; tv <= cyc;
            end

            if (pstate == 3'd0 && state == 3'd1) t0 <= cyc;   // round start
            if (done && !p_done) tdone <= cyc;                // round end
            p_done <= done;

            p_stim <= stim_trig; p_meas <= meas_trig;
            p_corr <= corr_trig; p_cons <= consume;
        end
    end

    // ---- helpers -----------------------------------------------------------
    task automatic clear_cnt;
        begin
            att = 0; stim_n = 0; meas_n = 0; corr_n = 0; cons_n = 0; ver_n = 0;
            stimc = 0; measc = 0; corrc = 0; consc = 0;
            ts = 0; tm = 0; tc = 0; tv = 0;
        end
    endtask

    task automatic do_reset;
        begin
            rst_n = 0; start_req = 0;
            repeat (3) @(posedge clk);
            rst_n = 1;
            repeat (2) @(posedge clk);
            clear_cnt;
        end
    endtask

    task automatic begin_round(input integer hold_start);
        begin
            start_req = 1;
            @(posedge clk);
            if (!hold_start) start_req = 0;
        end
    endtask

    task automatic wait_fin(input integer maxc);
        integer k;
        begin
            for (k = 0; k < maxc; k = k + 1) begin
                @(posedge clk);
                if (done) k = maxc;
            end
        end
    endtask

    task automatic report(input [255:0] name);
        integer ord;
        begin
            ord = (ts > 0 && ts < tm && tm < tc && tc < tv) ? 1 : 0;
            $display("SCN %0s st=%0d stim=%0d meas=%0d corr=%0d cons=%0d retry=%0d maxretry=%0d stimc=%0d measc=%0d corrc=%0d consc=%0d ver=%0d ord=%0d n=%0d",
                     name, state, stim_n, meas_n, corr_n, cons_n, retry_cnt, 2,
                     stimc, measc, corrc, consc, ver_n, ord, cyc - t0);
        end
    endtask

    // ---- scenarios ---------------------------------------------------------
    initial begin
        dg[0] = 8'hA1; dg[1] = 8'hB2; dg[2] = 8'hC3; dg[3] = 8'hD4;
        vo[0] = 1; vo[1] = 1; vo[2] = 1; vo[3] = 1;
        mv[0] = 1; mv[1] = 1; mv[2] = 1; mv[3] = 1;

        // S1  straight accept
        do_reset;
        begin_round(0);
        wait_fin(80);
        report("happy");

        // S2  one retry, accepted on the second attempt
        do_reset;
        dg[0] = 8'hA1; dg[1] = 8'hB2; vo[0] = 0; vo[1] = 1;
        mv[0] = 1; mv[1] = 1;
        begin_round(0);
        wait_fin(160);
        report("retry1");

        // S3  retry budget exhausted: FAILED after KMAX+1 attempts
        do_reset;
        dg[0] = 8'hA1; dg[1] = 8'hB2; dg[2] = 8'hC3; dg[3] = 8'hD4;
        vo[0] = 0; vo[1] = 0; vo[2] = 0;
        mv[0] = 1; mv[1] = 1; mv[2] = 1;
        begin_round(0);
        wait_fin(240);
        report("budget");

        // S4  the third attempt replays the FIRST attempt's reading.
        //     HIST=4 must refuse it (FAILED); HIST=1 cannot see it.
        do_reset;
        dg[0] = 8'hA1; dg[1] = 8'hB2; dg[2] = 8'hA1;   // dg[2] == dg[0]
        vo[0] = 0; vo[1] = 0; vo[2] = 1;
        mv[0] = 1; mv[1] = 1; mv[2] = 1;
        begin_round(0);
        wait_fin(240);
        report("stale");

        // S5  a conversion that produced no data
        do_reset;
        dg[0] = 8'hA1; vo[0] = 1; mv[0] = 0;
        begin_round(0);
        wait_fin(80);
        report("nodata");

        // S6  ACCEPTED is terminal: start held high must not re-enter
        do_reset;
        dg[0] = 8'hA1; vo[0] = 1; mv[0] = 1;
        begin_round(0);
        wait_fin(80);
        repeat (10) @(posedge clk);       // start_req is 0 here
        start_req = 1;                     // now hold it high in ACC
        repeat (10) @(posedge clk);
        start_req = 0;
        report("term_acc");

        // S7  FAILED is terminal, same test
        do_reset;
        dg[0] = 8'hA1; dg[1] = 8'hB2; dg[2] = 8'hC3; dg[3] = 8'hD4;
        vo[0] = 0; vo[1] = 0; vo[2] = 0;
        mv[0] = 1; mv[1] = 1; mv[2] = 1;
        begin_round(0);
        wait_fin(240);
        start_req = 1;
        repeat (10) @(posedge clk);
        start_req = 0;
        report("term_fail");

        // S8  reset in the middle of a round returns to IDLE
        do_reset;
        dg[0] = 8'hA1; dg[1] = 8'hB2; vo[0] = 0; vo[1] = 1; mv[0] = 1; mv[1] = 1;
        begin_round(0);
        repeat (30) @(posedge clk);        // somewhere in the retry
        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);
        $display("SCN reset_mid st=%0d stim=%0d meas=%0d corr=%0d cons=%0d retry=%0d maxretry=%0d stimc=%0d measc=%0d corrc=%0d consc=%0d ver=%0d ord=%0d n=%0d",
                 state, stim_n, meas_n, corr_n, cons_n, retry_cnt, 2,
                 stimc, measc, corrc, consc, ver_n, 1, 0);

        // S9  start held for the whole round: exactly one round runs
        do_reset;
        dg[0] = 8'hA1; vo[0] = 1; mv[0] = 1;
        begin_round(1);
        wait_fin(120);
        start_req = 0;
        report("start_held");

        $display("SIMDONE");
        $finish;
    end

    // global timeout
    initial begin
        #200000;
        $display("SIMTIMEOUT");
        $finish;
    end

endmodule
