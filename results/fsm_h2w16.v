// -----------------------------------------------------------------------------
// calib_fsm.v -- B3 characterization-vehicle supervisor: the calibration FSM.
//
// Protocol contract: twin/calib.py + bench/run_calib.py (62/62).  The state
// sequence and the terminal semantics are the contract; this file is the
// pin-level realisation that B3's test plan replays (G6/C3).
//
//   IDLE -> STIMULATE -> MEASURE -> CORRECT -> VERIFY -> ACCEPTED (terminal)
//                                                    \-> STIMULATE (retry)
//                                                    \-> FAILED   (terminal)
//
// Three semantics are load-bearing, and they are why this block carries a
// digest history instead of a single "last reading" register:
//   * FAILED is terminal and is never reported as an accept.  The twin refuses
//     a FAILED CalibrationResult (CalibratedTwin.apply_foreground), so a
//     certificate can never be gated on one.
//   * a retry must consume a FRESH measurement: a reading whose digest is
//     already in the history may not be re-used.  Replaying an old reading is
//     exactly the overfit failure bench/run_calib.py catches (the correction
//     is fitted to the training set and "passes" on it).
//   * VERIFY reads a held-out stimulus/reading pair (verify_ok), not the
//     training residual.  The RTL does not distinguish them -- it only sees
//     the verdict -- but the testbench drives verify_ok from a held-out pair,
//     so the pin-level protocol matches the software contract.
//
// pulse convention: every trigger is one clock per assertion, and every state
// re-uses `settle` as its own in-state cycle counter (reset on entry).
// -----------------------------------------------------------------------------
// The fsm_encoding attribute is rewritten per variant by bench/run_gf180_fsm.py
// (auto / one-hot / binary / gray) so the three encodings are one source file.
(* fsm_encoding = "auto" *)
module calib_fsm #(
    parameter KMAX = 2,     // retry budget: at most KMAX retries (3 attempts)
    parameter HIST = 2,     // digest history depth (1 = the rejected variant)
    parameter DW   = 16      // digest width
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,        // begin a round (only honoured in IDLE)
    input  wire          cmp_done,     // conversion finished strobe
    input  wire          meas_valid,   // a fresh reading actually arrived
    input  wire          verify_ok,    // held-out residual inside tolerance
    input  wire [DW-1:0] digest_in,    // digest of the reading being consumed
    output reg  [2:0]    state,
    output wire          stim_trig,
    output wire          meas_trig,
    output wire          corr_trig,
    output wire          consume,
    output wire          done,
    output wire          accepted,
    output wire          failed,
    output reg  [1:0]    retry_cnt
);

localparam S_IDLE = 3'd0;
localparam S_STIM = 3'd1;
localparam S_MEAS = 3'd2;
localparam S_CORR = 3'd3;
localparam S_VER  = 3'd4;
localparam S_ACC  = 3'd5;
localparam S_FAIL = 3'd6;

integer           k;
reg [1:0]         settle;                       // in-state cycle counter
reg [HIST*DW-1:0] hist;                         // consumed digests, [0] newest
wire [HIST-1:0]   dg_hit;

genvar i;
generate
    for (i = 0; i < HIST; i = i + 1) begin : g_cmp
        assign dg_hit[i] = (hist[i*DW +: DW] == digest_in);
    end
endgenerate

wire dg_match = |dg_hit;                        // this reading was already used

assign stim_trig = (state == S_STIM) && (settle == 2'd0);
assign meas_trig = (state == S_MEAS) && (settle == 2'd0);
assign corr_trig = (state == S_CORR) && (settle == 2'd0);
assign consume   = corr_trig;                   // one digest per measurement
assign accepted  = (state == S_ACC);
assign failed    = (state == S_FAIL);
assign done      = accepted | failed;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state     <= S_IDLE;
        settle    <= 2'd0;
        retry_cnt <= 2'd0;
        // A real cell resets the history to an impossible digest; here the
        // testbench uses non-zero digests, so all-zero is that value.
        hist      <= {HIST*DW{1'b0}};
    end else begin
        // `settle` is incremented by each state itself: a free-running
        // increment at the top of this block would still read the OLD value in
        // the same cycle, so a state that keys on settle==0 would fire twice.
        case (state)
            S_IDLE: begin
                if (start) begin
                    retry_cnt <= 2'd0;
                    settle    <= 2'd0;
                    state     <= S_STIM;
                end
            end

            S_STIM: begin                       // stimulus applied, then settle
                if (settle == 2'd2) begin
                    settle <= 2'd0;
                    state  <= S_MEAS;
                end else begin
                    settle <= settle + 2'd1;
                end
            end

            S_MEAS: begin                      // wait for the conversion
                if (settle == 2'd0)
                    settle <= 2'd1;             // meas_trig is one cycle only
                if (cmp_done) begin
                    if (!meas_valid || dg_match) begin
                        settle <= 2'd0;
                        state  <= S_FAIL;       // no data, or a stale reading
                    end else begin
                        settle <= 2'd0;
                        state  <= S_CORR;
                    end
                end
            end

            S_CORR: begin                      // apply the correction
                if (settle == 2'd0) begin
                    // word-wise shift, newest at hist[0]; a loop rather than a
                    // concatenation so HIST=1 needs no negative part-select
                    for (k = HIST - 1; k > 0; k = k - 1)
                        hist[k*DW +: DW] <= hist[(k-1)*DW +: DW];
                    hist[0 +: DW] <= digest_in;
                    settle <= 2'd1;             // one cycle only: hist shifts once
                end else begin
                    settle <= 2'd0;
                    state  <= S_VER;
                end
            end

            S_VER: begin                       // held-out verdict
                if (verify_ok) begin
                    settle <= 2'd0;
                    state  <= S_ACC;
                end else if (retry_cnt < KMAX) begin
                    retry_cnt <= retry_cnt + 2'd1;
                    settle    <= 2'd0;
                    state     <= S_STIM;
                end else begin
                    settle <= 2'd0;
                    state  <= S_FAIL;
                end
            end

            // terminal: only reset leaves ACCEPTED/FAILED.  `start` is ignored,
            // so a host that holds start high cannot re-enter a round.
            S_ACC, S_FAIL: begin
                // terminal: do NOT assign state (a `state <= state` here is
                // what makes fsm_detect reject the whole machine as a
                // non-constant next-state function)
                settle <= 2'd0;
            end

            default: begin
                settle <= 2'd0;
                state  <= S_IDLE;
            end
        endcase
    end
end

endmodule
