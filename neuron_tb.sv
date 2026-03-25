`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 25.03.2026 13:40:00
// Design Name: 
// Module Name: neuron_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


`timescale 1ns / 1ps

module neuron_tb;
  // Parameters
  localparam DATA_WIDTH = 8;
  localparam ACCU_WIDTH = 24;
  localparam CLK_PERIOD = 10;

  // DUT signals
  logic clk, rst_n, clear_accum, my_input_valid;
  logic signed [DATA_WIDTH-1:0] my_input, my_weight;
  logic signed [ACCU_WIDTH-1:0] out;

  // Instantiate DUT
  neuron #(.data_width(DATA_WIDTH), .accu_width(ACCU_WIDTH)) dut (
    .clk(clk), .rst_n(rst_n), .clear_accum(clear_accum),
    .my_input(my_input), .my_weight(my_weight),
    .my_input_valid(my_input_valid), .out(out)
  );

  // Clock generation
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // Helper task: send one input/weight pair
  task send_data(
    input logic signed [DATA_WIDTH-1:0] in_val, wt_val,
    input logic clr
  );
    @(negedge clk);
    my_input       = in_val;
    my_weight      = wt_val;
    my_input_valid = 1;
    clear_accum    = clr;
  endtask

  task idle();
    @(negedge clk);
    my_input_valid = 0;
    clear_accum    = 0;
  endtask

  // ─── Test sequence ───────────────────────────────────────────────────
  integer expected;

  initial begin
    // Initialise
    rst_n = 0; clear_accum = 0;
    my_input = 0; my_weight = 0; my_input_valid = 0;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // ── TEST 1: Basic accumulation ────────────────────────────────────
    // Dot product: (3*2) + (4*5) + (6*1) = 6 + 20 + 6 = 32
    $display("\n=== TEST 1: Basic Accumulation ===");
    send_data(3, 2, 1);   // clear=1: starts new accumulation
    send_data(4, 5, 0);
    send_data(6, 1, 0);
    idle();
    repeat(3) @(posedge clk);  // flush pipeline
    expected = 32;
    $display("Expected: %0d | Got: %0d | %s",
             expected, out, (out==expected) ? "PASS":"FAIL");

    // ── TEST 2: Clear mid-stream (new dot product) ────────────────────
    // Continue from above: clear then (2*3) + (1*1) = 7
    $display("\n=== TEST 2: Clear & Restart ===");
    send_data(2, 3, 1);   // clear=1: discards previous, starts fresh
    send_data(1, 1, 0);
    idle();
    repeat(3) @(posedge clk);
    expected = 7;
    $display("Expected: %0d | Got: %0d | %s",
             expected, out, (out==expected) ? "PASS":"FAIL");

    // ── TEST 3: ReLU — negative result should output 0 ───────────────
    // (-5)*3 = -15 → ReLU → 0
    $display("\n=== TEST 3: ReLU (negative → 0) ===");
    send_data(-5, 3, 1);
    idle();
    repeat(3) @(posedge clk);
    expected = 0;
    $display("Expected: %0d | Got: %0d | %s",
             expected, out, (out==expected) ? "PASS":"FAIL");

    // ── TEST 4: valid gate — idle cycles shouldn't change accumulator ─
    $display("\n=== TEST 4: Valid Gating ===");
    send_data(4, 4, 1);   // load 16
    idle(); idle(); idle(); // three dead cycles
    send_data(1, 2, 0);   // add 2 → total should be 18
    idle();
    repeat(3) @(posedge clk);
    expected = 18;
    $display("Expected: %0d | Got: %0d | %s",
             expected, out, (out==expected) ? "PASS":"FAIL");

    // ── TEST 5: Max positive values (no overflow check) ───────────────
    $display("\n=== TEST 5: Large values ===");
    send_data(127, 127, 1);  // 127*127 = 16129
    idle();
    repeat(3) @(posedge clk);
    expected = 16129;
    $display("Expected: %0d | Got: %0d | %s",
             expected, out, (out==expected) ? "PASS":"FAIL");

    // ── TEST 6: Reset during operation ────────────────────────────────
    $display("\n=== TEST 6: Async Reset ===");
    send_data(10, 10, 1);
    @(posedge clk);
    rst_n = 0;             // assert reset mid-operation
    idle();
    @(posedge clk);
    rst_n = 1;
    
    repeat(3) @(posedge clk);
    expected = 0;
    $display("Expected: %0d | Got: %0d | %s",
             expected, out, (out==expected) ? "PASS":"FAIL");

    $display("\n=== All tests done ===");
    $finish;
  end

  // Waveform dump (for GTKWave / Vivado sim)
  initial begin
    $dumpfile("neuron_tb.vcd");
    $dumpvars(0, neuron_tb);
  end

endmodule
