`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 24.03.2026 16:27:25
// Design Name: 
// Module Name: neuron
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


module neuron #(parameter data_width = 8,
                parameter accu_width = 24)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic                           clear_accum,
    input  logic signed [data_width-1:0]   my_input,
    input  logic signed [data_width-1:0]   my_weight,
    input  logic                           my_input_valid,
    output logic signed [accu_width-1:0]   out
);
 
    // ----------------------------------------------------------------
    // Stage 1 : register the INPUT only (weight is used combinationally)
    // ----------------------------------------------------------------
    logic signed [data_width-1:0]   input_delay;
    logic                           valid_d;
    // NOTE: weight_delay FF removed - my_weight is used directly below.
 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_delay <= '0;
            valid_d     <= 1'b0;
        end else begin
            input_delay <= my_input;
            valid_d     <= my_input_valid;
        end
    end
 
    // ----------------------------------------------------------------
    // Stage 2 : multiply  (input is 1-cycle delayed, weight is current)
    // Both are now aligned to the same inference step.
    // ----------------------------------------------------------------
    logic signed [(data_width*2)-1:0] mul_result;
 
    always_comb begin
        mul_result = input_delay * my_weight;
    end
 
    // ----------------------------------------------------------------
    // Stage 3 : accumulate
    // clear_accum used directly (not through an extra FF) so the clear
    // fires in the same cycle as the first valid mul product.
    // ----------------------------------------------------------------
    logic signed [accu_width-1:0] sum_result;
 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_result <= '0;
        end else if (valid_d) begin
            if (clear_accum)
                sum_result <= accu_width'(mul_result);   // start fresh
            else
                sum_result <= sum_result + accu_width'(mul_result);
        end
    end
 
    assign out = sum_result;
 
endmodule
