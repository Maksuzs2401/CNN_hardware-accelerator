`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 27.03.2026 20:03:44
// Design Name: 
// Module Name: l1_l2_buff
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
`include "config.vh"

module l1_l2_buff (
    input  logic clk,
    input  logic rst_n,
    input  logic l1_done, 
    input  logic signed [`accu_width-1:0] l1_parallel_in [`L1_neurons-1:0], 
    
    output logic signed [`data_width-1:0] l2_serial_out, 
    output logic                          l2_valid
    );
    
    logic signed [`data_width-1:0] shift_reg [`L1_neurons-1:0];
    logic [$clog2(`L1_neurons)-1:0] shift_counter;
    logic is_shifting;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            is_shifting   <= 1'b0;
            l2_valid      <= 1'b0;
            l2_serial_out <= 0;
            shift_counter <= 0;
            
            for (int i=0; i<`L1_neurons; i++) shift_reg[i] <= 0;
            
        end else if (l1_done && !is_shifting) begin
            // PHASE 1: Load and Quantize
            is_shifting <= 1'b1;
            
            for (int i = 0; i < `L1_neurons; i++) begin
                // Cap at maximum 8-bit signed value (127)
                if (l1_parallel_in[i] > 24'sd127) begin
                    shift_reg[i] <= 8'sd127;
                end else begin
                    shift_reg[i] <= l1_parallel_in[i][`data_width-1:0]; 
                end
            end
            
        end else if (is_shifting) begin
            // PHASE 2: Serial Shift
            l2_valid      <= 1'b1;
            l2_serial_out <= shift_reg[0]; 
            
            // Shift everything down by 1
            for (int i = 0; i < `L1_neurons-1; i++) begin
                shift_reg[i] <= shift_reg[i+1];
            end
            
            // Check if we have sent the last neuron's data
            if (shift_counter == `L1_neurons - 1) begin
                is_shifting   <= 1'b0;     
                shift_counter <= 0;
            end else begin
                shift_counter <= shift_counter + 1;
            end
            
        end else begin
            l2_valid <= 1'b0; 
        end
    end
endmodule
