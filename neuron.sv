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


module neuron #(parameter data_width=8,accu_width=24)(
input logic clk,
input logic rst_n,
input  logic                         clear_accum,
input logic signed  [data_width-1:0] my_input,
input logic signed  [data_width-1:0] my_weight,
input logic                          my_input_valid,
output logic signed [accu_width-1:0] out 
    );
    //==== Internal wires ====//
  logic signed [data_width-1:0] input_delay;
  logic signed [data_width-1:0] weight_delay;
  logic signed [(data_width*2)-1:0] mul_result;
  logic signed [accu_width-1:0]sum_result;
  logic clear_d;
  logic valid_d;

    //==== D flip flop acting as a buffer/delay unit ====//
  always_ff @(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
      input_delay <=0;
      clear_d <=0;
      valid_d <=0;
      weight_delay <=0;
    end else begin
      input_delay <= my_input;
      clear_d <= clear_accum;
      valid_d <= my_input_valid;
      weight_delay <= my_weight;
    end
  end

    //==== Combinational block for multiplication unit ====//
  always_comb begin
    mul_result = input_delay * weight_delay;
  end

    //==== Summation unit ====//
  always @(posedge clk or negedge rst_n)begin
    if(!rst_n)
      sum_result <=0;
    else if(valid_d)begin
      if(clear_d)
        sum_result <= accu_width'(mul_result);
      else
        sum_result <= sum_result + mul_result;
    end
  end

    //==== ReLU activation unit ====//
  assign out = sum_result;
  
endmodule
