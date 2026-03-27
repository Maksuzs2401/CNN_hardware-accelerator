`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 26.03.2026 15:30:58
// Design Name: 
// Module Name: layers
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

module layers #(parameter neuron_no = `L1_neurons,
weights = `L1_weights,
act_type = `L1_act,
parameter string weight_file = "default.hex")(
input logic clk,
input logic rst_n,
input logic valid_in,
input logic clear_accu,
input  logic signed [`data_width-1:0] sensor_data,
output logic signed [`accu_width-1:0]layer_out[neuron_no-1:0]
);

  logic signed [`data_width-1:0]weight_rom[0:neuron_no-1][0:weights-1];

  initial begin
    $readmemh(weight_file,weight_rom);
  end
  
  logic [$clog2(weights)-1:0] weight_ptr;
  
  always_ff @(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
      weight_ptr <= 0;
    end else if(valid_in)begin
      if(clear_accu)begin
        weight_ptr <= 1;
      end else if(weight_ptr < weights-1)begin
        weight_ptr <= weight_ptr+1;
      end
    end
  end
  
  logic [$clog2(weights)-1:0]current_ptr;
  assign current_ptr = (clear_accu)? 0:weight_ptr;
  
  genvar i;
  generate
        for(i=0;i<neuron_no;i=i+1)begin : neuron_array
          
          logic signed [`accu_width-1:0]raw_sum;
          logic signed [`data_width-1:0]current_weight;
          
          assign current_weight = weight_rom[i][current_ptr];
          
          neuron #(.data_width(`data_width),
          .accu_width(`accu_width))
          neuron_inst(.clk(clk),.rst_n(rst_n),.my_input(sensor_data),.out(raw_sum),
          .my_input_valid(valid_in),.clear_accum(clear_accu),.my_weight(current_weight));
          
          if(act_type == "relu")begin
            activation_funct relu_inst(.data_in(raw_sum),
            .data_out(layer_out[i]));
          end else
            assign layer_out[i] = raw_sum;
        end
   endgenerate
endmodule
