`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 26.03.2026 15:43:51
// Design Name: 
// Module Name: activation_funct
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


module activation_funct #(parameter data_width =24)(
input logic signed  [data_width-1:0] data_in;
output logic signed [data_width-1:0] data_out;
    );
    always_comb begin
      assign data_out = (data_in[data_width-1]==1'b1) '0:data_in;
    end
endmodule
