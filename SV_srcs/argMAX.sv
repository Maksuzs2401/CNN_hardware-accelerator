`timescale 1ns / 1ps
`include "config.vh"

module argMAX #(parameter num_classes = `L2_neurons,
data_width = `accu_width)(
input logic signed [data_width-1:0] data_in [num_classes-1:0],
output logic [$clog2(num_classes)-1:0] predicted_class
    );
    
    logic signed [data_width-1:0]current_maxval;
    logic [$clog2(num_classes)-1:0]current_idx;
    
    always_comb begin
       current_maxval = data_in[0];
       current_idx = 0;
       
       for(int i=0; i<num_classes; i++)begin
            if(data_in[i] > current_maxval)begin
                current_maxval = data_in[i];
                current_idx = i[$clog2(num_classes)-1:0];
            end
       end
    end
    assign predicted_class = current_idx;
endmodule
