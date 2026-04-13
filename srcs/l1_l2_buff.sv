`timescale 1ns / 1ps

`include "config.vh"

module l1_l2_buff #(parameter IN_CHANNELS = `L1_neurons,
                    parameter SCALE_SHIFT = 8)(
    input  logic clk,
    input  logic rst_n,
    
    input  logic s_axis_tvalid, 
    input  logic signed [`accu_width-1:0] s_axis_tdata [IN_CHANNELS-1:0], 
    output logic s_axis_tready, 
    output logic signed [`data_width-1:0] m_axis_tdata,
    output logic                          m_axis_tvalid,
    output logic                          m_axis_tlast, 
    input logic                           m_axis_tready
    );
    
    logic signed [`data_width-1:0] shift_reg [IN_CHANNELS-1:0];
    logic [$clog2(IN_CHANNELS)-1:0] shift_counter;
    logic is_shifting;
    logic signed [`accu_width-1:0] scaled_val;
    
    assign s_axis_tready = ~is_shifting;
    
    assign m_axis_tdata = shift_reg[0];
    assign m_axis_tlast = (shift_counter==IN_CHANNELS-1)? 1'b1:1'b0;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            is_shifting   <= 1'b0;
            m_axis_tvalid <= 1'b0;
            shift_counter <= 0;
            for(int i=0; i<IN_CHANNELS; i++) shift_reg[i]<=0;
            
        end else if (s_axis_tvalid && s_axis_tready) begin
            // PHASE 1: Load and Quantize
            is_shifting <= 1'b1;
            m_axis_tvalid <= 1'b1;
            for (int i = 0; i < IN_CHANNELS; i++) begin
                scaled_val = s_axis_tdata[i] >>> SCALE_SHIFT;
                if (i == 0) begin
                $display("[%0t] SERIALIZER PROBE | Raw In: %0d | Shift Param: %0d | Scaled: %0d", 
                         $time, s_axis_tdata[0], SCALE_SHIFT, scaled_val);end
                
                if (scaled_val > 24'sd127) begin   // Capped at maximum 8-bit signed value (127)
                    shift_reg[i] <= 8'sd127;
                end else begin
                    shift_reg[i] <= scaled_val[`data_width-1:0]; 
                end
            end
        end else if (is_shifting) begin
            // PHASE 2: Serial Shift
            m_axis_tvalid      <= 1'b1;
            if(m_axis_tready)begin
                for (int i = 0; i < IN_CHANNELS-1; i++) begin       // Shifting everything down by 1
                    shift_reg[i] <= shift_reg[i+1];
                end
                if (shift_counter == IN_CHANNELS - 1) begin          // Checking if we have sent the last neuron's data
                    is_shifting   <= 1'b0; 
                    m_axis_tvalid <= 1'b0;    
                    shift_counter <= 0;
            end else begin
                shift_counter <= shift_counter + 1;
            end
          end
        end else begin
            m_axis_tvalid <= 1'b0; 
        end
    end
endmodule
