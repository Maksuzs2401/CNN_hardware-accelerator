`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07.04.2026 02:17:29
// Design Name: 
// Module Name: l3_l4_buff
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

module l3_l4_buff #(parameter IN_CHANNELS = `L3_neurons,
                    parameter TIME_STEPS = 23,
                    parameter SCALE_SHIFT = 8,
                    parameter TOTAL_EL = TIME_STEPS * IN_CHANNELS)(
    input  logic clk,
    input  logic rst_n,
    
    input  logic s_axis_tvalid, 
    input  logic signed [`accu_width-1:0] s_axis_tdata [IN_CHANNELS-1:0], 
    output logic s_axis_tready, 
    input logic s_axis_tlast,
    output logic signed [`data_width-1:0] m_axis_tdata,
    output logic                          m_axis_tvalid,
    output logic                          m_axis_tlast, 
    input  logic                          m_axis_tready
    );
    
    logic signed [`data_width-1:0]one_dime_mem[0:TOTAL_EL-1];
    logic [4:0]step_count;
    logic [11:0]read_count;
    logic is_flattening;
    logic signed [`accu_width-1:0] scaled_val;
    
    assign s_axis_tready = is_flattening;
    assign m_axis_tlast = (!is_flattening && (read_count == TOTAL_EL - 1) && m_axis_tvalid) ? 1'b1 : 1'b0;
    
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
           step_count <= 0;
           is_flattening <= 1'b1;
           m_axis_tvalid <= 1'b0;
           read_count <= 0;
        end else begin
            if(s_axis_tvalid && s_axis_tready && is_flattening)begin
                for(int i=0; i<IN_CHANNELS; i++)begin
                    // FIX 5: Quantize the 24-bit input to 8-bit before saving
                    scaled_val = s_axis_tdata[i] >>> SCALE_SHIFT;
                    if (scaled_val > 24'sd127) begin
                        one_dime_mem[(step_count*IN_CHANNELS)+i] <= 8'sd127;
                    end else begin
                        one_dime_mem[(step_count*IN_CHANNELS)+i] <= scaled_val[`data_width-1:0];
                    end
                end
                if(s_axis_tlast)begin
                    step_count <= 0;
                    is_flattening <= 1'b0;
                end else begin
                    step_count <= step_count + 1;                   
                end
            end 
            
            else if(!is_flattening)begin
                if(m_axis_tready)begin
                    if(read_count < TOTAL_EL)begin
                        m_axis_tdata <= one_dime_mem[read_count];
                        read_count <= read_count +1;
                        m_axis_tvalid <= 1'b1;
                    end else begin
                        m_axis_tvalid <= 1'b0;
                        step_count <= 0;
                        read_count <= 0;
                        is_flattening <= 1'b1;
                    end
                end else begin
                    m_axis_tvalid <= 1'b0;  
                end
            end
        end
    end
endmodule
