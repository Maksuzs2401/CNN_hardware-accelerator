`timescale 1ns / 1ps
`include "config.vh"

module l3_l4_buff #(parameter IN_CHANNELS = `L3_neurons,
                    parameter TIME_STEPS = 19,
                    parameter SCALE_SHIFT = 8,
                    parameter TOTAL_EL = TIME_STEPS * IN_CHANNELS)(
    input  logic clk,
    input  logic rst_n,
    
    input  logic s_axis_tvalid, 
    input  logic signed [`accu_width-1:0] s_axis_tdata [IN_CHANNELS-1:0], 
    output logic s_axis_tready, 
    input  logic s_axis_tlast,
    output logic signed [`data_width-1:0] m_axis_tdata,
    output logic                          m_axis_tvalid,
    output logic                          m_axis_tlast, 
    input  logic                          m_axis_tready
);
    
    logic signed [`data_width-1:0] one_dime_mem [0:TOTAL_EL-1];
    logic [4:0]  step_count;
    logic [11:0] read_count;
    logic is_flattening;
    logic signed [`accu_width-1:0] scaled_val;
    
    assign s_axis_tready = is_flattening;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            step_count    <= 0;
            is_flattening <= 1'b1;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
            read_count    <= 0;
        end else begin
            // ==========================================
            // PHASE 1: Collect and quantize from L3
            // ==========================================
            if (s_axis_tvalid && s_axis_tready && is_flattening) begin
                for (int i = 0; i < IN_CHANNELS; i++) begin
                    scaled_val = s_axis_tdata[i] >>> SCALE_SHIFT;
                    if (scaled_val > 24'sd127)
                        one_dime_mem[(step_count * IN_CHANNELS) + i] <= 8'sd127;
                    else if (scaled_val < -24'sd128)
                        one_dime_mem[(step_count * IN_CHANNELS) + i] <= -8'sd128;
                    else
                        one_dime_mem[(step_count * IN_CHANNELS) + i] <= scaled_val[`data_width-1:0];
                end
                if (step_count == TIME_STEPS - 1) begin
                    step_count    <= 0;
                    is_flattening <= 1'b0;
                end else begin
                    step_count <= step_count + 1;
                end
            end
            
            // ==========================================
            // PHASE 2: Serial readout to dense layer
            // ==========================================
            else if (!is_flattening) begin
                if (m_axis_tvalid && !m_axis_tready) begin
                    // Hold valid+data until handshake completes (AXI-Stream rule)
                end else if (read_count < TOTAL_EL) begin
                    // Load next value - data and tlast are registered together
                    m_axis_tdata  <= one_dime_mem[read_count];
                    m_axis_tlast  <= (read_count == TOTAL_EL - 1);
                    m_axis_tvalid <= 1'b1;
                    read_count    <= read_count + 1;
                end else begin
                    // All data sent, reset for next inference
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                    read_count    <= 0;
                    step_count    <= 0;
                    is_flattening <= 1'b1;
                end
            end
        end
    end
endmodule