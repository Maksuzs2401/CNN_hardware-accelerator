`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 29.03.2026 16:27:00
// Design Name: 
// Module Name: max_polling
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

module max_polling #(parameter IN_CHANNELS = 32)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         s_axis_tvalid,
    //input  logic                         s_axis_tlast,
    output logic                         s_axis_tready,
    input  logic signed [`accu_width-1:0]s_axis_tdata[IN_CHANNELS-1:0],
    
    output logic signed [`accu_width-1:0]m_axis_tdata[IN_CHANNELS-1:0],
    output logic                         m_axis_tvalid,
    input  logic                         m_axis_tready
    //output logic                         m_axis_tlast
    );
    
    logic toggle_state;
    logic signed [`accu_width-1:0]temp_reg[IN_CHANNELS-1:0];
    
    assign s_axis_tready  = ~m_axis_tvalid;
    
    always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        toggle_state  <= 1'b0;
        m_axis_tvalid <= 1'b0;
        //m_axis_tlast  <= 1'b0;
        for (int i = 0; i < IN_CHANNELS; i++) begin
            temp_reg[i]    <= 0;
            m_axis_tdata[i]<= 0;
        end
    end else begin
        // Clear valid after downstream handshake
        if (m_axis_tvalid && m_axis_tready)begin
            m_axis_tvalid <= 1'b0;
           // m_axis_tlast  <= 1'b0;
        end
        // Accept new upstream data
        if (s_axis_tvalid && s_axis_tready) begin
            if (toggle_state == 1'b0) begin
                // FIRST sample of pair: just store, do NOT assert valid yet
                for (int i = 0; i < IN_CHANNELS; i++)
                    temp_reg[i] <= s_axis_tdata[i];
                toggle_state <= 1'b1;
            end else begin
                // SECOND sample of pair: compare and output max
                for (int i = 0; i < IN_CHANNELS; i++)
                    m_axis_tdata[i] <= (s_axis_tdata[i] > temp_reg[i])
                                        ? s_axis_tdata[i] : temp_reg[i];
                toggle_state  <= 1'b0;
              //  m_axis_tlast <= s_axis_tlast;
                m_axis_tvalid <= 1'b1;
            end
        end
    end
end
endmodule
