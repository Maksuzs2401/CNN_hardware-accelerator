`timescale 1ns / 1ps
`include "config.vh"

module neuron #(parameter data_width = 8,
                parameter accu_width = 24,parameter kernel_size = 5)(
    input  logic                           clk,
    input  logic                           rst_n,
    input  logic signed [data_width-1:0]   s_axis_tdata,   
    input  logic signed [data_width-1:0]   s_axis_tdata_wgt, 
    input  logic                           s_axis_tvalid,
    input  logic                           s_axis_tlast,
    output logic                           s_axis_tready,
    output logic signed [accu_width-1:0]   m_axis_tdata,
    output logic                           m_axis_tvalid
);
    logic signed [accu_width-1:0] accumulator;
    (*use_dsp = "yes"*)logic signed [accu_width-1:0] mul_result;
    
    assign s_axis_tready = 1'b1;

    always_comb begin
        mul_result = s_axis_tdata * s_axis_tdata_wgt;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axis_tdata   <=0;
            m_axis_tvalid  <=1'b0;
            accumulator <= 1'b0;
        end else begin
            m_axis_tvalid <= 1'b0;
            if(s_axis_tvalid && s_axis_tready)begin
                if(s_axis_tlast)begin
                    m_axis_tdata <= accumulator + mul_result;
                    m_axis_tvalid <= 1'b1;
                    accumulator <=0;
                end else begin
                    accumulator <= accumulator + mul_result;
                end
            end
        end
    end
endmodule
