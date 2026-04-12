`timescale 1ns / 1ps
`include "config.vh"

module line_buffer #(
    parameter KERNEL_SIZE = 5,
    parameter NUM_CHANNELS = 1  // Defaults to 1 for Layer 1. Set to 32 for Layer 2!
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [`data_width-1:0] s_axis_tdata,
    input  logic s_axis_tvalid,
    input  logic s_axis_tlast,
    output logic s_axis_tready,
    // The output is flattened: KERNEL_SIZE * NUM_CHANNELS
    output logic signed [`data_width-1:0] m_axis_tdata [(KERNEL_SIZE * NUM_CHANNELS)-1:0],
    output logic m_axis_tvalid,
    input  logic m_axis_tready
);
    
    // 2D Memory Grid: [Time Steps] [Channels]
    logic signed [`data_width-1:0] shift_mem [KERNEL_SIZE-1:0][NUM_CHANNELS-1:0];
    
    integer ch_count;
    integer time_count;
    logic is_full;
    
    assign s_axis_tready = ~m_axis_tvalid;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch_count       <= 0;
            time_count     <= 0;
            is_full        <= 1'b0;
            m_axis_tvalid <= 1'b0;
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                for (int j = 0; j < NUM_CHANNELS; j++) begin
                    shift_mem[i][j] <= 0;
                end
            end
        end else begin
            if(m_axis_tvalid && m_axis_tready)begin
                m_axis_tvalid <= 1'b0;
            end
            if(s_axis_tvalid && s_axis_tready)begin
                for(int i=0; i<KERNEL_SIZE; i++)begin
                    shift_mem[i][ch_count] <= shift_mem[i+1][ch_count];
                end
                shift_mem[KERNEL_SIZE-1][ch_count]<=s_axis_tdata;
                
                if(s_axis_tlast)begin
                    ch_count<=0;
                    if(time_count < KERNEL_SIZE-1)begin
                        time_count <= time_count+1;
                    end else begin
                        is_full <= 1'b1;
                    end
                    if(time_count == KERNEL_SIZE-1 || is_full)begin
                        m_axis_tvalid <= 1'b1;
                    end
                end else begin
                    ch_count <= ch_count +1;
                end
            end
        end
    end

    // 5. Flatten the 2D array into a 1D output for the neurons
    always_comb begin
        for (int i = 0; i < KERNEL_SIZE; i++) begin
            for (int j = 0; j < NUM_CHANNELS; j++) begin
                m_axis_tdata[(i * NUM_CHANNELS) + j] = shift_mem[i][j];
            end
        end
    end

endmodule