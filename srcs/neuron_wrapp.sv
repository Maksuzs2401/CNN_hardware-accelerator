`timescale 1ns / 1ps
`include "config.vh"

module neuron_wrapp(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [`data_width-1:0] s_axis_tdata_ecg,
    input  logic                          s_axis_tvalid_ecg,
    input  logic                          s_axis_tlast_ecg,
    output logic                          s_axis_tready_ecg,
    output logic signed [`dense_width-1:0]m_axis_tdata_out[`L4_neurons-1:0],
    output logic                          m_axis_tvalid_out
);
    
    // Buffer > MAC
    logic signed [`data_width-1:0]w_l1_window [(5*1)-1:0];
    logic w_l1_window_valid;
    logic w_l1_window_ready;
    
    // L1 > max pool
    logic signed [`accu_width-1:0]w_l1_mac_out[`L1_neurons-1:0];
    logic w_l1_mac_valid;
    logic w_maxpool_ready;
    
    // Max pool > Serializer
    logic signed [`accu_width-1:0]w_pool_out[`L1_neurons-1:0];
    logic w_pool_valid;
    logic w_serializer_ready;
    
    // Serializer > L2 Buffer
    logic signed[`data_width-1:0]w_serial_data;
    logic w_serial_valid;
    logic w_serial_last;
    logic w_l2_buffer_ready;
    
    //L2 buffer > MAC
    logic signed [`data_width-1:0]w_l2_window[(5*`L1_neurons)-1:0];
    logic w_l2_window_valid;
    logic w_l2_window_ready;
    
    //L2 mac > Max pool
    logic signed [`accu_width-1:0] w_l2_mac_out [`L2_neurons-1:0];
    logic w_l2_mac_valid;
    logic w_l2_maxpool_ready;
    
    // Max pool > serializer
    logic signed [`accu_width-1:0] w_l2_pool_out [`L2_neurons-1:0];
    logic w_l2_pool_valid;
    logic w_l2_serializer_ready;
    
    // Serializer > L3 Buffer
    logic signed[`data_width-1:0]w_l2_serial_data;
    logic w_l2_serial_valid;
    logic w_l2_serial_last;
    logic w_l3_buffer_ready;
    
    //L3 Buffer > MAC
    logic signed [`data_width-1:0] w_l3_window [(5*`L2_neurons)-1:0];
    logic w_l3_window_valid;
    logic w_l3_window_ready;
    
    //L3 mac > Max pool
    logic signed [`accu_width-1:0] w_l3_mac_out [`L3_neurons-1:0];
    logic w_l3_mac_valid;
    logic w_l3_maxpool_ready;
    
    //Max pool > serializer
    logic signed [`accu_width-1:0] w_l3_pool_out [`L3_neurons-1:0];
    logic w_l3_pool_valid;
    logic w_l3_serializer_ready;
    
    //Serializer > MAC
    logic signed [`data_width-1:0] w_l3_serial_data;
    logic w_l3_serial_valid;
    logic w_l3_serial_last;
    logic w_l4_ready;
    
    // Layer-1 stages
    line_buffer #(.KERNEL_SIZE(5),.NUM_CHANNELS(1))
        l1_buffer(.clk(clk),.rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata_ecg),.s_axis_tvalid(s_axis_tvalid_ecg),
        .s_axis_tready(s_axis_tready_ecg),.s_axis_tlast(s_axis_tlast_ecg),
        .m_axis_tdata(w_l1_window),.m_axis_tvalid(w_l1_window_valid),
        .m_axis_tready(w_l1_window_ready));
    
    layers #(.neuron_no(`L1_neurons),.weights(`L1_weights),.KERNEL_SIZE(5),
    .NUM_CHANNELS(1),.weight_file("l1_weights.hex"))
    l1_compute(.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_l1_window),
    .s_axis_tvalid(w_l1_window_valid),.s_axis_tready(w_l1_window_ready),
    .layer_out(w_l1_mac_out),.m_axis_tvalid(w_l1_mac_valid),
    .m_axis_tready(w_maxpool_ready));
    
    max_polling #(.IN_CHANNELS(`L1_neurons))
         pool_block (.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_l1_mac_out),
         .s_axis_tvalid(w_l1_mac_valid),.s_axis_tready(w_maxpool_ready),
         .m_axis_tdata(w_pool_out),.m_axis_tvalid(w_pool_valid),
         .m_axis_tready(w_serializer_ready));
    
    // Layer-2 instances 
       
    l1_l2_buff #(.IN_CHANNELS(`L1_neurons),.SCALE_SHIFT(0))
         serializer_block(.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_pool_out),
         .s_axis_tvalid(w_pool_valid),.s_axis_tready(w_serializer_ready),
         .m_axis_tdata(w_serial_data),.m_axis_tvalid(w_serial_valid),
         .m_axis_tlast(w_serial_last),.m_axis_tready(w_l2_buffer_ready));
        
    line_buffer #(
        .KERNEL_SIZE(5), .NUM_CHANNELS(`L1_neurons) // Takes all 64 channels from L1
    ) L2_Buffer (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(w_serial_data), .s_axis_tvalid(w_serial_valid), 
        .s_axis_tlast(w_serial_last), .s_axis_tready(w_l2_buffer_ready),
        .m_axis_tdata(w_l2_window), .m_axis_tvalid(w_l2_window_valid), .m_axis_tready(w_l2_window_ready)
    );

    layers #(
        .neuron_no(`L2_neurons), .weights(`L2_weights), .KERNEL_SIZE(5), .NUM_CHANNELS(`L1_neurons),
        .weight_file("l2_weights.hex") 
    ) L2_Compute (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(w_l2_window), .s_axis_tvalid(w_l2_window_valid), 
        .s_axis_tready(w_l2_window_ready),.layer_out(w_l2_mac_out),
        .m_axis_tvalid(w_l2_mac_valid),.m_axis_tready(w_l2_maxpool_ready)
    );
    
    max_polling #(.IN_CHANNELS(`L2_neurons))
         pool_block_l2 (.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_l2_mac_out),
         .s_axis_tvalid(w_l2_mac_valid),.s_axis_tready(w_l2_maxpool_ready),
         .m_axis_tdata(w_l2_pool_out),.m_axis_tvalid(w_l2_pool_valid),
         .m_axis_tready(w_l2_serializer_ready));
     
     // Layer-3 instances    
         
    l1_l2_buff #(.IN_CHANNELS(`L2_neurons),.SCALE_SHIFT(8))
         serializer_block_l2(.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_l2_pool_out),
         .s_axis_tvalid(w_l2_pool_valid),.s_axis_tready(w_l2_serializer_ready),
         .m_axis_tdata(w_l2_serial_data),.m_axis_tvalid(w_l2_serial_valid),
         .m_axis_tlast(w_l2_serial_last),.m_axis_tready(w_l3_buffer_ready));
    
    line_buffer #(
        .KERNEL_SIZE(5), .NUM_CHANNELS(`L2_neurons)
    ) L3_Buffer (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(w_l2_serial_data), .s_axis_tvalid(w_l2_serial_valid), 
        .s_axis_tlast(w_l2_serial_last), .s_axis_tready(w_l3_buffer_ready),
        .m_axis_tdata(w_l3_window), .m_axis_tvalid(w_l3_window_valid),
        .m_axis_tready(w_l3_window_ready)
    );
    
    layers #(
        .neuron_no(`L3_neurons), .weights(`L3_weights), 
        .KERNEL_SIZE(5), .NUM_CHANNELS(`L2_neurons),
        .weight_file("l3_weights.hex") 
    ) L3_Compute (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(w_l3_window), .s_axis_tvalid(w_l3_window_valid), 
        .s_axis_tready(w_l3_window_ready),.layer_out(w_l3_mac_out),
        .m_axis_tvalid(w_l3_mac_valid),.m_axis_tready(w_l3_maxpool_ready)
    );
    
    max_polling #(.IN_CHANNELS(`L3_neurons))
         pool_block_l3 (.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_l3_mac_out),
         .s_axis_tvalid(w_l3_mac_valid),.s_axis_tready(w_l3_maxpool_ready),
         .m_axis_tdata(w_l3_pool_out),.m_axis_tvalid(w_l3_pool_valid),
         .m_axis_tready(w_l3_serializer_ready));

    // Layer-4 instances
    
   l3_l4_buff #(.IN_CHANNELS(`L3_neurons))
         serializer_block_l3(.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_l3_pool_out),
         .s_axis_tvalid(w_l3_pool_valid),.s_axis_tready(w_l3_serializer_ready),
         .m_axis_tdata(w_l3_serial_data),.m_axis_tvalid(w_l3_serial_valid),
         .m_axis_tlast(w_l3_serial_last),.s_axis_tlast(1'b0),
         .m_axis_tready(w_l4_ready));
   
   dense_layer #(.weight_file("l4_weights.hex")) 
         layer4_compute(.clk(clk),.rst_n(rst_n),.s_axis_tdata(w_l3_serial_data),
        .s_axis_tvalid(w_l3_serial_valid),.s_axis_tlast(w_l3_serial_last),
        .s_axis_tready(w_l4_ready),.m_axis_tdata(m_axis_tdata_out), // Final parallel output
        .m_axis_tvalid(m_axis_tvalid_out));
endmodule
