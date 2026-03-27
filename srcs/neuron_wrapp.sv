`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 26.03.2026 22:54:44
// Design Name: 
// Module Name: neuron_wrapp
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

module neuron_wrapp(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [`data_width-1:0] sense_data,
    output logic signed [`accu_width-1:0] final_out[`L2_neurons-1:0]
);

    logic signed [`accu_width-1:0] L1_L2wires[`L1_neurons-1:0];
    
    // ==========================================
    // LAYER 1 FSM
    // ==========================================
    logic [$clog2(`L1_weights)-1:0] mac_counter;
    logic l1_clear_accu; 
    logic l1_done;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            mac_counter   <= 0;
            l1_clear_accu <= 1'b0;
            l1_done       <= 1'b0;
        end else if(valid_in) begin
            l1_done <= 1'b0;
            
            if(mac_counter == 0) begin
                l1_clear_accu <= 1'b1;
                mac_counter   <= mac_counter + 1;
            end else if(mac_counter == `L1_weights - 1) begin
                l1_clear_accu <= 1'b0;
                mac_counter   <= 0;
                l1_done       <= 1'b1;
            end else begin 
                l1_clear_accu <= 1'b0;
                mac_counter   <= mac_counter + 1;
            end
        end else begin // FIXED: Added begin...end here
            l1_clear_accu <= 1'b0;
            l1_done       <= 1'b0;
        end
    end
    
    // Pipeline Delay
    logic [1:0] l1_done_delayed; 
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) l1_done_delayed <= 0;
        else       l1_done_delayed <= {l1_done_delayed[0], l1_done}; 
    end
    
    logic buffer_trigger;
    assign buffer_trigger = l1_done_delayed[1];

    // ==========================================
    // LAYER 1 INSTANCE
    // ==========================================
    layers #(
        .neuron_no(`L1_neurons),
        .weights(`L1_weights),
        .act_type(`L1_act),
        .weight_file("l1_weights.hex")
    ) layer_1 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .clear_accu(l1_clear_accu),
        .sensor_data(sense_data),
        .layer_out(L1_L2wires) 
    );

    // ==========================================
    // BUFFER (L1 to L2 Bridge)
    // ==========================================
    logic signed [`data_width-1:0] L2_serial_in;
    logic                          L2_valid_in;

    l1_l2_buff buffer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .l1_done(buffer_trigger),
        .l1_parallel_in(L1_L2wires),
        .l2_serial_out(L2_serial_in),     // 8-bit wire out to Layer 2
        .l2_valid(L2_valid_in)            // Wakes up Layer 2
    );

    // ==========================================
    // LAYER 2 FSM
    // ==========================================
    logic [$clog2(`L2_weights)-1:0] l2_mac_counter;
    logic l2_clear_accu;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l2_mac_counter <= 0;
            l2_clear_accu  <= 1'b0;
        end else if (L2_valid_in) begin  // Triggered by the buffer!
            if (l2_mac_counter == 0) begin
                l2_clear_accu  <= 1'b1; 
                l2_mac_counter <= l2_mac_counter + 1;
            end else if (l2_mac_counter == `L2_weights - 1) begin
                l2_clear_accu  <= 1'b0;
                l2_mac_counter <= 0; 
            end else begin
                l2_clear_accu  <= 1'b0;
                l2_mac_counter <= l2_mac_counter + 1;
            end
        end else begin
            l2_clear_accu <= 1'b0; 
        end
    end

    // ==========================================
    // LAYER 2 INSTANCE
    // ==========================================
    layers #(
        .neuron_no(`L2_neurons),
        .weights(`L2_weights),
        .act_type(`L2_act),               // FIXED: typo "act_typ" to "act_type"
        .weight_file("l2_weights.hex")
    ) layer_2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(L2_valid_in),          // FIXED: Uses buffer's valid signal
        .clear_accu(l2_clear_accu),      // FIXED: Uses Layer 2 FSM clear
        .sensor_data(L2_serial_in),      // FIXED: Uses buffer's 8-bit serial out
        .layer_out(final_out)
    );

endmodule