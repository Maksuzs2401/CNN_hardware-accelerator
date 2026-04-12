`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 26.03.2026 15:30:58
// Design Name: 
// Module Name: layers
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

module layers #(parameter        neuron_no = `L1_neurons,
                                 weights = `L1_weights,
                                 act_type = `L1_act,
                                 KERNEL_SIZE = 5,
                                 NUM_CHANNELS = 1,
                parameter string weight_file = "default.hex")(
    input  logic clk,
    input  logic rst_n,
    // Split control signals for the Folded Architecture
    input  logic signed [`data_width-1:0] s_axis_tdata [(KERNEL_SIZE * NUM_CHANNELS)-1:0],
    input  logic s_axis_tvalid,
    output logic s_axis_tready,
    output logic signed [`accu_width-1:0] layer_out[neuron_no-1:0],
    output logic m_axis_tvalid,
    input  logic m_axis_tready
);
    // 1. Weight ROM Instantiation
    localparam TOTAL_WEIGHTS = neuron_no * weights;
    
    (* rom_style = "block" *) logic signed [`data_width-1:0] weight_rom [0:TOTAL_WEIGHTS-1];
    
    initial begin
        $readmemh(weight_file, weight_rom);
    end
    
    logic is_calculating;
    logic [$clog2(weights)-1:0] current_ptr;
    
    assign s_axis_tready = ~is_calculating && m_axis_tready;
    logic signed [`data_width-1:0] input_snapshot [(KERNEL_SIZE * NUM_CHANNELS)-1:0];
   
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            is_calculating <= 1'b0;
            current_ptr <= 0;
        end else begin
            if(s_axis_tready && s_axis_tvalid) begin
                is_calculating <= 1'b1;
                current_ptr <= 0;
                for (int i = 0; i < (KERNEL_SIZE * NUM_CHANNELS); i++) begin
                input_snapshot[i] <= s_axis_tdata[i];
                end
            end else if(is_calculating) begin
                if(current_ptr == weights -1)begin
                    is_calculating <= 1'b0;
                    current_ptr <= 0;
                end else begin
                    current_ptr <= current_ptr + 1;
                end
            end
        end
    end
    
    // ==========================================
    // 3. PIPELINE SYNCHRONIZATION (1 Cycle Latency)
    // ==========================================
    logic signed [`data_width-1:0] current_weight_reg [neuron_no-1:0];
    logic signed [`data_width-1:0] current_input_reg;
    logic neuron_tvalid_reg;
    logic neuron_tlast_reg;
    // NEW: Delay register to keep MAC alive 1 extra cycle

    // ==========================================
    // 4. PIPELINE SYNCHRONIZATION (1 Cycle Latency)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            neuron_tvalid_reg <= 1'b0; // HARD RESET
            neuron_tlast_reg  <= 1'b0;
            current_input_reg <= 0;
            // Clear weight regs to prevent 'X' propagation
            for(int i=0; i<neuron_no; i++) current_weight_reg[i] <= 0;
        end else begin
            // ONLY allow signals to propagate if we are actually in the CALC state
            if (is_calculating) begin
                for(int i=0; i<neuron_no; i=i+1) begin
                    current_weight_reg[i] <= weight_rom[(i * weights) + current_ptr];
                end
                //current_input_reg <= s_axis_tdata[current_ptr];
                current_input_reg <= input_snapshot[current_ptr];
                neuron_tvalid_reg <= 1'b1;
                neuron_tlast_reg  <= (current_ptr == weights - 1) ? 1'b1 : 1'b0;
            end else begin
                // FORCE SILENCE when not calculating
                neuron_tvalid_reg <= 1'b0;
                neuron_tlast_reg  <= 1'b0;
            end
        end
    end
  genvar i;
  generate
        for(i=0;i<neuron_no;i=i+1)begin : neuron_array
          
          logic signed [`accu_width-1:0]raw_sum;
          logic individual_valid;
          
          neuron #(.data_width(`data_width),.accu_width(`accu_width))
          neuron_inst(
            .clk(clk),.rst_n(rst_n),
            .s_axis_tdata(current_input_reg),
            .s_axis_tdata_wgt(current_weight_reg[i]),
            .s_axis_tvalid   (neuron_tvalid_reg),
            .s_axis_tlast    (neuron_tlast_reg),
            .s_axis_tready   (), // Left unconnected, neuron is always ready  
            .m_axis_tdata    (raw_sum),
            .m_axis_tvalid   (individual_valid)
          );                
          
          if(act_type == "relu")begin
            activation_funct relu_inst(.data_in(raw_sum),
            .data_out(layer_out[i]));
          end else
            assign layer_out[i] = raw_sum;
        end
   endgenerate
   
   assign m_axis_tvalid = neuron_array[0].individual_valid;
endmodule
