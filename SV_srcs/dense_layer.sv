`timescale 1ns / 1ps
`include "config.vh"

module dense_layer #(parameter neuron_no=`L4_neurons,
                     parameter weights = `L4_weights,
                     parameter act_type = `L1_act,
                     parameter KERNEL_SIZE = 5,
                     parameter NUM_CHANNELS = 1,
                     parameter dense_width = 32,   
                     parameter string weight_file = "default.hex")(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [`data_width-1:0] s_axis_tdata,
    input  logic s_axis_tvalid,
    output logic s_axis_tready,
    input  logic s_axis_tlast,
    output logic signed [dense_width-1:0] m_axis_tdata[neuron_no-1:0],
    output logic m_axis_tvalid,
    input  logic m_axis_tready
);

    assign s_axis_tready = 1'b1;

    // ==========================================
    // 1. FLAT INT8 WEIGHT ROM (same format as layers.sv)
    //    Layout: neuron_0_weight_0, neuron_0_weight_1, ..., neuron_1_weight_0, ...
    // ==========================================
    localparam TOTAL_WEIGHTS = neuron_no * weights;

    (* rom_style = "block" *) logic signed [`data_width-1:0] weight_rom [0:TOTAL_WEIGHTS-1];

    initial begin
        $readmemh(weight_file, weight_rom);
    end

    // ==========================================
    // 2. ADDRESS COUNTER
    // ==========================================
    logic [11:0] addr_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_count <= 0;
        end else if (s_axis_tvalid) begin
            if (s_axis_tlast)
                addr_count <= 0;
            else
                addr_count <= addr_count + 1;
        end
    end

    // ==========================================
    // 3. PIPELINE REGISTER (1 cycle latency for ROM read)
    // ==========================================
    logic signed [`data_width-1:0] current_weight_reg [neuron_no-1:0];
    logic signed [`data_width-1:0] delayed_tdata;
    logic                          delayed_tvalid;
    logic                          delayed_tlast;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            delayed_tdata  <= 0;
            delayed_tvalid <= 1'b0;
            delayed_tlast  <= 1'b0;
            for (int i = 0; i < neuron_no; i++) current_weight_reg[i] <= 0;
        end else begin
            // Each neuron gets its own weight from the flat ROM
            for (int i = 0; i < neuron_no; i++) begin
                current_weight_reg[i] <= weight_rom[(i * weights) + addr_count];
            end
            // Delay data/control by 1 cycle to match ROM read
            delayed_tdata  <= s_axis_tdata;
            delayed_tvalid <= s_axis_tvalid;
            delayed_tlast  <= s_axis_tlast;
        end
    end

    // ==========================================
    // 4. NEURON ARRAY
    // ==========================================
    logic [neuron_no-1:0] w_valid_out;
    assign m_axis_tvalid = w_valid_out[0];

    genvar i;
    generate
        for (i = 0; i < neuron_no; i++) begin : gen_neurons
            neuron #(
                .data_width(`data_width),
                .accu_width(dense_width),
                .kernel_size(1)
            ) mac_inst (
                .clk(clk),
                .rst_n(rst_n),
                .s_axis_tdata(delayed_tdata),
                .s_axis_tdata_wgt(current_weight_reg[i]),
                .s_axis_tvalid(delayed_tvalid),
                .s_axis_tlast(delayed_tlast),
                .s_axis_tready(),
                .m_axis_tdata(m_axis_tdata[i]),
                .m_axis_tvalid(w_valid_out[i])
            );
        end
    endgenerate

endmodule