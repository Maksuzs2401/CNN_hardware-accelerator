`timescale 1ns / 1ps

module dense_layer #(parameter neuron_no=`L4_neurons,
                     parameter weights = `L1_weights,
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
    output logic signed [dense_width-1:0]m_axis_tdata[neuron_no-1:0],
    output logic m_axis_tvalid,
    input  logic m_axis_tready
    );
    
    assign s_axis_tready = 1'b1;
    logic  [11:0]addr_count;
    
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            addr_count <= 0;
        end
        else if(s_axis_tvalid)begin
                if(s_axis_tlast)begin
                    addr_count <= 0;
                end
                else begin
                    addr_count <= addr_count + 1;
                end
        end
    end
    
localparam ROM_WIDTH = neuron_no * `data_width; // 64 * 8 = 512 bits wide
    
    // Explicit BRAM declaration
    (* rom_style = "block" *) logic [ROM_WIDTH-1:0] weight_rom [0:weights-1];
    
    initial begin
        $readmemh(weight_file, weight_rom);
    end
    
    // 3. PIPELINE SYNC (1 Cycle Latency)
    // ==========================================
    logic [ROM_WIDTH-1:0]          packed_weight_reg;
    logic signed [`data_width-1:0] delayed_tdata;
    logic                          delayed_tvalid;
    logic                          delayed_tlast;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            packed_weight_reg <= 0;
            delayed_tdata     <= 0;
            delayed_tvalid    <= 1'b0;
            delayed_tlast     <= 1'b0;
        end else begin
           
          packed_weight_reg <= weight_rom[addr_count];         // Fetching 512-bit data (Takes 1 cycle)
            
            // Delay incoming data and controls by 1 cycle to match the ROM
            delayed_tdata     <= s_axis_tdata;
            delayed_tvalid    <= s_axis_tvalid;
            delayed_tlast     <= s_axis_tlast;
        end
    end    
    logic [neuron_no-1:0] w_valid_out;
    assign m_axis_tvalid = w_valid_out[0];
    
    genvar i;
    generate
        for(i=0; i<neuron_no; i++)begin : gen_neurons
            logic signed [`data_width-1:0] w_weights;
            assign w_weights = packed_weight_reg[(i * `data_width) +: `data_width];
            
            neuron #(.data_width(`data_width),.accu_width(dense_width),
                     .kernel_size(1)) 
            mac_inst (.clk(clk),
                .rst_n(rst_n),
                      .s_axis_tdata(delayed_tdata),          // Broadcasting scalar data
                      .s_axis_tdata_wgt(w_weights),          // Scalar weight for this neuron
                      .s_axis_tvalid(delayed_tvalid),
                      .s_axis_tlast(delayed_tlast),
                      .s_axis_tready(),                     // Handled by top layer
                      .m_axis_tdata(m_axis_tdata[i]),       // Parallel output array
                      .m_axis_tvalid(w_valid_out[i]));
         end
     endgenerate
    
endmodule
