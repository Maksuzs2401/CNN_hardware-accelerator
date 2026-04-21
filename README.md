# CNN Hardware Accelerator for ECG Arrhythmia Classification
This project implements a 5-layer Convolutional neural network for real-time ECG heartbeat classification on an FPGA. The model is trained in python  
and implemented on an FPGA in systemverilog. The five different states of classifications are: Normal, Supraventricular, Ventricular,   
Fusion and Unknown/Paced.   

> **Current Status:** Active Development. The Python software training pipeline is under mdification, decided to reduce the number of dense layers to 1. The FPGA hardware pipeline is complete. **The FPGA model passed the test with dummy/known values.** The final testing is to be done on actual trained weights. 

## Project Team & Roles    
- **_Sujal Makwana (B.Tech ECE, LDCE):_** Lead Hardware Engineer    
– Contribution: Designing FPGA architecture, RTL design (SystemVerilog).  
- **_Devankit Shukla (DCP,Seneca Polytechnic):_** Lead Machine Learning Engineer  
– Contribution: ECG data preprocessing, software training in Python,   
and exporting the quantized weights for hardware use.
---  

## Architecture Overview
This project bridges the gap between software-based machine learning and hardware-accelerated digital logic.
The system pipeline is divided into three main phases: offline training, data handoff, and hardware simulation.  

### 1. Phase 1: Software Training & Quantization (Python)
Process: Raw ECG heartbeat datasets available on Kaggle are preprocessed and used to train a custom 1D Convolutional Neural Network (CNN) in Python.

Hardware Prep: Once the desired accuracy is achieved, the floating-point model is quantized into INT-8 format. 
The final weights and biases for all layers are exported as standard .hex and .mem files.

### 2. Phase 2: The Data Handoff (The Bridge)
The Python environment and the FPGA environment do not communicate live. Instead, the exported .hex and .mem files act as the static memory bridge.
The hardware testbench reads these files directly into the simulated FPGA Block RAM (BRAM) to initialize the neural network before 
the ECG signal begins streaming.

### 3. Phase 3: Hardware Simulation & Inference (SystemVerilog)
Process: A custom RTL architecture is designed to stream the ECG data through multiple pipelined hardware layers.
Current Architecture: The hardware utilizes line buffers, Multiply-Accumulate (MAC) arrays, and a "Wide-ROM" technique to process 
the Dense Layer efficiently. All the individual modules are verified seperately and in-loop. 

## FPGA SYSTEM 
<img width="900" height="522" alt="cnn_diagram" src="https://github.com/user-attachments/assets/e492ef39-0ec5-4d97-9762-96640a4e98fe" />
---
The hardware accelerator is a custom-designed, fully pipelined Convolutional Neural Network written in SystemVerilog, 
to achieve real-time ECG classification through deeply optimized data streaming, custom Multiply-Accumulate (MAC) 
arrays, and folded resource sharing.

### 1. Module Hierarchy & Data Flow
The top-level module `(neuron_wrapp.sv)` integrates all processing stages into a continuous streaming pipeline. 
Data enters as a serialized ECG stream and is passed through alternating layers of line buffers, compute 
nodes, and serializers.
** Top Level: ** `neuron_wrapp.sv`
** Compute Modules: ** `layers.sv`, `dense_layer.sv`, `neuron.sv`, `activation_funct.sv`, `argMAX.sv`
** Memory & Routing: ** `line_buffer.sv`, `max_polling.sv`, `l1_l2_buff.sv`, `l3_l4_buff.sv`

### 2. Data Representation & Hardware Quantization
To fit the model within FPGA logic limits without sacrificing speed, the network utilizes strict fixed-point arithmetic 
with custom hardware scaling: 
** Inputs & Weights: ** Processed as signed `data_width` integers.
** Accumulators: ** Extended to wider `accu_width` signed integers during deep MAC operations to prevent arithmetic overflow.
** Dynamic Scaling & Saturation `(l1_l2_buff.sv)`: ** Before parallel feature maps are passed to the next layer, 
the wide accumulated values are scaled back down using an arithmetic right shift `(>>> SCALE_SHIFT)`. 
To prevent catastrophic overflow wrapping, a hardware saturation limits values to a maximum of 
`8'sd127` before casting them back to the standard data width.

### 3. Core Architectural Mechanisms
*** A. Folded Convolution Architecture (`layers.sv` & `neuron.sv`) ***  
To conserve DSP slices, the convolutional layers utilize a time-multiplexed "folded" architecture. When a valid data window arrives, 
`layers.sv` captures a static input_snapshot of the entire receptive field. The module enters an `is_calculating` state, 
sequentially fetching weights from a dedicated Block RAM (`weight_rom`) and feeding them to the parallel MAC units 
(`neuron.sv`) over multiple clock cycles.The MAC units multiply the data (`(*use_dsp = "yes"*)`) and accumulate the
partial sums, asserting a valid output only on the final cycle (`s_axis_tlast`).

***B. Streaming 1D Line Buffers (`line_buffer.sv`)***  
Standard neural networks require loading full matrices into memory. This design uses a custom 2D shift-register memory (`shift_mem`) 
that continuously shifts incoming serial data. Once the buffer reaches the `KERNEL_SIZE`, it combinationally flattens the 2D grid 
into a 1D parallel array, allowing the MAC units to perform convolution immediately without stalling the data stream.

***C. Temporal Max Pooling (`max_polling.sv`)***  
The max-pooling layers achieve 2:1 data decimation natively in the data stream. By utilizing a lightweight state machine (`toggle_state`), 
the module captures the first sample in a temporary register. On the subsequent clock cycle, it evaluates 
`(s_axis_tdata[i] > temp_reg[i]) ? s_axis_tdata[i] : temp_reg[i]`, outputting the maximum value and resetting.

***D. Non-Linearity (`activation_funct.sv`)***  
The ReLU activation function is implemented using ultra-low-latency combinational logic. Instead of mathematical comparisons, it simply
checks the Most Significant Bit (MSB/Sign Bit) of the accumulator. If the MSB is 1 (negative), it outputs 0; otherwise, it passes 
the data.

***E. The "Wide-ROM" Dense Layer (`dense_layer.sv`)***  
The final fully connected layer requires parallel access to massive amounts of weights, which would normally bottleneck a standard 2-port 
BRAM.To bypass this, the design infers a custom ROM_WIDTH calculated as `neuron_no * data_width` (e.g., 512 bits wide). A single read 
address fetches an entire 512-bit block into a synchronization register.A generate loop then combinationally slices this wide bus 
(`[(i * data_width) +: data_width]`) to feed all 64 parallel neurons simultaneously in exactly one clock cycle.

***F. Final Classification (`argMAX.sv`)***   
The terminal stage of the accelerator uses a combinational comparison b the array of output classes. It iterates through the final dense 
features to locate the highest activated value (`current_maxval`), outputting its corresponding index (`current_idx`) as the final predicted
Arrhythmia class.  
**1) Normal,**   
**2) Supraventricular,**  
**3) Ventricular,**  
**4) Fusion,**  
**5) Unknown/Paced.**

# Design Summary

## Target Device
| Parameter | Value |
|-----------|-------|
| FPGA | Xilinx Artix-7 XC7A100T-CSG324-1 |
| Tool | Vivado 2024.1 |
| Clock Frequency | 100 MHz (10 ns period) |
| Top Module | `neuron_wrapp` |

## Network Architecture
| Layer | Type | Neurons | Kernel | Channels In | Weights/Neuron | Activation |
|-------|------|---------|--------|-------------|----------------|------------|
| L1 | Conv1D | 32 | 5 | 1 | 5 | ReLU |
| — | MaxPool (2×) | — | — | 32 | — | — |
| L2 | Conv1D | 64 | 5 | 32 | 160 | ReLU |
| — | MaxPool (2×) | — | — | 64 | — | — |
| L3 | Conv1D | 128 | 5 | 64 | 320 | ReLU |
| — | MaxPool (2×) | — | — | 128 | — | — |
| L4 | Dense | 5 | — | 2432 | 2432 | — |
| — | ArgMax | 5 classes | — | — | — | — |

## Resource Utilization (Post-Implementation)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| Slice LUTs | 24,940 | 63,400 | 39.34% |
| Slice Registers | 49,665 | 126,800 | 39.17% |
| Block RAM (RAMB36E1) | 24 | 135 | 17.78% |
| DSP48E1 | 229 | 240 | 95.42% |
| Bonded IOBs | 17 | 210 | 8.10% |
| BUFG | 1 | 32 | 3.13% |

## Timing Summary (Post-Implementation, Post `phys_opt_design`)
| Metric | Value | Status |
|--------|-------|--------|
| Clock Period | 10.000 ns | — |
| Clock Frequency | 100 MHz | — |
| Worst Negative Slack (Setup) | +0.038 ns | ✅ Met |
| Worst Hold Slack | +0.024 ns | ✅ Met |
| Worst Pulse Width Slack | +4.500 ns | ✅ Met |
| Failing Endpoints | 0 | ✅ |
| Total Endpoints | 100,159 | — |
| Critical Path | L3 BRAM → DSP48 → Accumulator | — |

## Power Estimate (Post-Route)
| Component | Power (W) |
|-----------|-----------|
| Clocks | 0.066 |
| Slice Logic | 0.048 |
| Signals | 0.208 |
| Block RAM | 0.044 |
| DSP | 0.245 |
| I/O | < 0.001 |
| **Dynamic Total** | **0.611** |
| Static | 0.094 |
| **Total On-Chip** | **0.705** |
| Junction Temperature | 28.2°C |

## I/O Interface
| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `clk` | Input | 1 | 100 MHz system clock |
| `rst_n` | Input | 1 | Active-low async reset |
| `s_axis_tdata_ecg` | Input | 8 | INT8 ECG sample (AXI-Stream) |
| `s_axis_tvalid_ecg` | Input | 1 | Input data valid |
| `s_axis_tlast_ecg` | Input | 1 | Last channel in group |
| `s_axis_tready_ecg` | Output | 1 | Backpressure to source |
| `predicted_class` | Output | 3 | Classification result (0–4) |
| `prediction_valid` | Output | 1 | Result ready strobe |

## Data Format
| Parameter | Value |
|-----------|-------|
| Input Precision | INT8 (signed) |
| Weight Precision | INT8 (signed) |
| Accumulator Width | 24-bit (L1–L3), 32-bit (L4) |
| Input Length | 187 samples per inference |
| Output Classes | 5 |

> Final documentation under development.

## FUTURE MODIFICATIONS
Reduce the number of layers to reduce the silicon area and overall power usage, while maintaining the sensitivity and specificity.
Reduce the critical path.
Modify the whole pipeline such that it can be deployed on resource constrained chip. 

