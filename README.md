# CNN Hardware Accelerator for ECG Arrhythmia Classification
This project implements a 5-layer Convolutional neural network for real-time ECG heartbeat classification on an FPGA. The model is trained in python  
and implemented on FPGA in systemverilog. The four different states of classifications are: Normal, Supraventricular, Ventricular,   
Fusion and Unknown/Paced.   

> **Current Status:** Active Development. The Python software training pipeline is complete. The FPGA hardware pipeline is complete
up to Layer 4 (Dense Layer). The final Argmax classification layer is currently being implemented in RTL.

## Project Team & Roles    
- **_Sujal Makwana:_** Lead Hardware Engineer    
– Contribution: Designing FPGA architecture, RTL design (SystemVerilog).  
- **_Devankit Shukla:_** Lead Machine Learning Engineer  
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
(Note: The final Argmax classification layer to process these 64 features into a single prediction is currently under active development).

## FPGA SYSTEM 
<img width="1169" height="522" alt="CNN drawio" src="https://github.com/user-attachments/assets/8b257261-54b0-489e-b89d-7518d8ba3f80" />
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

## Future Modifications
Reduce the number of layers to reduce the silicon area and overall power usage.
Reduce the critical path.
Modify the whole pipeline such that it can be deployed on resource constrained chip. 

