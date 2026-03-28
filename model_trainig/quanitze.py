
"""
step3_quantize.py
==================
Convert trained CNN weights from float32 → int8.

Why?
  Your SV neuron hardware uses 8-bit signed integers (data_width=8).
  The trained CNN uses 32-bit floats.
  We must convert (quantize) the weights to int8 so they
  match exactly what the hardware expects.

How (symmetric quantization)?
  scale     = max(|weight|) / 127
  int8_weight = round(weight / scale)
  range: -128 to 127

Run:
  python step3_quantize.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.config.list_physical_devices('GPU'))
# ─── Settings ─────────────────────────────────────────────────────────────────
MODEL_PATH    = "./model/ecg_cnn.keras"
PROCESSED_DIR = "./processed"
OUTPUT_DIR    = "./quantized"
AAMI_CLASSES  = ['N', 'S', 'V', 'F', 'Q']

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Quantize float32 array to int8 ──────────────────────────────────────────
def quantize_to_int8(weights):
    """
    Symmetric min-max quantization.

    Example:
      weight   = 0.0234   (float32)
      max_val  = 0.75
      scale    = 0.75 / 127 = 0.00591
      int8_val = round(0.0234 / 0.00591) = round(3.96) = 4

    To recover float: 4 * 0.00591 = 0.02364  (close to original)
    """
    max_val = np.max(np.abs(weights))

    if max_val == 0:
        return np.zeros_like(weights, dtype=np.int8), 1.0

    scale     = max_val / 127.0
    q_weights = np.clip(np.round(weights / scale), -128, 127)
    return q_weights.astype(np.int8), float(scale)


# ─── Quantize all layers ──────────────────────────────────────────────────────
def quantize_model(model):
    """
    Go through every Conv1D and Dense layer.
    Quantize kernel (weights) and bias separately.
    Save each as .npy file.
    """
    print("\n── Quantizing Layers ────────────────────────────────")
    print(f"{'Layer':<22} {'Shape':<18} {'Float range':<22} {'Scale'}")
    print("─" * 80)

    quantized = {}

    for layer in model.layers:
        if not isinstance(layer, (keras.layers.Conv1D, keras.layers.Dense)):
            continue

        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            continue

        kernel = layer_weights[0]
        bias   = layer_weights[1] if len(layer_weights) > 1 else None

        q_kernel, k_scale = quantize_to_int8(kernel)
        q_bias,   b_scale = quantize_to_int8(bias) if bias is not None else (None, 1.0)

        quantized[layer.name] = {
            'kernel':       q_kernel,
            'bias':         q_bias,
            'kernel_scale': k_scale,
            'bias_scale':   b_scale,
        }

        # Save to disk
        np.save(os.path.join(OUTPUT_DIR, f"{layer.name}_kernel_int8.npy"), q_kernel)
        if q_bias is not None:
            np.save(os.path.join(OUTPUT_DIR, f"{layer.name}_bias_int8.npy"), q_bias)

        fmin = kernel.min()
        fmax = kernel.max()
        print(f"{layer.name:<22} {str(kernel.shape):<18} [{fmin:+.4f}, {fmax:+.4f}]        {k_scale:.6f}")

    return quantized


# ─── Show quantization error ──────────────────────────────────────────────────
def quantization_error(model, quantized):
    """
    Compare original float weight vs dequantized weight.
    Small MSE = good quantization (little information lost).
    """
    print("\n── Quantization Error (MSE) ─────────────────────────")
    for layer in model.layers:
        if layer.name not in quantized:
            continue
        original = layer.get_weights()[0]
        q        = quantized[layer.name]
        recovered = q['kernel'].astype(np.float32) * q['kernel_scale']
        mse = np.mean((original - recovered) ** 2)
        print(f"  {layer.name:<22} MSE = {mse:.8f}")


# ─── Export first layer weights for SV neuron testbench ──────────────────────
def export_for_hardware(quantized, layer_name='conv1'):
    """
    The SV neuron receives one weight per clock cycle (serial input).
    We flatten the conv1 kernel and write each weight as a signed integer.

    Your testbench reads these values as my_weight input.
    """
    if layer_name not in quantized:
        print(f"[WARN] Layer {layer_name} not found.")
        return

    kernel      = quantized[layer_name]['kernel']   # shape: (5, 1, 32)
    kernel_flat = kernel.flatten().tolist()

    hw_path = os.path.join(OUTPUT_DIR, f"{layer_name}_weights_for_hw.txt")
    with open(hw_path, 'w') as f:
        f.write(f"# Layer: {layer_name}\n")
        f.write(f"# Kernel shape: {kernel.shape}\n")
        f.write(f"# Total weights: {len(kernel_flat)}\n")
        f.write(f"# Format: signed int8 decimal (one per line)\n")
        f.write(f"# Feed these as my_weight to SV neuron serially\n#\n")
        for w in kernel_flat:
            f.write(f"{int(w)}\n")

    print(f"\nHardware file saved → {hw_path}")
    print(f"  Total weights : {len(kernel_flat)}")
    print(f"  Value range   : [{int(min(kernel_flat))}, {int(max(kernel_flat))}]")


# ─── Quick inference check ────────────────────────────────────────────────────
def inference_check(model, X_test, y_test, n=10):
    """Run a few test beats and show predictions."""
    X_in  = X_test[:n][..., np.newaxis] if X_test.ndim == 2 else X_test[:n]
    preds = model.predict(X_in, verbose=0)

    print(f"\n── Sample Predictions ({n} beats) ───────────────────")
    print(f"  {'#':<4} {'True':>5} {'Predicted':>10} {'Confidence':>12}")
    print("  " + "─" * 34)
    for i in range(n):
        true_cls = AAMI_CLASSES[int(y_test[i])]
        pred_cls = AAMI_CLASSES[np.argmax(preds[i])]
        conf     = np.max(preds[i])
        match    = "✓" if true_cls == pred_cls else "✗"
        print(f"  {i:<4} {true_cls:>5} {pred_cls:>10} {conf:>11.1%}  {match}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load model
    print(f"Loading model from {MODEL_PATH} ...")
    model = keras.models.load_model(MODEL_PATH)

    # Quantize all layers
    quantized = quantize_model(model)

    # Show error
    quantization_error(model, quantized)

    # Quick inference
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    inference_check(model, X_test, y_test)

    # Export first layer for hardware
    export_for_hardware(quantized, layer_name='conv1')

    print(f"\nAll quantized weights saved → {OUTPUT_DIR}/")
    print("Step 3 complete. Run step4_verify.py next.")