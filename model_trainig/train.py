"""
step2_train.py — Matched to Hardware Architecture
=====================================================
Architecture matches FPGA hardware EXACTLY as per README:

L1 : Conv1D(32, kernel=5, channels_in=1) → ReLU → MaxPool(2)
L2 : Conv1D(64, kernel=5, channels_in=32) → ReLU → MaxPool(2)
L3 : Conv1D(128, kernel=5, channels_in=64) → ReLU → MaxPool(2)
L4 : Dense(5, inputs=2432) → ArgMax

Shape flow:
Input : (187, 1)
After Conv1 : (187, 32) → MaxPool → (93, 32)
After Conv2 : (93, 64) → MaxPool → (46, 64)
After Conv3 : (46, 128) → MaxPool → (23, 128)
After Flatten: (2944,) ← NOTE: README says 2432
This is because hardware uses
padding='valid' not 'same'
We use padding='valid' to match.
Dense input : (2432,) ← matches hardware exactly
Output : (5,)

Weight counts (INT8 after quantization):
L1 kernel : 5 × 1 × 32 = 160
L1 bias : 32 = 32
L2 kernel : 5 × 32 × 64 = 10,240
L2 bias : 64 = 64
L3 kernel : 5 × 64 × 128 = 40,960
L3 bias : 128 = 128
L4 kernel : 2432 × 5 = 12,160
L4 bias : 5 = 5
─────────────────────────────────────
Total : = 63,749 (~63 KB)

Accumulator widths (matching hardware):
L1-L3 : 24-bit
L4 : 32-bit

Run:
pip install imbalanced-learn tensorflow scikit-learn matplotlib
python step2_train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None
    print("imblearn not found. Install with 'pip install imbalanced-learn' to enable SMOTE.")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# ─── Settings ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = "./processed"
MODEL_DIR = "./model"
SEGMENT_LENGTH = 187
N_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0003
AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']

os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(42)
np.random.seed(42)


# ─── Build model ─────────────────────────────────────────────────────────────
def build_model():
    inputs = keras.Input(shape=(SEGMENT_LENGTH, 1), name="ecg_input")

    # L1
    x = layers.Conv1D(32, 5, padding='valid', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling1D(2, name='pool1')(x)
    x = layers.Dropout(0.1, name='drop1')(x)

    # L2
    x = layers.Conv1D(64, 5, padding='valid', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling1D(2, name='pool2')(x)
    x = layers.Dropout(0.1, name='drop2')(x)

    # L3
    x = layers.Conv1D(128, 5, padding='valid', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.MaxPooling1D(2, name='pool3')(x)
    x = layers.Dropout(0.2, name='drop3')(x)

    # Flatten
    x = layers.Flatten(name='flatten')(x)

    # Dense
    x = layers.Dropout(0.4, name='drop4')(x)
    outputs = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name="ECG_CNN_HW_MATCHED")
    return model


# ─── Load data ───────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    print(f"X_train : {X_train.shape} y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape} y_test  : {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ─── SMOTE ───────────────────────────────────────────────────────────────────
def oversample(X_train, y_train):
    print("\nApplying SMOTE to balance classes ...")
    print("Before:")

    counts = Counter(y_train.tolist())
    for i, cls in enumerate(AAMI_CLASSES):
        n = counts.get(i, 0)
        bar = "█" * int(20 * n / len(y_train))
        print(f" {cls}: {n:>6} {bar}")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("After:")
    counts_new = Counter(y_res.tolist())
    for i, cls in enumerate(AAMI_CLASSES):
        print(f" {cls}: {counts_new.get(i, 0):>6}")

    print(f"Total training samples: {len(X_res)}")
    return X_res, y_res


# ─── Print architecture ──────────────────────────────────────────────────────
def print_architecture(model):
    print("\n── Architecture ──────────────────")
    print(f"{'Layer':<14} {'Output Shape':<20} {'Weights':>10} {'Bias':>8} {'Total':>10}")
    print(" " + "─" * 66)

    total = 0
    for layer in model.layers:
        # Handle layers without output_shape (e.g. InputLayer in newer Keras)
        try:
            out_shape = str(layer.output_shape[1:])
        except AttributeError:
            try:
                out_shape = str(layer.output.shape[1:])
            except AttributeError:
                out_shape = "—"

        w = layer.get_weights()

        if len(w) == 0:
            print(f"{layer.name:<14} {out_shape:<20} {'—':>10} {'—':>8} {'—':>10}")
            continue

        kernel = w[0].size
        bias = w[1].size if len(w) > 1 else 0
        t = kernel + bias
        total += t

        print(f"{layer.name:<14} {out_shape:<20} {kernel:>10} {bias:>8} {t:>10}")

    print(" " + "─" * 66)
    print(f"{'TOTAL':<14} {'':<20} {'':>10} {'':>8} {total:>10} (~{total/1024:.1f} KB)")

# ─── Train ───────────────────────────────────────────────────────────────────
def train(model, X_train, y_train):
    X_in = X_train[..., np.newaxis]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "ecg_cnn.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_in, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ─── Evaluate ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    X_in = X_test[..., np.newaxis]
    y_pred = np.argmax(model.predict(X_in, verbose=0), axis=1)

    print("\n── Test Results ─────────────────")
    print(classification_report(y_test, y_pred, target_names=AAMI_CLASSES, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


# ─── Plot ────────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Accuracy')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_data()

    X_train_bal, y_train_bal = oversample(X_train, y_train)

    model = build_model()

    print_architecture(model)

    history = train(model, X_train_bal, y_train_bal)

    evaluate(model, X_test, y_test)

    plot_history(history)

    print(f"\nModel saved → {MODEL_DIR}/ecg_cnn.keras")
print("Step 2 complete. Run step3_quantize.py next.")
