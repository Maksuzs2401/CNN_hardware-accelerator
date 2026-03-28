"""
step2_train.py
===============
Build and train a 1D CNN for ECG arrhythmia classification.

Input  : X_train.npy  (87554 beats, 187 samples each)
Output : ecg_cnn.keras (trained model)

5 classes (AAMI standard):
  0 = N  Normal
  1 = S  Supraventricular
  2 = V  Ventricular
  3 = F  Fusion
  4 = Q  Unknown/Paced

Run:
  pip install tensorflow scikit-learn matplotlib numpy
  python step2_train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


# ─── Settings ─────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "./processed"
MODEL_DIR      = "./model"
SEGMENT_LENGTH = 187
N_CLASSES      = 5
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 0.0003
AAMI_CLASSES   = ['N', 'S', 'V', 'F', 'Q']

os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(42)
np.random.seed(42)


# ─── Build the 1D CNN ─────────────────────────────────────────────────────────
def build_model():
    """
    1D CNN architecture for ECG beat classification.

    Why 1D Conv?
      ECG is a signal over time (not an image).
      Conv1D slides a small window along the 187 samples
      looking for patterns like peaks, dips, and shapes.

    3 blocks: each block finds deeper patterns than the last.
    """
    inputs = keras.Input(shape=(SEGMENT_LENGTH, 1), name="ecg_input")

    # ── Block 1: find basic patterns (peaks, dips) ────────────────────
    x = layers.Conv1D(32, kernel_size=5, padding='same', name='conv1')(inputs) #32 neurons 
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)
    # Shape: (93, 32)

    # ── Block 2: find complex patterns (wave shapes) ──────────────────
    x = layers.Conv1D(64, kernel_size=5, padding='same', name='conv2')(x) #64 neurons
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)
    x = layers.Dropout(0.2, name='drop2')(x)
    # Shape: (46, 64)

    # ── Block 3: find high-level features ────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding='same', name='conv3')(x) #128 neurons
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool3')(x)
    x = layers.Dropout(0.3, name='drop3')(x)
    # Shape: (23, 128)

    # Classifier head
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='drop4')(x)
    outputs = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return keras.Model(inputs, outputs, name="ECG_CNN_v2")


# ─── Load data ────────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    print(f"Loaded  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"Loaded  X_test : {X_test.shape}    y_test : {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ─── SMOTE oversampling ───────────────────────────────────────────────────────
def oversample(X_train, y_train):
    """
    SMOTE = Synthetic Minority Oversampling Technique.

    WHY we need this:
      N = 72471 beats  (82.8%)  ← model only learns this
      F =   641 beats  ( 0.7%)  ← model ignores this completely

    WHAT SMOTE does:
      Creates synthetic (fake but realistic) beats for minority classes
      until all 5 classes have equal numbers.

    RESULT:
      All 5 classes → ~72471 beats each
      Model is forced to learn F and Q properly.
    """
    print("\nApplying SMOTE oversampling ...")
    print("Class counts BEFORE:")
    counts = Counter(y_train.tolist())
    for i, cls in enumerate(AAMI_CLASSES):
        print(f"  {cls} (class {i}): {counts.get(i, 0):>6} beats")

    smote        = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("\nClass counts AFTER SMOTE:")
    counts_new = Counter(y_res.tolist())
    for i, cls in enumerate(AAMI_CLASSES):
        print(f"  {cls} (class {i}): {counts_new.get(i, 0):>6} beats")

    print(f"\nTotal training beats: {len(X_res)}")
    return X_res, y_res


# ─── Train ────────────────────────────────────────────────────────────────────
def train(model, X_train, y_train):
    # Add channel dimension: (N, 187) → (N, 187, 1)
    X_in = X_train[..., np.newaxis]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    callbacks = [
        # Stop if val_accuracy stops improving
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR if stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "ecg_cnn.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

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
    X_in   = X_test[..., np.newaxis]
    y_pred = np.argmax(model.predict(X_in, verbose=0), axis=1)

    print("\n── Test Set Results ─────────────────────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names=AAMI_CLASSES,
        digits=4,
    ))

    # Confusion matrix
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':6}", end="")
    for cls in AAMI_CLASSES:
        print(f"{cls:>8}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{AAMI_CLASSES[i]:6}", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()


# ─── Plot ────────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved → {path}")
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    X_train, y_train, X_test, y_test = load_data()

    # 2. Balance classes with SMOTE
    X_train, y_train = oversample(X_train, y_train)

    # 3. Build model
    model = build_model()

    # 4. Train
    history = train(model, X_train, y_train)

    # 5. Evaluate on real test data
    evaluate(model, X_test, y_test)

    # 6. Plot training curves
    plot_history(history)

    print(f"\nModel saved → {MODEL_DIR}/ecg_cnn.keras")
    print("Step 2 complete. Run step3_quantize.py next.")