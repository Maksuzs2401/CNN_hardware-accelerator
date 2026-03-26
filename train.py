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
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─── Settings ─────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "./processed"
MODEL_DIR      = "./model"
SEGMENT_LENGTH = 187
N_CLASSES      = 5
BATCH_SIZE     = 64
EPOCHS         = 30
LEARNING_RATE  = 0.001
AAMI_CLASSES   = ['N', 'S', 'V', 'F', 'Q']

os.makedirs(MODEL_DIR, exist_ok=True)


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
    # Shape: (93, 32)

    # ── Block 2: find complex patterns (wave shapes) ──────────────────
    x = layers.Conv1D(64, kernel_size=5, padding='same', name='conv2')(x) #64 neurons
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)
    # Shape: (46, 64)

    # ── Block 3: find high-level features ────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding='same', name='conv3')(x) #128 neurons
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool3')(x)
    # Shape: (23, 128)

    # ── Classifier head ───────────────────────────────────────────────
    x = layers.Flatten(name='flatten')(x)           # → 2944
    x = layers.Dense(128, activation='relu',
                     name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)      # prevent overfitting
    outputs = layers.Dense(N_CLASSES,
                           activation='softmax',
                           name='output')(x)         # → 5 class probabilities

    model = keras.Model(inputs, outputs, name="ECG_CNN")
    return model


# ─── Load data ────────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Add channel dimension: (N, 187) → (N, 187, 1)
    # Required by Conv1D
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ─── Class weights (fix imbalance) ───────────────────────────────────────────
def get_class_weights(y_train):
    """
    MIT-BIH is ~90% class N (Normal).
    Without class weights the model just predicts N for everything.
    Class weights penalize wrong predictions on minority classes more.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    print("\nClass weights:")
    for i, cls in enumerate(AAMI_CLASSES):
        print(f"  {cls}: {cw.get(i, 0):.3f}")
    return cw


# ─── Train ────────────────────────────────────────────────────────────────────
def train(model, X_train, y_train, class_weights):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    callbacks = [
        # Stop early if val_loss stops improving (saves time)
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        # Reduce learning rate if stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        # Save best model to disk
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "ecg_cnn.keras"),
            monitor='val_loss', save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,        # use 10% of train as validation
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─── Evaluate ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n── Test Set Evaluation ─────────────────────────────")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(classification_report(
        y_test, y_pred,
        target_names=AAMI_CLASSES,
        digits=4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ─── Plot training curves ────────────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    print(f"\nTraining plot saved → {path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Class weights
    class_weights = get_class_weights(y_train)

    # Build model
    model = build_model()

    # Train
    history = train(model, X_train, y_train, class_weights)

    # Evaluate on test set
    evaluate(model, X_test, y_test)

    # Plot
    plot_history(history)

    print(f"\nModel saved → {MODEL_DIR}/ecg_cnn.keras")
    print("Step 2 complete. Run step3_quantize.py next.")