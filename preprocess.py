import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Default Settings
DATA_DIR = "./data"
OUTPUT_DIR = "./processed"
TRAIN_CSV = os.path.join(DATA_DIR,"mitbih_train.csv")
TEST_CSV = os.path.join(DATA_DIR,"mitbih_test.csv")
AAMI_CLASSES = ['N','S','V','F','Q']

# loading csv file
#mit_test = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\CNN\mitbih_test.csv")
#mit_train = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\CNN\mitbih_train.csv")

def load_csv(filepath):
    """
    Load CSV and spit it into X(signal) and Y(label).
    Each row = one heartbeat:
    X : coloumms 0-186 -> 187 ECG samples (float32,range 0-1)
    Y : column 187 -> class label (int 0,1,2,3,4)
    """
    print(f"loading {filepath}...")
    df = pd.read_csv(filepath,header=None)
    
    X = df.iloc[:,:-1].values.astype(np.float32)  # shape:(N,187)
    Y = df.iloc[:,-1].values.astype(np.float32)   # shape:(N,)
    print(f" Loaded {len(X)} beats")

    return X,Y

# Showing Class Distribution
def print_distribution(Y, name):
    print(f"\n{name} class distribution:")
    counts = Counter(Y.tolist())
    total  = len(Y)
    for i, cls in enumerate(AAMI_CLASSES):
        n   = counts.get(i, 0)
        bar = "█" * int(30 * n / total)
        print(f"  {cls} (class {i}): {n:>6} beats  ({100*n/total:5.1f}%)  {bar}")
 # Sanity Check 
def sanity_check(X, Y, name):
    print(f"\n── {name} Sanity Check ──────────────")
    print(f"  X shape     : {X.shape}")
    print(f"  Y shape     : {Y.shape}")
    print(f"  Signal range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Any NaN?    : {np.isnan(X).any()}")
    print(f"  Labels found: {np.unique(Y).tolist()}")

# Main
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load both CSV files
    X_train, Y_train = load_csv(TRAIN_CSV)
    X_test,  Y_test  = load_csv(TEST_CSV)

    # Sanity check
    sanity_check(X_train, Y_train, "Train")
    sanity_check(X_test,  Y_test,  "Test")

    # Class distribution
    print_distribution(Y_train, "Train")
    print_distribution(Y_test,  "Test")

    # Save as numpy arrays for step 2
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), Y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  Y_test)

    print(f"\nSaved 4 files to {OUTPUT_DIR}/")
    print("  X_train.npy   Y_train.npy")
    print("  X_test.npy    Y_test.npy")
    print("\nStep 1 complete. Run step2_train.py next.")