# evaluate_model.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from timbre_transfer_trainer import TimbreTransferTrainer
from sklearn.metrics import mean_squared_error

# ── CONFIG ─────────────────────────────────────────────
FLUTE_FILE    = 'DDSP Reaper/DDSP Reaper_stems_Viloin-001.wav'
GUITAR_FILE   = 'DDSP Reaper/Guitar_dataset.wav'
MODEL_PATH    = 'models/Guitar_model.h5'

# These must match what you used in train_models.py
SR            = 16000
HOP_LENGTH    = 512
FRAME_RATE    = 250
NUM_HARMONICS = 5

# ── EXTRACT TRAINING FEATURES ──────────────────────────
trainer = TimbreTransferTrainer(
    source_files  = [FLUTE_FILE],
    target_files  = [GUITAR_FILE],
    sr            = SR,
    hop_length    = HOP_LENGTH,
    frame_rate    = FRAME_RATE,
    num_harmonics = NUM_HARMONICS
)

print("Preparing feature dataset for evaluation…")
X_train, Y_train = trainer.prepare_dataset()
print(f"  → X_train shape: {X_train.shape}")
print(f"  → Y_train shape: {Y_train.shape}")

# ── LOAD MODEL & PREDICT ───────────────────────────────
model = load_model(MODEL_PATH)
print("\nModel summary:")
model.summary()

print("\nRunning forward pass on training data…")
Y_pred = model.predict(X_train, batch_size=32, verbose=1)

# ── COMPUTE & PRINT f0 MSE ─────────────────────────────
true_f0 = Y_train[:, 0]
pred_f0 = Y_pred[:, 0]
mse_f0 = mean_squared_error(true_f0, pred_f0)
print(f"\nFundamental-frequency MSE on training set: {mse_f0:.4f} Hz^2")

# ── PLOT true vs pred ──────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(true_f0[:500], label='target f0 (true)', linewidth=2)
plt.plot(pred_f0[:500], label='predicted f0', alpha=0.8)
plt.title('Target vs Predicted f₀ (first 500 frames)')
plt.xlabel('Frame index')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.tight_layout()
plt.show()

# Optionally, check one harmonic (e.g. H1 at index=1)
true_h1 = Y_train[:,1]
pred_h1 = Y_pred[:,1]
mse_h1  = mean_squared_error(true_h1, pred_h1)
print(f"Harmonic-1 amplitude MSE: {mse_h1:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(true_h1[:500], label='target H1', linewidth=2)
plt.plot(pred_h1[:500], label='pred H1', alpha=0.8)
plt.title('Target vs Predicted First Harmonic (H1)')
plt.xlabel('Frame index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
