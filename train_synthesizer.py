# train_synthesizer.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from losses import SpectralLoss
from audio_processor import AudioProcessor
from ddsp_model import Encoder, Decoder, DDSPAutoencoder
from synths import HarmonicPlusNoiseSynth

# ─── Paths & Hyperparams ───────────────────────────────────────
GUITAR_WAV   = "DDSP Reaper/Guitar_dataset.wav"
MODELS_DIR   = "models"
WEIGHTS_FILE = os.path.join(MODELS_DIR, "GuitarDDSP.weights.h5")
os.makedirs(MODELS_DIR, exist_ok=True)

SR           = 16000
FRAME_RATE   = 250
HOP          = SR // FRAME_RATE        # 512
WINDOW_SEC   = 4
SAMPLES_WIN  = WINDOW_SEC * SR         # 64 000
FRAMES_WIN   = WINDOW_SEC * FRAME_RATE # 1000
BATCH        = 4
EPOCHS       = 50
LR           = 1e-4

# ─── 1) LOAD + EXTRACT FRAME-LEVEL FEATURES ────────────────────
proc = AudioProcessor(
    GUITAR_WAV,
    sr=SR,
    frame_rate=FRAME_RATE,
    model_capacity='full'
)
data      = proc.extract_features()
audio     = data['audio'].astype(np.float32)     # [N_samples]
f0_frames = data['f0'].astype(np.float32)        # [N_frames]
loudness  = data['loudness'].astype(np.float32)  # [N_frames]

# ─── 2) BUILD WINDOWED DATASETS ────────────────────────────────
def make_windows(x, size, step):
    return (
        tf.data.Dataset
          .from_tensor_slices(x)
          .window(size, step, drop_remainder=True)
          .flat_map(lambda w: w.batch(size))
    )

audio_windows    = make_windows(audio,    SAMPLES_WIN,  SAMPLES_WIN)
f0_windows       = make_windows(f0_frames,FRAMES_WIN,   FRAMES_WIN)
loudness_windows = make_windows(loudness, FRAMES_WIN,   FRAMES_WIN)

ds = tf.data.Dataset.zip((audio_windows, f0_windows, loudness_windows))

# ─── 3) COMPUTE steps_per_epoch ───────────────────────────────
NUM_WINDOWS = len(audio) // SAMPLES_WIN
print(f"Computed NUM_WINDOWS = {NUM_WINDOWS}")

# ─── 4) PREPARE FOR TRAINING ───────────────────────────────────
ds = (
    ds
    .shuffle(512)
    .repeat()
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

# ─── 5) PACK INPUTS & LABELS ───────────────────────────────────
def pack(aud_batch, f0_batch, loud_batch):
    # Add channel dims for f0 & loudness
    f0_in   = f0_batch[..., tf.newaxis]   # [B, FRAMES_WIN, 1]
    loud_in = loud_batch[..., tf.newaxis] # [B, FRAMES_WIN, 1]

    # Return a dict mapping input names to tensors, plus target audio
    x = {
        "audio":    aud_batch,  # model arg #1
        "f0":       f0_in,      # model arg #2
        "loudness": loud_in     # model arg #3
    }
    return x, aud_batch

ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)

# Optional sanity-check:
print("DATASET ELEMENT SPEC:", ds.element_spec)
# Should print something like:
# ({'audio': TensorSpec((None,64000),float32),
#   'f0':    TensorSpec((None,1000,1),float32),
#   'loudness':TensorSpec((None,1000,1),float32)},
#  TensorSpec((None,64000),float32))

# ─── 6) BUILD DDSP AUTOENCODER & FUNCTIONAL TRAINING MODEL ─────
# Instantiate your components
enc   = Encoder(conv_channels=64, num_layers=4, kernel_size=3, latent_dim=128)
dec   = Decoder(latent_dim=128, n_harmonics=64, hidden_units=256, upsample_rate=HOP)
synth = HarmonicPlusNoiseSynth(
    n_samples=SAMPLES_WIN,
    sample_rate=SR,
    n_harmonics=64,
    window_size=257
)

# Core autoencoder (subclassed)
ae = DDSPAutoencoder(enc, dec, synth)

# Build a functional Model so Keras knows there are 3 separate inputs
audio_in = tf.keras.Input(shape=(SAMPLES_WIN,),     name="audio")
f0_in    = tf.keras.Input(shape=(FRAMES_WIN,1),     name="f0")
loud_in  = tf.keras.Input(shape=(FRAMES_WIN,1),     name="loudness")
recon    = ae(audio_in, f0_in, loud_in)             # calls ae.call(...)
model    = tf.keras.Model(
    inputs=[audio_in, f0_in, loud_in],
    outputs=recon,
    name="ddsp_autoencoder"
)

model.summary()

# ─── 7) LOSS & COMPILATION ─────────────────────────────────────
spectral_loss = SpectralLoss(
    fft_sizes=(2048,1024,512,256),
    loss_type='L1',
    mag_weight=1.0,
    logmag_weight=1.0
)
def loss_fn(y_true, y_pred):
    return spectral_loss(y_true, y_pred)

model.compile(
    optimizer=Adam(LR),
    loss=loss_fn
)

ckpt = ModelCheckpoint(
    WEIGHTS_FILE,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss',
    verbose=1
)

# ─── 8) TRAIN ───────────────────────────────────────────────────
print("Starting training…")
model.fit(
    ds,
    epochs=EPOCHS,
    steps_per_epoch=NUM_WINDOWS // BATCH,
    callbacks=[ckpt]
)
print("Training complete. Weights saved to", WEIGHTS_FILE)
