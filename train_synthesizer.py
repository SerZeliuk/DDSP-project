"""Train gated WaveNet on guitar dataset with snapped‑MIDI f0 + pitch loss."""
import os, numpy as np, tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import ModelCheckpoint

from audio_processor       import AudioProcessor, make_per_sample_features
from wavenet_synthesizer   import WaveNetSynthesizer

# ─── paths ───────────────────────────────────────────────────
GUITAR_WAV   = "DDSP Reaper/Guitar_dataset.wav"
WEIGHTS_FILE = "models/GuitarSynth.weights.h5"
STATS_FILE   = "models/feature_stats.npz"
os.makedirs("models", exist_ok=True)

# ─── constants ───────────────────────────────────────────────
SR=16000; HOP=512; FR=250; HARM=5
WIN=16384; STRIDE=8192; BATCH=1; EPOCHS=30; LR=1e-4

# ─── util: snap Hz → MIDI within ±50 cents ───────────────────
def hz_to_midi_quantised(f0_hz: np.ndarray):
    midi = 69 + 12*np.log2(np.maximum(f0_hz, 1e-6)/440.0)
    midi_q = np.round(midi)
    cents  = (midi - midi_q)*100
    midi_q[np.abs(cents) > 50] = 0   # treat far bins as unvoiced 0
    return midi_q.astype(np.float32)

# ─── build dataset ───────────────────────────────────────────
proc  = AudioProcessor(GUITAR_WAV, SR, HOP, FR, HARM)
audio, _  = proc.load_audio()
# per-sample features: [f0(Hz), harms, loud]
feats = make_per_sample_features(audio, proc)   
# replace first column with snapped MIDI & store auxiliary target
f0_hz = feats[:, 0]
feats[:,0] = hz_to_midi_quantised(f0_hz)

# normalise columns
mean, std = feats.mean(0), feats.std(0)+1e-6
feats = (feats-mean)/std
np.savez(STATS_FILE, mean=mean, std=std)
print("Saved feature stats →", STATS_FILE)

# prepare windowed tf.data
feat_ds  = tf.data.Dataset.from_tensor_slices(feats)
aud_ds   = tf.data.Dataset.from_tensor_slices(audio.astype(np.float32))
full_ds  = tf.data.Dataset.zip((feat_ds, aud_ds))
full_ds  = full_ds.window(WIN, STRIDE, drop_remainder=True)
full_ds  = full_ds.flat_map(lambda f,y: tf.data.Dataset.zip((f.batch(WIN), y.batch(WIN))))
full_ds  = full_ds.shuffle(512).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ─── model & custom loss ─────────────────────────────────────
feat_dim = feats.shape[1]
model = WaveNetSynthesizer(num_blocks=10, filters=64, kernel_size=2,
                           dilation_rates=[1,2,4,8,16,32])
model(tf.zeros((1,100,feat_dim)))   # build

mse = tf.keras.losses.MeanSquaredError()

@tf.function
def loss_fn(y_true, y_pred):
    # y_true: waveform (B,T)
    # Add aux pitch target as snapped MIDI in first channel of features
    f0_target = tf.expand_dims(feats[:,0], 0)  # quick broadcast
    f0_pred, audio_pred = model.split_outputs(y_pred)
    loss_wave = mse(y_true, audio_pred)
    loss_f0   = mse(f0_target, f0_pred)
    return loss_wave + 0.1*loss_f0

model.compile(optimizer=Adam(LR), loss=loss_fn)

ckpt = ModelCheckpoint(WEIGHTS_FILE, save_weights_only=True,
                       save_best_only=True, monitor="loss", verbose=1)

print("Starting training …")
model.fit(full_ds, epochs=EPOCHS, callbacks=[ckpt])
print("Training done. Weights →", WEIGHTS_FILE)
