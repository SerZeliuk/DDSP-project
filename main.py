# main.py

import os
import numpy as np
import soundfile as sf
import tensorflow as tf

from audio_processor import AudioProcessor
from wavenet_synthesizer import WaveNetSynthesizer
import visualize

# ─── Hyperparameters ─────────────────────────────────────────
SR         = 16000

# ─── Paths ────────────────────────────────────────────────────
INPUT_WAV   = "DDSP Reaper/DDSP Reaper_stems_Viloin-001.wav"
OUTPUT_WAV  = 'out/violin_to_guitar_work_please.wav'
WEIGHTS     = 'models/GuitarSynthEncoder.weights.h5'
STATS_FILE  = 'models/feature_stats.npz'

os.makedirs(os.path.dirname(OUTPUT_WAV), exist_ok=True)

# ─── Load normalization stats ─────────────────────────────────
stats = np.load(STATS_FILE)
mean, std = stats['mean'], stats['std']

# ─── Load audio + frame-level features via extract_features() ────
proc = AudioProcessor(
    INPUT_WAV,
    sr=SR,
    frame_rate=250,          # matches your training FRAME_RATE
    model_capacity='full'    # or 'tiny', etc.
)
data = proc.extract_features()
audio_in     = data['audio']        # [N_samples]
f0_frames    = data['f0']           # [N_frames]
conf_frames  = data['confidence']    # [N_frames]
loud_frames  = data['loudness']      # [N_frames]
hop_length   = data['hop_length']    # samples per frame

# ─── Build per-sample feature array by repeating each frame ───────
frame_feats = np.stack([f0_frames, conf_frames, loud_frames], axis=1)
feats_in = np.repeat(frame_feats, hop_length, axis=0)
feats_in = feats_in[: len(audio_in)]  # trim to match audio length

# ─── Normalize using training stats ──────────────────────────────
feats_norm = (feats_in - mean) / std
n_samples, feat_dim = feats_norm.shape

# ─── Visualize the input diagnostics ─────────────────────────────
visualize.plot_input(
    audio=audio_in,
    sr=SR,
    f0=f0_frames,
    confidence=conf_frames,
    loudness=loud_frames,
    hop_length=hop_length,
    frame_rate=int(SR / hop_length)
)

# ─── Build & load your WaveNet model ─────────────────────────────
model = WaveNetSynthesizer(
    num_blocks=10,
    filters=64,
    kernel_size=2,
    dilation_rates=[1, 2, 4, 8, 16, 32]
)
_ = model(tf.zeros((1, 100, feat_dim)))  # build
model.load_weights(WEIGHTS)

# ─── Run inference ───────────────────────────────────────────────
y = model.predict(feats_norm[None, ...], batch_size=1)[0]  # [T, 2]
f0_aux, audio_out = WaveNetSynthesizer.split_outputs(y)
audio_out = audio_out[:n_samples]

# ─── Recompute output frame-level features for plotting ──────────
# We’ll use the same hop_length to downsample audio_out
loud_out = proc.compute_loudness(audio_out)

# ─── Visualize the output diagnostics ────────────────────────────
visualize.plot_output(
    audio=audio_out,
    sr=SR,
    f0=f0_aux,
    loudness=loud_out,
    hop_length=hop_length,
    frame_rate=int(SR / hop_length)
)

# ─── Save the synthesized audio ──────────────────────────────────
sf.write(OUTPUT_WAV, audio_out, SR)
print(f"Saved → {OUTPUT_WAV}")
