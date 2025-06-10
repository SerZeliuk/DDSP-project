"""Inference using gated WaveNet with snapped-MIDI features & f0 plot."""
import os, numpy as np, soundfile as sf, matplotlib.pyplot as plt, librosa
from audio_processor       import AudioProcessor
from wavenet_synthesizer   import WaveNetSynthesizer

# ─── paths ───────────────────────────────────────────────────
WEIGHTS_FILE = "models/GuitarSynth.weights.h5"
STATS_FILE   = "models/feature_stats.npz"
SOURCE_FILE  = "DDSP Reaper/DDSP Reaper_stems_Viloin-001.wav"
OUTPUT_FILE  = "out/violin_as_guitar.wav"

SR=16000; HOP=512; FR=250; HARM=5

# util: snap Hz to MIDI ±50 cents
def hz_to_midi_quantised(f):
    midi = 69 + 12*np.log2(np.maximum(f,1e-6)/440.0)
    midi_q = np.round(midi)
    cents  = (midi-midi_q)*100
    midi_q[np.abs(cents)>50] = 0
    return midi_q.astype(np.float32)

def main():
    # stats
    stats = np.load(STATS_FILE)
    mean, std = stats['mean'], stats['std']

    # extract feats
    proc = AudioProcessor(SOURCE_FILE, SR, HOP, FR, HARM)
    audio,_ = proc.load_audio()
    feats   = proc.make_per_sample_features(audio)
    feats[:,0] = hz_to_midi_quantised(feats[:,0])
    feats = (feats-mean)/std
    n_samples, feat_dim = feats.shape

    # model
    model = WaveNetSynthesizer(num_blocks=10, filters=64, kernel_size=2,
                               dilation_rates=[1,2,4,8,16,32])
    model(np.zeros((1,100,feat_dim)))
    model.load_weights(WEIGHTS_FILE)

    # synthesis
    y_pred = model.predict(feats[None,...], batch_size=1)[0]
    _, y   = model.split_outputs(y_pred)   # use only audio channel
    y = y[:n_samples].numpy().squeeze()

    # f0 curves
    f0_src, _, _ = librosa.pyin(audio.astype(np.float32), sr=SR,
                                fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=HOP)
    f0_syn, _, _ = librosa.pyin(y.astype(np.float32), sr=SR,
                                fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=HOP)
    f0_src, f0_syn = np.nan_to_num(f0_src), np.nan_to_num(f0_syn)
    n_frames = min(len(f0_src), len(f0_syn))
    t_fr = np.arange(n_frames)*(HOP/SR)

    # plots
    plt.figure(figsize=(10,3)); plt.plot(np.arange(n_samples)/SR, audio); plt.title('Source'); plt.tight_layout(); plt.show()
    plt.figure(figsize=(10,3)); plt.plot(np.arange(n_samples)/SR, y); plt.title('Synthesised'); plt.tight_layout(); plt.show()
    plt.figure(figsize=(10,3));
    plt.plot(t_fr, f0_src[:n_frames], label='source f₀');
    plt.plot(t_fr, f0_syn[:n_frames], label='synth  f₀');
    plt.ylabel('Hz'); plt.xlabel('Time (s)'); plt.title('f₀ comparison'); plt.legend(); plt.tight_layout(); plt.show()

    # save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    sf.write(OUTPUT_FILE, y, SR)
    print('Saved →', OUTPUT_FILE)

if __name__=='__main__':
    main()
