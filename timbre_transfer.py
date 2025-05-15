import os
import numpy as np
import soundfile as sf
from audio_processor import AudioProcessor
from wavenet_synthesizer import WaveNetSynthesizer
from tensorflow.keras.models import load_model

class TimbreTransfer:
    """
    Load a trained timbre-transfer model and apply it to new audio.
    Uses additive synthesis to reconstruct waveform from predicted f0 and harmonics,
    correctly matching the input duration.
    """

    def __init__(self, model_path):
        # Load the Keras model
        self.model = load_model(model_path)

    def transfer_file(self, source_path, output_path):
        """
        Convert source audio to target timbre and save the output WAV.

        Args:
          source_path: Path to input audio file (e.g., flute.wav).
          output_path: Path where to write the timbre-transferred audio.
        """
        # 1) Load and inspect source audio
        proc = AudioProcessor(audio_path=source_path)
        audio, sr = proc.load_audio()
        print(f"[DEBUG] Loaded audio: {len(audio)} samples at {sr} Hz")

        # 2) Extract frame-based features
        f0, _     = proc.compute_f0(audio)
        loudness  = proc.compute_loudness(audio)
        harmonics = proc.compute_harmonics(f0)
        feats = np.concatenate([f0[:, None], harmonics, loudness[:, None]], axis=1)
        n_frames, feat_dim = feats.shape
        print(f"[DEBUG] Extracted features: {n_frames} frames × {feat_dim} features")

        # 3) Predict target features
        pred = self.model.predict(feats, verbose=1)
        print(f"[DEBUG] Model output: {pred.shape[0]} frames × {pred.shape[1]} features")

        # 4) Reconstruct audio via additive synthesis
        # Use frame_rate to compute hop size (samples per frame)
        hop = int(sr // proc.frame_rate)
        n_samples = len(audio)
        y = np.zeros(n_samples, dtype=np.float32)
        phase = np.zeros(proc.num_harmonics)
        for i in range(n_frames):
            frame_feat = pred[i]
            f0_i = frame_feat[0]
            start = i * hop
            end = start + hop
            if start >= n_samples:
                break
            end = min(end, n_samples)
            # Synthesize harmonics for this frame
            for h in range(proc.num_harmonics):
                amp = frame_feat[1 + h]
                freq = f0_i * (h + 1)
                phase_inc = 2 * np.pi * freq / sr
                for n in range(start, end):
                    y[n] += amp * np.sin(phase[h])
                    phase[h] += phase_inc
            # apply loudness gain
            loud_db = frame_feat[-1]
            gain = 10 ** (loud_db / 20.0)
            y[start:end] *= gain
        print(f"[DEBUG] Synthesized waveform: {y.shape[0]} samples, min={y.min():.3e}, max={y.max():.3e}")

        # 5) Write predictions to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y, sr)
        print(f"[INFO] Written {y.shape[0]} samples to {output_path}")
