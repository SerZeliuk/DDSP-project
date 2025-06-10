import librosa
import numpy as np
from spectral_operations import pad, compute_f0 as ddsp_compute_f0, compute_loudness as ddsp_compute_loudness

def make_per_sample_features(audio, proc):
    # 1) frame‐level features
    f0, _       = proc.compute_f0(audio)
    loudness    = proc.compute_loudness(audio)
    harmonics   = proc.compute_harmonics(f0)          # shape (n_frames, H)
    feats_frame = np.concatenate([f0[:,None], harmonics, loudness[:,None]], axis=1)

    # 2) upsample to per‐sample
    n_frames, feat_dim = feats_frame.shape
    hop = proc.hop_length
    feats_sample = np.repeat(feats_frame, hop, axis=0) 
    feats_sample = feats_sample[:len(audio)]         # trim to exact length
    return feats_sample  # shape (n_samples, feat_dim)

class AudioProcessor:
    """
    Load audio and extract per-frame features:
      - f0 and confidence via DDSP/CREPE
      - Perceptual loudness
      - Harmonic sinusoids
    """

    def __init__(self, audio_path, sr=16000, hop_length=512, frame_rate=250, num_harmonics=100):
        self.audio_path = audio_path
        self.sr = sr
        self.hop_length = hop_length
        self.frame_rate = frame_rate
        self.num_harmonics = num_harmonics

    def load_audio(self, sr=None):
        audio, sr = librosa.load(self.audio_path, sr=sr or self.sr)
        return audio, sr

    def compute_f0(self, audio, viterbi=True, padding='center'):
        """
        Returns:
          f0_hz: ndarray [n_frames]
          f0_confidence: ndarray [n_frames]
        """
        f0_hz, f0_confidence = ddsp_compute_f0(
            audio, frame_rate=self.frame_rate, viterbi=viterbi, padding=padding
        )
        return f0_hz, f0_confidence

    def compute_loudness(self, audio):
        """
        Returns:
          loudness_db: ndarray [n_frames]
        """
        loudness_db = ddsp_compute_loudness(
            audio,
            sample_rate=self.sr,
            frame_rate=self.frame_rate,
            n_fft=self.hop_length*2,
            use_tf=False,
            padding='center'
        )
        return loudness_db

    def compute_harmonics(self, f0_hz):
        """
        Generate per-frame harmonics:
          returns H of shape [n_frames, num_harmonics]
        """
        n_frames = len(f0_hz)
        H = np.zeros((n_frames, self.num_harmonics), dtype=np.float32)
        # phase per harmonic
        for h in range(1, self.num_harmonics + 1):
            # simple sinusoid harmonic
            H[:, h-1] = np.sin(2 * np.pi * f0_hz * h * (self.hop_length / self.sr))
        return H

  