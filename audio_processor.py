# audio_processor.py
import numpy as np
import librosa
import crepe

class AudioProcessor:
    """
    Load audio and extract frame-level f0 (via CREPE) and loudness (via RMS → dB).
    """
    def __init__(self, audio_path, sr=16000, frame_rate=250, hop_length=512, num_harmonics=100, model_capacity='full'):
        self.audio_path = audio_path
        self.sr = sr
        self.frame_rate = frame_rate
        self.hop_length = int(sr / frame_rate) 
        self.num_harmonics = num_harmonics
        self.model_capacity = model_capacity  # 'tiny', 'full', etc.

    def load_audio(self):
        audio, sr = librosa.load(self.audio_path, sr=self.sr, mono=True)
        return audio, sr

    def compute_f0(self, audio):
        """
        Returns:
          f0_hz:      np.ndarray, shape [n_frames]
          confidence: np.ndarray, shape [n_frames]
        """
        # crepe.predict expects audio in [-1,1], and returns arrays sampled at step_size ms
        step_size = 1000 * self.hop_length / self.sr  # in milliseconds
        # crepe returns (time, frequency, confidence, activation)
        time, frequency, confidence, _ = crepe.predict(
            audio,
            sr=self.sr,
            step_size=step_size,
            model_capacity=self.model_capacity,
            viterbi=True
        )
        return frequency.astype(np.float32), confidence.astype(np.float32)

    def compute_loudness(self, audio):
        """
        Returns:
          loudness_db: np.ndarray, shape [n_frames]
        """
        # Compute frame-level RMS energy
        # hop_length determines frame count = len(audio)/hop_length
        rms = librosa.feature.rms(y=audio, frame_length=self.hop_length*2,
                                  hop_length=self.hop_length, center=True)[0]
        # Convert amplitude → dB
        loudness_db = librosa.amplitude_to_db(rms, ref=1.0)
        return loudness_db.astype(np.float32)

    def compute_harmonics(self, audio):
        H = []
        f0, conf = self.compute_f0(audio)
        for h in range(1, self.num_harmonics+1):
            # instantaneous phase increment per frame in radians
            H.append(np.sin(2*np.pi * f0 * h * (self.hop_length/self.sr)))
        harmonics = np.stack(H, axis=1)  # [n_frames, num_harmonics]
        return harmonics

    def extract_features(self):
        """
        Convenience: load audio, compute f0, confidence, and loudness.
        Returns dict with 'audio', 'f0', 'confidence', 'loudness'.
        """
        audio, _ = self.load_audio()
        f0, conf = self.compute_f0(audio)
        loud   = self.compute_loudness(audio)
        harmonics = self.compute_harmonics(audio)
        # Trim or pad so that len(loud)=len(f0)
        min_len = min(len(f0), len(loud))
        return {
            'audio':    audio,
            'f0':       f0[:min_len],
            'confidence': conf[:min_len],
            'harmonics':  harmonics[:min_len],
            'loudness': loud[:min_len],
            'hop_length': self.hop_length,
        }
