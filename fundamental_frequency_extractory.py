import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import audio_processor

def f0_to_sine_wave_with_harmonics(f0, sr, hop_length=512, unvoiced_freq=0.0, num_harmonics=15):
    """
    Generate a signal composed of the fundamental frequency (f0) and its harmonics.
    """
    n_frames = len(f0)
    n_samples = n_frames * hop_length

    y = np.zeros(n_samples, dtype=np.float32)
    phase = np.zeros(num_harmonics)  # Phase for each harmonic

    for i in range(n_frames):
        current_f0 = f0[i]
        if np.isnan(current_f0) or current_f0 <= 0:
            current_f0 = unvoiced_freq

        # Generate signal by summing harmonics
        for harmonic in range(1, num_harmonics + 1):
            harmonic_f0 = current_f0 * harmonic
            phase_increment = 2.0 * np.pi * (harmonic_f0 / sr)

            start_sample = i * hop_length
            end_sample = start_sample + hop_length

            for n in range(start_sample, end_sample):
                y[n] += np.sin(phase[harmonic - 1])
                phase[harmonic - 1] += phase_increment

    return y

def save_audio(y, sr, out_path):
    """
    Save the synthesized audio signal to a file.
    """
    sf.write(out_path, y, sr)
    print(f"Synthesized audio written to {out_path}")

def save_tfrecord(f0, harmonics, loudness, out_path):
    """
    Save the features (f0, harmonics, loudness) to a TFRecord file.
    """
    with tf.io.TFRecordWriter(out_path) as writer:
        for i in range(len(f0)):
            # Create a feature dictionary
            feature = {
                'f0': tf.train.Feature(float_list=tf.train.FloatList(value=f0[i:i+1])),
                'harmonics': tf.train.Feature(float_list=tf.train.FloatList(value=harmonics[i].flatten())),
                'loudness': tf.train.Feature(float_list=tf.train.FloatList(value=loudness[i:i+1])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"Features saved to {out_path}")

# ----------------------------------
# Example usage
# ----------------------------------
def main():
    # Step 1: Load audio file
    audio_path = "DDSP Reaper/DDSP Reaper_stems_Viloin-001.wav"
    out_path_wave = "DDSP Reaper/harmonics_wave.wav"
    out_path_tfrecord = "DDSP Reaper/timbre_features.tfrecord"

    audio_processor = audio_processor.AudioProcessor(
        audio_path=audio_path,
        out_path="DDSP Reaper/harmonics_wave.wav",
        sr=22050,
        hop_length=512,
        fmin='C2',
        fmax='C7',
        num_harmonics=5
    )

    audio, sr = audio_processor.load_audio()

    # Step 2: Extract fundamental frequency (f0)
    f0 = audio_processor.extract_f0(audio, sr)

    # Step 3: Compute loudness (RMS)
    loudness = audio_processor.compute_loudness(audio)

    # Step 4: Generate the audio signal with harmonics
    hop_length = 512
    y_harmonics = f0_to_sine_wave_with_harmonics(f0, sr, hop_length=hop_length, unvoiced_freq=0.0, num_harmonics=5)

    # Step 5: Save the generated audio with harmonics
    
    save_audio(y_harmonics, sr, out_path_wave)

    # Step 6: Save the features (f0, harmonics, loudness) to a TFRecord
    
    save_tfrecord(f0, y_harmonics, loudness, out_path_tfrecord)

if __name__ == "__main__":
    main()
