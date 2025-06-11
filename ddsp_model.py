# ddsp_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
import core_functions as core
from synths import HarmonicPlusNoiseSynth

def build_ddsp_autoencoder(
    sample_rate: int = 16000,
    frame_rate:   int = 250,
    window_sec:   int = 4,
    conv_channels:   int = 64,
    num_layers:      int = 4,
    kernel_size:     int = 3,
    latent_dim:      int = 128,
    decoder_hidden:  int = 256,
    n_harmonics:     int = 64,
    synth_window:    int = 257
) -> tf.keras.Model:
    """
    Returns a Functional Keras Model:
      Inputs:
        audio    : [B, window_sec*sample_rate]
        f0       : [B, window_sec*frame_rate, 1]
        loudness : [B, window_sec*frame_rate, 1]
      Output:
        recon    : [B, window_sec*sample_rate]
    All lengths are computed once (statically) so there is no None/%% trouble.
    """
    SR = sample_rate
    FR = frame_rate
    HOP = SR // FR
    SAMPLES_WIN = window_sec * SR
    FRAMES_WIN = window_sec * FR

    # ─── 1) Inputs ──────────────────────────────────────────────────────────
    audio_in = tf.keras.Input(shape=(SAMPLES_WIN,),   name="audio")
    f0_in    = tf.keras.Input(shape=(FRAMES_WIN, 1),  name="f0")
    loud_in  = tf.keras.Input(shape=(FRAMES_WIN, 1),  name="loudness")

    # ─── 2) Encoder ─────────────────────────────────────────────────────────
    # Add channel dim for convs
    x = layers.Reshape((SAMPLES_WIN, 1), name="reshape_audio")(audio_in)
    # Strided conv stack
    for i in range(num_layers):
        x = layers.Conv1D(
            conv_channels,
            kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            name=f"enc_conv_{i}"
        )(x)
    # Global pooling + FC → z
    z = layers.GlobalAveragePooling1D(name="enc_pool")(x)
    z = layers.Dense(latent_dim, name="enc_fc")(z)
    z = layers.Lambda(lambda t: tf.zeros_like(t),
                     name="fixed_guitar_z")(z) 

    # ─── 3) Decoder ─────────────────────────────────────────────────────────
    # Tile z to frame-rate
    z_t = layers.Lambda(
        lambda t: tf.repeat(tf.expand_dims(t, 1), FRAMES_WIN, axis=1),
        name="tile_z"
    )(z)                                  # [B, FRAMES_WIN, latent_dim]
    # Concat f0 + loud → [B,FRAMES_WIN,2]
    controls = layers.Concatenate(name="concat_controls")([f0_in, loud_in])
    # Combine z_t + controls → [B,FRAMES_WIN,latent_dim+2]
    d = layers.Concatenate(name="decoder_input")([z_t, controls])
    # Two FC layers at frame-rate
    d = layers.Dense(decoder_hidden, activation="relu", name="dec_fc1")(d)
    d = layers.Dense(decoder_hidden * HOP, activation="relu", name="dec_fc2")(d)
    # Reshape to sample-rate timeline
    d = layers.Reshape((SAMPLES_WIN, decoder_hidden), name="dec_reshape")(d)

    # ─── 4) DSP Parameter Projections ───────────────────────────────────────
    amp   = layers.Conv1D(1, 1, activation="relu",     name="amplitude")(d)
    harm  = layers.Conv1D(
        n_harmonics, 1, activation="softmax", name="harmonic_distribution"
    )(d)
    noise = layers.Conv1D(1, 1, activation="sigmoid",  name="noise_magnitudes")(d)

    # ─── 5) Resample f0 to sample rate ─────────────────────────────────────
    f0_up = layers.Lambda(
        lambda f: core.resample(f, SAMPLES_WIN, method="linear"),
        name="resample_f0"
    )(f0_in)                              # [B, SAMPLES_WIN, 1]

    # ─── 6) Synthesizer ────────────────────────────────────────────────────
    synth = HarmonicPlusNoiseSynth(
        n_samples=SAMPLES_WIN,
        sample_rate=SR,
        n_harmonics=n_harmonics,
        window_size=synth_window
    )
    recon = synth(amp, harm, noise, f0_up)  # → [B, SAMPLES_WIN]
    recon = layers.Lambda(
    lambda x: tf.clip_by_value(x, -1.0, 1.0),
    name="clip_output"
    )(recon)
    # ─── 7) Build & return Functional Model ───────────────────────────────
    model = Model(
        inputs=[audio_in, f0_in, loud_in],
        outputs=recon,
        name="ddsp_autoencoder"
    )
    return model
