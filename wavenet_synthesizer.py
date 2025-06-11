# wavenet_synthesizer.py

"""Gated WaveNet synthesizer with residual & skip connections.
Matches DDSP-style vocoder but lightweight.
Outputs 1-channel audio plus an auxiliary f0 channel, with strict causality
and tanh-constrained audio output.
"""
import tensorflow as tf
from tensorflow.keras import layers

class WaveNetBlock(layers.Layer):
    def __init__(self, filters: int, kernel_size: int, dilation: int):
        super().__init__()
        self.filt = layers.Conv1D(filters, kernel_size,
                                  dilation_rate=dilation,
                                  padding="causal")
        self.gate = layers.Conv1D(filters, kernel_size,
                                  dilation_rate=dilation,
                                  padding="causal")
        self.res  = layers.Conv1D(filters, 1, padding="causal")
        self.skip = layers.Conv1D(filters, 1, padding="causal")

    def call(self, x):
        z = tf.nn.tanh(self.filt(x)) * tf.nn.sigmoid(self.gate(x))
        res = self.res(z) + x               # residual connection
        skip = self.skip(z)                 # skip connection
        return res, skip

class WaveNetSynthesizer(tf.keras.Model):
    def __init__(self,
                 num_blocks: int = 10,
                 filters: int = 64,
                 kernel_size: int = 2,
                 dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [2 ** i for i in range(num_blocks)]

        # Project features, ensure no future leakage
        self.input_proj = layers.Conv1D(filters, 1, padding="causal")

        # Stacked gated blocks
        self.blocks = [WaveNetBlock(filters, kernel_size, d)
                       for d in dilation_rates]

        self.relu  = layers.ReLU()

        # Post-processing, still causal
        self.post1 = layers.Conv1D(filters, 1,
                                   activation="relu",
                                   padding="causal")
        # Final two outputs: [f0_aux, audio_raw]
        self.post2 = layers.Conv1D(2, 1, padding="causal")

    def call(self, x):
        # x: [batch, time, feat_dim]
        x = self.input_proj(x)

        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)

        x = tf.add_n(skips)
        x = self.relu(x)
        x = self.post1(x)

        # Linear heads
        y = self.post2(x)  # [B, T, 2]
        f0_aux = y[..., 0]           # [B, T]
        audio_raw = y[..., 1]        # [B, T]

        # Constrain audio to [-1, 1]
        audio = tf.tanh(audio_raw)

        # Return stacked [f0_aux, audio]
        return tf.stack([f0_aux, audio], axis=-1)  # [B, T, 2]

    @staticmethod
    def split_outputs(y):
        # Splits into (f0_aux, audio) both [B, T]
        return y[..., 0], y[..., 1]
