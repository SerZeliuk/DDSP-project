"""Gated WaveNet synthesizer with residual & skip connections.
Matches DDSP-style vocoder but lightweight.
Outputs 1‑channel audio plus the first channel copied from input (f0) so we
can add an auxiliary pitch‑tracking loss during training.
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
        self.res  = layers.Conv1D(filters, 1)
        self.skip = layers.Conv1D(filters, 1)

    def call(self, x):
        z = tf.nn.tanh(self.filt(x)) * tf.nn.sigmoid(self.gate(x))
        res = self.res(z) + x               # residual
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
        self.input_proj = layers.Conv1D(filters, 1)   # project features
        self.blocks = [WaveNetBlock(filters, kernel_size, d) for d in dilation_rates]
        self.relu  = layers.ReLU()
        self.post1 = layers.Conv1D(filters, 1, activation="relu")
        # two outputs: 0=f0_aux, 1=audio
        self.post2 = layers.Conv1D(2, 1)

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
        return self.post2(x)                # [B,T,2]

    # helper to split outputs
    @staticmethod
    def split_outputs(y):
        # y: [B,T,2]
        return y[..., 0], y[..., 1]
