# legacy_wavenet.py
import tensorflow as tf
from tensorflow.keras import layers

class LegacyWaveNet(tf.keras.Model):
    """
    Simple WaveNet-like stack exactly as used during your original training:
      * ReLU dilated Conv1D layers (no gating)
      * No residual/skip connections
      * No input 1×1 projection
    """
    def __init__(self,
                 num_layers=10,
                 filters=64,
                 kernel_size=2,
                 dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [2**i for i in range(num_layers)]
        self.dilated_convs = [
            layers.Conv1D(filters,
                          kernel_size,
                          dilation_rate=d,
                          padding='causal',
                          activation='relu')
            for d in dilation_rates
        ]
        self.final_conv = layers.Conv1D(1, 1)

    def call(self, x):
        # x shape: [batch, time, feat_dim]
        for conv in self.dilated_convs:
            x = conv(x)
        return self.final_conv(x)

    def synthesize(self, features):
        # features shape: [time, feat_dim]
        y = self(features[None, ...])     # [1, time, 1]
        return tf.squeeze(y, 0).numpy()   # [time]
