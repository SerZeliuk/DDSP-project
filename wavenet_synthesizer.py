import tensorflow as tf
from tensorflow.keras import layers

class WaveNetSynthesizer(tf.keras.Model):
    """
    A simple WaveNet-like synthesizer in TF2, 
    which converts feature sequences into raw audio.
    """

    def __init__(self, num_layers=8, num_filters=64, kernel_size=2, dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            # e.g. [1,2,4,8,...]
            dilation_rates = [2**i for i in range(num_layers)]
        self.dilated_convs = []
        for d in dilation_rates:
            self.dilated_convs.append(
                layers.Conv1D(num_filters,
                              kernel_size,
                              dilation_rate=d,
                              padding='causal',
                              activation='relu')
            )
        # final to 1 channel audio
        self.final_conv = layers.Conv1D(1, 1)

    def call(self, x):
        # x: [batch, time, features]
        for conv in self.dilated_convs:
            x = conv(x)
        return self.final_conv(x)

    def synthesize(self, features):
        """
        features: np.ndarray [time, feat_dim]
        returns: np.ndarray [time] raw waveform
        """
        # add batch dim
        x = tf.expand_dims(features, 0)
        y = self(x)                   # [1, time, 1]
        y = tf.squeeze(y, -1)         # [1, time]
        return y.numpy()[0]
