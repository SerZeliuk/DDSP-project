import tensorflow as tf
from tensorflow.keras import layers, models

class WaveNet(tf.keras.Model):
    def __init__(self, num_layers=10, num_filters=64, kernel_size=2, dilation_rates=[1, 2, 4, 8, 16], output_channels=1):
        super(WaveNet, self).__init__()

        # Number of layers in the WaveNet
        self.num_layers = num_layers

        # Build the dilated convolution layers
        self.dilated_convs = []
        for i in range(self.num_layers):
            dilation_rate = dilation_rates[i % len(dilation_rates)]
            self.dilated_convs.append(layers.Conv1D(
                num_filters, 
                kernel_size, 
                dilation_rate=dilation_rate,
                padding="causal", 
                activation="relu"
            ))

        # Final convolution layer to produce the output audio (same size as input)
        self.final_conv = layers.Conv1D(output_channels, 1)

    def call(self, inputs):
        x = inputs
        for conv in self.dilated_convs:
            x = conv(x)
        
        output = self.final_conv(x)
        return output
