from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class TimbreTransferModel:
    """
    MLP that maps input feature vectors (f0 + harmonics + loudness)
    from a source sound to the corresponding features of the target sound.
    """

    def __init__(self, input_shape):
        """
        input_shape: tuple (num_features,)
        """
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.input_shape[0], activation='linear')(x)
        return Model(inputs, outputs)

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model
