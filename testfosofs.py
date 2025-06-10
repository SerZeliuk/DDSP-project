from timbre_transfer_trainer import TimbreTransferTrainer
from tensorflow.keras.models import load_model

model = load_model('models/Guitar_model.h5')
print(model.input_shape)   # e.g. (None, 7)
