from timbre_transfer_model import TimbreTransferModel
from timbre_transfer_trainer import TimbreTransferTrainer
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from train_models import FLUTE_DATASET, GUITAR_DATASET, SR, HOP_LENGTH, FRAME_RATE, NUM_HARMONICS

# 1) summary
N = 7  # or your actual num_features-
mtm = TimbreTransferModel(input_shape=(N,))
model = mtm.compile_model()
model.summary()

# 2) random forward pass
x = np.random.randn(5, N).astype(np.float32)
print("Random test output shape:", model.predict(x).shape)

# 3) evaluate on training set
trainer = TimbreTransferTrainer(
    source_files  = [FLUTE_DATASET],
    target_files  = [GUITAR_DATASET],
    sr            = SR,
    hop_length    = HOP_LENGTH,
    frame_rate    = FRAME_RATE,
    num_harmonics = NUM_HARMONICS
)
X,y = trainer.prepare_dataset()
model = load_model('models/Guitar_model.h5')
loss = model.evaluate(X, y, batch_size=32)
print("Train loss:", loss)

# 4) plot a few frames of f0
pred = model.predict(X[:200])
plt.plot(y[:200,0], label='true f0')
plt.plot(pred[:,0], label='pred f0', alpha=0.7)
plt.legend(); plt.show()
