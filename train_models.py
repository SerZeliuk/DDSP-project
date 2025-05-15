# train_models.py

import os
from timbre_transfer_trainer import TimbreTransferTrainer

# ── CONFIGURE YOUR DATASETS HERE ──────────────────
# A long flute recording used as the *source* timbre for training both models:
FLUTE_DATASET = 'DDSP Reaper/DDSP Reaper_stems_Viloin-001.wav'

# Long recordings carrying the *target* timbres you want to learn:
GUITAR_DATASET = 'DDSP Reaper/Guitar_dataset.wav'
VOICE_DATASET  = 'DDSP Reaper/Voice_dataset.wav'

# Where to save your two trained MLPs:
OUT_DIR = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    'Guitar_model': {
        'source_files': [FLUTE_DATASET],
        'target_files': [GUITAR_DATASET],
        'model_path':   os.path.join(OUT_DIR, 'Guitar_model.h5')
    },
    'Voice_model': {
        'source_files': [FLUTE_DATASET],
        'target_files': [VOICE_DATASET],
        'model_path':   os.path.join(OUT_DIR, 'Voice_model.h5')
    }
}

# ── HYPERPARAMETERS ───────────────────────────────
SR            = 16000
HOP_LENGTH    = 512
FRAME_RATE    = 250
NUM_HARMONICS = 5
BATCH_SIZE    = 32
EPOCHS        = 20


def train_all():
    for name, cfg in DATASETS.items():
        print(f"\n⏳ Training {name!r} → {cfg['model_path']}")
        trainer = TimbreTransferTrainer(
            source_files  = cfg['source_files'],
            target_files  = cfg['target_files'],
            sr            = SR,
            hop_length    = HOP_LENGTH,
            frame_rate    = FRAME_RATE,
            num_harmonics = NUM_HARMONICS
        )
        trainer.train(
            batch_size      = BATCH_SIZE,
            epochs          = EPOCHS,
            checkpoint_path = cfg['model_path']
        )
        print(f"✅ Finished {name!r}. Saved to {cfg['model_path']}")

if __name__ == '__main__':
    print("=== Train all timbre-transfer models ===")
    train_all()
    print("\nAll models trained and saved under ./models/")
