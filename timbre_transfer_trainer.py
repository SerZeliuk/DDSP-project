import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from audio_processor import AudioProcessor
from timbre_transfer_model import TimbreTransferModel

class TimbreTransferTrainer:
    """
    Class responsible for preparing data, building, and training the timbre transfer model
    using source-target audio file pairs, with progress logging.
    """
    def __init__(self,
                 source_files,
                 target_files,
                 sr=16000,
                 hop_length=512,
                 frame_rate=250,
                 num_harmonics=5):
        self.source_files = source_files
        self.target_files = target_files
        self.sr = sr
        self.hop_length = hop_length
        self.frame_rate = frame_rate
        self.num_harmonics = num_harmonics
        self.model = None

    def _extract_features(self, file_path):
        """
        Extract features (f0, harmonics, loudness) from a single audio file.
        Returns a 2D array of shape [n_frames, num_features].
        """
        print(f"    ▶ Extracting features from: {file_path}")
        proc = AudioProcessor(audio_path=file_path,
                              sr=self.sr,
                              hop_length=self.hop_length,
                              frame_rate=self.frame_rate,
                              num_harmonics=self.num_harmonics)
        audio, _ = proc.load_audio()

        print("      - Computing f0 and confidence...")
        f0, _ = proc.compute_f0(audio, viterbi=True)

        print("      - Computing loudness...")
        loudness = proc.compute_loudness(audio)

        print("      - Computing harmonics...")
        harmonics = proc.compute_harmonics(f0)

        # Align lengths
        n_frames = len(f0)
        harmonics = harmonics[:n_frames]
        loudness = loudness[:n_frames]

        # Stack features: [f0, harmonic_1, ..., harmonic_N, loudness]
        features = np.concatenate(
            [f0[:, None], harmonics, loudness[:, None]],
            axis=1
        )
        print(f"      ✓ Extracted {n_frames} frames of features.\n")
        return features

    def prepare_dataset(self):
        """
        Prepare the training dataset by extracting and pairing features
        from source and target audio files, with progress logs.
        Returns:
          X: numpy array of shape [total_frames, num_features]
          y: numpy array of shape [total_frames, num_features]
        """
        print("Preparing dataset:")
        X_list, y_list = [], []
        total_pairs = len(self.source_files)

        for idx, (src, tgt) in enumerate(zip(self.source_files, self.target_files), start=1):
            print(f"Pair {idx}/{total_pairs}:")
            # Source
            src_feat = self._extract_features(src)
            # Target
            tgt_feat = self._extract_features(tgt)

            # Trim to the shorter length
            min_len = min(len(src_feat), len(tgt_feat))
            print(f"    Trimming to {min_len} frames (min of source/target)")
            X_list.append(src_feat[:min_len])
            y_list.append(tgt_feat[:min_len])

        print("Concatenating all feature pairs into arrays...")
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        print(f"Prepared dataset with {X.shape[0]} total frames and {X.shape[1]} features per frame.\n")
        return X, y

    def build_model(self):
        """
        Build and compile the timbre transfer model.
        Returns the compiled model.
        """
        # Prepare one pass of dataset to get input dimension
        X, _ = self.prepare_dataset()
        input_dim = X.shape[1]
        print(f"Building model with input dimension: {input_dim}")
        model_builder = TimbreTransferModel(input_shape=(input_dim,))
        self.model = model_builder.compile_model()
        print("Model compiled and ready.\n")
        return self.model

    def train(self,
              batch_size=32,
              epochs=20,
              checkpoint_path=None):
        """
        Train the model on the prepared dataset.
        Optionally save best model to checkpoint_path.
        Returns the training history.
        """
        print("Starting training...")
        X, y = self.prepare_dataset()

        if self.model is None:
            self.build_model()

        callbacks = []
        if checkpoint_path:
            print(f"Checkpoint will be saved to: {checkpoint_path}")
            callbacks.append(
                ModelCheckpoint(checkpoint_path,
                                save_best_only=True,
                                monitor='loss',
                                verbose=1)
            )

        history = self.model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete.\n")
        return history
