# nn.py
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Activation, Flatten, MaxPooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class Collaborative_Filtering_Neural_Net:
    """
    A tiny one-hot collaborative–filtering network.
    """

    def __init__(self, train_data, val_data, mask,
                 num_layers: int = 3,
                 learn_rate: float = 0.2) -> None:

        self.train_data = train_data.astype(np.float32)
        self.val_data   = val_data.astype(np.float32)
        self.mask       = mask                # boolean mask for validation cells
        self.num_layers = num_layers
        self.learn_rate = learn_rate

        self.m, self.n = self.train_data.shape  # users, movies
        self.construct_input()                  # build self.train_x / self.train_y …

    # ------------------------------------------------------------------ #
    # 1. DATA PREP
    # ------------------------------------------------------------------ #
    def construct_input(self) -> None:
        """
        Build one-hot (user ⊕ movie) inputs and one-hot 11-bucket outputs
        for both training and validation sets.
        """
        def as_one_hot(score: float) -> np.ndarray:
            vec = np.zeros(11, dtype=np.float32)          # scores 0 … 5 in 0.5 steps
            vec[int(score / 0.5)] = 1.0
            return vec

        m, n = self.m, self.n

        # ---------- TRAIN SAMPLES ----------
        u_idx, v_idx = np.where(self.train_data > 0)
        n_train      = u_idx.shape[0]

        self.train_x = np.zeros((n_train, m + n), dtype=np.float32)
        self.train_y = np.zeros((n_train, 11),     dtype=np.float32)

        for k, (u, v) in enumerate(zip(u_idx, v_idx)):
            self.train_x[k, u]     = 1.0          # user one-hot
            self.train_x[k, m + v] = 1.0          # movie one-hot
            self.train_y[k]        = as_one_hot(self.train_data[u, v])

        # ---------- VALIDATION SAMPLES ----------
        u_idx, v_idx = np.where(self.mask)
        n_val        = u_idx.shape[0]

        self.val_x = np.zeros((n_val, m + n), dtype=np.float32)
        self.val_y = np.zeros((n_val, 11),     dtype=np.float32)

        for k, (u, v) in enumerate(zip(u_idx, v_idx)):
            self.val_x[k, u]     = 1.0
            self.val_x[k, m + v] = 1.0
            self.val_y[k]        = as_one_hot(self.val_data[u, v])

    # ------------------------------------------------------------------ #
    # 2. MODEL BUILD
    # ------------------------------------------------------------------ #
    def construct_model(self, hidden_layer_pattern: str = "exponential") -> None:
        """
        Build a fully-connected net whose width shrinks either linearly
        or exponentially toward the output.
        """
        model      = Sequential()
        input_size = self.m + self.n

        # Input layer (gets rid of "Don’t pass input_shape" warning)
        model.add(Input(shape=(input_size,)))
        model.add(Dense(input_size, activation="relu"))

        if hidden_layer_pattern == "linear":
            delta = input_size // self.num_layers
            for _ in range(self.num_layers):
                input_size -= delta
                model.add(Dense(input_size, activation="relu"))

        elif hidden_layer_pattern == "exponential":
            # divide width by roughly the same ratio each layer
            ratio = int(np.exp(np.log(input_size) / (self.num_layers + 2)))
            for _ in range(self.num_layers):
                input_size = max(8, input_size // ratio)
                model.add(Dense(input_size, activation="relu"))

        # Final 11-way softmax
        model.add(Dense(11, activation="softmax"))

        # Optimizer (new API)
        adam = Adam(learning_rate=self.learn_rate)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=adam,
            metrics=["accuracy"]
        )
        self.model = model

    # ------------------------------------------------------------------ #
    # 3. TRAINING
    # ------------------------------------------------------------------ #

    def train_model(self,
                    epochs: int = 1,
                    batch_size: int = 128,
                    model_tag: str = "run0"):

        # overwrite the same file every epoch  ⇢ never grows past ~400 MB
        ckpt = ModelCheckpoint(
            filepath=f"nn_{model_tag}.weights.h5",   # no {epoch:02d}
            monitor="val_loss",
            save_best_only=False,   # or True if you only want the best
            save_weights_only=True, # weights only ≈¼ the size
            verbose=1
        )

        # (optional) stop early when val_loss stops improving
        early = EarlyStopping(patience=3, restore_best_weights=True)

        self.model.fit(
            self.train_x, self.train_y,
            validation_data=(self.val_x, self.val_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[ckpt, early],
            verbose=1
        )




    # ------------------------------------------------------------------ #
    # 4. INFERENCE
    # ------------------------------------------------------------------ #
    def predict_values(self, on_validation: bool = True):
        """
        Return (predicted_softmax, ground_truth_one_hot) for either the
        validation or training set.
        """
        if on_validation:
            return self.model.predict(self.val_x, verbose=1), self.val_y
        return self.model.predict(self.train_x, verbose=1), self.train_y

    # ------------------------------------------------------------------ #
    # 5. CHECKPOINT LOAD
    # ------------------------------------------------------------------ #
    def load_weights(self, filename: str) -> None:
        self.model.load_weights(filename)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learn_rate),
            metrics=["accuracy"]
        )
