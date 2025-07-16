# run_nn.py
import time
import numpy as np
import util                              # your helper that loads data + k-folds
from nn import Collaborative_Filtering_Neural_Net as CFNN
import os
import glob


# --------------------------------------------------------------------------- #
# 1. TRAIN ONE FOLD AND CHECKPOINT EVERY EPOCH
# --------------------------------------------------------------------------- #
def train_one_fold(
    fold_id: int       = 0,
    num_layers: int    = 3,
    learn_rate: float  = 0.10,
    epochs: int        = 1,
    batch_size: int    = 128,
) -> None:
    """
    Train the collaborative-filtering NN on a single k-fold split and write
    checkpoints such as  nn_model_fold0_lr0.100_epoch02.h5
    """
    train_mats, val_mats, masks = util.k_cross(k=4)
    train_mat = train_mats[fold_id]
    val_mat   = val_mats[fold_id]
    mask      = masks[fold_id]

    net = CFNN(
        train_mat,
        val_mat,
        mask,
        num_layers = num_layers,
        learn_rate = learn_rate,
    )

    net.construct_model(hidden_layer_pattern="exponential")

    start = time.time()
    net.train_model(
        epochs      = epochs,
        batch_size  = batch_size,
        model_tag   = f"fold{fold_id}"
    )
    print(f"Training time (s): {time.time() - start:.1f}")


# --------------------------------------------------------------------------- #
# 2. EVALUATION HELPER
# --------------------------------------------------------------------------- #

def evaluate_checkpoint(
    checkpoint_path: str | None = None,   # pass None ⇒ auto-select latest
    fold_id: int        = 0,
    learn_rate: float   = 0.10,
    on_validation: bool = True,
) -> None:
    """
    Report classification accuracy and integer-bucket MSE for a saved model.
    If `checkpoint_path` is None, the newest  *.weights.h5  file in the cwd
    (or sub-dir) is used automatically.
    """
    # ---------------------------------------------------- #
    # 1. Resolve which checkpoint file to load
    # ---------------------------------------------------- #
    if checkpoint_path is None:
        weight_files = sorted(
            glob.glob("**/*weights.h5", recursive=True),
            key=os.path.getmtime
        )
        if not weight_files:
            raise FileNotFoundError(
                "No *.weights.h5 checkpoint found. "
                "Train a model first or pass checkpoint_path explicitly."
            )
        checkpoint_path = weight_files[-1]
        print(f"[evaluate] Using latest checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ---------------------------------------------------- #
    # 2. Re-create the fold-id data split
    # ---------------------------------------------------- #
    train_mats, val_mats, masks = util.k_cross(k=4)
    train_mat = train_mats[fold_id]
    val_mat   = val_mats[fold_id]
    mask      = masks[fold_id]

    # ---------------------------------------------------- #
    # 3. Re-build identical architecture and load weights
    # ---------------------------------------------------- #
    net = CFNN(
        train_mat,
        val_mat,
        mask,
        learn_rate=learn_rate
    )
    net.construct_model(hidden_layer_pattern="exponential")
    net.load_weights(checkpoint_path)

    # ---------------------------------------------------- #
    # 4. Predict & compute metrics
    # ---------------------------------------------------- #
    preds_soft, targets_onehot = net.predict_values(
        on_validation=on_validation
    )
    preds   = preds_soft.argmax(axis=1)
    targets = targets_onehot.argmax(axis=1)

    acc = (preds == targets).mean() * 100
    mse = np.mean((preds - targets) ** 2)

    split = "validation" if on_validation else "training"
    print(f"{split.capitalize()} accuracy: {acc:.2f}%")
    print(f"{split.capitalize()} MSE:       {mse:.4f}")

    """
    Reports classification accuracy and integer-bucket MSE of a saved model.
    `checkpoint_path` must be one of the *.h5 files produced by train_one_fold.
    """
    train_mats, val_mats, masks = util.k_cross(k=4)
    train_mat = train_mats[fold_id]
    val_mat   = val_mats[fold_id]
    mask      = masks[fold_id]

    net = CFNN(
        train_mat,
        val_mat,
        mask,
        learn_rate = learn_rate
    )
    net.construct_model(hidden_layer_pattern="exponential")
    net.load_weights(checkpoint_path)

    preds_soft, targets_onehot = net.predict_values(on_validation=on_validation)
    preds   = preds_soft.argmax(axis=1)
    targets = targets_onehot.argmax(axis=1)

    acc  = (preds == targets).mean() * 100
    mse  = np.mean((preds - targets) ** 2)

    split = "validation" if on_validation else "training"
    print(f"{split.capitalize()} accuracy: {acc:.2f}%")
    print(f"{split.capitalize()} MSE:       {mse:.4f}")


# --------------------------------------------------------------------------- #
# 3. MAIN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ---------- 1. train ----------
    train_one_fold(
        fold_id     = 0,
        learn_rate  = 0.10,
        epochs      = 1,
        batch_size  = 128,
    )

    # ---------- 2. evaluate ----------
    # adjust the filename to whichever epoch you want to test
    evaluate_checkpoint(on_validation=False)
    evaluate_checkpoint(on_validation=True)
