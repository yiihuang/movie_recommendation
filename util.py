import csv
import numpy as np
import pickle
import time


"""
The code herein will separate the A matrix into k separate data matrices,
with 1/k of the values held as validation error:
A -> A1, A2, A3, A4, A5 ... Ak   training matrices
  -> Y1, Y2, Y3, Y4, Y5 ... Yk   validation matrices
"""


# Transplant every k-th + c non-zero value from a training matrix to validation matrix c
# Therefore, every k-th + c non-zero value in A should now be 0 in training matrix c
#                                           and now be A[i,j] in validation matrix c


def k_cross(k: int = 10):
    """
    Create the training and validation matrices for k-fold cross-validation.

    Returns
    -------
    training_matrices   : list[np.ndarray]
    prediction_matrices : list[np.ndarray]
    index_matrices      : list[np.ndarray]   (boolean masks)
    """
    A = load_data_matrix()
    m, n = A.shape

    print(f"A.shape = {A.shape}")

    prediction_matrices = []
    training_matrices = []
    index_lists = []

    for _ in range(k):
        prediction_matrices.append(np.zeros((m, n)))
        training_matrices.append(A.copy())
        index_lists.append(np.zeros((m, n), dtype=bool))

    it = 0
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:
                fold = it % k
                training_matrices[fold][i, j] = 0
                prediction_matrices[fold][i, j] = A[i, j]
                index_lists[fold][i, j] = True
                it += 1

    return training_matrices, prediction_matrices, index_lists


def load_data_matrix(filename: str = "data_matrix.p", path: str = "data"):
    """
    Quick helper that loads the ratings matrix from <path>/<filename>.
    """
    filepath = filename if path == "" else f"{path}/{filename}"
    return pickle.load(open(filepath, "rb"))


def get_MSE(predicted: np.ndarray, mask: np.ndarray, original=None) -> float:
    """
    Mean-squared error of *predicted* w.r.t. *original* on positions where *mask* is True.
    """
    if original is None:
        original = load_data_matrix()

    diff = original[mask] - predicted[mask]
    mse = np.mean(diff ** 2)          # <-- fixed line  ✓
    return mse


if __name__ == "__main__":
    """
    Quick sanity test:
    * run the partitioning
    * compute MSE for the first fold (baseline predictor = 0)
    * check that no entry appears in both a training and its corresponding validation matrix
    """

    k = 10
    train_mats, val_mats, masks = k_cross(k=k)
    print(f"MSE = {get_MSE(train_mats[0], masks[0]):.6f}")

    m, n = train_mats[0].shape
    start = time.time()

    for i in range(m):
        for j in range(n):
            for fold in range(k):
                if train_mats[fold][i, j] != 0 and val_mats[fold][i, j] != 0:
                    print("we have a problem")

    elapsed = time.time() - start
    print(f"you wasted {elapsed:.2f} seconds of my life")
