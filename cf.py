import pickle
import numpy as np
import util


class CollaborativeFiltering:
    """
    Memory-friendly user–based (or item–based) collaborative filtering.

    Parameters
    ----------
    method : {"neighborhood", "item"}
        "neighborhood"  → user-based CF  
        "item"          → item-based CF (just runs CF on A.T)
    k : int
        Number of neighbours.
    s : int
        Significance threshold (minimum common ratings to give full weight).
    """

    def __init__(self, method: str = "neighborhood", k: int = 10, s: int = 50):
        self.method = method
        self.k = k
        self.s = s

    # ------------------------------------------------------------------ #
    # static helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pearson(r_a: np.ndarray, r_i: np.ndarray) -> float:
        mask = (r_a > 0) & (r_i > 0)
        if mask.sum() == 0:
            return 0.0               # no overlap → no correlation

        r_a, r_i = r_a[mask], r_i[mask]
        a_centred = r_a - r_a.mean()
        i_centred = r_i - r_i.mean()

        denom = np.sqrt(np.dot(a_centred, a_centred) * np.dot(i_centred, i_centred))
        return 0.0 if denom == 0 else float(np.dot(a_centred, i_centred) / denom)

    @staticmethod
    def _significance(r_a: np.ndarray, r_i: np.ndarray, thresh: int) -> float:
        common = ((r_a > 0) & (r_i > 0)).sum()
        return 1.0 if common >= thresh else common / thresh

    @staticmethod
    def _prediction(R_neigh: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Return offset vector  (mean-centred)  for the active user."""
        centred = R_neigh - R_neigh.mean(axis=1, keepdims=True)
        denom = w.sum()
        if denom == 0:
            return np.zeros(R_neigh.shape[1], dtype=np.float32)
        return (centred.T @ w) / denom

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def fit(self, A: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Return a *new* matrix with all zero entries replaced by predictions.
        """
        if self.method == "item":
            A_filled = self._neighbourhood_based(A.T, verbose).T
        else:
            A_filled = self._neighbourhood_based(A, verbose)

        # clip and round to half-stars
        A_filled = np.clip(A_filled, 0.5, 5.0)
        return np.round(A_filled * 2) / 2

    # ------------------------------------------------------------------ #
    # internal routine
    # ------------------------------------------------------------------ #
    def _neighbourhood_based(self, A: np.ndarray, verbose: bool) -> np.ndarray:
        A_new = A.copy().astype(np.float32)

        for u, r_u in enumerate(A):
            # ---- 1. compute similarity weights to ALL other users ----
            w = np.array([
                0.0 if i == u else
                self._pearson(r_u, r_i) * self._significance(r_u, r_i, self.s)
                for i, r_i in enumerate(A)
            ], dtype=np.float32)

            # ---- 2. pick the k highest-weight neighbours ----
            top_idx = w.argsort()[-self.k:]           # largest k
            w_k     = np.maximum(w[top_idx], 0.0)     # discard negative sims
            if w_k.sum() == 0:
                continue                              # nothing to add

            # ---- 3. fill the zeros for user u ----
            pred_offset = self._prediction(A[top_idx], w_k)
            user_mean   = r_u[r_u > 0].mean() if (r_u > 0).any() else 3.0
            fill_values = user_mean + pred_offset

            missing = r_u == 0
            A_new[u, missing] = fill_values[missing]

            if verbose and u % 50 == 0:
                print(f"processed {u}/{A.shape[0]} users", end="\r")

        if verbose:
            print("\nDone.")
        return A_new

if __name__ == "__main__":
    A = util.load_data_matrix()
    cf = CollaborativeFiltering(k=50, s=10)
    A_pred = cf.fit(A, verbose=True)

    rec_cols = np.argsort(A_pred[1])[:5]

    B = pickle.load(open('data/data_dicts.p', 'rb'))
    print("\nRecommendations for user-1:")
    for col in rec_cols:
        movie_id = next(k for k, v in B['movieId_movieCol'].items() if v == col)
        print(B['movieId_movieName'][movie_id])
