# -*- coding: utf-8 -*-

import numpy as np
import util
import matplotlib.pyplot as plt
import pickle

class Ridge():
    """Weighted Ridge Regression"""
    def __init__(self, f=200, alpha=20, lambd=0.1, thres=0, epsilon=0.1):
        self.f = f
        self.alpha = alpha
        self.lambd = lambd
        self.thres = thres          # ← save it!
        self.epsilon = epsilon


    # Input: Rating matrix A
    # Hyperparameters: Vector Space Dimension: f ; alpha ; lambd
    def fit(self, A, max_iter=30):
        """
        Memory-safe Alternating Least Squares for implicit feedback.
        Works in < 200 MB for the full 671 × 9 066 matrix.
        """
        k, n = A.shape
        P = (A > self.thres).astype(np.float32)        # preference matrix
        C = (1 + self.alpha * A).astype(np.float32)    # confidence matrix

        self.X = np.random.rand(k, self.f).astype(np.float32)
        self.Y = np.random.rand(self.f, n).astype(np.float32)
        eye_f = np.eye(self.f, dtype=np.float32)

        for _ in range(max_iter):
            # ---------- update X ----------
            YTY = self.Y @ self.Y.T                    # (f × f)
            for u in range(k):
                cu = C[u]                              # (n,)
                A_u = YTY + (self.Y * cu) @ self.Y.T + self.lambd * eye_f
                b_u = (self.Y * cu) @ P[u]
                self.X[u] = np.linalg.solve(A_u, b_u)

            # ---------- update Y ----------
            XTX = self.X.T @ self.X                    # (f × f)
            for m in range(n):
                cm = C[:, m]                           # (k,)
                A_m = XTX + (self.X.T * cm) @ self.X + self.lambd * eye_f
                b_m = (self.X.T * cm) @ P[:, m]
                self.Y[:, m] = np.linalg.solve(A_m, b_m)
      
    # K is the number of recommended movies   
    def predict(self, u, K=20):
        """
        Get top K movie recommendations for user u.
        Default K=20 for user-facing recommendations.
        """
        P_u_hat = np.dot(self.X[u] , self.Y)
        indices = np.argsort(P_u_hat)[::-1]  # Sort in descending order for top recommendations
        
        recommended_movies = indices[:K].tolist()
        return recommended_movies
    
    def predict_all(self, u):
        """
        Get all movie rankings for user u (for evaluation purposes).
        Returns movies sorted by preference score (highest first).
        """
        P_u_hat = np.dot(self.X[u] , self.Y)
        indices = np.argsort(P_u_hat)[::-1]  # Sort in descending order
        return indices.tolist()
            
def rank(mat1, r):
    k = len(mat1) #number of users
    n = len(mat1[0]) #number of movies
    sum_numerator = 0
    sum_denominator = np.sum(mat1)
    for u in range(k):
        # Use predict_all for evaluation to get all movie rankings
        recommendations = r.predict_all(u)
        K = len(recommendations)
        
        # Create efficient rank lookup: O(n) instead of O(n²)
        rank_lookup = {movie: idx/(K-1) for idx, movie in enumerate(recommendations)}
        
        for m in range(n):
            if m in rank_lookup:
                sum_numerator += mat1[u, m] * rank_lookup[m]
    
    return(sum_numerator / sum_denominator)
    
if __name__ == "__main__":
    
    '''Basic test of the algorithm'''
    # A = util.load_data_matrix()
    A = util.load_data_matrix()[:,:100] # if tested on a laptop, please use the first 100 movies 
    print(A, A.shape)
    r = Ridge()
    r.fit(A, max_iter=10)  # Reduced iterations for faster training
    
    # Get top 20 recommendations for user 1 (user-facing)
    recommendations = r.predict(1, K=20)
    print(f"\nTop 20 recommendations for user 1:")
    print(recommendations)
    
    B = pickle.load( open('{}'.format('data/data_dicts.p'), 'rb'))
    
    print("\nUser 2's 5-star rated movies:")
    for movie_id,rating in B['userId_rating'][2]:
        if rating ==5 :
            print(B['movieId_movieName'][movie_id] , ", rating:" , rating )
        
    # Convert column indices to movie names for recommendations
    l = recommendations
    k_list =[]
    for movie_column in l :
        for k, v in B['movieId_movieCol'].items():
            if v == movie_column:
                k_list.append(k)
    print("")
    print("Top 20 Recommendations for User 1:")
    for movie_id in k_list :
        print(B['movieId_movieName'][movie_id])
    
    '''Choice of hyperparameters (optimized for speed)'''
    # A = util.load_data_matrix()
    A = util.load_data_matrix()[:,:100] # if tested on a laptop, please use the first 100 movies 
    
    # Reduced parameter ranges for faster execution
    f_range = np.arange(50, 151, 50)  # Only test 3 values: 50, 100, 150
    ranks_f = []
    alpha_range = np.arange(10, 51, 20)  # Only test 3 values: 10, 30, 50
    ranks_alpha = []
    lambd_range = [0.01, 0.1, 1.0]  # Only test 3 values
    ranks_lambd = []
    thres_range = [0, 1.0, 2.0]  # Only test 3 values
    ranks_thres = []

    k = 2  # Reduced from 4 to 2 folds for faster cross-validation
    train_mats, val_mats, masks = util.k_cross(k=k)

    print("\nRunning hyperparameter optimization (this may take a few minutes)...")

    '''Choice of f'''
    print("Testing f values...")
    for f in f_range :
        print(f"  Testing f={f}")
        x=[]
        for i in range(k):
            print(f"    Fold {i+1}/{k}...", end=" ")
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(f=f)
            r.fit(train_mat, max_iter=5)  # Reduced iterations
            x.append(rank(val_mat, r))
            print("✓")
            
        ranks_f.append(np.mean(x)*100)
        
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(f_range,ranks_f, 'o-')
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('f')
    plt.title('Factor Dimension (f)')

    '''Choice of alpha'''
    print("Testing alpha values...")
    for alpha in alpha_range :
        print(f"  Testing alpha={alpha}")
        x=[]
        for i in range(k):
            print(f"    Fold {i+1}/{k}...", end=" ")
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(alpha=alpha)
            r.fit(train_mat, max_iter=5)  # Reduced iterations
            x.append(rank(val_mat, r))
            print("✓")
            
        ranks_alpha.append(np.mean(x)*100)
        
    plt.subplot(2, 2, 2)
    plt.plot(alpha_range,ranks_alpha, 'o-')
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('alpha')
    plt.title('Confidence Parameter (alpha)')
    
    '''Choice of lambda'''
    print("Testing lambda values...")
    for lambd in lambd_range :
        print(f"  Testing lambda={lambd}")
        x=[]
        for i in range(k):
            print(f"    Fold {i+1}/{k}...", end=" ")
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(lambd=lambd)
            r.fit(train_mat, max_iter=5)  # Reduced iterations
            x.append(rank(val_mat, r))
            print("✓")
            
        ranks_lambd.append(np.mean(x)*100)
        
    plt.subplot(2, 2, 3)
    plt.semilogx(lambd_range,ranks_lambd, 'o-')
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('lambda')
    plt.title('Regularization (lambda)')

    '''Choice of threshold'''
    print("Testing threshold values...")
    for thres in thres_range :
        print(f"  Testing threshold={thres}")
        x=[]
        for i in range(k):
            print(f"    Fold {i+1}/{k}...", end=" ")
            train_mat = train_mats[i]
            val_mat = val_mats[i]
            r = Ridge(thres=thres)
            r.fit(train_mat, max_iter=5)  # Reduced iterations
            x.append(rank(val_mat, r))
            print("✓")
            
        ranks_thres.append(np.mean(x)*100)
        
    plt.subplot(2, 2, 4)
    plt.plot(thres_range,ranks_thres, 'o-')
    plt.ylabel('expected percentile ranking (%)')
    plt.xlabel('threshold')
    plt.title('Preference Threshold')

    plt.tight_layout()
    plt.show()
    
    print("\nHyperparameter optimization complete!")
    print(f"Best f: {f_range[np.argmin(ranks_f)]}")
    print(f"Best alpha: {alpha_range[np.argmin(ranks_alpha)]}")
    print(f"Best lambda: {lambd_range[np.argmin(ranks_lambd)]}")
    print(f"Best threshold: {thres_range[np.argmin(ranks_thres)]}")
