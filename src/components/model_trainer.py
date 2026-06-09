# =====================================================================
# PATH CONFIGURATION (CRUCIAL FIX FOR MODULE NOT FOUND ERROR)
# =====================================================================
import sys
import os
# This dynamically finds your 'recommendation_system' folder and makes it discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# This will now load perfectly because of the configuration lines above!
from src.utils import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k

# =====================================================================
# NEIGHBORHOOD-BASED METHODS (KNN)
# =====================================================================

def build_user_item_matrix(ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
    """
    Build user-item matrix in CSR sparse format.
    """
    user_ids = ratings_df[user_col].astype('category')
    item_ids = ratings_df[item_col].astype('category')
    user_idx = user_ids.cat.codes
    item_idx = item_ids.cat.codes
    
    rating_values = ratings_df[rating_col].values
    num_users = user_ids.cat.categories.size
    num_items = item_ids.cat.categories.size
    
    matrix = csr_matrix((rating_values, (user_idx, item_idx)), shape=(num_users, num_items))
    return matrix, user_ids.cat.categories, item_ids.cat.categories

def mean_center(matrix):
    """
    Mean-center each user's ratings; unrated entries remain zero.
    """
    dense = matrix.toarray()
    user_means = np.true_divide(dense.sum(axis=1), (dense != 0).sum(axis=1))
    user_means = np.nan_to_num(user_means)
    
    mean_centered = dense - user_means[:, np.newaxis]
    mean_centered[dense == 0] = 0
    return user_means, mean_centered

def train_knn(mean_centered_matrix, n_neighbors=5, metric='cosine'):
    """
    Fit KNN model on mean-centered user-item matrix.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    knn.fit(mean_centered_matrix)
    return knn

def predict_rating(user_index, item_index, user_means, mean_centered_matrix, knn, k=5):
    """
    Predict rating for user_index on item_index using user-user CF.
    """
    distances, indices = knn.kneighbors(mean_centered_matrix[user_index].reshape(1, -1), n_neighbors=k+1)
    distances = distances.flatten()
    indices = indices.flatten()
    
    mask = indices != user_index
    neighbors = indices[mask][:k]
    sims = 1 - distances[mask][:k]
    
    numerator = 0
    denominator = 0
    for sim, neighbor_idx in zip(sims, neighbors):
        neighbor_rating = mean_centered_matrix[neighbor_idx, item_index]
        if neighbor_rating != 0:
            numerator += sim * neighbor_rating
            denominator += abs(sim)
    if denominator == 0:
        return user_means[user_index]
    
    pred = user_means[user_index] + (numerator / denominator)
    return pred


# =====================================================================
# MODEL-BASED METHODS (MATRIX FACTORIZATION VIA SGD)
# =====================================================================

def train_sgd_matrix_factorization(ratings_df, user_col='user_id', item_col='item_id', rating_col='rating',
                                   n_factors=20, n_epochs=20, lr=0.01, reg=0.1, verbose=True):
    """
    Train matrix factorization using SGD with fixed simultaneous matrix updates.
    """
    user_ids = ratings_df[user_col].unique()
    item_ids = ratings_df[item_col].unique()
    
    user_mapper = {uid: idx for idx, uid in enumerate(user_ids)}
    item_mapper = {iid: idx for idx, iid in enumerate(item_ids)}

    n_users = len(user_ids)
    n_items = len(item_ids)
    
    U = np.random.normal(scale=0.1, size=(n_users, n_factors))
    V = np.random.normal(scale=0.1, size=(n_items, n_factors))
    
    user_idx = ratings_df[user_col].map(user_mapper).values
    item_idx = ratings_df[item_col].map(item_mapper).values
    ratings = ratings_df[rating_col].values
    
    for epoch in range(n_epochs):
        total_loss = 0
        for u, i, r in zip(user_idx, item_idx, ratings):
            prediction = U[u, :].dot(V[i, :])
            e = r - prediction
            total_loss += e**2
            
            u_old = U[u, :].copy()
            
            U[u, :] += lr * (e * V[i, :] - reg * U[u, :])
            V[i, :] += lr * (e * u_old - reg * V[i, :])
        
        epoch_rmse = rmse(ratings, np.array([U[u].dot(V[i]) for u, i in zip(user_idx, item_idx)]))
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} - Training RMSE: {epoch_rmse:.4f}", flush=True)
            
    return U, V, user_mapper, item_mapper

def predict_rating_svd(U, V, user_mapper, item_mapper, user_id, item_id):
    """
    Predict rating for user and item using matrix factorization latent vectors.
    """
    if user_id not in user_mapper or item_id not in item_mapper:
        return None
    
    u_idx = user_mapper[user_id]
    i_idx = item_mapper[item_id]
    return float(U[u_idx, :].dot(V[i_idx, :]))


# =====================================================================
# INTEGRATED MOVIELENS 100K TESTING ENGINE
# =====================================================================

if __name__ == "__main__":
    import time
    
    try:
        from src.components.data_ingestion import download_movielens, load_ratings
    except ImportError:
        print("Error: Could not import data_ingestion! Check your folder workspace pathways.")
        sys.exit(1)

    print("==================================================")
    print("   RUNNING PRODUCTION MOVIELENS PIPELINE TEST      ")
    print("==================================================\n")

    data_path = download_movielens()
    df = load_ratings(data_path)
    print(f"Ingested Real Dataset Shape: {df.shape}\n")

    # Define a test target pair
    target_user_id = 196
    target_item_id = 242

    # Find what the actual historical rating was in MovieLens for validation
    actual_row = df[(df['user_id'] == target_user_id) & (df['item_id'] == target_item_id)]
    actual_rating = actual_row['rating'].values[0] if not actual_row.empty else 4.0
    print(f"Ground Truth Reality: User {target_user_id} rated Movie {target_item_id} a {actual_rating:.1f}\n")
    print("-" * 50)

    # ==================================================
    # PIPELINE 1: USER-USER KNN
    # ==================================================
    print("Executing Pipeline 1: User-User KNN...")
    matrix, user_cats, item_cats = build_user_item_matrix(df)
    user_means, mean_centered_matrix = mean_center(matrix)
    
    try:
        user_index = list(user_cats).index(target_user_id)
        item_index = list(item_cats).index(target_item_id)
        
        knn_model = train_knn(mean_centered_matrix, n_neighbors=5)
        knn_pred = predict_rating(user_index, item_index, user_means, mean_centered_matrix, knn_model, k=5)
        
        knn_error = rmse(np.array([actual_rating]), np.array([knn_pred]))
        print(f"-> KNN Predicted rating: {knn_pred:.2f} (Single Point RMSE: {knn_error:.4f})\n")
    except ValueError:
        print("Target mapping error inside KNN sparse categories.")
        
    print("-" * 50)

    # ==================================================
    # PIPELINE 2: SGD MATRIX FACTORIZATION
    # ==================================================
    print("Executing Pipeline 2: SGD Matrix Factorization...")
    U, V, u_map, i_map = train_sgd_matrix_factorization(
        df, n_factors=10, n_epochs=5, lr=0.01, reg=0.05, verbose=True
    )
    
    svd_pred = predict_rating_svd(U, V, u_map, i_map, target_user_id, target_item_id)
    
    svd_error = rmse(np.array([actual_rating]), np.array([svd_pred]))
    print(f"-> SVD Predicted rating: {svd_pred:.2f} (Single Point RMSE: {svd_error:.4f})\n")
    print("==================================================")
    
    time.sleep(1)