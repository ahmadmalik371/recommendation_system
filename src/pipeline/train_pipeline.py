import os
import sys
import pickle

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

import pandas as pd
from src.components.data_ingestion import download_movielens, load_ratings, load_item_metadata
from src.components.data_transformation import one_hot_encode_genres, create_user_profile, recommend_items
from src.components.model_trainer import train_sgd_matrix_factorization, predict_rating_svd
from src.utils import rmse, mae

def main():
    print("==================================================")
    # Step 1: Data Ingestion
    print("[STEP 1/4] Starting Data Ingestion Layer...")
    data_path = download_movielens()
    ratings_df = load_ratings(data_path)
    items_df = load_item_metadata(data_path)
    print(f"Loaded {len(ratings_df)} ratings and {len(items_df)} movie details.\n")
    
    # Step 2: Train Matrix Factorization Model
    print("[STEP 2/4] Training SGD Matrix Factorization Model...")
    U, V, user_mapper, item_mapper = train_sgd_matrix_factorization(
        ratings_df, n_factors=10, n_epochs=5, lr=0.01, reg=0.05, verbose=True
    )
    
    # --- HARD IMPLEMENTATION: FORCE WRITE TO HARD DRIVE ---
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_file_path = os.path.join(models_dir, "mf_model.pkl")
    
    with open(model_file_path, "wb") as f:
        pickle.dump((U, V, user_mapper, item_mapper), f)
    
    print(f"--> SUCCESS: Model matrices saved safely to disk at: {model_file_path}\n")
    
    # Step 3: Evaluate on a sample hold-out
    print("[STEP 3/4] Evaluating Model Quality on Hold-out Sample...")
    test_df = ratings_df.sample(100, random_state=42)
    
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        pred = predict_rating_svd(U, V, user_mapper, item_mapper, row['user_id'], row['item_id'])
        if pred is not None:
            y_true.append(row['rating'])
            y_pred.append(pred)
    
    print(f"-> Evaluation Complete | Test RMSE: {rmse(y_true, y_pred):.4f} | Test MAE: {mae(y_true, y_pred):.4f}\n")
    
    # Step 4: Cold-Start Recommendation
    print("[STEP 4/4] Calculating Cold-Start Global Recommendations...")
    pop_threshold = 50
    item_stats = ratings_df.groupby('item_id').agg(
        rating_count=('rating', 'count'),
        rating_mean=('rating', 'mean')
    ).reset_index()
    
    trusted_items = item_stats[item_stats['rating_count'] >= pop_threshold]
    top_popular_items = trusted_items.sort_values(by='rating_mean', ascending=False).head(10)
    top_recommendations = pd.merge(top_popular_items, items_df, left_on='item_id', right_on='movie_id')
    
    print("\n--- Top 5 Popular Movies For Cold-Start Users ---")
    for idx, row in top_recommendations.head(5).iterrows():
        print(f"{idx+1}. {row['title']} (Avg Rating: {row['rating_mean']:.2f}, Reviews: {int(row['rating_count'])})")
    print("==================================================")

if __name__ == "__main__":
    main()