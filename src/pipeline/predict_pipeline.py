import os
import sys
import pickle
import numpy as np
import pandas as pd

# =====================================================================
# PATH CONFIGURATION (Ensures 'src' can be imported seamlessly)
# =====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.components.model_trainer import predict_rating_svd
from src.components.data_ingestion import download_movielens, load_item_metadata

COLD_START_POPULAR_ITEMS = None  # global placeholder for popularity fallback

def load_model_artifacts():
    """Load model matrices and dictionary encoders securely from the system path."""
    model_file_path = os.path.join(PROJECT_ROOT, 'models', 'mf_model.pkl')
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(
            f"Could not locate model artifact at: '{model_file_path}'. "
            "Please run 'python src\\pipeline\\train_pipeline.py' first to build and save the model."
        )
    with open(model_file_path, 'rb') as f:
        U, V, user_mapper, item_mapper = pickle.load(f)
    return U, V, user_mapper, item_mapper

def get_popular_items(ratings_df, threshold=50):
    """Fallback logic to grab top rated items for unmapped users."""
    item_popularity = ratings_df.groupby('item_id').size()
    popular = item_popularity[item_popularity >= threshold].sort_values(ascending=False).index.tolist()
    return popular

def recommend_for_user(user_id, U, V, user_mapper, item_mapper, ratings_df, top_k=10):
    if user_id in user_mapper:
        # Known user: recommend top predicted ratings for unrated items
        u_idx = user_mapper[user_id]
        user_vec = U[u_idx]
        
        all_items = list(item_mapper.keys())
        preds = []
        for item_id in all_items:
            i_idx = item_mapper[item_id]
            pred = user_vec.dot(V[i_idx])
            preds.append((item_id, pred))
        
        # Exclude items this user has already rated in history
        rated_items = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'])
        preds = [p for p in preds if p[0] not in rated_items]
        preds.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in preds[:top_k]]
    else:
        # Cold-start user: fallback to global popular items
        global COLD_START_POPULAR_ITEMS
        if COLD_START_POPULAR_ITEMS is None:
            COLD_START_POPULAR_ITEMS = get_popular_items(ratings_df)
        return COLD_START_POPULAR_ITEMS[:top_k]

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Recommendation Prediction Pipeline")
    parser.add_argument('--user_id', type=int, required=True, help='User ID for recommendations')
    args = parser.parse_args()
    
    # 1. Gather raw data resources
    data_path = download_movielens()
    ratings_df = pd.read_csv(
        os.path.join(data_path, "u.data"), 
        sep='\t', 
        names=['user_id', 'item_id', 'rating', 'timestamp']
    ).drop(columns=['timestamp'])
    
    items_df = load_item_metadata(data_path)
    
    # 2. Extract trained weights
    try:
        U, V, user_mapper, item_mapper = load_model_artifacts()
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    # 3. Process Predictions
    recommendations = recommend_for_user(args.user_id, U, V, user_mapper, item_mapper, ratings_df)
    
    print("==================================================")
    print(f"   RECOMMENDATIONS FOR USER: {args.user_id}       ")
    print("==================================================")
    
    # Map raw numeric item IDs back to string movie names for cleaner readability
    matched_movies = items_df[items_df['movie_id'].isin(recommendations)].copy()
    
    # Ensure display order respects recommendation matrix sequence output rank
    matched_movies['movie_id'] = matched_movies['movie_id'].astype('category')
    matched_movies['movie_id'] = matched_movies['movie_id'].cat.set_categories(recommendations, ordered=True)
    matched_movies = matched_movies.sort_values('movie_id').reset_index(drop=True)
    
    if matched_movies.empty:
        print(f"Raw Recommended Item IDs: {recommendations}")
    else:
        for idx, row in matched_movies.iterrows():
            print(f"{idx+1}. {row['title']}")
    print("==================================================")

if __name__ == "__main__":
    main()