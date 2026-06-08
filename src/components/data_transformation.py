import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def one_hot_encode_genres(items_df, genre_col_prefix="genre_"):
    """
    From binary genre columns, create an item feature matrix.
    """
    genre_cols = [col for col in items_df.columns if col.startswith(genre_col_prefix)]
    feature_matrix = items_df[genre_cols].astype(int).values
    return feature_matrix

def create_user_profile(ratings_df, item_features, user_id, user_col='user_id', item_col='item_id', rating_col='rating'):
    """
    Compute user profile vector as weighted sum of item features by user ratings.
    
    Parameters:
    - ratings_df: DataFrame with user-item ratings.
    - item_features: numpy array (n_items x n_features)
    - user_id: ID of the user for profile calculation.
    
    Returns:
    - user_profile: 1D numpy array (feature vector)
    """
    # Get ratings of selected user
    user_ratings = ratings_df[ratings_df[user_col] == user_id]
    if user_ratings.empty:
        raise ValueError(f"No ratings found for user {user_id}")

    # Map item ids in ratings to indices in item_features matrix
    # Assuming item_features rows correspond to items sorted by 'item_id' ascending
    # Consistent mapping must be maintained during ingestion; here we assume item_id - 1 indexing (MovieLens 100k)
    
    indices = user_ratings[item_col].values - 1  # MovieLens 100k items start at 1
    ratings = user_ratings[rating_col].values
    
    # Weighted sum of item feature vectors
    weighted_features = np.multiply(item_features[indices].T, ratings).T
    user_profile = weighted_features.sum(axis=0)
    return user_profile

def recommend_items(user_profile, item_features, rated_item_indices, top_k=10):
    """
    Recommend top_k items based on cosine similarity with user profile.
    
    Parameters:
    - user_profile: user feature vector
    - item_features: matrix of all item features
    - rated_item_indices: indices of items already rated by the user (to exclude)
    - top_k: number of recommendations
    
    Returns:
    - indices of recommended items
    """
    # Compute cosine similarity between user profile and all items
    user_profile = user_profile.reshape(1, -1)
    similarities = cosine_similarity(user_profile, item_features).flatten()
    
    # Exclude items already rated
    similarities[rated_item_indices] = -np.inf