import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================================
# RATING PREDICTION METRICS (Error-Based)
# =====================================================================

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


# =====================================================================
# TOP-K RECOMMENDATION METRICS (Ranking-Based)
# =====================================================================

def precision_at_k(recommended_items, relevant_items, k):
    """Calculate Precision at K."""
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / k

def recall_at_k(recommended_items, relevant_items, k):
    """Calculate Recall at K."""
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / len(relevant_items) if len(relevant_items) > 0 else 0

def ndcg_at_k(recommended_items, relevant_items, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at K.
    Assumes binary relevance (1 if item is in relevant_items, 0 otherwise).
    """
    # 1. Calculate Actual DCG for top-K recommendations
    actual_dcg = 0
    for i, item in enumerate(recommended_items[:k]):
        rel = 1 if item in relevant_items else 0
        actual_dcg += (2 ** rel - 1) / np.log2(i + 2)
        
    # 2. Calculate Ideal DCG (Assuming perfect ordering where all hits are ranked at the top)
    n_relevant_in_k = min(len(relevant_items), k)
    ideal_dcg = 0
    for i in range(n_relevant_in_k):
        # In a perfect recommendation sequence, the top positions are filled with hits (rel = 1)
        ideal_dcg += (2 ** 1 - 1) / np.log2(i + 2)
        
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0