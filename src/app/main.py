import os
import sys
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

# Root path alignment
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.pipeline.predict_pipeline import load_model_artifacts, recommend_for_user
from src.components.data_ingestion import download_movielens, load_item_metadata

app = FastAPI(title="Recommendation System Insights")

templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "src", "app", "templates"))

# Pre-load core artifacts
data_path = download_movielens()
ratings_df = pd.read_csv(
    os.path.join(data_path, "u.data"), 
    sep='\t', 
    names=['user_id', 'item_id', 'rating', 'timestamp']
).drop(columns=['timestamp'])
items_df = load_item_metadata(data_path)

U, V, user_mapper, item_mapper = load_model_artifacts()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user_id: int = Query(default=196)):
    # 1. Pipeline Execution
    recs = recommend_for_user(user_id, U, V, user_mapper, item_mapper, ratings_df, top_k=10)
    
    # 2. Extract matches
    matched_movies = items_df[items_df['movie_id'].isin(recs)].copy()
    matched_movies['movie_id'] = matched_movies['movie_id'].astype('category')
    matched_movies['movie_id'] = matched_movies['movie_id'].cat.set_categories(recs, ordered=True)
    matched_movies = matched_movies.sort_values('movie_id').reset_index(drop=True)
    
    # 3. Create flat datatypes
    movie_list = []
    for idx, row in matched_movies.iterrows():
        movie_list.append({
            "rank": int(idx + 1),
            "title": str(row['title']),
            "release": str(row.get('release_date', 'Unknown')),
            "delay_index": int(idx)
        })
        
    is_cold_start = user_id not in user_mapper

    # 4. FIX: Passing 'request' directly as the first argument prevents the cache dict crash
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user_id": int(user_id),
            "movies": movie_list,
            "is_cold_start": is_cold_start
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Modified to run universally on cloud containers or local networks
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)