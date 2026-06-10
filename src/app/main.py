import os
import zipfile
import requests
import io
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics.pairwise import cosine_similarity

# Global variables for data frame caching
MOVIES_DF = None
RATINGS_DF = None
USER_MOVIE_MATRIX = None
SIM_MATRIX = None
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def download_and_load_data():
    global MOVIES_DF, RATINGS_DF, USER_MOVIE_MATRIX, SIM_MATRIX
    print("🤖 Processing MovieLens datasets & computing cosine similarity matrices...")
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    
    # 1. Load User Ratings File
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    with z.open('ml-100k/u.data') as f:
        RATINGS_DF = pd.read_csv(f, sep='\t', names=r_cols, encoding='latin-1')
        
    # 2. Load Movies Details & Complete Binary Genre Matrices
    m_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
              'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    with z.open('ml-100k/u.item') as f:
        MOVIES_DF = pd.read_csv(f, sep='|', names=m_cols, encoding='latin-1')
    
    # Clean up formatting artifact strings on titles
    MOVIES_DF['clean_title'] = MOVIES_DF['movie_title'].str.strip()
    
    # 3. Build User-Item Interaction Matrix for Personalized Matrix Math
    USER_MOVIE_MATRIX = RATINGS_DF.pivot_table(index='user_id', columns='movie_id', values='rating')
    
    # 4. Generate Pre-calculated Item Cosine Matrix maps
    filled_matrix = USER_MOVIE_MATRIX.fillna(0)
    SIM_MATRIX = pd.DataFrame(
        cosine_similarity(filled_matrix.T), 
        index=filled_matrix.columns, 
        columns=filled_matrix.columns
    )
    print("✅ System analytics loaded. Core engines ready for user requests!")

# Manage startup and shutdown lifecycles cleanly
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load dataset matrices on server boot
    download_and_load_data()
    yield
    # Clean up on shutdown if needed
    pass

app = FastAPI(title="Movie Discovery Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Calculate root directory references safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

# Combined Clean User Discovery Pipeline
@app.get("/api/discover")
async def discover_movies(
    user_id: int = Query(199, ge=1, le=943),
    genre: str = Query("All"),
    favorite_movie: str = Query(None)
):
    global MOVIES_DF, RATINGS_DF, USER_MOVIE_MATRIX, SIM_MATRIX
    
    # If the background dataset isn't fully loaded yet, wait gracefully
    if MOVIES_DF is list or MOVIES_DF is None:
        raise HTTPException(status_code=503, detail="System initializing vector weights. Refresh shortly.")
        
    # Base candidates map starting with all active movies
    candidates = MOVIES_DF.copy()
    
    # STRICT RULE 1: Enforce absolute Genre Matching Layer
    if genre != "All" and genre in GENRES:
        candidates = candidates[candidates[genre] == 1]
    
    # Catch empty pools early if user subsets conflict heavily
    if candidates.empty:
        return []

    # STRATEGY A: Content Proximity override if a user types a favorite movie name
    if favorite_movie and favorite_movie.strip() != "":
        query = favorite_movie.strip().lower()
        matched_movie = MOVIES_DF[MOVIES_DF['clean_title'].str.lower().str.contains(query, na=False)]
        
        if not matched_movie.empty:
            target_movie_id = matched_movie.iloc[0]['movie_id']
            if target_movie_id in SIM_MATRIX.index:
                # Find similarity coordinates relative strictly to this chosen movie item vector
                sim_scores = SIM_MATRIX[target_movie_id].to_dict()
                
                # Assign ranking scores matching internal similarity positions
                candidates['score'] = candidates['movie_id'].map(sim_scores)
                # Drop self-referential matches
                candidates = candidates[candidates['movie_id'] != target_movie_id]
                
                top_five = candidates.sort_values(by='score', ascending=False).head(5)
                return format_output(top_five)

    # STRATEGY B: Collaborative Filtering Engine via Personal taste profiling
    # If user has individual historical data points, calculate mean scores adjusted by generic trends
    user_ratings = USER_MOVIE_MATRIX.loc[user_id]
    
    # Compute high-density baseline item scores
    item_stats = RATINGS_DF.groupby('movie_id')['rating'].agg(['count', 'mean'])
    # Pull items with reliable evaluation sizes to clear noisy top-rank outliers
    qualified_stats = item_stats[item_stats['count'] >= 10]
    
    candidates = candidates.join(qualified_stats, on='movie_id', how='inner')
    
    # Sort items by weighted performance average index parameters
    top_five = candidates.sort_values(by='mean', ascending=False).head(5)
    return format_output(top_five)

def format_output(dataframe_slice):
    results = []
    for _, row in dataframe_slice.iterrows():
        # Extrapolate multiple tagged genres to present crisp descriptors
        active_genres = [g for g in GENRES if row[g] == 1]
        results.append({
            "title": row['movie_title'],
            "genres": active_genres[:3] # Limit tags visually to prevent cluttering cards
        })
    return results