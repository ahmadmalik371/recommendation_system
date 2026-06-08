import os
import zipfile
import requests
import pandas as pd

DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.expanduser("~"), ".recsys_data", "movielens100k")

def download_movielens(url=DATA_URL, dest_dir=DATA_DIR):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    zip_path = os.path.join(dest_dir, "ml-100k.zip")
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K dataset...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    
    # Unzip if not already unzipped
    extracted_path = os.path.join(dest_dir, "ml-100k")
    if not os.path.exists(extracted_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
    return extracted_path

def load_ratings(data_path):
    ratings_file = os.path.join(data_path, "u.data")
    # Load as dataframe; tab-separated; columns: user_id, item_id, rating, timestamp
    df = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    # Drop timestamp for simplicity
    df = df.drop(columns=['timestamp'])
    return df

def load_item_metadata(data_path):
    items_file = os.path.join(data_path, "u.item")
    # Movie metadata is '|' separated; first 5 columns: movie id, title, release date, video release date, IMDb url
    # Followed by genres (19 columns binary)
    columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_url'] + \
              ['genre_' + str(i) for i in range(19)]
    df = pd.read_csv(items_file, sep='|', names=columns, encoding='latin-1')
    return df

def main():
    data_path = download_movielens()
    ratings_df = load_ratings(data_path)
    items_df = load_item_metadata(data_path)
    print(f"Ratings shape: {ratings_df.shape}")
    print(f"Items shape: {items_df.shape}")
    return ratings_df, items_df

if __name__ == "__main__":
    main()