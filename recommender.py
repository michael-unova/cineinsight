import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_movie_data():
    """Load and prepare MovieLens data."""
    movies  = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    
    # Average rating and count per movie
    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    movies = movies.merge(stats, on='movieId', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0).round(2)
    movies['num_ratings'] = movies['num_ratings'].fillna(0).astype(int)
    
    # Clean up year from title e.g. "Toy Story (1995)" -> year=1995
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
    movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    
    return movies

def build_recommender(movies):
    """Build TF-IDF matrix on genres for content-based filtering."""
    # Replace | with space so 'Action|Adventure' becomes 'Action Adventure'
    movies['genre_text'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genre_text'])
    
    # Cosine similarity matrix - this is the core of the recommender
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Map movie titles to their index
    indices = pd.Series(movies.index, index=movies['clean_title'].str.lower())
    
    return cosine_sim, indices

def get_recommendations(title, movies, cosine_sim, indices, top_n=6):
    """Get top N movie recommendations for a given title."""
    title_lower = title.lower().strip()
    
    # Find closest match
    matches = [idx for t, idx in indices.items() if title_lower in t]
    if not matches:
        return pd.DataFrame()  # No match found
    
    idx = matches[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Skip the movie itself
    
    movie_indices = [i[0] for i in sim_scores]
    result = movies.iloc[movie_indices][['clean_title', 'genres', 'avg_rating', 'num_ratings', 'year']].copy()
    result['similarity'] = [round(s[1], 3) for s in sim_scores]
    
    return result

def search_movies(query, movies, top_n=10):
    """Search movies by title."""
    query = query.lower()
    mask = movies['clean_title'].str.lower().str.contains(query, na=False)
    results = movies[mask].sort_values('num_ratings', ascending=False)
    return results.head(top_n)

if __name__ == "__main__":
    print("Loading movie data...")
    movies = load_movie_data()
    print(f"Loaded {len(movies)} movies")
    
    print("Building recommender...")
    cosine_sim, indices = build_recommender(movies)
    
    print("\nTest recommendations for 'Toy Story':")
    recs = get_recommendations("Toy Story", movies, cosine_sim, indices)
    print(recs[['clean_title', 'genres', 'avg_rating']].to_string())