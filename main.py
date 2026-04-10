import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import requests
from datetime import datetime
import sys
from dotenv import load_dotenv

# Try to import surprise, fall back to mock if it fails
try:
    from surprise import SVD
    SURPRISE_AVAILABLE = True
except ImportError:
    from surprise_mock import SVD
    SURPRISE_AVAILABLE = False
    st.warning("Using mock surprise module - scikit-surprise not available")

# Load environment variables
load_dotenv()

# TMDB API integration from environment variables
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TMDB_IMG = os.getenv("TMDB_IMG", "https://image.tmdb.org/t/p/w500")
TMDB_BASE = os.getenv("TMDB_BASE", "https://api.themoviedb.org/3")
TMDB_IMG_500 = os.getenv("TMDB_IMG_500", "https://image.tmdb.org/t/p/w500")

# TMDB API key input (fallback to env)
TMDB_API_KEY = st.sidebar.text_input("🎬 TMDB API Key:", 
                                  value=TMDB_BEARER_TOKEN or "", 
                                  type="password", 
                                  help="Get your key from https://www.themoviedb.org/settings/api")

def get_tmdb_movie_data(title, year=None):
    """Get TMDB data for a movie"""
    if not TMDB_API_KEY or TMDB_API_KEY == "":
        return None
    
    try:
        # Search for movie
        search_url = f"{TMDB_BASE}/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'year': year
        }
        response = requests.get(search_url, params=params)
        
        if response.status_code == 200 and response.json()['results']:
            movie_data = response.json()['results'][0]
            movie_id = movie_data['id']
            
            # Get detailed movie info
            details_url = f"{TMDB_BASE}/movie/{movie_id}"
            details_params = {'api_key': TMDB_API_KEY}
            details_response = requests.get(details_url, params=details_params)
            
            if details_response.status_code == 200:
                details = details_response.json()
                return {
                    'tmdb_id': movie_id,
                    'title': details.get('title'),
                    'overview': details.get('overview'),
                    'release_date': details.get('release_date'),
                    'runtime': details.get('runtime'),
                    'vote_average': details.get('vote_average'),
                    'vote_count': details.get('vote_count'),
                    'popularity': details.get('popularity'),
                    'poster_url': f"{TMDB_IMG_500}{details.get('poster_path')}" if details.get('poster_path') else None,
                    'backdrop_url': f"https://image.tmdb.org/t/p/w1280{details.get('backdrop_path')}" if details.get('backdrop_path') else None,
                    'genres': [g['name'] for g in details.get('genres', [])],
                    'budget': details.get('budget'),
                    'revenue': details.get('revenue')
                }
    except Exception as e:
        st.error(f"TMDB API error: {str(e)}")
    
    return None

def display_movie_card(movie_id, title, predicted_rating=None, show_tmdb=True):
    """Display movie with optional TMDB data"""
    # TMDB data
    tmdb_data = None
    if show_tmdb and TMDB_API_KEY:
        tmdb_data = get_tmdb_movie_data(title)
    
    # Layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if tmdb_data and tmdb_data.get('poster_url'):
            st.image(tmdb_data['poster_url'], width=200)
        else:
            st.markdown("🎬\n\n*No poster*")
    
    with col2:
        st.markdown(f"### {title}")
        
        if predicted_rating:
            st.markdown(f"⭐ **Predicted Rating:** {predicted_rating:.2f}")
        
        if tmdb_data:
            if tmdb_data.get('overview'):
                st.markdown(f"📝 **Overview:** {tmdb_data['overview']}")
            
            if tmdb_data.get('release_date'):
                st.markdown(f"📅 **Release:** {tmdb_data['release_date']}")
            
            if tmdb_data.get('vote_average'):
                st.markdown(f"🎯 **TMDB Rating:** {tmdb_data['vote_average']}/10")
            
            if tmdb_data.get('genres'):
                st.markdown(f"🎭 **Genres:** {', '.join(tmdb_data['genres'])}")
            
            if tmdb_data.get('runtime'):
                st.markdown(f"⏱️ **Runtime:** {tmdb_data['runtime']} min")
        
        st.markdown(f"🆔 **Movie ID:** {movie_id}")

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_components():
    """Load all necessary model components"""
    try:
        # Load model metadata
        with open('models/model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
        
        # Load mappings
        with open('models/api_mappings.json', 'r') as f:
            mappings = json.load(f)
        
        # Load movies data
        movies_df = pd.read_csv('models/movies_api.csv')
        
        # Load recommendation engine
        with open('models/recommendation_engine.pkl', 'rb') as f:
            recommender = pickle.load(f)
        
        # Load SVD model
        with open('models/svd_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        return {
            'model_metadata': model_metadata,
            'mappings': mappings,
            'movies_df': movies_df,
            'recommender': recommender,
            'model': model
        }
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        return None

def get_recommendations_for_user(user_id, components, n_recommendations=10):
    """Get recommendations for a specific user"""
    try:
        recommender = components['recommender']
        recommendations = recommender.get_user_recommendations(user_id, n_recommendations)
        return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

def get_similar_movies_for_movie(movie_id, components, n_similar=10):
    """Get similar movies for a specific movie"""
    try:
        recommender = components['recommender']
        similar_movies = recommender.get_similar_movies(movie_id, n_similar)
        return similar_movies
    except Exception as e:
        st.error(f"Error getting similar movies: {str(e)}")
        return []

def search_movies_by_title(query, components, limit=20):
    """Search movies by title"""
    try:
        movies_df = components['movies_df']
        if not query:
            return movies_df.head(limit).to_dict('records')
        
        mask = movies_df['title'].str.contains(query, case=False, na=False)
        results = movies_df[mask].head(limit)
        return results.to_dict('records')
    except Exception as e:
        st.error(f"Error searching movies: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="Movie Recommender API",
        page_icon="🎬",
        layout="centered"
    )
    
    st.title("🎬 Movie Recommender with TMDB")
    st.markdown("Collaborative filtering + TMDB movie data")
    
    # TMDB API status
    if TMDB_API_KEY:
        st.sidebar.success("✅ TMDB API connected")
    else:
        st.sidebar.warning("⚠️ Enter TMDB API key for enhanced features")
    
    # Load model components
    components = load_model_components()
    
    if not components:
        st.error("❌ Failed to load model components. Please ensure models are trained and saved.")
        st.stop()
    
    st.success("✅ Model components loaded successfully!")
    
    # API endpoints
    st.markdown("## 🚀 Available Features")
    
    feature = st.selectbox("Choose a feature:", [
        "Get User Recommendations",
        "Get Similar Movies", 
        "Search Movies",
        "Model Info"
    ])
    
    if feature == "Get User Recommendations":
        st.markdown("### 📋 Get User Recommendations")
        
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.number_input("User ID:", min_value=1, max_value=200000, value=1)
        with col2:
            n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("🎯 Get Recommendations"):
            recommendations = get_recommendations_for_user(user_id, components, n_recommendations)
            
            if recommendations:
                st.success(f"Found {len(recommendations)} recommendations for User {user_id}")
                
                for i, (movie_id, pred_rating, title) in enumerate(recommendations, 1):
                    st.markdown("---")
                    st.markdown(f"### 🎬 Recommendation {i}")
                    display_movie_card(movie_id, title, pred_rating, show_tmdb=True)
            else:
                st.warning("No recommendations available for this user.")
    
    elif feature == "Get Similar Movies":
        st.markdown("### 🔍 Get Similar Movies")
        
        # Movie search first
        search_query = st.text_input("Search for a movie to find similar ones:")
        
        if search_query:
            search_results = search_movies_by_title(search_query, components, limit=10)
            
            if search_results:
                selected_movie = st.selectbox(
                    "Select a movie:",
                    options=search_results,
                    format_func=lambda x: f"{x['title']} (ID: {x['movieId']})"
                )
                
                n_similar = st.slider("Number of similar movies:", 5, 20, 10)
                
                if st.button("🔍 Find Similar Movies"):
                    similar_movies = get_similar_movies_for_movie(
                        selected_movie['movieId'], components, n_similar
                    )
                    
                    if similar_movies:
                        st.success(f"Movies similar to '{selected_movie['title']}':")
                        
                        for i, (movie_id, similarity, title) in enumerate(similar_movies, 1):
                            st.markdown("---")
                            st.markdown(f"### 🎬 Similar Movie {i}")
                            st.markdown(f"**Similarity Score:** {similarity:.3f}")
                            display_movie_card(movie_id, title, show_tmdb=True)
                    else:
                        st.warning("No similar movies found.")
            else:
                st.warning("No movies found matching your search.")
    
    elif feature == "Search Movies":
        st.markdown("### 🔎 Search Movies")
        
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Search query:")
        with col2:
            limit = st.slider("Results limit:", 10, 100, 20)
        
        if st.button("🔍 Search Movies"):
            search_results = search_movies_by_title(search_query, components, limit)
            
            if search_results:
                st.success(f"Found {len(search_results)} movies")
                
                # Display results with TMDB data
                for movie in search_results:
                    st.markdown("---")
                    display_movie_card(movie['movieId'], movie['title'], show_tmdb=True)
            else:
                st.warning("No movies found.")
    
    elif feature == "Model Info":
        st.markdown("### 📊 Model Information")
        
        model_metadata = components['model_metadata']
        
        st.json({
            "model_info": {
                "algorithm": model_metadata['model_type'],
                "version": model_metadata['version'],
                "training_date": model_metadata['training_date'],
                "parameters": model_metadata['best_params']
            },
            "performance": model_metadata['performance'],
            "dataset": {
                "users": model_metadata['dataset_info']['n_users'],
                "movies": model_metadata['dataset_info']['n_items'],
                "ratings": model_metadata['dataset_info']['n_ratings'],
                "sparsity": model_metadata['dataset_info']['sparsity']
            },
            "tmdb_integration": "enabled" if TMDB_API_KEY else "disabled"
        })

if __name__ == "__main__":
    main()
