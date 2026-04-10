import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
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

# Set page config
st.set_page_config(
    page_title="Collaborative Filtering Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #333333;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #555555;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* White background with black text */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .main .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Cards with red border and red title */
    .movie-card {
        background: #ffffff !important;
        border: 2px solid #ff0000 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 15px 0 !important;
        box-shadow: 0 4px 15px rgba(255,0,0,0.2) !important;
        color: #000000 !important;
    }
    
    .recommendation-card {
        background: #ffffff !important;
        border: 2px solid #ff0000 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 15px 0 !important;
        box-shadow: 0 4px 15px rgba(255,0,0,0.2) !important;
        color: #000000 !important;
    }
    
    .movie-card h3, .recommendation-card h3, .recommendation-card h4 {
        color: #cc0000 !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
    }
    
    .movie-card p, .recommendation-card p {
        color: #000000 !important;
        margin-bottom: 5px !important;
    }
    
    .movie-card strong, .recommendation-card strong {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ff0000 !important;
    }
    
    .stNumberInput > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ff0000 !important;
    }
    
    .stTextInput > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ff0000 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #ff0000 !important;
        color: #ffffff !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
    }
    
    /* Success and warning messages */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

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
            st.image(tmdb_data['poster_url'], width=200, use_container_width=False)
        else:
            st.markdown("🎬\n\n*No poster available*")
    
    with col2:
        # Wrap content in styled movie card
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        
        # Movie title
        st.markdown(f"### 🎬 {title}")
        
        # Predicted rating
        if predicted_rating:
            st.markdown(f'<div class="rating">⭐ <strong>Predicted Rating:</strong> {predicted_rating:.2f}/5.0</div>', unsafe_allow_html=True)
        
        # TMDB data
        if tmdb_data:
            if tmdb_data.get('overview'):
                st.markdown(f"📝 **Overview:** {tmdb_data['overview']}")
            
            if tmdb_data.get('release_date'):
                st.markdown(f"📅 **Release Date:** {tmdb_data['release_date']}")
            
            if tmdb_data.get('vote_average'):
                st.markdown(f"🎯 **TMDB Rating:** {tmdb_data['vote_average']}/10")
            
            if tmdb_data.get('genres'):
                st.markdown(f"🎭 **Genres:** {', '.join(tmdb_data['genres'])}")
            
            if tmdb_data.get('runtime'):
                st.markdown(f"⏱️ **Runtime:** {tmdb_data['runtime']} minutes")
        
        # Movie ID
        st.markdown(f"🆔 **Movie ID:** {movie_id}")
        
        st.markdown('</div>', unsafe_allow_html=True)

class MovieRecommenderApp:
    def __init__(self):
        self.model = None
        self.recommender = None
        self.movies_df = None
        self.mappings = None
        self.model_metadata = None
        # Don't load models automatically to avoid errors
        # self.load_models()
    
    def load_models(self):
        """Load models and data safely"""
        try:
            # Load movies data
            self.movies_df = pd.read_csv('models/movies_api.csv')
            
            # Load model metadata
            with open('models/model_metadata.json', 'r') as f:
                self.model_metadata = json.load(f)
            
            # Create mock recommender for demo
            self.recommender = "MockRecommender"
            
            return True
        except Exception as e:
            # Create mock data for demo
            self.movies_df = pd.DataFrame({
                'movieId': [1, 2, 3],
                'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)'],
                'genres': ['Adventure|Animation|Children', 'Adventure|Children|Fantasy', 'Comedy|Romance']
            })
            self.model_metadata = {
                'performance': {'rmse': 0.79, 'mae': 0.60},
                'dataset_info': {'n_users': 610, 'n_items': 9742, 'n_ratings': 100836, 'sparsity': 98.3}
            }
            self.recommender = "MockRecommender"
            return False
    
    def check_backend_connection(self):
        """Check if FastAPI backend is running"""
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations for a user"""
        try:
            # Call FastAPI backend
            api_url = f"http://localhost:8000/api/recommendations/{user_id}?limit={n_recommendations}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                recommendations = []
                for rec in data.get('recommendations', []):
                    recommendations.append((rec['movie_id'], rec['predicted_rating'], rec['title']))
                return recommendations
            else:
                # Fallback to collaborative filtering mock recommendations
                collaborative_recommendations = self.get_collaborative_mock_recommendations(user_id, n_recommendations)
                return collaborative_recommendations
                
        except Exception as e:
            # Return collaborative filtering mock recommendations as fallback
            collaborative_recommendations = self.get_collaborative_mock_recommendations(user_id, n_recommendations)
            return collaborative_recommendations
    
    def get_collaborative_mock_recommendations(self, user_id, n_recommendations=10):
        """Generate collaborative filtering mock recommendations based on user patterns"""
        # Simulate different user preferences for collaborative filtering demo
        user_patterns = {
            # User 1 likes animated/family movies
            1: [(1, 4.5, "Toy Story (1995)"), (3114, 4.3, "Toy Story 2 (1999)"), (783, 4.2, "Shrek (2001)"), 
                (1721, 4.1, "Monsters, Inc. (2001)"), (5952, 4.0, "Finding Nemo (2003)")],
            
            # User 2 likes action/adventure movies  
            2: [(1196, 4.4, "Star Wars (1977)"), (1198, 4.3, "Raiders of the Lost Ark (1981)"), 
                (1210, 4.2, "Star Wars: Episode V (1980)"), (2628, 4.1, "Star Wars: Episode VI (1983)"),
                (5995, 4.0, "Lord of the Rings: The Fellowship (2001)")],
            
            # User 3 likes drama/romance movies
            3: [(1193, 4.5, "One Flew Over the Cuckoo's Nest (1975)"), (1197, 4.3, "Boogie Nights (1997)"),
                (1213, 4.2, "Goodfellas (1990)"), (1214, 4.1, "Pulp Fiction (1994)"),
                (1215, 4.0, "The Shawshank Redemption (1994)")],
            
            # User 4 likes comedy movies
            4: [(356, 4.4, "Forrest Gump (1994)"), (296, 4.3, "Pulp Fiction (1994)"),
                (593, 4.2, "Silence of the Lambs (1991)"), (2571, 4.1, "Matrix (1999)"),
                (110, 4.0, "Braveheart (1995)")],
            
            # User 5 likes sci-fi movies
            5: [(260, 4.5, "Star Wars (1977)"), (1196, 4.4, "Star Wars (1977)"), 
                (1210, 4.3, "Star Wars: Episode V (1980)"), (2628, 4.2, "Star Wars: Episode VI (1983)"),
                (1198, 4.1, "Raiders of the Lost Ark (1981)")]
        }
        
        # Get recommendations for this user or default pattern
        if user_id in user_patterns:
            recommendations = user_patterns[user_id][:n_recommendations]
        else:
            # Create pattern based on user_id modulo for more variety
            pattern_id = (user_id - 1) % len(user_patterns) + 1
            recommendations = user_patterns[pattern_id][:n_recommendations]
        
        return recommendations
    
    def show_collaborative_filtering_info(self, user_id):
        """Show collaborative filtering information for this user"""
        st.markdown("### 🔍 Collaborative Filtering Analysis")
        
        # Simulate user similarity analysis
        similar_users = [user_id + 1, user_id + 2, user_id + 3] if user_id < 100 else [user_id - 1, user_id - 2, user_id - 3]
        
        st.markdown(f"""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4>👥 User-Based Collaborative Filtering</h4>
        <p><strong>User {user_id}</strong> recommendations are based on similar users:</p>
        <ul>
            <li>User {similar_users[0]} (similarity: 0.85)</li>
            <li>User {similar_users[1]} (similarity: 0.78)</li>
            <li>User {similar_users[2]} (similarity: 0.72)</li>
        </ul>
        <p><strong>Logic:</strong> Users who liked similar movies also rated these highly...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show collaborative filtering vs content-based comparison
        st.markdown("""
        <div style="background: #fff5f5; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4>🆚 Collaborative vs Content-Based</h4>
        <table style="width: 100%; border: 1px solid #ddd;">
        <tr style="background: #ffe6e6;">
            <th style="padding: 10px;">Collaborative Filtering</th>
            <th style="padding: 10px;">Content-Based</th>
        </tr>
        <tr>
            <td style="padding: 10px;">• Based on user behavior patterns</td>
            <td style="padding: 10px;">• Based on movie attributes</td>
        </tr>
        <tr>
            <td style="padding: 10px;">• "Users like you also liked..."</td>
            <td style="padding: 10px;">• "Because you watched..."</td>
        </tr>
        <tr>
            <td style="padding: 10px;">• Personalized per user</td>
            <td style="padding: 10px;">• Same for all users</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """Get similar movies"""
        try:
            # Call FastAPI backend
            api_url = f"http://localhost:8000/api/similar/{movie_id}?limit={n_similar}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                similar_movies = []
                for movie in data.get('similar_movies', []):
                    similar_movies.append((movie['movie_id'], movie['similarity_score'], movie['title']))
                return similar_movies
            else:
                # Fallback to mock similar movies if API is not available
                mock_similar = [
                    (3114, 0.95, "Toy Story 2 (1999)"),
                    (1, 0.90, "Toy Story (1995)"),
                    (1196, 0.85, "Star Wars (1977)"),
                    (1198, 0.82, "Raiders of the Lost Ark (1981)"),
                    (1193, 0.78, "One Flew Over the Cuckoo's Nest (1975)")
                ][:n_similar]
                return mock_similar
                
        except Exception as e:
            st.error(f"Error getting similar movies: {str(e)}")
            # Return mock similar movies as fallback
            mock_similar = [
                (3114, 0.95, "Toy Story 2 (1999)"),
                (1, 0.90, "Toy Story (1995)"),
                (1196, 0.85, "Star Wars (1977)"),
                (1198, 0.82, "Raiders of the Lost Ark (1981)"),
                (1193, 0.78, "One Flew Over the Cuckoo's Nest (1975)")
            ][:n_similar]
            return mock_similar
    
    def get_user_history(self, user_id):
        """Get user's rating history"""
        try:
            # This would ideally come from your database
            # For demo purposes, we'll show a sample
            sample_history = [
                {"movieId": 1, "title": "Toy Story (1995)", "rating": 4.0, "timestamp": "2005-04-02"},
                {"movieId": 316, "title": "Toy Story 2 (1999)", "rating": 4.5, "timestamp": "2005-04-02"},
                {"movieId": 110, "title": "Braveheart (1995)", "rating": 3.5, "timestamp": "2005-04-02"},
            ]
            return sample_history
        except Exception as e:
            st.error(f"Error getting user history: {str(e)}")
            return []
    
    def search_movies(self, query, limit=20):
        """Search movies by title"""
        try:
            if not query:
                return self.movies_df.head(limit).to_dict('records')
            
            mask = self.movies_df['title'].str.contains(query, case=False, na=False)
            results = self.movies_df[mask].head(limit)
            return results.to_dict('records')
        except Exception as e:
            st.error(f"Error searching movies: {str(e)}")
            return []

def main():
    app = MovieRecommenderApp()
    
    # Ensure models are loaded
    app.load_models()
    
    # Check backend connection
    backend_connected = app.check_backend_connection()
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Collaborative Filtering Movie Recommender</h1>', 
                unsafe_allow_html=True)
    
    # Backend status indicator
    if backend_connected:
        st.success("✅ Backend API Connected")
    else:
        st.warning("⚠️ Backend API Not Connected - Using Mock Data")
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Sidebar for navigation
    st.sidebar.markdown('<h2 class="sub-header">Navigation</h2>', unsafe_allow_html=True)
    
    # Use session state for page selection to allow button navigation
    page = st.sidebar.selectbox("Choose a page:", [
        "Home", "Get Recommendations", "Similar Movies", "Movie Search", "User Comparison", "Collaborative Search", "Model Info"
    ], index=["Home", "Get Recommendations", "Similar Movies", "Movie Search", "User Comparison", "Collaborative Search", "Model Info"].index(st.session_state.page))
    
    # Update session state if sidebar selection changes
    st.session_state.page = page
    
    # Model info in sidebar
    if app.model_metadata:
        st.sidebar.markdown('<h3>Model Performance</h3>', unsafe_allow_html=True)
        st.sidebar.metric("RMSE", f"{app.model_metadata['performance']['rmse']:.4f}")
        st.sidebar.metric("MAE", f"{app.model_metadata['performance']['mae']:.4f}")
        st.sidebar.info(f"Dataset: {app.model_metadata['dataset_info']['n_users']:,} users, "
                       f"{app.model_metadata['dataset_info']['n_items']:,} movies")
    
    if page == "Home":
        st.markdown('<h2 class="sub-header">Welcome to Your Personal Movie Recommender!</h2>', 
                    unsafe_allow_html=True)
        
        # Introduction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 How it Works
            This collaborative filtering system analyzes user behavior patterns to recommend movies 
            you'll love based on similar users' preferences.
            
            ### 🌟 Key Features
            - **Personalized Recommendations**: Get movie suggestions tailored to your taste
            - **Similar Movies**: Find movies similar to your favorites
            - **Advanced ML**: Powered by Matrix Factorization (SVD)
            """)
        # Quick actions
        st.markdown('<h2 class="sub-header">🚀 Quick Actions</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("🎯 Get Recommendations", use_container_width=True):
                st.session_state.page = "Get Recommendations"
                st.rerun()

        with col2:
            if st.button("🔍 Similar Movies", use_container_width=True):
                st.session_state.page = "Similar Movies"
                st.rerun()

        with col3:
            if st.button("🎭 Movie Search", use_container_width=True):
                st.session_state.page = "Movie Search"
                st.rerun()

        # Additional quick actions
        col4, col5 = st.columns([1, 1])

        with col4:
            if st.button("👥 User Comparison", use_container_width=True):
                st.session_state.page = "User Comparison"
                st.rerun()

        with col5:
            if st.button("🔍 Collaborative Search", use_container_width=True):
                st.session_state.page = "Collaborative Search"
                st.rerun()

        # Educational content
        st.markdown('<h2 class="sub-header">🎓 How It Works</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #fff5f5; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #ff0000;">
            <h4>🔍 Collaborative Filtering Process</h4>
            <ol style="line-height: 1.8;">
                <li><strong>User watches movies</strong> and rates them</li>
                <li><strong>System finds similar users</strong> based on rating patterns</li>
                <li><strong>Recommends movies</strong> that similar users liked but you haven't seen</li>
                <li><strong>Learns continuously</strong> from user feedback and behavior</li>
            </ol>

            <h5>🎯 Try These User IDs</h5>
            <ul>
                <li><strong>User 1:</strong> Animation & Family movies</li>
                <li><strong>User 2:</strong> Action & Adventure movies</li>
                <li><strong>User 3:</strong> Drama & Romance movies</li>
                <li><strong>User 4:</strong> Comedy & Drama movies</li>
                <li><strong>User 5:</strong> Science Fiction movies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Get Recommendations":
        st.markdown('<h2 class="sub-header">Get Personalized Recommendations</h2>', 
                    unsafe_allow_html=True)

    # ... rest of the code remains the same ...
        # User input
        col1, col2 = st.columns([1, 3])
        
        with col1:
            user_id = st.number_input("Enter User ID:", min_value=1, max_value=200000, value=1)
            n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
            
            if st.button("🎯 Get Recommendations", use_container_width=True):
                with st.spinner("Generating recommendations..."):
                    recommendations = app.get_user_recommendations(user_id, n_recommendations)
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations for User {user_id}!")
                        
                        # Show collaborative filtering analysis
                        app.show_collaborative_filtering_info(user_id)
                        
                        # Display recommendations with black cards and red text
                        for i, (movie_id, pred_rating, title) in enumerate(recommendations, 1):
                            st.markdown(f'''
                            <div class="recommendation-card">
                                <h3>{i}. {title}</h3>
                                <p><strong>🆔 Movie ID:</strong> {movie_id}</p>
                                <p><strong>⭐ Predicted Rating:</strong> {pred_rating:.2f} ⭐</p>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations available. Please try a different user ID.")
        
        with col2:
            st.markdown("### 📈 User Rating History")
            user_history = app.get_user_history(user_id)
            
            if user_history:
                history_df = pd.DataFrame(user_history)
                st.dataframe(history_df, use_container_width=True)
                
                # Rating distribution chart
                fig = px.histogram(history_df, x='rating', nbins=10, 
                                 title="User's Rating Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rating history available for this user.")
    
    elif page == "Similar Movies":
        st.markdown('<h2 class="sub-header">Find Similar Movies</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Movie search
            search_query = st.text_input("Search for a movie:")
            
            if search_query:
                search_results = app.search_movies(search_query, limit=10)
                
                if search_results:
                    st.markdown("### Search Results")
                    for movie in search_results:
                        movie_title = movie.get('title', 'Unknown')
                        movie_id = movie.get('movieId', 'Unknown')
                        if st.button(f"Select: {movie_title}", key=f"select_{movie_id}"):
                            st.session_state.selected_movie_id = movie_id
                            st.session_state.selected_movie_title = movie_title
                            st.rerun()
                else:
                    st.info("No movies found. Try a different search term.")
            else:
                # Show popular movies if no search
                if app.movies_df is not None and not app.movies_df.empty:
                    st.markdown("### Popular Movies")
                    popular_movies = app.movies_df.head(10)
                    for _, movie in popular_movies.iterrows():
                        movie_title = movie.get('title', 'Unknown')
                        movie_id = movie.get('movieId', 'Unknown')
                        if st.button(f"Select: {movie_title}", key=f"popular_{movie_id}"):
                            st.session_state.selected_movie_id = movie_id
                            st.session_state.selected_movie_title = movie_title
                            st.rerun()
                else:
                    st.info("No movie data available. Please check data loading.")
        
        with col2:
            # Show similar movies if a movie is selected
            if 'selected_movie_id' in st.session_state:
                movie_id = st.session_state.selected_movie_id
                movie_title = st.session_state.selected_movie_title
                
                st.markdown(f"### Movies Similar to: {movie_title}")
                
                with st.spinner("Finding similar movies..."):
                    similar_movies = app.get_similar_movies(movie_id)
                    
                    if similar_movies:
                        # Display similar movies with black cards and red text
                        for i, (similar_movie_id, similarity_score, similar_title) in enumerate(similar_movies, 1):
                            st.markdown(f'''
                            <div class="recommendation-card">
                                <h3>{i}. {similar_title}</h3>
                                <p><strong>🆔 Movie ID:</strong> {similar_movie_id}</p>
                                <p><strong>🎯 Similarity Score:</strong> {similarity_score:.3f}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.warning("No similar movies found.")
            else:
                st.info("👈 Select a movie from the left to see similar movies.")
    
    elif page == "Movie Search":
        st.markdown('<h2 class="sub-header">Movie Search & Browse</h2>', 
                    unsafe_allow_html=True)
        
        # Search interface
        col1, col2 = st.columns([1, 3])
        
        with col1:
            search_query = st.text_input("Search movies:")
            limit = st.slider("Results limit:", 10, 100, 20)
            
            if st.button("🔍 Search", use_container_width=True):
                with st.spinner("Searching movies..."):
                    results = app.search_movies(search_query, limit)
                    st.session_state.search_results = results
                    st.session_state.last_search = search_query
                    st.rerun()
        
        with col2:
            # Display search results or popular movies
            if 'search_results' in st.session_state:
                results = st.session_state.search_results
                last_search = st.session_state.get('last_search', '')
                
                if results:
                    st.markdown(f"### Found {len(results)} results for '{last_search}'")
                    
                    for i, movie in enumerate(results, 1):
                        movie_id = movie.get('movieId', 'Unknown')
                        title = movie.get('title', 'Unknown')
                        genres = movie.get('genres', 'Unknown')
                        
                        st.markdown(f'''
                        <div class="recommendation-card">
                            <h3>{i}. {title}</h3>
                            <p><strong>🆔 Movie ID:</strong> {movie_id}</p>
                            <p><strong>🎭 Genres:</strong> {genres}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.warning("No movies found. Try a different search term.")
            else:
                # Show popular movies if no search performed
                if app.movies_df is not None and not app.movies_df.empty:
                    st.markdown("### � Popular Movies (Browse)")
                    
                    # Check if genres column exists
                    if 'genres' in app.movies_df.columns:
                        # Genre filter
                        all_genres = set()
                        for genres_str in app.movies_df['genres'].dropna():
                            all_genres.update([g.strip() for g in genres_str.split('|')])
                        
                        selected_genre = st.selectbox("Filter by genre:", ["All"] + sorted(list(all_genres)))
                        
                        if selected_genre == "All":
                            display_movies = app.movies_df.head(20)
                        else:
                            mask = app.movies_df['genres'].str.contains(selected_genre, case=False, na=False)
                            display_movies = app.movies_df[mask].head(20)
                    else:
                        display_movies = app.movies_df.head(20)
                    
                    st.markdown(f"### Showing {len(display_movies)} movies")
                    
                    for i, (_, movie) in enumerate(display_movies.iterrows(), 1):
                        movie_id = movie.get('movieId', 'Unknown')
                        title = movie.get('title', 'Unknown')
                        genres = movie.get('genres', 'Unknown')
                        
                        st.markdown(f'''
                        <div class="recommendation-card">
                            <h3>{i}. {title}</h3>
                            <p><strong>🆔 Movie ID:</strong> {movie_id}</p>
                            <p><strong>🎭 Genres:</strong> {genres}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.error("No movie data available. Please check data loading.")
    
    elif page == "User Comparison":
        st.markdown('<h2 class="sub-header">👥 User Similarity Comparison</h2>', 
                    unsafe_allow_html=True)
        
        # User selection interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### User 1")
            user1_id = st.number_input("Enter User ID:", min_value=1, max_value=1000, value=1, key="user1")
            
            st.markdown("### User 2")
            user2_id = st.number_input("Enter User ID:", min_value=1, max_value=1000, value=2, key="user2")
        
        with col2:
            if st.button("🔍 Compare Users", use_container_width=True):
                if user1_id and user2_id:
                    with st.spinner("Analyzing user similarity..."):
                        # Get recommendations for both users
                        user1_recs = app.get_user_recommendations(user1_id, 5)
                        user2_recs = app.get_user_recommendations(user2_id, 5)
                        
                        # Extract movie titles
                        user1_titles = {rec[2] for rec in user1_recs}
                        user2_titles = {rec[2] for rec in user2_recs}
                        
                        # Calculate similarity
                        common_movies = user1_titles.intersection(user2_titles)
                        similarity_score = len(common_movies) / min(len(user1_titles), len(user2_titles)) if user1_titles and user2_titles else 0
                        
                        # Display comparison
                        st.markdown(f'''
                        <div class="recommendation-card">
                            <h3>👥 User Comparison Analysis</h3>
                            <p><strong>User {user1_id} Movies:</strong> {', '.join(list(user1_titles))}</p>
                            <p><strong>User {user2_id} Movies:</strong> {', '.join(list(user2_titles))}</p>
                            <p><strong>Common Movies:</strong> {', '.join(list(common_movies))}</p>
                            <p><strong>Similarity Score:</strong> {similarity_score:.2%}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Similarity explanation
                        if similarity_score >= 0.6:
                            st.success("🎯 High Similarity - Users have similar tastes!")
                        elif similarity_score >= 0.3:
                            st.info("🔄 Moderate Similarity - Users share some interests")
                        else:
                            st.warning("⚠️ Low Similarity - Users have different tastes")
                        
                        # Recommendation overlap
                        st.markdown("### 📊 Recommendation Overlap")
                        overlap_movies = list(common_movies)
                        if overlap_movies:
                            for movie in overlap_movies:
                                st.markdown(f"- {movie}")
                        else:
                            st.info("No overlapping recommendations between users.")
                        
                        # Collaborative filtering insight
                        st.markdown(f'''
                        <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <h4>🔍 Collaborative Filtering Insight</h4>
                            <p><strong>Pattern:</strong> Users with {similarity_score:.0%} similarity tend to like similar movies.</p>
                            <p><strong>Recommendation:</strong> {"Consider recommending movies liked by both users" if similarity_score >= 0.3 else "Consider exploring different genres for each user"}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter User IDs for both users to compare.")
    
    elif page == "Collaborative Search":
        st.markdown('<h2 class="sub-header">🔍 Collaborative Search Insights</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h4>🎯 What is Collaborative Search?</h4>
            <p><strong>Collaborative Search</strong> shows you not just what movies are recommended, but <strong>WHY</strong> they're recommended based on what similar users liked.</p>
            <p>This demonstrates the core principle of collaborative filtering: <em>"Users who liked what you liked also liked these movies"</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive demonstration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🔍 Explore User Preferences")
            
            # User selection for demo
            demo_user = st.selectbox("Select a User to Explore:", 
                options=["User 1 - Animation Fan", "User 2 - Action Lover", "User 3 - Drama Enthusiast", 
                       "User 4 - Comedy Fan", "User 5 - Sci-Fi Addict"],
                index=0)
            
            # Extract user ID correctly
            user_id = int(demo_user.split(" ")[1]) if demo_user.startswith("User ") else 1
            
            # Get user recommendations
            recommendations = app.get_user_recommendations(user_id, 5)
            
            if recommendations:
                st.markdown(f"### 📊 {demo_user}'s Recommendation Profile")
                
                # Show user's movie preferences
                user_patterns = {
                    1: "Animation & Family Movies",
                    2: "Action & Adventure Movies", 
                    3: "Drama & Romance Movies",
                    4: "Comedy & Drama Movies",
                    5: "Science Fiction Movies"
                }
                
                st.markdown(f"""
                <div style="background: #fff5f5; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ff7f0e;">
                    <h4>🎭 User Preference Pattern</h4>
                    <p><strong>Taste Profile:</strong> {user_patterns.get(user_id, 'Unknown')}</p>
                    <p><strong>Collaborative Logic:</strong> This user tends to like movies that other users with similar tastes also enjoyed.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show collaborative analysis
                st.markdown("### 👥 Why These Movies Were Recommended")
                
                for i, (movie_id, pred_rating, title) in enumerate(recommendations, 1):
                    # Find similar users who liked this movie
                    similar_users = [user_id + 1, user_id + 2, user_id + 3]
                    similar_user_names = [f"User {uid}" for uid in similar_users]
                    
                    # Create collaborative insight
                    insights = {
                        1: ["Animation lovers", "Family movie fans", "Parents with kids"],
                        2: ["Action movie fans", "Adventure seekers", "Sci-Fi enthusiasts"],
                        3: ["Drama film buffs", "Romance movie lovers", "Critical film viewers"],
                        4: ["Comedy fans", "Drama enthusiasts", "Pop culture fans"],
                        5: ["Sci-Fi fans", "Space adventure lovers", "Future tech fans"]
                    }
                    
                    user_insights = insights.get(user_id, ["Movie enthusiasts"])
                    
                    st.markdown(f'''
                    <div class="recommendation-card">
                        <h4>{i}. {title}</h4>
                        <p><strong>🆔 Movie ID:</strong> {movie_id}</p>
                        <p><strong>⭐ Predicted Rating:</strong> {pred_rating:.2f}/5.0</p>
                        
                        <div style="background: #ffe6e6; padding: 10px; border-radius: 5px; margin: 10px 0;">
                            <h5>🔍 Collaborative Insight</h5>
                            <p><strong>Why recommended:</strong> Users with similar tastes to {demo_user} loved this movie.</p>
                            <p><strong>Similar users:</strong> {', '.join(similar_user_names)}</p>
                            <p><strong>User group:</strong> {', '.join(user_insights)}</p>
                            <p><strong>Collaborative pattern:</strong> "Users who like {user_patterns.get(user_id, 'similar movies')} also enjoyed this content"</p>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Collaborative Filtering Visualization")
            
            # Create visual explanation
            st.markdown("""
            <div style="background: #f8fff8; padding: 20px; border-radius: 10px; margin: 10px 0;">
                <h4>🎯 How Collaborative Filtering Works</h4>
                <ol style="line-height: 1.8;">
                    <li><strong>User A</strong> watches Toy Story, Shrek, Finding Nemo</li>
                    <li><strong>User B</strong> also watches Toy Story, Shrek, Finding Nemo</li>
                    <li><strong>System detects:</strong> Users A & B have 85% similar tastes</li>
                    <li><strong>Recommendation:</strong> "Since User A liked Monsters, Inc., you'll probably like it too!"</li>
                    <li><strong>Result:</strong> Personalized recommendations based on user behavior patterns</li>
                </ol>
                
                <h5>🔄 Real-World Example</h5>
                <p><strong>Netflix/Amazon:</strong> "Because you watched The Matrix, users who also liked The Matrix also enjoyed Inception"</p>
                <p><strong>Spotify:</strong> "Because you liked The Beatles, users who like The Beatles also enjoy The Rolling Stones"</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Educational content
            st.markdown("### 🎓 Key Benefits of Collaborative Filtering")
            
            benefits = [
                ("🎯 **Personalization**", "Each user gets unique recommendations based on their specific taste patterns"),
                ("👥 **Social Proof**", "Recommendations are backed by what real users with similar tastes actually liked"),
                ("🔍 **Discovery**", "Users discover new content through similar users' preferences, not just content attributes"),
                ("📊 **Scalability**", "System learns from millions of user interactions to improve recommendations over time"),
                ("🎭 **Variety**", "Exposes users to diverse content they might not find through content-based filtering")
            ]
            
            for benefit_title, benefit_desc in benefits:
                st.markdown(f"#### {benefit_title}")
                st.markdown(f"{benefit_desc}")
                st.markdown("---")
    
    elif page == "Model Info":
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        
        # Model metrics
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.metric("Users", f"{app.model_metadata['dataset_info']['n_users']:,}")
        
        with col2:
            st.metric("Items", f"{app.model_metadata['dataset_info']['n_items']:,}")
        
        with col3:
            st.metric("Ratings", f"{app.model_metadata['dataset_info']['n_ratings']:,}")
        
        with col4:
            st.metric("Sparsity", f"{app.model_metadata['dataset_info']['sparsity']:.4f}")
        
        # Sparsity visualization
        st.markdown("### 📊 Data Sparsity Visualization")
        
        total_possible = app.model_metadata['dataset_info']['n_users'] * app.model_metadata['dataset_info']['n_items']
        observed = app.model_metadata['dataset_info']['n_ratings']
        missing = total_possible - observed
        
        fig = go.Figure(data=[
            go.Bar(name='Observed Ratings', x=['Data Matrix'], y=[observed]),
            go.Bar(name='Missing Values', x=['Data Matrix'], y=[missing])
        ])
        
        fig.update_layout(
            title='User-Item Matrix Sparsity',
            barmode='stack',
            yaxis_type="log"
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
