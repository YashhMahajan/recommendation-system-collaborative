"""
Movie Recommendation System - Production Flask API

A production-ready REST API for movie recommendations using collaborative filtering.
Built with Flask and optimized for performance and scalability.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv
import logging
from functools import wraps
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    MODEL_DIR = os.getenv('MODEL_DIR', 'models')
    DATA_DIR = os.getenv('DATA_DIR', 'processed_data')
    DEBUG = os.getenv('FLASK_ENV', 'production') == 'development'
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', 5000))
    MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', 50))
    CACHE_TTL = int(os.getenv('CACHE_TTL', 300))  # 5 minutes

app.config.from_object(Config)

# Global variables for loaded models
model_components = None
model_load_time = None

def load_models():
    """Load all required models and data"""
    global model_components, model_load_time
    
    if model_components is not None:
        return model_components
    
    logger.info("Loading models and data...")
    start_time = time.time()
    
    try:
        # Load SVD model
        with open(os.path.join(app.config['MODEL_DIR'], 'svd_model.pkl'), 'rb') as f:
            svd_model = pickle.load(f)
        
        # Load recommendation engine
        with open(os.path.join(app.config['MODEL_DIR'], 'recommendation_engine.pkl'), 'rb') as f:
            recommendation_engine = pickle.load(f)
        
        # Load mappings
        with open(os.path.join(app.config['MODEL_DIR'], 'api_mappings.json'), 'r') as f:
            mappings = json.load(f)
        
        # Load movies data
        movies_df = pd.read_csv(os.path.join(app.config['MODEL_DIR'], 'movies_api.csv'))
        
        # Load model metadata
        with open(os.path.join(app.config['MODEL_DIR'], 'model_metadata.json'), 'r') as f:
            model_metadata = json.load(f)
        
        model_components = {
            'svd_model': svd_model,
            'recommendation_engine': recommendation_engine,
            'user_id_map': {int(k): int(v) for k, v in mappings['user_id_map'].items()},
            'movie_id_map': {int(k): int(v) for k, v in mappings['movie_id_map'].items()},
            'reverse_user_id_map': {int(k): int(v) for k, v in mappings['reverse_user_id_map'].items()},
            'reverse_movie_id_map': {int(k): int(v) for k, v in mappings['reverse_movie_id_map'].items()},
            'movies_df': movies_df,
            'model_metadata': model_metadata
        }
        
        model_load_time = time.time() - start_time
        logger.info(f"Models loaded successfully in {model_load_time:.2f} seconds")
        
        return model_components
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def validate_user_exists(user_id):
    """Check if user exists in our training data"""
    components = load_models()
    return user_id in components['user_id_map']

def validate_movie_exists(movie_id):
    """Check if movie exists in our data"""
    components = load_models()
    return movie_id in components['movie_id_map']

def handle_errors(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({'error': str(e), 'type': 'validation_error'}), 400
        except KeyError as e:
            return jsonify({'error': f'Resource not found: {str(e)}', 'type': 'not_found'}), 404
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            return jsonify({'error': 'Internal server error', 'type': 'server_error'}), 500
    return wrapper

@app.route('/')
def index():
    """API documentation and health check"""
    return jsonify({
        'name': 'Movie Recommendation API',
        'version': '1.0.0',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'recommendations': '/api/recommendations/<user_id>',
            'similar_movies': '/api/similar/<movie_id>',
            'search_movies': '/api/movies',
            'model_info': '/api/model/info',
            'health': '/api/health'
        }
    })

@app.route('/api/health')
def health_check():
    """Detailed health check endpoint"""
    try:
        components = load_models()
        return jsonify({
            'status': 'healthy',
            'models_loaded': True,
            'model_load_time': model_load_time,
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'total_users': components['model_metadata']['dataset_info']['n_users'],
                'total_movies': components['model_metadata']['dataset_info']['n_items'],
                'total_ratings': components['model_metadata']['dataset_info']['n_ratings']
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/model/info')
@handle_errors
def get_model_info():
    """Get detailed model information and performance metrics"""
    components = load_models()
    
    metadata = components['model_metadata']
    
    return jsonify({
        'model_info': {
            'algorithm': metadata['model_type'],
            'version': metadata['version'],
            'training_date': metadata['training_date'],
            'parameters': metadata['best_params']
        },
        'performance': metadata['performance'],
        'dataset': metadata['dataset_info'],
        'api_stats': {
            'model_load_time_seconds': model_load_time,
            'max_recommendations': app.config['MAX_RECOMMENDATIONS']
        }
    })

@app.route('/api/recommendations/<int:user_id>')
@handle_errors
def get_recommendations(user_id):
    """Get personalized movie recommendations for a user"""
    
    # Validate input
    limit = min(int(request.args.get('limit', 10)), app.config['MAX_RECOMMENDATIONS'])
    
    if not validate_user_exists(user_id):
        return jsonify({
            'error': f'User {user_id} not found in training data',
            'type': 'not_found',
            'suggestion': 'Try a different user ID or ensure the user exists in the dataset'
        }), 404
    
    # Get recommendations
    components = load_models()
    recommender = components['recommendation_engine']
    
    recommendations = recommender.get_user_recommendations(user_id, n_recommendations=limit)
    
    if not recommendations:
        return jsonify({
            'user_id': user_id,
            'message': 'No recommendations available for this user',
            'recommendations': [],
            'total_count': 0
        })
    
    # Format response
    formatted_recommendations = []
    for movie_id, predicted_rating, title in recommendations:
        movie_info = components['movies_df'][components['movies_df']['movieId'] == movie_id]
        genres = movie_info['genres'].iloc[0] if not movie_info.empty else 'Unknown'
        
        formatted_recommendations.append({
            'movie_id': movie_id,
            'title': title,
            'predicted_rating': round(predicted_rating, 2),
            'genres': genres,
            'confidence': 'high' if predicted_rating >= 4.0 else 'medium' if predicted_rating >= 3.0 else 'low'
        })
    
    return jsonify({
        'user_id': user_id,
        'recommendations': formatted_recommendations,
        'total_count': len(formatted_recommendations),
        'generated_at': datetime.now().isoformat()
    })

@app.route('/api/similar/<int:movie_id>')
@handle_errors
def get_similar_movies(movie_id):
    """Get movies similar to the specified movie"""
    
    # Validate input
    limit = min(int(request.args.get('limit', 5)), app.config['MAX_RECOMMENDATIONS'])
    
    if not validate_movie_exists(movie_id):
        return jsonify({
            'error': f'Movie {movie_id} not found in dataset',
            'type': 'not_found',
            'suggestion': 'Try a different movie ID or search for movies first'
        }), 404
    
    # Get similar movies
    components = load_models()
    recommender = components['recommendation_engine']
    
    similar_movies = recommender.get_similar_movies(movie_id, n_similar=limit)
    
    if not similar_movies:
        return jsonify({
            'movie_id': movie_id,
            'message': 'No similar movies found',
            'similar_movies': [],
            'total_count': 0
        })
    
    # Get original movie info
    movie_info = components['movies_df'][components['movies_df']['movieId'] == movie_id]
    original_title = movie_info['title'].iloc[0] if not movie_info.empty else 'Unknown'
    original_genres = movie_info['genres'].iloc[0] if not movie_info.empty else 'Unknown'
    
    # Format response
    formatted_similar = []
    for sim_movie_id, similarity, title in similar_movies:
        sim_movie_info = components['movies_df'][components['movies_df']['movieId'] == sim_movie_id]
        genres = sim_movie_info['genres'].iloc[0] if not sim_movie_info.empty else 'Unknown'
        
        formatted_similar.append({
            'movie_id': sim_movie_id,
            'title': title,
            'similarity_score': round(similarity, 3),
            'genres': genres,
            'similarity_level': 'very_high' if similarity >= 0.8 else 'high' if similarity >= 0.6 else 'medium'
        })
    
    return jsonify({
        'movie_id': movie_id,
        'movie_title': original_title,
        'movie_genres': original_genres,
        'similar_movies': formatted_similar,
        'total_count': len(formatted_similar),
        'generated_at': datetime.now().isoformat()
    })

@app.route('/api/movies')
@handle_errors
def search_movies():
    """Search for movies by title, filter by genre, or get popular movies"""
    
    components = load_models()
    movies_df = components['movies_df'].copy()
    
    # Get query parameters
    search_query = request.args.get('search', '').strip()
    genre_filter = request.args.get('genre', '').strip()
    limit = min(int(request.args.get('limit', 20)), 100)
    offset = max(int(request.args.get('offset', 0)), 0)
    
    # Apply filters
    filtered_movies = movies_df
    
    if search_query:
        filtered_movies = filtered_movies[
            filtered_movies['title'].str.contains(search_query, case=False, na=False)
        ]
    
    if genre_filter:
        filtered_movies = filtered_movies[
            filtered_movies['genres'].str.contains(genre_filter, case=False, na=False)
        ]
    
    # Get total count before pagination
    total_count = len(filtered_movies)
    
    # Sort by title if searching, otherwise return popular movies
    if search_query or genre_filter:
        filtered_movies = filtered_movies.sort_values('title')
    else:
        # Return some popular movies (you could modify this to return actual popular movies)
        filtered_movies = filtered_movies.head(limit)
        total_count = len(movies_df)
    
    # Apply pagination
    paginated_movies = filtered_movies.iloc[offset:offset + limit]
    
    # Format response
    movies = []
    for _, movie in paginated_movies.iterrows():
        movies.append({
            'movie_id': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres']
        })
    
    return jsonify({
        'query': search_query,
        'genre': genre_filter,
        'movies': movies,
        'total_count': total_count,
        'limit': limit,
        'offset': offset,
        'has_more': offset + limit < total_count
    })

@app.route('/api/users/<int:user_id>/history')
@handle_errors
def get_user_history(user_id):
    """Get user's rating history (if available)"""
    
    if not validate_user_exists(user_id):
        return jsonify({
            'error': f'User {user_id} not found in training data',
            'type': 'not_found'
        }), 404
    
    # Note: This would require access to the original ratings data
    # For now, return a placeholder response
    return jsonify({
        'user_id': user_id,
        'message': 'User rating history not available in current deployment',
        'suggestion': 'This feature requires additional data loading'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'type': 'not_found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'type': 'server_error'}), 500

if __name__ == '__main__':
    # Load models at startup
    try:
        load_models()
        logger.info("Application startup successful")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        exit(1)
    
    # Run the application
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
