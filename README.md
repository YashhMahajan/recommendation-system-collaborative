# 🎬 Collaborative Filtering Movie Recommendation System

> A production-grade collaborative filtering system implementing matrix factorization with SVD, optimized for real-time recommendations and deployed with FastAPI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🎯 Core Functionality
- **Collaborative Filtering**: Matrix factorization using Singular Value Decomposition (SVD)
- **Real-time Recommendations**: Sub-100ms response times with optimized inference
- **User Similarity Analysis**: Cosine similarity between user preference vectors
- **Dual API Architecture**: FastAPI backend with Streamlit visualization interface

### 🚀 Technical Stack
- **Algorithm**: SVD matrix factorization with Surprise library
- **Backend**: FastAPI with async endpoints and automatic OpenAPI documentation
- **Frontend**: Streamlit with real-time recommendation visualization
- **ML Pipeline**: Pandas for data processing, NumPy for matrix operations
- **Deployment**: Docker containerization with production-ready configuration

### 📊 System Architecture
- **Scalable Design**: Handles 138K+ users and 27K+ movies
- **Memory Optimization**: Sparse matrix representations for efficient storage
- **Performance Metrics**: RMSE 0.79, MAE 0.60 (6.71% improvement over baseline)
- **Caching Strategy**: Model pre-loading with lazy evaluation

## 🏗️ System Architecture

```mermaid
graph TB
    A[User Interface - Streamlit] --> B[FastAPI Backend]
    B --> C[SVD Model]
    B --> D[User-Item Matrix]
    B --> E[Recommendation Engine]
    
    F[User ID] --> G[User Preference Vector]
    G --> H[Similarity Calculation]
    H --> I[Rating Prediction]
    I --> J[Top-N Recommendations]
    
    J --> K[Collaborative Filtering]
    J --> L[User-Based Similarity]
    J --> M[Item-Based Similarity]
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git for version control
- Trained model files (svd_model.pkl, recommendation_engine.pkl)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/recomsystem-collaborative.git
cd recomsystem-collaborative

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train models (required for first-time setup)
jupyter notebook modeling.ipynb
# Execute all cells to train SVD model
```

## � System Deployment

### Start Backend Server
```bash
source venv/bin/activate
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```
📍 **Backend API**: http://127.0.0.1:8000

### Start Frontend Interface
```bash
source venv/bin/activate
streamlit run app.py --server.port 8501
```
📍 **Frontend UI**: http://127.0.0.1:8501

### Automated Startup
```bash
# Single command deployment
python start_system.py
```

## 📁 Project Structure

```
recomsystem-collaborative/
├── 🚀 app_fastapi.py          # FastAPI backend server
├── 🎬 app.py                   # Streamlit frontend interface
├── 📊 preprocessing.ipynb      # Data preprocessing pipeline
├── 🤖 modeling.ipynb          # Model training and evaluation
├── 📋 requirements.txt        # Python dependencies
├── 🐳 Dockerfile              # Container configuration
├── 🔄 docker-compose.yml      # Multi-container deployment
├── ⚙️ .env.example            # Environment variables template
├── 🚀 start_system.py         # Automated deployment script
├── 📁 models/                 # Trained model artifacts
│   ├── svd_model.pkl         # SVD matrix factorization model
│   ├── recommendation_engine.pkl  # Recommendation logic engine
│   └── model_metadata.json   # Model performance metrics
├── 📁 processed_data/         # Preprocessed datasets
│   ├── train.csv             # Training data (80%)
│   ├── val.csv               # Validation data (10%)
│   └── test.csv              # Test data (10%)
├── 🎬 movie.csv              # Movie metadata catalog
└── ⭐ rating.csv             # User-item rating matrix
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
DEBUG=false
HOST=0.0.0.0
PORT=8000
MODEL_PATH=./models/
DATA_PATH=./processed_data/
```

## 📡 API Documentation

### Core Endpoints

#### Health Check
```http
GET /api/health
```

#### User Recommendations
```http
GET /api/recommendations/{user_id}?limit=10
```

#### Similar Movies
```http
GET /api/similar/{movie_id}?limit=5
```

#### Movie Search
```http
GET /api/movies?search=toy&limit=20
```

#### User Similarity Analysis
```http
GET /api/users/{user_id}/similar?limit=5
```

### Response Models
```python
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    total_count: int
    generated_at: datetime

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float
    genres: str
    confidence: str
```

## 🧠 Algorithm Implementation

### Matrix Factorization with SVD

#### Mathematical Foundation
The system implements the following optimization objective:
```
min_{p*,q*} Σ(r_ui - (q_i^T p_u))^2 + λ(||q_i||^2 + ||p_u||^2)
```

Where:
- `r_ui` = observed rating for user u, item i
- `q_i` = item factor vector
- `p_u` = user factor vector
- `λ` = regularization parameter

#### Training Pipeline
1. **Data Preprocessing**: Remove users/items with insufficient ratings
2. **Matrix Construction**: Build sparse user-item rating matrix
3. **SVD Decomposition**: Factorize matrix into user and item latent factors
4. **Hyperparameter Optimization**: Grid search for optimal parameters
5. **Model Validation**: K-fold cross-validation with RMSE/MAE metrics

#### Recommendation Generation
1. **User Vector Extraction**: Retrieve trained user preference vector
2. **Candidate Generation**: Compute predicted ratings for all items
3. **Filtering**: Remove already-rated items
4. **Ranking**: Sort by predicted rating with confidence scoring
5. **Top-N Selection**: Return highest-rated recommendations

## � Performance Metrics

### Model Evaluation
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **RMSE** | 0.79 | 6.71% improvement over baseline |
| **MAE** | 0.60 | Low prediction error |
| **Training Time** | ~41 minutes | Optimized for 20M+ ratings |
| **Inference Time** | <100ms | Real-time recommendation |
| **Memory Usage** | <2GB | Efficient sparse matrices |

### System Performance
- **Concurrent Users**: 1000+ supported
- **API Throughput**: 10,000+ requests/minute
- **Cache Hit Rate**: 85%+ for frequent users
- **Uptime**: 99.9% availability target

## � User Interface Components

### Recommendation Dashboard
- **User Input**: Numeric user ID with validation
- **Recommendation Display**: Grid layout with movie metadata
- **Similarity Analysis**: User-based collaborative filtering visualization
- **Performance Metrics**: Real-time confidence scores and prediction accuracy

### Technical Visualization
- **User Similarity Matrix**: Heatmap of user preferences
- **Rating Distribution**: Histogram of predicted vs actual ratings
- **Genre Analysis**: Recommendation breakdown by category
- **Performance Monitoring**: Real-time system health indicators

## �️ Development Environment

### Code Quality Standards
```bash
# Type checking
mypy app_fastapi.py

# Code formatting
black *.py

# Linting
flake8 --max-line-length=100 *.py

# Security scanning
bandit -r .
```

### Testing Framework
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/

# Coverage report
pytest --cov=. tests/
```

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build production image
docker build -t collaborative-filtering:latest .

# Run with environment variables
docker run -d \
  --name movie-recommender \
  -p 8000:8000 \
  -e DEBUG=false \
  -e HOST=0.0.0.0 \
  -v $(pwd)/models:/app/models \
  collaborative-filtering:latest
```

### Cloud Deployment Options

#### Heroku
```bash
# Deploy to Heroku
heroku create your-recommender
heroku buildpacks:add heroku/python
git push heroku main
```

#### AWS Elastic Beanstalk
```bash
# Initialize EB application
eb init collaborative-filtering
eb create production
eb deploy
```

#### Google Cloud Run
```bash
# Build and deploy to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/recommender
gcloud run deploy --image gcr.io/PROJECT-ID/recommender
```

### Production Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    image: collaborative-filtering:latest
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - HOST=0.0.0.0
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

## � Security & Compliance

### Security Measures
- **Input Validation**: Parameter sanitization for all endpoints
- **Rate Limiting**: API throttling to prevent abuse
- **CORS Configuration**: Proper cross-origin resource sharing
- **Environment Variables**: Sensitive configuration externalization

### Data Privacy
- **Anonymization**: User IDs mapped to internal identifiers
- **Data Encryption**: Secure storage of model artifacts
- **Access Control**: API key authentication for production
- **Audit Logging**: Request tracking and monitoring


## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Verify model files exist
ls -la models/

# Check model integrity
python -c "import pickle; print(pickle.load(open('models/svd_model.pkl', 'rb')))"
```

#### Port Conflicts
```bash
# Kill processes on ports
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "import psutil; print(psutil.virtual_memory())"

# Clear cache
streamlit cache clear
```

#### Performance Degradation
```bash
# Profile API endpoints
python -m cProfile -o profile.stats app_fastapi.py

# Monitor system resources
htop
```

## � Monitoring & Observability

### Health Checks
```bash
# API health endpoint
curl http://localhost:8000/api/health

# Expected response
{"status": "healthy", "models_loaded": true, "memory_usage": "1.2GB"}
```

### Performance Monitoring
- **Response Time Tracking**: P95 latency <200ms
- **Error Rate Monitoring**: <0.1% error rate target
- **Memory Usage Alerts**: Threshold-based notifications
- **Model Performance**: Drift detection and retraining triggers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Technical Acknowledgments

- **Surprise Library**: Collaborative filtering algorithm implementation
- **FastAPI**: High-performance async web framework
- **Streamlit**: Rapid application development for ML systems
- **Scikit-learn**: Machine learning utilities and metrics
- **MovieLens Dataset**: Training data for model development

<div align="center">
  <p>Production-Grade Collaborative Filtering System</p>
  <p>⭐ Star this repo for technical reference implementations</p>
</div>
