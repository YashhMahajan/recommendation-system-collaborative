#!/bin/bash

# Movie Recommendation System - Deployment Script (FastAPI)
# This script helps deploy the FastAPI application to various platforms

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required files exist
check_requirements() {
    print_status "Checking deployment requirements..."
    
    required_files=("requirements.txt" "app_fastapi.py" "Dockerfile" ".env.example")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing required files: ${missing_files[*]}"
        exit 1
    fi
    
    # Check if models directory exists and has files
    if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
        print_warning "Models directory is empty. Make sure to run the modeling notebook first!"
    fi
    
    print_success "All requirements checked"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Copy .env.example to .env if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_status "Created .env file from template"
        print_warning "Please update .env file with your configuration"
    fi
    
    # Create logs directory
    mkdir -p logs
    
    print_success "Environment setup completed"
}

# Local deployment
deploy_local() {
    print_status "Starting local deployment..."
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Start the FastAPI application
    print_status "Starting FastAPI server..."
    export DEBUG=true
    uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
}

# Docker deployment
deploy_docker() {
    print_status "Starting Docker deployment..."
    
    # Build and run with Docker Compose
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    
    print_success "Docker deployment started"
    print_status "Application is running at: http://localhost:8000"
    print_status "API documentation: http://localhost:8000/docs"
    print_status "Check logs with: docker-compose logs -f"
}

# Production deployment with Gunicorn and Uvicorn
deploy_production() {
    print_status "Starting production deployment..."
    
    # Install production dependencies
    pip install gunicorn uvicorn
    
    # Set production environment
    export DEBUG=false
    
    # Start with Gunicorn and Uvicorn workers
    print_status "Starting production server with Gunicorn + Uvicorn..."
    gunicorn app_fastapi:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
}

# Heroku deployment
deploy_heroku() {
    print_status "Starting Heroku deployment..."
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        print_error "Heroku CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Create Heroku app if not exists
    app_name="movie-recommender-$(date +%s)"
    print_status "Creating Heroku app: $app_name"
    heroku create $app_name
    
    # Set environment variables
    heroku config:set DEBUG=false
    heroku config:set HOST=0.0.0.0
    heroku config:set PORT=8000
    
    # Create Procfile for FastAPI
    echo "web: gunicorn app_fastapi:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:\$PORT" > Procfile
    
    # Deploy
    git add .
    git commit -m "Deploy FastAPI to Heroku"
    git push heroku main
    
    print_success "Deployed to Heroku: https://$app_name.herokuapp.com"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Wait for the server to start
    sleep 5
    
    # Check if the FastAPI is responding
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_success "Health check passed - FastAPI is responding"
    else
        print_error "Health check failed - FastAPI is not responding"
        exit 1
    fi
}

# Test API endpoints
test_api() {
    print_status "Testing API endpoints..."
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    curl -s http://localhost:8000/api/health | jq .
    
    # Test root endpoint
    print_status "Testing root endpoint..."
    curl -s http://localhost:8000/ | jq .
    
    # Test model info endpoint
    print_status "Testing model info endpoint..."
    curl -s http://localhost:8000/api/model/info | jq .
    
    print_success "API endpoints tested"
}

# Show help
show_help() {
    echo "Movie Recommendation System - FastAPI Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  local      Deploy locally for development"
    echo "  docker     Deploy using Docker and Docker Compose"
    echo "  production Deploy for production using Gunicorn + Uvicorn"
    echo "  heroku     Deploy to Heroku"
    echo "  check      Check deployment requirements"
    echo "  health     Perform health check on running application"
    echo "  test       Test API endpoints"
    echo "  help       Show this help message"
    echo ""
    echo "FastAPI Features:"
    echo "  - Automatic API documentation at /docs"
    echo "  - ReDoc documentation at /redoc"
    echo "  - Type hints and validation"
    echo "  - High performance with async support"
    echo ""
    echo "Examples:"
    echo "  $0 local      # Start local development server"
    echo "  $0 docker     # Deploy with Docker"
    echo "  $0 production # Deploy for production"
    echo "  $0 test       # Test API endpoints"
    echo ""
}

# Main script logic
main() {
    case "${1:-help}" in
        "check")
            check_requirements
            ;;
        "local")
            check_requirements
            setup_environment
            deploy_local
            ;;
        "docker")
            check_requirements
            setup_environment
            deploy_docker
            health_check
            ;;
        "production")
            check_requirements
            setup_environment
            deploy_production
            ;;
        "heroku")
            check_requirements
            setup_environment
            deploy_heroku
            ;;
        "health")
            health_check
            ;;
        "test")
            test_api
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run the main function with all arguments
main "$@"
