#!/usr/bin/env python
"""
🎬 Movie Recommendation System - Quick Start Script
One command to get everything running!
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def print_banner():
    """Print a nice startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🎬  Movie Recommendation System                              ║
║                                                              ║
║  🚀  Starting your personalized movie discovery journey!      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['fastapi', 'streamlit', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found!")
    return True

def start_backend():
    """Start FastAPI backend server"""
    print("🚀 Starting FastAPI Backend...")
    print("   📍 Location: http://localhost:8000")
    print("   📖 API Docs: http://localhost:8000/docs")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app_fastapi:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n✅ Backend server stopped gracefully")

def start_frontend():
    """Start Streamlit frontend"""
    print("🎬 Starting Streamlit Frontend...")
    time.sleep(3)  # Wait for backend to start
    print("   📍 Location: http://localhost:8501")
    print("   🎯 What you'll find: Personalized recommendations, movie search, user comparison!")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n✅ Frontend server stopped gracefully")

def open_browser():
    """Open browser tabs automatically"""
    time.sleep(8)  # Wait for servers to fully start
    print("🌐 Opening your browser...")
    
    try:
        # Open main frontend
        webbrowser.open("http://localhost:8501")
        print("   🎬 Frontend opened in your browser")
        
        # Open API docs in new tab
        time.sleep(1)
        webbrowser.open("http://localhost:8000/docs")
        print("   📖 API documentation opened")
        
    except Exception as e:
        print(f"   ⚠️  Could not open browser automatically: {e}")
        print("   💡 Please open these URLs manually:")
        print("      🎬 Frontend: http://localhost:8501")
        print("      📖 API Docs: http://localhost:8000/docs")

def show_tips():
    """Show helpful tips for first-time users"""
    tips = """
💡 Quick Start Tips:
   • Try User IDs 1-5 to see different movie preferences
   • User 1: Animation movies (Toy Story, Shrek)
   • User 2: Action movies (Star Wars, Raiders)
   • User 3: Drama movies (Shawshank, Pulp Fiction)
   • User 4: Comedy movies (Forrest Gump, Matrix)
   • User 5: Sci-Fi movies (Star Wars series)

🎯 Explore Features:
   • Get Recommendations: Personalized movie suggestions
   • Similar Movies: Find movies like your favorites
   • User Comparison: See who has similar taste
   • Collaborative Search: Learn how recommendations work

⚠️  Press Ctrl+C to stop all services
    """
    print(tips)

def main():
    """Main startup function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("📋 Starting services...")
    print()
    
    # Show tips
    show_tips()
    print()
    
    print("⏳ Starting up your movie recommendation system...")
    print("   This usually takes 10-15 seconds...")
    print()
    
    # Start backend in background thread
    backend_thread = Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend in background thread  
    frontend_thread = Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    # Open browser automatically
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print("✨ System is starting up...")
    print("🎬 Get ready for some amazing movie recommendations!")
    print()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n")
        print("🛑 Shutting down movie recommendation system...")
        print("👋 Thanks for using the Movie Recommendation System!")
        print("✅ All services stopped gracefully")
        print()
        print("🎬 Happy movie watching! 🍿")

if __name__ == "__main__":
    main()
