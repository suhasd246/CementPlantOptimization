# ~/Documents/cement-operations-optimization/run_dashboard.py
#!/usr/bin/env python3
"""
Run the Dashboard server
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the dashboard
from src.dashboard.main import app

if __name__ == "__main__":
    print("📊 Starting Cement Plant Optimization Dashboard...")
    print("🔗 Dashboard: http://localhost:8050")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
