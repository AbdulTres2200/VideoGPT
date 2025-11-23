#!/usr/bin/env python3
"""
Simple script to run the FastAPI server.
"""
import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    import os
    # Disable reload in production to prevent database connection issues
    # Set RELOAD=false in environment to disable auto-reload
    enable_reload = os.getenv("RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "api.router:app",
        host="0.0.0.0",
        port=8000,
        reload=enable_reload  # Auto-reload on code changes (can cause DB issues)
    )

