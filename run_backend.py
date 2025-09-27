#!/usr/bin/env python3
"""
Main entry point for the Multimodal Design Analysis Suite backend.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run the FastAPI backend server."""
    try:
        # Import after path setup
        import uvicorn
        from backend.config import settings
        
        logger.info("Starting Multimodal Design Analysis Suite Backend")
        logger.info(f"Server will run on {settings.host}:{settings.port}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"PORT environment variable: {os.getenv('PORT', 'Not set')}")
        
        # Check environment variables
        if not settings.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not set - LLM functionality may not work")
        
        if not settings.huggingface_api_token:
            logger.warning("HUGGINGFACE_API_TOKEN not set - some models may not be accessible")
        
        # Run the server
        logger.info(f"Binding to {settings.host}:{settings.port}")
        uvicorn.run(
            "backend.api.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info" if not settings.debug else "debug"
        )
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
