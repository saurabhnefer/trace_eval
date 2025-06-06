"""
Configuration settings for the RAG evaluation pipeline.

This module contains all the configuration parameters, API keys, and constants
used throughout the evaluation pipeline.
"""

import os
import argparse
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Langfuse Configuration
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# MongoDB Connection String
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "genieai")

# API Endpoints
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "https://coachs-production.onrender.com/service/test/transcribe-search")
CHAT_API_URL = os.getenv("CHAT_API_URL", "https://coachs-production.onrender.com/service/master_coach/chat/stream")

# Default Business Context
DEFAULT_BUSINESS_CONTEXT: Dict[str, str] = {
    'roleOrBackground': 'Aspiring entrepreneur',
    'annualRevenue': 'Pre-revenue',
    'primaryBusinessGoal': 'Launch New Product',
    'businessStage': 'Ideation',
    'targetMarket': 'B2B',
    'primaryAspiration': 'Develop an innovative product/service'
}

# Command line argument parsing
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the evaluation pipeline.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='RAG Evaluation Pipeline')
    parser.add_argument('--guest', action='store_true',
                      help='Use Guest_Message_History collection instead of Message_History')
    parser.add_argument('--limit', type=int, default=20,
                      help='Maximum number of conversations to evaluate (default: 20)')
    parser.add_argument('--no-date-filter', action='store_true',
                      help='Disable filtering by current date (evaluate all conversations)')
    parser.add_argument('--force-run', action='store_true',
                      help='Force the script to run regardless of the day of the week')
    parser.add_argument('--start-date', type=str,
                      help='Start date for filtering conversations (format: YYYY-MM-DDT00:00:00)')
    parser.add_argument('--end-date', type=str,
                      help='End date for filtering conversations (format: YYYY-MM-DDT00:00:00)')
    return parser.parse_args()

# Initialize environment variables
def init_environment() -> None:
    """Initialize environment variables for APIs."""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 