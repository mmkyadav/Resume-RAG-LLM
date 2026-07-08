import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESUMES_DIR = BASE_DIR / "Resumes"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# OpenRouter Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLM and Embedding Models
# Default to stable Qwen 2.5 models if Qwen3 is not fully propagated or has different IDs
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen-2.5-72b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

# Chroma DB Collection Name
COLLECTION_NAME = "resumes_collection"

def validate_config():
    """Validates that crucial settings like OpenRouter API Key are set."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set in the environment or .env file. "
            "Please create a .env file and set OPENROUTER_API_KEY."
        )
    if not RESUMES_DIR.exists():
        raise FileNotFoundError(
            f"Resumes directory does not exist at {RESUMES_DIR}. "
            "Please create it and add resume files."
        )
