import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse common boolean env var values."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

# Google Gemini API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# Groq API Configuration (optional fallback)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "groq"

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Vector Store Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "smart-finance-docs")

# Validation
if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file")
if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env file")

if _as_bool(os.getenv("VERBOSE_CONFIG"), default=True):
    print(f"Config loaded - Embedding model: {EMBEDDING_MODEL}, Chat model: {CHAT_MODEL}")
