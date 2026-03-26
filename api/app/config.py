import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / os.getenv("DB_PATH", "data/agentic_honeypot.sqlite3")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_PATH", "data/raw/telemetry.jsonl")

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# API
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))

# Security
MAX_PAYLOAD_BYTES = int(os.getenv("MAX_PAYLOAD_BYTES", 10240))  # 10KB
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")