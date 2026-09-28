"""Chunk sizes are characters, not tokens.

500 tokens is about 2000 characters (4 characters per token).
Overlap of 200 characters is about 50 tokens.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(__file__).resolve().parent / "index"

CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200

EMBEDDING_MODEL = "local-token-hash-v1"
EMBEDDING_DIM = 256

TEXT_EXTENSIONS = {".md", ".txt", ".py"}
SKIP_DIRS = {".venv", ".git", "__pycache__", "index", "data"}
