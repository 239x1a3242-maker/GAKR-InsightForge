"""Database configuration (file-based mode - SQLAlchemy not used)."""
# This project uses a file-based JSON database (see filedb.py).
# This module is kept for compatibility but does not initialize any SQL engine.

Base = None
AsyncSessionLocal = None


async def get_db():
    """Stub - not used in file-based mode."""
    raise NotImplementedError("SQL database is not configured. Using file-based storage.")
