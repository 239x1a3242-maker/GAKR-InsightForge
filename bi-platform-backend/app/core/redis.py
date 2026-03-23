"""Redis configuration."""
import redis.asyncio as redis
from app.core.config import settings

redis_client: redis.Redis | None = None


async def init_redis():
    """Initialize Redis connection."""
    global redis_client
    redis_client = redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    if redis_client is None:
        await init_redis()
    return redis_client
