"""Redis cache wrapper for rate limiting, blocked IPs, and hot data."""

import os
import json
from typing import Optional


class Cache:
    def __init__(self):
        try:
            import redis
            self._r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True,
            )
            self._r.ping()
            self._available = True
        except Exception:
            self._available = False
            self._store: dict = {}

    def get(self, key: str) -> Optional[str]:
        if self._available:
            return self._r.get(key)
        return self._store.get(key)

    def set(self, key: str, value: str, ttl: int = None):
        if self._available:
            if ttl:
                self._r.setex(key, ttl, value)
            else:
                self._r.set(key, value)
        else:
            self._store[key] = value

    def incr(self, key: str, ttl: int = 60) -> int:
        if self._available:
            pipe = self._r.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl)
            results = pipe.execute()
            return results[0]
        count = int(self._store.get(key, 0)) + 1
        self._store[key] = str(count)
        return count

    def exists(self, key: str) -> bool:
        if self._available:
            return self._r.exists(key) > 0
        return key in self._store

    def delete(self, key: str):
        if self._available:
            self._r.delete(key)
        else:
            self._store.pop(key, None)
