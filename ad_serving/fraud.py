"""Fraud detection guard with Redis sorted set sliding window.

Upgraded from simple counter to Redis sorted set sliding window
(inspired by BetterAds FraudService). Also includes campaign-wide
velocity cap to catch botnet/proxy rotation attacks.
"""

import hashlib
import time
import uuid
from collections import defaultdict
from typing import Optional


class FraudGuard:
    """Real-time bot and fraud detection.

    Uses Redis sorted sets for distributed sliding window rate limiting.
    Falls back to in-memory tracking when Redis is unavailable.
    """

    MAX_VIEWS_PER_MINUTE = 30
    MAX_CAMPAIGN_VIEWS_PER_MINUTE = 200
    WINDOW_SECONDS = 60
    CAMPAIGN_WINDOW_SECONDS = 60

    def __init__(self, cache=None):
        self.cache = cache
        self._ip_views: dict[str, list[float]] = defaultdict(list)
        self._blocked: set[str] = set()
        self._campaign_views: dict[str, int] = defaultdict(int)

    def is_bot(self, ip: str, user_agent: str) -> bool:
        if self._is_blocked(ip):
            return True

        if self._is_known_bot(user_agent):
            self._block_ip(ip)
            return True

        if self._sliding_window_exceeded(ip):
            self._block_ip(ip)
            return True

        return False

    def is_campaign_over_velocity(self, campaign_id: int) -> bool:
        """Detect botnet attacks on a specific campaign.

        Catches the pattern a per-IP window can't: many IPs each under
        the per-IP cap, all inflating one campaign's view count.
        """
        if campaign_id is None:
            return False

        key = f"fraud:campaign:{campaign_id}"

        if self.cache and hasattr(self.cache, "_r") and self.cache._available:
            # Redis-backed: atomic increment with TTL
            pipe = self.cache._r.pipeline()
            count = pipe.incr(key)
            pipe.expire(key, self.CAMPAIGN_WINDOW_SECONDS)
            results = pipe.execute()
            total = results[0]
            return total > self.MAX_CAMPAIGN_VIEWS_PER_MINUTE

        # In-memory fallback
        now = time.time()
        window_key = f"campaign:{campaign_id}"
        self._campaign_views[window_key] = self._campaign_views.get(window_key, 0) + 1
        if self._campaign_views[window_key] > self.MAX_CAMPAIGN_VIEWS_PER_MINUTE:
            return True
        return False

    def is_suspicious(self, ip: str, location: str, window_seconds: int = 300) -> bool:
        """Detect impossible travel: same IP appearing from distant locations."""
        if self.cache:
            key = f"fraud:geo:{self._hash(ip)}"
            prev_loc = self.cache.get(key)
            if prev_loc and prev_loc != location:
                self.cache.set(key, location, ttl=window_seconds)
                return True
            self.cache.set(key, location, ttl=window_seconds)
        return False

    def _is_blocked(self, ip: str) -> bool:
        if ip in self._blocked:
            return True
        if self.cache and self.cache.exists(f"fraud:blocked:{self._hash(ip)}"):
            self._blocked.add(ip)
            return True
        return False

    def _block_ip(self, ip: str):
        self._blocked.add(ip)
        if self.cache:
            self.cache.set(f"fraud:blocked:{self._hash(ip)}", "1", ttl=3600)

    def _is_known_bot(self, user_agent: str) -> bool:
        if not user_agent:
            return True
        ua_lower = user_agent.lower()
        bot_patterns = [
            "bot", "crawler", "spider", "headless", "phantom",
            "curl", "wget", "python-requests", "go-http-client",
            "scrapy", "httpie", "semrush", "ahref",
        ]
        return any(p in ua_lower for p in bot_patterns)

    def _sliding_window_exceeded(self, ip: str) -> bool:
        """Redis sorted set sliding window (BetterAds pattern)."""
        h = self._hash(ip)
        key = f"fraud:ip:{h}"

        if self.cache and hasattr(self.cache, "_r") and self.cache._available:
            now_ms = int(time.time() * 1000)
            cutoff_ms = now_ms - (self.WINDOW_SECONDS * 1000)

            r = self.cache._r
            r.zremrangebyscore(key, 0, cutoff_ms)
            r.zadd(key, {uuid.uuid4().hex: now_ms})
            r.expire(key, self.WINDOW_SECONDS)
            count = r.zcard(key)
            return count > self.MAX_VIEWS_PER_MINUTE

        # In-memory fallback
        now = time.time()
        self._ip_views[h].append(now)
        self._ip_views[h] = [t for t in self._ip_views[h] if now - t < self.WINDOW_SECONDS]
        return len(self._ip_views[h]) > self.MAX_VIEWS_PER_MINUTE

    @staticmethod
    def _hash(ip: str) -> str:
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
