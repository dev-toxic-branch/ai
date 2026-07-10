"""Fraud detection guard - blocks bots and suspicious traffic."""

import hashlib
import time
from collections import defaultdict
from typing import Optional


class FraudGuard:
    """Real-time bot and fraud detection using heuristic rules."""

    def __init__(self, cache=None):
        self.cache = cache
        self._ip_views: dict[str, list[float]] = defaultdict(list)
        self._blocked: set[str] = set()

    def is_bot(self, ip: str, user_agent: str) -> bool:
        if self._is_blocked(ip):
            return True

        if self._is_known_bot(user_agent):
            self._block_ip(ip)
            return True

        if self._too_many_requests(ip):
            self._block_ip(ip)
            return True

        return False

    def is_suspicious(self, ip: str, location: str, window_seconds: int = 300) -> bool:
        """Detect impossible travel: same IP appearing from distant locations within a short window."""
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

    def _too_many_requests(self, ip: str) -> bool:
        now = time.time()
        h = self._hash(ip)

        if self.cache:
            count = self.cache.incr(f"fraud:rate:{h}", ttl=60)
            return count > 15

        self._ip_views[h].append(now)
        self._ip_views[h] = [t for t in self._ip_views[h] if now - t < 60]
        return len(self._ip_views[h]) > 15

    @staticmethod
    def _hash(ip: str) -> str:
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
