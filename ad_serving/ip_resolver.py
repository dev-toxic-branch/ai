"""Client IP resolver - extracts real client IP behind proxies.

Inspired by BetterAds ClientIpResolver. Handles X-Forwarded-For,
X-Real-IP, and other proxy headers.
"""

import os
from typing import Optional


class ClientIpResolver:
    """Resolves the real client IP from request headers, respecting trusted proxies."""

    def __init__(self, trusted_proxies: Optional[list[str]] = None):
        proxy_env = os.getenv("TRUSTED_PROXIES", "")
        if trusted_proxies is not None:
            self.trusted_proxies = set(trusted_proxies)
        elif proxy_env:
            self.trusted_proxies = set(proxy_env.split(","))
        else:
            self.trusted_proxies = set()

    def resolve(self, headers: dict, remote_addr: str = "0.0.0.0") -> str:
        """Extract client IP from headers.

        Checks X-Forwarded-For, X-Real-IP in order.
        Only trusts proxy headers if the connecting IP is a known proxy.
        """
        # If the direct connection is from a trusted proxy, look at headers
        if self.trusted_proxies and remote_addr in self.trusted_proxies:
            # X-Forwarded-For: client, proxy1, proxy2
            xff = headers.get("x-forwarded-for", "")
            if xff:
                first_ip = xff.split(",")[0].strip()
                if first_ip:
                    return first_ip

            xri = headers.get("x-real-ip", "")
            if xri:
                return xri.strip()

        # Direct connection or untrusted proxy
        return remote_addr
