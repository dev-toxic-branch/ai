"""Targeting filter - selects ads matching the impression request criteria."""

from typing import Optional
from .models import ImpressionRequest


class TargetingFilter:
    """Filter ads based on targeting criteria set by advertisers."""

    def filter(self, ads: list[dict], request: ImpressionRequest) -> list[dict]:
        matched = []
        for ad in ads:
            if self._matches(ad, request):
                matched.append(ad)
        return matched

    def _matches(self, ad: dict, req: ImpressionRequest) -> bool:
        # Location targeting
        locations = ad.get("target_locations")
        if locations and req.location not in locations:
            return False

        # Device targeting
        devices = ad.get("target_devices")
        if devices and req.device not in devices:
            return False

        # Language targeting
        languages = ad.get("target_languages")
        if languages and req.language not in languages:
            return False

        # Browser targeting
        browsers = ad.get("target_browsers")
        if browsers and req.browser not in browsers:
            return False

        # OS targeting
        os_list = ad.get("target_os")
        if os_list and req.os not in os_list:
            return False

        # Time window
        now = req.timestamp
        start = ad.get("start_time", 0)
        end = ad.get("end_time", float("inf"))
        if not (start <= now <= end):
            return False

        # Connection type targeting
        conn_types = ad.get("target_connection_types")
        if conn_types and req.connection_type not in conn_types:
            return False

        # Screen size targeting (min width)
        min_width = ad.get("min_screen_width", 0)
        if min_width and req.screen_width < min_width:
            return False

        return True
