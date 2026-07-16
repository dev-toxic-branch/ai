"""Data models for the ad serving system."""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class ImpressionRequest:
    """Incoming request when someone views an ad link."""
    link_id: str
    ip: str
    user_agent: str
    device: str          # mobile | desktop | tablet
    browser: str         # chrome | safari | firefox | edge | other
    os: str              # ios | android | windows | mac | linux | other
    location: str        # ISO country code (US, EG, SA ...)
    city: str
    language: str        # en | es | fr | ar | de | ...
    referrer: str
    timestamp: float = field(default_factory=time.time)
    connection_type: str = "unknown"  # wifi | 4g | 3g | ethernet
    screen_width: int = 0
    screen_height: int = 0
    view_token: Optional[str] = None  # HMAC signed one-time token

    @classmethod
    def from_dict(cls, data: dict) -> "ImpressionRequest":
        return cls(
            link_id=data.get("link_id", ""),
            ip=data.get("ip", "0.0.0.0"),
            user_agent=data.get("user_agent", ""),
            device=data.get("device", "unknown"),
            browser=data.get("browser", "other"),
            os=data.get("os", "other"),
            location=data.get("location", "unknown"),
            city=data.get("city", ""),
            language=data.get("language", "en"),
            referrer=data.get("referrer", ""),
            timestamp=data.get("timestamp", time.time()),
            connection_type=data.get("connection_type", "unknown"),
            screen_width=data.get("screen_width", 0),
            screen_height=data.get("screen_height", 0),
            view_token=data.get("view_token"),
        )


@dataclass
class AdResponse:
    """What the agent returns after picking an ad."""
    ad_id: int
    video_url: str
    cta_text: str
    cta_url: str
    variant: Optional[str] = None
    tracking_pixel: str = ""
    score: float = 0.0
    campaign_id: Optional[int] = None
    view_token: Optional[str] = None
