"""Tracking pixel and event tracking for ad impressions and clicks."""

import hashlib
import time


class TrackingPixel:
    """Generates tracking pixel HTML and handles event recording."""

    def generate(self, ad_id: int, impression_id: int, cta_url: str) -> str:
        return f"""
<script>
(function() {{
  var t = window._adTrack = {{
    adId: {ad_id},
    impId: {impression_id},
    ctaUrl: '{cta_url}',
    startTime: Date.now(),
    track: function(endpoint) {{
      var img = new Image();
      img.src = '/api/track/' + endpoint + '?ad={ad_id}&imp={impression_id}&t=' + Date.now();
    }},
    trackClick: function() {{
      this.track('click');
    }},
    trackQuartile: function(q) {{
      this.track('quartile&q=' + q);
    }},
    trackComplete: function() {{
      this.track('complete');
    }},
    trackSkip: function() {{
      this.track('skip');
    }}
  }};

  window._adTrackReady = true;
  window.dispatchEvent(new Event('adTrackReady'));
}})();
</script>
<img src="/api/track/impression?ad={ad_id}&imp={impression_id}" 
     width="1" height="1" style="display:none" />
"""


def generate_view_hash(ip: str, ad_id: int, timestamp: float) -> str:
    """Generate a unique hash for deduplication."""
    raw = f"{ip}:{ad_id}:{int(timestamp)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
