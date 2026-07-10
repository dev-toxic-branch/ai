"""API routes for ad serving, tracking, and analytics."""

import json
import time
from typing import Optional

from fastapi import APIRouter, Request, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse

from ad_serving.agent import AdServingAgent
from ad_serving.models import ImpressionRequest
from ad_serving.db import Database
from ad_serving.cache import Cache

router = APIRouter()

# Initialize agent (singleton)
_agent: Optional[AdServingAgent] = None


def get_agent() -> AdServingAgent:
    global _agent
    if _agent is None:
        _agent = AdServingAgent()
    return _agent


def _parse_ua(ua: str) -> tuple[str, str]:
    """Parse browser and OS from user agent string."""
    ua_lower = ua.lower()

    browser = "other"
    if "edg" in ua_lower:
        browser = "edge"
    elif "chrome" in ua_lower:
        browser = "chrome"
    elif "firefox" in ua_lower:
        browser = "firefox"
    elif "safari" in ua_lower:
        browser = "safari"

    os_name = "other"
    if "iphone" in ua_lower or "ipad" in ua_lower:
        os_name = "ios"
    elif "android" in ua_lower:
        os_name = "android"
    elif "windows" in ua_lower:
        os_name = "windows"
    elif "mac os" in ua_lower or "macintosh" in ua_lower:
        os_name = "mac"
    elif "linux" in ua_lower:
        os_name = "linux"

    return browser, os_name


def _parse_device(ua: str, sw: int) -> str:
    ua_lower = ua.lower()
    if "ipad" in ua_lower or "tablet" in ua_lower:
        return "tablet"
    if sw > 0 and sw < 768:
        return "mobile"
    if "mobile" in ua_lower or "android" in ua_lower or "iphone" in ua_lower:
        return "mobile"
    return "desktop"


# ─── Ad Serving ───────────────────────────────────────────────────────────────

@router.get("/{link_id}", response_class=HTMLResponse)
async def serve_ad(
    link_id: str,
    request: Request,
    loc: str = Query("unknown", alias="loc"),
    city: str = Query("", alias="city"),
    lang: str = Query("en", alias="lang"),
    conn: str = Query("unknown", alias="conn"),
    sw: int = Query(0, alias="sw"),
    sh: int = Query(0, alias="sh"),
):
    agent = get_agent()
    ua = request.headers.get("user-agent", "")
    browser, os_name = _parse_ua(ua)
    device = _parse_device(ua, sw)

    req = ImpressionRequest(
        link_id=link_id,
        ip=request.client.host,
        user_agent=ua,
        device=device,
        browser=browser,
        os=os_name,
        location=loc,
        city=city,
        language=lang,
        referrer=request.headers.get("referer", ""),
        timestamp=time.time(),
        connection_type=conn,
        screen_width=sw,
        screen_height=sh,
    )

    result = agent.serve(req)

    if result.get("error"):
        return HTMLResponse(
            f"<html><body><h3>{result['error']}</h3></body></html>",
            status_code=200,
        )

    ad = result["ad"]

    return HTMLResponse(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ad</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            background: #000;
            font-family: system-ui, sans-serif;
            overflow: hidden;
        }}
        .ad-container {{
            position: relative;
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        video {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .cta-button {{
            position: absolute;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            padding: 14px 32px;
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.2s;
            z-index: 10;
        }}
        .cta-button:hover {{ background: #1d4ed8; }}
        .skip-btn {{
            position: absolute;
            top: 16px;
            right: 16px;
            padding: 8px 16px;
            background: rgba(0,0,0,0.5);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <div class="ad-container">
        <video autoplay muted playsinline id="adVideo"
               src="{ad.video_url}">
        </video>
        <button class="skip-btn" onclick="window._adTrack.trackSkip(); window.history.back();">
            Skip Ad
        </button>
        <a class="cta-button" href="{ad.cta_url}" onclick="window._adTrack.trackClick();">
            {ad.cta_text}
        </a>
    </div>

    {ad.tracking_pixel}

    <script>
    document.addEventListener('adTrackReady', function() {{
        var v = document.getElementById('adVideo');
        var t = window._adTrack;
        var reported = {{}};

        v.addEventListener('timeupdate', function() {{
            if (!v.duration) return;
            var pct = (v.currentTime / v.duration) * 100;
            [25, 50, 75, 100].forEach(function(q) {{
                if (pct >= q && !reported[q]) {{
                    reported[q] = true;
                    t.trackQuartile(q);
                }}
            }});
        }});

        v.addEventListener('ended', function() {{
            t.trackComplete();
        }});
    }});
    </script>
</body>
</html>
""")


# ─── Tracking Endpoints ──────────────────────────────────────────────────────

@router.get("/track/impression")
async def track_impression(ad: int, imp: int, t: Optional[str] = None):
    return Response(status_code=204)


@router.get("/track/click")
async def track_click(ad: int, imp: int, t: Optional[str] = None):
    agent = get_agent()
    agent.track_click(ad, imp)
    return Response(status_code=204)


@router.get("/track/quartile")
async def track_quartile(ad: int, imp: int, q: int, t: Optional[str] = None):
    agent = get_agent()
    agent.track_quartile(ad, q)
    return Response(status_code=204)


@router.get("/track/complete")
async def track_complete(ad: int, imp: int, t: Optional[str] = None):
    agent = get_agent()
    agent.track_complete(ad, imp)
    return Response(status_code=204)


@router.get("/track/skip")
async def track_skip(ad: int, imp: int, t: Optional[str] = None):
    db = Database()
    db.execute(
        "UPDATE impressions SET skipped = 1 WHERE id = %s",
        (imp,),
    )
    return Response(status_code=204)


# ─── Analytics Endpoints ─────────────────────────────────────────────────────

@router.get("/analytics/ad/{ad_id}")
async def ad_analytics(ad_id: int):
    db = Database()
    ad = db.fetchone("SELECT * FROM ads WHERE id = %s", (ad_id,))
    if not ad:
        return JSONResponse({"error": "ad not found"}, status_code=404)

    stats = db.fetchone(
        """SELECT
            COUNT(*) as total_impressions,
            SUM(clicked) as total_clicks,
            SUM(completed) as total_completes,
            SUM(skipped) as total_skips,
            AVG(score) as avg_score
        FROM impressions WHERE ad_id = %s""",
        (ad_id,),
    )

    revenue = db.fetchone(
        "SELECT SUM(amount) as total FROM revenue WHERE ad_id = %s",
        (ad_id,),
    )

    return {
        "ad": ad,
        "stats": stats,
        "revenue": float(revenue["total"]) if revenue and revenue["total"] else 0.0,
        "ctr": (stats["total_clicks"] / stats["total_impressions"]
                if stats and stats["total_impressions"] else 0.0),
    }


@router.get("/analytics/revenue")
async def revenue_analytics(days: int = 7):
    db = Database()
    rows = db.fetchall(
        """SELECT DATE(created_at) as date, type, SUM(amount) as total
           FROM revenue
           WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
           GROUP BY DATE(created_at), type
           ORDER BY date""",
        (days,),
    )
    return {"daily_revenue": rows}


@router.get("/analytics/top-ads")
async def top_ads(limit: int = 10):
    db = Database()
    rows = db.fetchall(
        """SELECT id, video_url, cta_text, total_impressions, total_clicks,
                  (total_clicks / GREATEST(total_impressions, 1)) as ctr,
                  spent, budget
           FROM ads
           WHERE status = 'active'
           ORDER BY total_clicks DESC
           LIMIT %s""",
        (limit,),
    )
    return {"top_ads": rows}
