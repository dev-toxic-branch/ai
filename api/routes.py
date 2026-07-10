"""API routes for ad serving, tracking, and analytics.

Integrates BetterAds patterns: view tokens, embed widget, locale-based
billing, campaign management, and campaign-level analytics.
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, Request, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse

from ad_serving.agent import AdServingAgent
from ad_serving.models import ImpressionRequest
from ad_serving.db import Database
from ad_serving.cache import Cache
from ad_serving.billing import BillingService
from ad_serving.ip_resolver import ClientIpResolver

router = APIRouter()

_agent: Optional[AdServingAgent] = None
_ip_resolver = ClientIpResolver()


def get_agent() -> AdServingAgent:
    global _agent
    if _agent is None:
        _agent = AdServingAgent()
    return _agent


def _parse_ua(ua: str) -> tuple[str, str]:
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
    vt: Optional[str] = Query(None, alias="vt"),
):
    agent = get_agent()
    ua = request.headers.get("user-agent", "")
    browser, os_name = _parse_ua(ua)
    device = _parse_device(ua, sw)
    ip = _ip_resolver.resolve(dict(request.headers), request.client.host)

    req = ImpressionRequest(
        link_id=link_id,
        ip=ip,
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
        view_token=vt,
    )

    result = agent.serve(req)

    if result.get("error"):
        if result["error"] in ("bot_detected", "campaign_velocity_exceeded"):
            return HTMLResponse(
                f"<html><body><h3>Ad unavailable</h3></body></html>",
                status_code=200,
            )
        return HTMLResponse(
            f"<html><body><h3>{result['error']}</h3></body></html>",
            status_code=200,
        )

    ad = result["ad"]
    new_vt = ad.view_token or ""

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


# ─── Embed Widget (BetterAds pattern) ────────────────────────────────────────

@router.get("/embed/{token}", response_class=HTMLResponse)
async def embed_widget(token: str, request: Request):
    agent = get_agent()
    db = Database()
    row = db.fetchone("SELECT ad_id FROM ad_links WHERE token = %s", (token,))
    if not row:
        return HTMLResponse("<html><body><h3>Ad not found</h3></body></html>", status_code=404)

    ad_id = row["ad_id"]
    view_token = agent.view_tokens.issue_token(ad_id)
    locale = request.query_params.get("locale", "en")

    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #000; display: flex; align-items: center;
               justify-content: center; width: 100vw; height: 100vh; }}
        video {{ width: 100%; height: 100%; object-fit: contain; }}
        #error {{ color: #fff; font-family: sans-serif; font-size: 14px; }}
    </style>
</head>
<body>
    <video id="ad" autoplay muted playsinline></video>
    <div id="error" style="display:none">Ad unavailable</div>
    <script>
    (function() {{
        var adId = {ad_id};
        var vt = '{view_token}';
        var locale = '{locale}';
        fetch('/api/' + adId + '?lang=' + locale + '&vt=' + encodeURIComponent(vt))
            .then(function(r) {{ return r.ok ? r.text() : Promise.reject(r.status); }})
            .then(function(html) {{
                document.open();
                document.write(html);
                document.close();
            }})
            .catch(function() {{
                document.getElementById('ad').style.display = 'none';
                document.getElementById('error').style.display = 'block';
            }});
    }})();
    </script>
</body>
</html>
""")


@router.get("/embed/{token}/snippet")
async def embed_snippet(token: str):
    db = Database()
    row = db.fetchone("SELECT ad_id FROM ad_links WHERE token = %s", (token,))
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "snippet": f'<iframe src="/api/embed/{token}" width="640" height="360" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>',
        "embed_url": f"/api/embed/{token}",
    }


# ─── Tracking Endpoints ──────────────────────────────────────────────────────

@router.get("/track/impression")
async def track_impression(ad: int, imp: int, t: Optional[str] = None):
    return Response(status_code=204)


@router.get("/track/click")
async def track_click(ad: int, imp: int, t: Optional[str] = None, locale: str = "US"):
    agent = get_agent()
    agent.track_click(ad, imp, locale)
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
    db.execute("UPDATE impressions SET skipped = 1 WHERE id = %s", (imp,))
    return Response(status_code=204)


# ─── Campaign Endpoints ──────────────────────────────────────────────────────

@router.post("/campaigns")
async def create_campaign(body: dict):
    db = Database()
    campaign_id = db.execute(
        """INSERT INTO campaigns (advertiser_id, name, budget, status)
           VALUES (%s, %s, %s, 'draft')""",
        (body.get("advertiser_id"), body.get("name", ""), body.get("budget", 0)),
    )
    return {"id": campaign_id, "status": "draft"}


@router.get("/campaigns/{campaign_id}")
async def get_campaign(campaign_id: int):
    db = Database()
    campaign = db.fetchone("SELECT * FROM campaigns WHERE id = %s", (campaign_id,))
    if not campaign:
        return JSONResponse({"error": "not found"}, status_code=404)
    billing = BillingService(db)
    spend_info = billing.get_campaign_spend(campaign_id)
    return {**campaign, **spend_info}


@router.get("/campaigns/{campaign_id}/analytics")
async def campaign_analytics(campaign_id: int):
    db = Database()
    stats = db.fetchone(
        """SELECT
            COUNT(DISTINCT i.id) as total_impressions,
            SUM(i.clicked) as total_clicks,
            SUM(i.completed) as total_completes,
            SUM(i.skipped) as total_skips
        FROM impressions i
        JOIN ads a ON i.ad_id = a.id
        WHERE a.campaign_id = %s""",
        (campaign_id,),
    )
    revenue = db.fetchone(
        """SELECT SUM(r.amount) as total
           FROM revenue r
           JOIN ads a ON r.ad_id = a.id
           WHERE a.campaign_id = %s""",
        (campaign_id,),
    )
    return {
        "campaign_id": campaign_id,
        "stats": stats,
        "revenue": float(revenue["total"]) if revenue and revenue["total"] else 0.0,
    }


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
           WHERE status = 'live'
           ORDER BY total_clicks DESC
           LIMIT %s""",
        (limit,),
    )
    return {"top_ads": rows}
