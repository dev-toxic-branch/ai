"""Main Ad Serving Agent - orchestrates ad selection and delivery."""

import time
from typing import Optional

from .models import ImpressionRequest, AdResponse
from .db import Database
from .cache import Cache
from .fraud import FraudGuard
from .budget import BudgetKeeper
from .targeting import TargetingFilter
from .ctr_model import CTRPredictor
from .tracking import TrackingPixel, generate_view_hash


class AdServingAgent:
    """Core agent that selects and serves the best ad for a given impression request.

    Pipeline:
        1. Fraud check (bot detection, rate limiting)
        2. Fetch candidate ads for the link_id
        3. Targeting filter (location, device, language, etc.)
        4. Budget check (remove exhausted ads)
        5. CTR prediction (rank remaining ads)
        6. A/B variant selection
        7. Record impression + revenue
        8. Return ad with tracking pixel
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        cache: Optional[Cache] = None,
        ctr_model: Optional[CTRPredictor] = None,
    ):
        self.db = db or Database()
        self.cache = cache or Cache()
        self.fraud = FraudGuard(cache=self.cache)
        self.budget = BudgetKeeper(self.db)
        self.targeting = TargetingFilter()
        self.ctr = ctr_model or CTRPredictor()
        self.tracker = TrackingPixel()

    def serve(self, request: ImpressionRequest) -> dict:
        # 1. Fraud guard
        if self.fraud.is_bot(request.ip, request.user_agent):
            return {"error": "bot_detected", "ad": None}

        if self.fraud.is_suspicious(request.ip, request.location):
            return {"error": "suspicious_activity", "ad": None}

        # 2. Rate limiting
        rate_key = f"rate:{request.ip}"
        count = self.cache.incr(rate_key, ttl=60)
        if count > 30:
            return {"error": "rate_limited", "ad": None}

        # 3. Fetch candidate ads
        ads = self._get_candidate_ads(request.link_id)
        if not ads:
            return {"error": "no_ads", "ad": None}

        # 4. Targeting filter
        matched = self.targeting.filter(ads, request)
        if not matched:
            matched = ads  # fallback: show any ad

        # 5. Budget filter
        within_budget = [a for a in matched if self.budget.has_budget(a["id"])]
        if not within_budget:
            return {"error": "budget_exceeded", "ad": None}

        # 6. CTR prediction + ranking
        ranked = self.ctr.predict(within_budget, request)
        if not ranked:
            return {"error": "no_ranked_ads", "ad": None}

        best_ad, best_score = ranked[0]

        # 7. A/B variant selection
        selected = self._select_variant(best_ad, request)

        # 8. Record impression
        impression_id = self._record_impression(selected, request, best_score)

        # 9. Generate tracking pixel
        pixel = self.tracker.generate(selected["id"], impression_id, selected.get("cta_url", ""))

        return {
            "ad": AdResponse(
                ad_id=selected["id"],
                video_url=selected["video_url"],
                cta_text=selected.get("cta_text", "Learn More"),
                cta_url=selected.get("cta_url", "#"),
                variant=selected.get("variant_group"),
                tracking_pixel=pixel,
                score=best_score,
            ),
            "error": None,
        }

    def track_click(self, ad_id: int, impression_id: int):
        self.db.execute(
            "UPDATE ads SET total_clicks = total_clicks + 1 WHERE id = %s",
            (ad_id,),
        )
        self.db.execute(
            "UPDATE impressions SET clicked = 1 WHERE id = %s",
            (impression_id,),
        )
        self.db.execute(
            "INSERT INTO revenue (ad_id, amount, type, impression_id) VALUES (%s, 0.005, 'click', %s)",
            (ad_id, impression_id),
        )

    def track_quartile(self, ad_id: int, quartile: int):
        self.db.execute(
            "INSERT INTO quartile_events (ad_id, quartile, recorded_at) VALUES (%s, %s, NOW())",
            (ad_id, quartile),
        )

    def track_complete(self, ad_id: int, impression_id: int):
        self.db.execute(
            "UPDATE impressions SET completed = 1 WHERE id = %s",
            (impression_id,),
        )

    def _get_candidate_ads(self, link_id: str) -> list[dict]:
        cache_key = f"ads:link:{link_id}"
        cached = self.cache.get(cache_key)
        if cached:
            import json
            return json.loads(cached)

        rows = self.db.fetchall(
            """SELECT id, video_url, cta_text, cta_url, category_id,
                      sentiment_score, duration_seconds, total_impressions,
                      total_clicks, target_locations, target_devices,
                      target_languages, target_browsers, target_os,
                      target_connection_types, min_screen_width,
                      budget, spent, ab_test_id, variant_count,
                      start_time, end_time, created_at,
                      TIMESTAMPDIFF(HOUR, created_at, NOW()) AS age_hours
               FROM ads
               WHERE link_id = %s AND status = 'active'""",
            (link_id,),
        )

        import json
        for row in rows:
            for key in ("target_locations", "target_devices", "target_languages",
                        "target_browsers", "target_os", "target_connection_types"):
                if row.get(key) and isinstance(row[key], str):
                    row[key] = json.loads(row[key])

        if rows and self.cache:
            self.cache.set(cache_key, json.dumps(rows, default=str), ttl=30)

        return rows

    def _select_variant(self, ad: dict, request: ImpressionRequest) -> dict:
        ab_test_id = ad.get("ab_test_id")
        if not ab_test_id:
            return ad

        variant_count = ad.get("variant_count", 1)
        if variant_count <= 1:
            return ad

        # Deterministic variant selection based on IP + day
        import hashlib
        day_bucket = int(request.timestamp // 86400)
        seed = f"{ad['id']}:{request.ip}:{day_bucket}"
        h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        variant_index = h % variant_count

        variants = self.db.fetchall(
            "SELECT * FROM ab_variants WHERE ab_test_id = %s ORDER BY variant_index",
            (ab_test_id,),
        )
        if variants and variant_index < len(variants):
            return variants[variant_index]
        return ad

    def _record_impression(self, ad: dict, request: ImpressionRequest, score: float) -> int:
        import hashlib
        ip_hash = hashlib.sha256(request.ip.encode()).hexdigest()

        impression_id = self.db.execute(
            """INSERT INTO impressions
               (ad_id, link_id, ip_hash, device, os, browser, location, city,
                language, referrer, connection_type, screen_width, screen_height,
                score, user_agent, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
            (ad["id"], request.link_id, ip_hash, request.device, request.os,
             request.browser, request.location, request.city, request.language,
             request.referrer, request.connection_type, request.screen_width,
             request.screen_height, score, request.user_agent),
        )

        self.db.execute(
            "UPDATE ads SET total_impressions = total_impressions + 1 WHERE id = %s",
            (ad["id"],),
        )

        self.db.execute(
            "INSERT INTO revenue (ad_id, amount, type, impression_id) VALUES (%s, 0.001, 'impression', %s)",
            (ad["id"], impression_id),
        )

        return impression_id
