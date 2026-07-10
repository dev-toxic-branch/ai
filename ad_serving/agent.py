"""Main Ad Serving Agent - orchestrates ad selection and delivery.

Integrates: fraud detection, view tokens, campaign velocity, locale-based
billing, targeting, CTR prediction, and A/B variant selection.
"""

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
from .view_token import ViewTokenService
from .billing import BillingService
from .ip_resolver import ClientIpResolver


class AdServingAgent:
    """Core agent that selects and serves the best ad for a given impression request.

    Pipeline (inspired by BetterAds + our CTR model):
        1. Fraud check (bot detection, view tokens, campaign velocity)
        2. Fetch candidate ads for the link_id
        3. Campaign budget check
        4. Targeting filter (location, device, language, etc.)
        5. CTR prediction (rank remaining ads)
        6. A/B variant selection
        7. Record impression + billing (locale-based rates)
        8. Return ad with tracking pixel + view token for next request
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
        self.view_tokens = ViewTokenService()
        self.billing = BillingService(self.db)
        self.ip_resolver = ClientIpResolver()

    def serve(self, request: ImpressionRequest) -> dict:
        # 1. Fraud guard - view token or IP-based
        trusted_by_token = False
        suspicious_ua = not request.user_agent or not request.user_agent.strip()
        if not suspicious_ua and request.view_token:
            ad_id_from_link = self._get_ad_id_from_link(request.link_id)
            if ad_id_from_link:
                trusted_by_token = self.view_tokens.validate_and_consume(
                    request.view_token, ad_id_from_link, cache=self.cache
                )

        if not trusted_by_token:
            if self.fraud.is_bot(request.ip, request.user_agent):
                return {"error": "bot_detected", "ad": None}

        # Campaign velocity check
        campaign_id = self._get_campaign_id(request.link_id)
        if campaign_id and self.fraud.is_campaign_over_velocity(campaign_id):
            return {"error": "campaign_velocity_exceeded", "ad": None}

        # 2. Fetch candidate ads
        ads = self._get_candidate_ads(request.link_id)
        if not ads:
            return {"error": "no_ads", "ad": None}

        # 3. Campaign budget check
        if campaign_id:
            spend = self.billing.get_campaign_spend(campaign_id)
            if spend["remaining"] <= 0:
                return {"error": "budget_exceeded", "ad": None}

        # 4. Targeting filter
        matched = self.targeting.filter(ads, request)
        if not matched:
            matched = ads  # fallback: show any ad

        # 5. Budget filter (per-ad budget)
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

        # 8. Record impression + billing
        impression_id = self._record_impression(selected, request, best_score)

        # 9. Generate tracking pixel
        pixel = self.tracker.generate(selected["id"], impression_id, selected.get("cta_url", ""))

        # 10. Issue view token for next request
        new_view_token = self.view_tokens.issue_token(selected["id"])

        return {
            "ad": AdResponse(
                ad_id=selected["id"],
                video_url=selected["video_url"],
                cta_text=selected.get("cta_text", "Learn More"),
                cta_url=selected.get("cta_url", "#"),
                variant=selected.get("variant_group"),
                tracking_pixel=pixel,
                score=best_score,
                campaign_id=campaign_id,
                view_token=new_view_token,
            ),
            "error": None,
        }

    def track_click(self, ad_id: int, impression_id: int, locale: str = "US"):
        self.db.execute(
            "UPDATE ads SET total_clicks = total_clicks + 1 WHERE id = %s",
            (ad_id,),
        )
        self.db.execute(
            "UPDATE impressions SET clicked = 1 WHERE id = %s",
            (impression_id,),
        )

        # Find campaign_id for billing
        ad = self.db.fetchone("SELECT campaign_id FROM ads WHERE id = %s", (ad_id,))
        campaign_id = ad["campaign_id"] if ad else None
        if campaign_id:
            self.billing.record_click(ad_id, campaign_id, locale)

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

    def generate_embed_snippet(self, ad_id: int) -> str:
        """Generate iframe embed snippet (BetterAds pattern)."""
        ad = self.db.fetchone("SELECT id FROM ads WHERE id = %s", (ad_id,))
        if not ad:
            return ""
        view_token = self.view_tokens.issue_token(ad_id)
        return (
            f'<iframe src="/api/{ad_id}?vt={view_token}" '
            f'width="640" height="360" frameborder="0" '
            f'allow="autoplay; fullscreen" allowfullscreen></iframe>'
        )

    def _get_ad_id_from_link(self, link_id: str) -> Optional[int]:
        row = self.db.fetchone(
            "SELECT id FROM ads WHERE link_id = %s LIMIT 1",
            (link_id,),
        )
        return row["id"] if row else None

    def _get_campaign_id(self, link_id: str) -> Optional[int]:
        row = self.db.fetchone(
            "SELECT campaign_id FROM ads WHERE link_id = %s LIMIT 1",
            (link_id,),
        )
        return row["campaign_id"] if row and row.get("campaign_id") else None

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
                      budget, spent, campaign_id, status,
                      ab_test_id, variant_count, variant_group,
                      start_time, end_time, created_at,
                      TIMESTAMPDIFF(HOUR, created_at, NOW()) AS age_hours
               FROM ads
               WHERE link_id = %s AND status = 'live'""",
            (link_id,),
        )

        import json
        for row in rows:
            for key in ("target_locations", "target_devices", "target_languages",
                        "target_browsers", "target_os", "target_connection_types"):
                if row.get(key) and isinstance(row[key], str):
                    try:
                        row[key] = json.loads(row[key])
                    except (json.JSONDecodeError, TypeError):
                        row[key] = None

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

        # Record billing
        campaign_id = ad.get("campaign_id")
        if campaign_id:
            self.billing.record_impression(
                ad["id"], campaign_id, request.location,
                request.ip, request.user_agent,
            )
        else:
            # Fallback: direct revenue recording
            self.db.execute(
                "INSERT INTO revenue (ad_id, amount, type, impression_id) VALUES (%s, 0.001, 'impression', %s)",
                (ad["id"], impression_id),
            )

        return impression_id
