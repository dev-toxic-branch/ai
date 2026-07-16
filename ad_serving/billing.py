"""Billing service with locale-based CPM rates.

Inspired by BetterAds BillingService. Different countries pay different
CPM rates based on advertiser value. Also handles campaign budget
locking to prevent overspend.
"""

from decimal import Decimal
from typing import Optional


# CPM rates by country (USD per 1000 impressions)
LOCALE_RATES = {
    "US": Decimal("0.0015"),
    "GB": Decimal("0.0014"),
    "CA": Decimal("0.0013"),
    "AU": Decimal("0.0013"),
    "DE": Decimal("0.0013"),
    "FR": Decimal("0.0012"),
    "JP": Decimal("0.0012"),
    "KR": Decimal("0.0011"),
    "BR": Decimal("0.0010"),
    "IN": Decimal("0.0008"),
    "EG": Decimal("0.0007"),
    "SA": Decimal("0.0009"),
    "NG": Decimal("0.0006"),
    "ZA": Decimal("0.0007"),
}
DEFAULT_RATE = Decimal("0.001")

# Click bonus multiplier
CLICK_MULTIPLIER = Decimal("5")


class BillingService:
    def __init__(self, db=None):
        self.db = db

    def rate_for(self, locale: str) -> Decimal:
        return LOCALE_RATES.get(locale.upper(), DEFAULT_RATE)

    def calculate_charge(self, locale: str, views: int) -> Decimal:
        return self.rate_for(locale) * Decimal(views)

    def record_impression(self, ad_id: int, campaign_id: int, locale: str,
                          ip: str, device_info: str = "") -> bool:
        """Record an impression and deduct from campaign budget.

        Returns False if campaign budget is exhausted.
        """
        if not self.db:
            return True

        rate = self.rate_for(locale)

        # Check budget with row-level locking (SELECT ... FOR UPDATE)
        campaign = self.db.fetchone(
            "SELECT id, budget, spent FROM campaigns WHERE id = %s FOR UPDATE",
            (campaign_id,),
        )
        if not campaign:
            return False

        new_spent = Decimal(str(campaign["spent"])) + rate
        if new_spent > Decimal(str(campaign["budget"])):
            return False

        import hashlib
        ip_hash = hashlib.sha256(ip.encode()).hexdigest()

        self.db.execute(
            """INSERT INTO impressions
               (ad_id, link_id, ip_hash, device, location, score, user_agent, created_at)
               VALUES (%s, '', %s, %s, %s, 0, %s, NOW())""",
            (ad_id, ip_hash, device_info, locale, device_info),
        )

        self.db.execute(
            "UPDATE ads SET total_impressions = total_impressions + 1 WHERE id = %s",
            (ad_id,),
        )

        self.db.execute(
            "UPDATE campaigns SET spent = %s WHERE id = %s",
            (str(new_spent), campaign_id),
        )

        self.db.execute(
            "INSERT INTO revenue (ad_id, amount, type, created_at) VALUES (%s, %s, 'impression', NOW())",
            (ad_id, str(rate)),
        )

        return True

    def record_click(self, ad_id: int, campaign_id: int, locale: str) -> bool:
        """Record a click (worth 5x an impression)."""
        if not self.db:
            return True

        rate = self.rate_for(locale) * CLICK_MULTIPLIER

        campaign = self.db.fetchone(
            "SELECT id, budget, spent FROM campaigns WHERE id = %s FOR UPDATE",
            (campaign_id,),
        )
        if not campaign:
            return False

        new_spent = Decimal(str(campaign["spent"])) + rate
        if new_spent > Decimal(str(campaign["budget"])):
            return False

        self.db.execute(
            "UPDATE ads SET total_clicks = total_clicks + 1 WHERE id = %s",
            (ad_id,),
        )

        self.db.execute(
            "UPDATE campaigns SET spent = %s WHERE id = %s",
            (str(new_spent), campaign_id),
        )

        self.db.execute(
            "INSERT INTO revenue (ad_id, amount, type, created_at) VALUES (%s, %s, 'click', NOW())",
            (ad_id, str(rate)),
        )

        return True

    def get_campaign_spend(self, campaign_id: int) -> dict:
        row = self.db.fetchone(
            "SELECT budget, spent FROM campaigns WHERE id = %s",
            (campaign_id,),
        )
        if not row:
            return {"budget": 0, "spent": 0, "remaining": 0}
        budget = Decimal(str(row["budget"]))
        spent = Decimal(str(row["spent"]))
        return {"budget": float(budget), "spent": float(spent), "remaining": float(budget - spent)}
