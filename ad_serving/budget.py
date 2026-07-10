"""Budget keeper - tracks and enforces ad spending limits."""

from .db import Database


class BudgetKeeper:
    def __init__(self, db: Database):
        self.db = db

    def has_budget(self, ad_id: int) -> bool:
        row = self.db.fetchone(
            "SELECT budget, spent FROM ads WHERE id = %s AND status = 'active'",
            (ad_id,),
        )
        if not row:
            return False
        return float(row["spent"]) < float(row["budget"])

    def deduct(self, ad_id: int, amount: float = 0.001) -> float:
        self.db.execute(
            "UPDATE ads SET spent = spent + %s WHERE id = %s",
            (amount, ad_id),
        )
        return amount

    def get_remaining(self, ad_id: int) -> float:
        row = self.db.fetchone(
            "SELECT budget - spent AS remaining FROM ads WHERE id = %s",
            (ad_id,),
        )
        return float(row["remaining"]) if row else 0.0

    def get_ads_within_budget(self, ad_ids: list[int]) -> list[int]:
        if not ad_ids:
            return []
        placeholders = ",".join(["%s"] * len(ad_ids))
        rows = self.db.fetchall(
            f"SELECT id FROM ads WHERE id IN ({placeholders}) AND budget > spent",
            ad_ids,
        )
        return [r["id"] for r in rows]
