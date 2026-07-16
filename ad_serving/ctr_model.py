"""CTR prediction model using XGBoost for ad ranking."""

import hashlib
import os
from typing import Optional

import numpy as np


class CTRPredictor:
    """Lightweight CTR prediction model.

    Uses a trained XGBoost model if available, otherwise falls back to
    a heuristic scorer based on historical CTR and recency.
    """

    FEATURE_NAMES = [
        "category_id",
        "sentiment_score",
        "duration_seconds",
        "total_impressions",
        "total_clicks",
        "ctr",
        "ad_age_hours",
        "device_mobile",
        "device_desktop",
        "device_tablet",
        "hour_of_day",
        "day_of_week",
        "location_tier1",
        "location_tier2",
        "screen_width",
        "is_wifi",
        "language_tier1",
        "referrer_social",
        "referrer_search",
        "referrer_direct",
    ]

    def __init__(self, model_path: str = "models/ctr_model.json"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            return
        try:
            import xgboost as xgb
            self.model = xgb.XGBRanker()
            self.model.load_model(self.model_path)
        except Exception:
            self.model = None

    def extract_features(self, ad: dict, request) -> np.ndarray:
        """Extract feature vector for one (ad, request) pair."""
        device = [0, 0, 0]
        if request.device == "mobile":
            device[0] = 1
        elif request.device == "desktop":
            device[1] = 1
        elif request.device == "tablet":
            device[2] = 1

        tier1_langs = {"en", "es", "fr", "de", "ja", "zh"}
        tier1_locs = {"US", "GB", "CA", "AU", "DE", "FR", "JP", "KR"}
        tier2_locs = {"EG", "SA", "AE", "IN", "BR", "MX", "NG", "ZA"}

        from datetime import datetime
        dt = datetime.fromtimestamp(request.timestamp)

        social_ref = any(s in request.referrer.lower() for s in
                         ["facebook", "instagram", "twitter", "tiktok", "linkedin", "snapchat"])
        search_ref = any(s in request.referrer.lower() for s in
                         ["google", "bing", "yahoo", "duckduckgo", "baidu"])

        ctr = (ad["total_clicks"] / ad["total_impressions"]
               if ad["total_impressions"] > 0 else 0.0)

        features = [
            ad.get("category_id", 0),
            ad.get("sentiment_score", 0.5),
            ad.get("duration_seconds", 15),
            ad["total_impressions"],
            ad["total_clicks"],
            ctr,
            ad.get("age_hours", 0),
            device[0], device[1], device[2],
            dt.hour,
            dt.weekday(),
            1 if request.location in tier1_locs else 0,
            1 if request.location in tier2_locs else 0,
            request.screen_width,
            1 if request.connection_type == "wifi" else 0,
            1 if request.language in tier1_langs else 0,
            1 if social_ref else 0,
            1 if search_ref else 0,
            1 if not social_ref and not search_ref else 0,
        ]
        return np.array(features, dtype=np.float32)

    def predict(self, ads: list[dict], request) -> list[tuple[dict, float]]:
        """Rank ads by predicted CTR. Returns list of (ad, score) tuples."""
        if self.model is not None:
            return self._predict_ml(ads, request)
        return self._predict_heuristic(ads, request)

    def _predict_ml(self, ads, request) -> list[tuple[dict, float]]:
        import xgboost as xgb
        features = np.array([self.extract_features(ad, request) for ad in ads])
        dmat = xgb.DMatrix(features, feature_names=self.FEATURE_NAMES)
        scores = self.model.predict(dmat)
        ranked = sorted(zip(ads, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked

    def _predict_heuristic(self, ads, request) -> list[tuple[dict, float]]:
        """Fallback heuristic: weighted combination of CTR, recency, and ad freshness."""
        scored = []
        for ad in ads:
            ctr = ad["total_clicks"] / max(ad["total_impressions"], 1)
            impressions_boost = min(ad["total_impressions"] / 10000, 1.0)
            freshness = max(0, 1.0 - ad.get("age_hours", 0) / 168)

            score = (ctr * 0.5 +
                     impressions_boost * 0.2 +
                     freshness * 0.2 +
                     ad.get("sentiment_score", 0.5) * 0.1)
            scored.append((ad, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def train(self, impression_data: list[dict], model_path: Optional[str] = None):
        """Train the model on historical impression data."""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        X = np.array([d["features"] for d in impression_data])
        y = np.array([1 if d["clicked"] else 0 for d in impression_data])
        groups = np.array([d["group_size"] for d in impression_data])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
        )
        self.model.fit(X_train, y_train, group=groups)

        save_to = model_path or self.model_path
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        self.model.save_model(save_to)
        return self.model
