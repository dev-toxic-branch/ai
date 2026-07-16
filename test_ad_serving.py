"""Tests for the ad serving agent system."""

import time
import hashlib
from decimal import Decimal
from unittest.mock import MagicMock, patch, PropertyMock
from ad_serving.models import ImpressionRequest, AdResponse
from ad_serving.fraud import FraudGuard
from ad_serving.targeting import TargetingFilter
from ad_serving.tracking import TrackingPixel, generate_view_hash
from ad_serving.ctr_model import CTRPredictor
from ad_serving.view_token import ViewTokenService
from ad_serving.billing import BillingService, LOCALE_RATES, DEFAULT_RATE
from ad_serving.ip_resolver import ClientIpResolver


# ─── ImpressionRequest Tests ─────────────────────────────────────────────────

def test_impression_request_from_dict():
    req = ImpressionRequest.from_dict({
        "link_id": "abc123",
        "ip": "192.168.1.1",
        "user_agent": "Mozilla/5.0 Chrome/120",
        "device": "desktop",
        "browser": "chrome",
        "os": "windows",
        "location": "US",
        "city": "New York",
        "language": "en",
        "referrer": "https://google.com",
        "connection_type": "wifi",
        "screen_width": 1920,
        "screen_height": 1080,
    })
    assert req.link_id == "abc123"
    assert req.ip == "192.168.1.1"
    assert req.device == "desktop"
    assert req.browser == "chrome"
    assert req.os == "windows"
    assert req.location == "US"
    assert req.language == "en"
    assert req.screen_width == 1920


def test_impression_request_defaults():
    req = ImpressionRequest(
        link_id="test",
        ip="127.0.0.1",
        user_agent="test",
        device="mobile",
        browser="safari",
        os="ios",
        location="EG",
        city="Cairo",
        language="ar",
        referrer="",
    )
    assert req.connection_type == "unknown"
    assert req.screen_width == 0
    assert req.timestamp > 0


# ─── FraudGuard Tests ────────────────────────────────────────────────────────

def test_fraud_bot_detection_no_user_agent():
    guard = FraudGuard()
    assert guard.is_bot("10.0.0.1", "") is True


def test_fraud_bot_detection_known_bot():
    guard = FraudGuard()
    assert guard.is_bot("10.0.0.2", "Googlebot/2.1") is True
    assert guard.is_bot("10.0.0.3", "python-requests/2.28.0") is True
    assert guard.is_bot("10.0.0.4", "curl/7.84.0") is True


def test_fraud_rate_limiting():
    guard = FraudGuard()
    ip = "192.168.100.1"
    for _ in range(30):
        guard.is_bot(ip, "Mozilla/5.0 Chrome/120")
    assert guard.is_bot(ip, "Mozilla/5.0 Chrome/120") is True


def test_fraud_clean_traffic():
    guard = FraudGuard()
    assert guard.is_bot("10.0.0.5", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120") is False
    assert guard.is_bot("10.0.0.6", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Safari/604.1") is False


def test_fraud_impossible_travel():
    guard = FraudGuard(cache=MagicMock())
    guard.cache.get.return_value = None
    assert guard.is_suspicious("1.2.3.4", "US") is False
    guard.cache.get.return_value = "US"
    assert guard.is_suspicious("1.2.3.4", "EG") is True


# ─── TargetingFilter Tests ───────────────────────────────────────────────────

def _make_ad(**overrides):
    ad = {
        "id": 1,
        "video_url": "http://example.com/ad.mp4",
        "cta_text": "Shop Now",
        "cta_url": "http://example.com",
        "category_id": 1,
        "sentiment_score": 0.8,
        "duration_seconds": 30,
        "total_impressions": 1000,
        "total_clicks": 50,
        "target_locations": None,
        "target_devices": None,
        "target_languages": None,
        "target_browsers": None,
        "target_os": None,
        "target_connection_types": None,
        "min_screen_width": 0,
        "budget": 100.0,
        "spent": 0.0,
        "ab_test_id": None,
        "variant_count": 1,
        "variant_group": None,
        "start_time": 0,
        "end_time": 9999999999,
        "age_hours": 1,
    }
    ad.update(overrides)
    return ad


def _make_req(**overrides):
    defaults = {
        "link_id": "test",
        "ip": "1.2.3.4",
        "user_agent": "Mozilla/5.0 Chrome/120",
        "device": "desktop",
        "browser": "chrome",
        "os": "windows",
        "location": "US",
        "city": "New York",
        "language": "en",
        "referrer": "",
        "timestamp": time.time(),
        "connection_type": "wifi",
        "screen_width": 1920,
        "screen_height": 1080,
    }
    defaults.update(overrides)
    return ImpressionRequest(**defaults)


def test_targeting_no_constraints():
    f = TargetingFilter()
    ads = [_make_ad(), _make_ad(id=2)]
    req = _make_req()
    assert len(f.filter(ads, req)) == 2


def test_targeting_location_match():
    f = TargetingFilter()
    ad = _make_ad(target_locations=["US", "CA"])
    req = _make_req(location="US")
    assert len(f.filter([ad], req)) == 1


def test_targeting_location_reject():
    f = TargetingFilter()
    ad = _make_ad(target_locations=["US", "CA"])
    req = _make_req(location="EG")
    assert len(f.filter([ad], req)) == 0


def test_targeting_device_match():
    f = TargetingFilter()
    ad = _make_ad(target_devices=["mobile"])
    assert len(f.filter([ad], _make_req(device="mobile"))) == 1
    assert len(f.filter([ad], _make_req(device="desktop"))) == 0


def test_targeting_language_match():
    f = TargetingFilter()
    ad = _make_ad(target_languages=["en", "es"])
    assert len(f.filter([ad], _make_req(language="en"))) == 1
    assert len(f.filter([ad], _make_req(language="ar"))) == 0


def test_targeting_time_window():
    f = TargetingFilter()
    now = time.time()
    ad = _make_ad(start_time=now - 100, end_time=now + 100)
    assert len(f.filter([ad], _make_req())) == 1
    ad_expired = _make_ad(start_time=now - 200, end_time=now - 100)
    assert len(f.filter([ad_expired], _make_req())) == 0


def test_targeting_min_screen_width():
    f = TargetingFilter()
    ad = _make_ad(min_screen_width=768)
    assert len(f.filter([ad], _make_req(screen_width=1920))) == 1
    assert len(f.filter([ad], _make_req(screen_width=375))) == 0


def test_targeting_combined():
    f = TargetingFilter()
    ad = _make_ad(
        target_locations=["US"],
        target_devices=["mobile"],
        target_languages=["en"],
    )
    req_match = _make_req(location="US", device="mobile", language="en")
    assert len(f.filter([ad], req_match)) == 1

    req_no_loc = _make_req(location="EG", device="mobile", language="en")
    assert len(f.filter([ad], req_no_loc)) == 0

    req_no_dev = _make_req(location="US", device="desktop", language="en")
    assert len(f.filter([ad], req_no_dev)) == 0


# ─── TrackingPixel Tests ─────────────────────────────────────────────────────

def test_tracking_pixel_contains_ids():
    pixel = TrackingPixel()
    html = pixel.generate(ad_id=42, impression_id=999, cta_url="http://shop.com")
    assert "ad=42" in html
    assert "imp=999" in html
    assert "http://shop.com" in html
    assert "_adTrack" in html


def test_tracking_pixel_quartile_hooks():
    pixel = TrackingPixel()
    html = pixel.generate(ad_id=1, impression_id=2, cta_url="#")
    assert "trackQuartile" in html
    assert "trackClick" in html
    assert "trackComplete" in html
    assert "trackSkip" in html


def test_generate_view_hash():
    h1 = generate_view_hash("1.2.3.4", 1, 1000.0)
    h2 = generate_view_hash("1.2.3.4", 1, 1000.0)
    assert h1 == h2  # deterministic
    h3 = generate_view_hash("1.2.3.5", 1, 1000.0)
    assert h1 != h3  # different IP


# ─── CTRPredictor Tests ─────────────────────────────────────────────────────

def test_ctr_heuristic_fallback():
    model = CTRPredictor()
    req = _make_req()
    ads = [
        _make_ad(id=1, total_impressions=1000, total_clicks=100, age_hours=1, sentiment_score=0.9),
        _make_ad(id=2, total_impressions=100, total_clicks=5, age_hours=48, sentiment_score=0.3),
    ]
    ranked = model.predict(ads, req)
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]  # first has higher score
    assert ranked[0][0]["id"] == 1  # higher CTR ad ranks first


def test_ctr_feature_extraction():
    model = CTRPredictor()
    req = _make_req(device="mobile", connection_type="wifi", language="en", location="US")
    ad = _make_ad(category_id=3, sentiment_score=0.7, duration_seconds=15,
                  total_impressions=500, total_clicks=25, age_hours=6)
    features = model.extract_features(ad, req)
    assert len(features) == 20
    assert features[0] == 3  # category_id
    assert features[7] == 1  # device_mobile
    assert features[14] == 1920  # screen_width
    assert features[15] == 1  # is_wifi


def test_ctr_empty_ads():
    model = CTRPredictor()
    req = _make_req()
    ranked = model.predict([], req)
    assert ranked == []


# ─── AdResponse Tests ────────────────────────────────────────────────────────

def test_ad_response():
    resp = AdResponse(
        ad_id=1,
        video_url="http://s3.amazonaws.com/video.mp4",
        cta_text="Buy Now",
        cta_url="http://shop.com",
        variant="A",
        tracking_pixel="<img>",
        score=0.85,
    )
    assert resp.ad_id == 1
    assert resp.variant == "A"
    assert resp.score == 0.85


def test_ad_response_defaults():
    resp = AdResponse(ad_id=1, video_url="x", cta_text="x", cta_url="x")
    assert resp.variant is None
    assert resp.tracking_pixel == ""
    assert resp.score == 0.0


# ─── ViewTokenService Tests ──────────────────────────────────────────────────

def test_view_token_issue_and_validate():
    svc = ViewTokenService(secret="test-secret-key-123")
    token = svc.issue_token(ad_id=42)
    assert len(token) > 10
    assert svc.validate_and_consume(token, ad_id=42) is True


def test_view_token_reject_wrong_ad_id():
    svc = ViewTokenService(secret="test-secret-key-123")
    token = svc.issue_token(ad_id=42)
    assert svc.validate_and_consume(token, ad_id=99) is False


def test_view_token_one_time_use():
    svc = ViewTokenService(secret="test-secret-key-123")
    token = svc.issue_token(ad_id=42)
    assert svc.validate_and_consume(token, ad_id=42) is True
    assert svc.validate_and_consume(token, ad_id=42) is False  # consumed


def test_view_token_invalid_signature():
    svc = ViewTokenService(secret="test-secret-key-123")
    token = svc.issue_token(ad_id=42)
    # Tamper with the token
    parts = token.rsplit(".", 1)
    tampered = parts[0] + ".0000" + parts[1][4:]
    assert svc.validate_and_consume(tampered, ad_id=42) is False


def test_view_token_empty():
    svc = ViewTokenService(secret="test-secret-key-123")
    assert svc.validate_and_consume("", ad_id=42) is False
    assert svc.validate_and_consume(None, ad_id=42) is False


def test_view_token_malformed():
    svc = ViewTokenService(secret="test-secret-key-123")
    assert svc.validate_and_consume("not-a-token", ad_id=42) is False
    assert svc.validate_and_consume("a.b", ad_id=42) is False


def test_view_token_with_cache():
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    svc = ViewTokenService(secret="test-secret-key-123")
    token = svc.issue_token(ad_id=42)
    assert svc.validate_and_consume(token, ad_id=42, cache=cache) is True
    cache.set.assert_called_once()


def test_view_token_with_cache_already_used():
    cache = MagicMock()
    cache.get.return_value = "1"  # already used
    svc = ViewTokenService(secret="test-secret-key-123")
    token = svc.issue_token(ad_id=42)
    assert svc.validate_and_consume(token, ad_id=42, cache=cache) is False


# ─── BillingService Tests ────────────────────────────────────────────────────

def test_billing_rate_us():
    billing = BillingService()
    assert billing.rate_for("US") == Decimal("0.0015")


def test_billing_rate_eg_default():
    billing = BillingService()
    assert billing.rate_for("EG") == Decimal("0.0007")


def test_billing_rate_unknown_locale():
    billing = BillingService()
    assert billing.rate_for("XX") == DEFAULT_RATE


def test_billing_calculate_charge():
    billing = BillingService()
    charge = billing.calculate_charge("US", 1000)
    assert charge == Decimal("0.0015") * 1000


def test_billing_calculate_charge_unknown():
    billing = BillingService()
    charge = billing.calculate_charge("ZZ", 5000)
    assert charge == DEFAULT_RATE * 5000


def test_billing_locale_rates_covered():
    assert "US" in LOCALE_RATES
    assert "EG" in LOCALE_RATES
    assert "JP" in LOCALE_RATES
    assert "BR" in LOCALE_RATES


# ─── ClientIpResolver Tests ──────────────────────────────────────────────────

def test_ip_resolver_direct():
    resolver = ClientIpResolver()
    assert resolver.resolve({}, "1.2.3.4") == "1.2.3.4"


def test_ip_resolver_x_forwarded_for_trusted():
    resolver = ClientIpResolver(trusted_proxies=["10.0.0.1"])
    headers = {"x-forwarded-for": "203.0.113.50, 10.0.0.1"}
    assert resolver.resolve(headers, "10.0.0.1") == "203.0.113.50"


def test_ip_resolver_x_forwarded_for_untrusted():
    resolver = ClientIpResolver(trusted_proxies=["10.0.0.1"])
    headers = {"x-forwarded-for": "203.0.113.50, 10.0.0.1"}
    assert resolver.resolve(headers, "192.168.1.1") == "192.168.1.1"


def test_ip_resolver_x_real_ip():
    resolver = ClientIpResolver(trusted_proxies=["10.0.0.1"])
    headers = {"x-real-ip": "203.0.113.50"}
    assert resolver.resolve(headers, "10.0.0.1") == "203.0.113.50"


def test_ip_resolver_no_headers():
    resolver = ClientIpResolver()
    assert resolver.resolve({"x-forwarded-for": "1.2.3.4"}, "5.6.7.8") == "5.6.7.8"
