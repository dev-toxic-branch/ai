-- ============================================================
-- Ad Serving Agent - MySQL Database Schema
-- ============================================================

CREATE DATABASE IF NOT EXISTS ad_serving
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE ad_serving;

-- ─── Users ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    name            VARCHAR(255) NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            ENUM('advertiser', 'admin') DEFAULT 'advertiser',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email)
) ENGINE=InnoDB;

-- ─── Links ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS links (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    link_code       VARCHAR(64) NOT NULL UNIQUE,
    user_id         BIGINT UNSIGNED NOT NULL,
    target_locations JSON,
    target_devices   JSON,
    target_languages JSON,
    description     TEXT,
    status          ENUM('active', 'paused', 'deleted') DEFAULT 'active',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_link_code (link_code),
    INDEX idx_user (user_id)
) ENGINE=InnoDB;

-- ─── Ads ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ads (
    id                      BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    link_id                 VARCHAR(64) NOT NULL,
    user_id                 BIGINT UNSIGNED NOT NULL,
    video_url               VARCHAR(512) NOT NULL,
    cta_text                VARCHAR(255) DEFAULT 'Learn More',
    cta_url                 VARCHAR(512) DEFAULT '#',
    category_id             INT DEFAULT 0,
    sentiment_score         FLOAT DEFAULT 0.5,
    duration_seconds        INT DEFAULT 0,

    -- targeting (JSON arrays of allowed values; NULL = any)
    target_locations        JSON,
    target_devices          JSON,
    target_languages        JSON,
    target_browsers         JSON,
    target_os               JSON,
    target_connection_types JSON,
    min_screen_width        INT DEFAULT 0,

    -- budget & spend
    budget                  DECIMAL(12,4) NOT NULL DEFAULT 100.0,
    spent                   DECIMAL(12,4) NOT NULL DEFAULT 0.0,

    -- performance
    total_impressions       BIGINT UNSIGNED DEFAULT 0,
    total_clicks            BIGINT UNSIGNED DEFAULT 0,

    -- A/B testing
    ab_test_id              VARCHAR(64),
    variant_count           INT DEFAULT 1,
    variant_group           VARCHAR(64),

    -- scheduling
    start_time              DOUBLE DEFAULT 0,
    end_time                DOUBLE DEFAULT 9999999999,

    status                  ENUM('active', 'paused', 'rejected', 'expired') DEFAULT 'active',
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_link_id (link_id),
    INDEX idx_user (user_id),
    INDEX idx_status (status),
    INDEX idx_ab_test (ab_test_id)
) ENGINE=InnoDB;

-- ─── Impressions ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS impressions (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ad_id           BIGINT UNSIGNED NOT NULL,
    link_id         VARCHAR(64) NOT NULL,
    ip_hash         VARCHAR(64) NOT NULL,
    device          VARCHAR(32),
    os              VARCHAR(32),
    browser         VARCHAR(32),
    location        VARCHAR(8),
    city            VARCHAR(128),
    language        VARCHAR(8),
    referrer        VARCHAR(512),
    connection_type VARCHAR(16),
    screen_width    INT DEFAULT 0,
    screen_height   INT DEFAULT 0,
    user_agent      VARCHAR(512),
    score           FLOAT DEFAULT 0,
    clicked         TINYINT(1) DEFAULT 0,
    completed       TINYINT(1) DEFAULT 0,
    skipped         TINYINT(1) DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ad_id) REFERENCES ads(id) ON DELETE CASCADE,
    INDEX idx_ad (ad_id),
    INDEX idx_link (link_id),
    INDEX idx_ip (ip_hash),
    INDEX idx_created (created_at),
    INDEX idx_ad_date (ad_id, created_at)
) ENGINE=InnoDB;

-- ─── A/B Variants ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ab_variants (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ab_test_id      VARCHAR(64) NOT NULL,
    variant_index   INT NOT NULL,
    variant_name    VARCHAR(64),
    video_url       VARCHAR(512),
    cta_text        VARCHAR(255),
    cta_url         VARCHAR(512),
    description     TEXT,

    FOREIGN KEY (ab_test_id) REFERENCES ads(ab_test_id) ON DELETE CASCADE,
    UNIQUE KEY uq_ab_variant (ab_test_id, variant_index)
) ENGINE=InnoDB;

-- ─── Quartile Events ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS quartile_events (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ad_id           BIGINT UNSIGNED NOT NULL,
    quartile        INT NOT NULL,   -- 25, 50, 75, 100
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ad_id) REFERENCES ads(id) ON DELETE CASCADE,
    INDEX idx_ad_quartile (ad_id, quartile)
) ENGINE=InnoDB;

-- ─── Revenue ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS revenue (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ad_id           BIGINT UNSIGNED NOT NULL,
    impression_id   BIGINT UNSIGNED,
    amount          DECIMAL(10,6) NOT NULL,
    type            ENUM('impression', 'click', 'conversion') NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (ad_id) REFERENCES ads(id) ON DELETE CASCADE,
    INDEX idx_ad (ad_id),
    INDEX idx_type (type),
    INDEX idx_created (created_at),
    INDEX idx_daily (created_at, type)
) ENGINE=InnoDB;

-- ─── Video Processing Jobs ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS processing_jobs (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ad_id           BIGINT UNSIGNED,
    user_id         BIGINT UNSIGNED NOT NULL,
    job_type        VARCHAR(64) NOT NULL,  -- translate, voiceover, subtitle, upscale, etc.
    input_url       VARCHAR(512) NOT NULL,
    output_url      VARCHAR(512),
    status          ENUM('queued', 'processing', 'completed', 'failed') DEFAULT 'queued',
    error_message   TEXT,
    metadata        JSON,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP,

    FOREIGN KEY (ad_id) REFERENCES ads(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user (user_id),
    INDEX idx_status (status)
) ENGINE=InnoDB;
