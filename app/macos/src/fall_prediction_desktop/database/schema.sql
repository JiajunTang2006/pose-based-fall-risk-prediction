-- FallGuard SQLite Schema V1.0
-- All timestamps are ISO 8601 UTC strings.
-- Videos/screenshots are stored on disk; DB stores paths + metadata only.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Application-level key-value settings ──────────────────────────
CREATE TABLE IF NOT EXISTS app_settings (
    key         TEXT PRIMARY KEY NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ── User profiles (one per monitored person) ──────────────────────
CREATE TABLE IF NOT EXISTS profiles (
    id              TEXT PRIMARY KEY NOT NULL,
    name            TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 0,
    sensitivity     TEXT NOT NULL DEFAULT 'medium',   -- low | medium | high
    prefall_threshold   REAL NOT NULL DEFAULT 0.45,
    fall_threshold      REAL NOT NULL DEFAULT 0.72,
    consecutive_frames  INTEGER NOT NULL DEFAULT 3,
    cooldown_seconds    INTEGER NOT NULL DEFAULT 30,
    camera_index        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ── Monitoring sessions (one per Start→Stop cycle) ────────────────
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id              TEXT PRIMARY KEY NOT NULL,
    profile_id      TEXT NOT NULL,
    source_type     TEXT NOT NULL DEFAULT 'camera',  -- camera | video | images
    source_path     TEXT,
    status          TEXT NOT NULL DEFAULT 'running', -- running | stopped | error | cancelled
    model_version   TEXT,
    pose_backend    TEXT NOT NULL DEFAULT 'yolo',
    predictor_type  TEXT NOT NULL DEFAULT 'ml',
    total_frames    INTEGER NOT NULL DEFAULT 0,
    total_events    INTEGER NOT NULL DEFAULT 0,
    peak_risk       REAL NOT NULL DEFAULT 0.0,
    avg_risk        REAL NOT NULL DEFAULT 0.0,
    fps_avg         REAL NOT NULL DEFAULT 0.0,
    resolution      TEXT,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    error_message   TEXT,
    FOREIGN KEY (profile_id) REFERENCES profiles(id)
);

-- ── Risk samples (periodic, every 0.5–1 s) ────────────────────────
CREATE TABLE IF NOT EXISTS risk_samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    frame_index     INTEGER NOT NULL,
    timestamp       REAL NOT NULL,          -- seconds since session start
    risk_score      REAL NOT NULL,
    visibility      REAL NOT NULL,
    state           TEXT NOT NULL,          -- Normal | Pre-fall | Fall | Unknown
    confidence      REAL NOT NULL DEFAULT 0.0,
    recorded_at     TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_risk_samples_session ON risk_samples(session_id);

-- ── Events (business-level fall/pre-fall detection events) ────────
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY NOT NULL,
    session_id      TEXT NOT NULL,
    profile_id      TEXT NOT NULL,
    event_type      TEXT NOT NULL,           -- pre-fall | fall
    status          TEXT NOT NULL DEFAULT 'open', -- open | ended | reviewed
    peak_risk       REAL NOT NULL DEFAULT 0.0,
    avg_risk        REAL NOT NULL DEFAULT 0.0,
    min_confidence  REAL NOT NULL DEFAULT 0.0,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    thumbnail_path  TEXT,
    video_clip_path TEXT,
    user_feedback   TEXT,                    -- confirmed | near_fall | normal | false_alarm | unsure
    notes           TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_profile ON events(profile_id);

-- ── Media files (imported videos/images + event clips) ────────────
CREATE TABLE IF NOT EXISTS media_files (
    id              TEXT PRIMARY KEY NOT NULL,
    session_id      TEXT,
    event_id        TEXT,
    file_path       TEXT NOT NULL,
    media_type      TEXT NOT NULL,           -- video | image | event_clip | thumbnail
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    width           INTEGER,
    height          INTEGER,
    fps             REAL,
    duration_seconds REAL,
    status          TEXT NOT NULL DEFAULT 'pending', -- pending | processing | complete | failed
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE SET NULL,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_media_files_session ON media_files(session_id);
