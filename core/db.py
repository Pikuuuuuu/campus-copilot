"""SQLite logging + analytics.

Every question is logged so the product can be managed with data:
- answer rate and "not found" questions (= content gaps the admin should fill)
- helpfulness from thumbs up/down
- latency and category mix

The analytics SQL is kept in one place (ANALYTICS_QUERIES) and shown in the admin
dashboard so anyone reviewing the product can see exactly how each metric is computed.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    question      TEXT NOT NULL,
    search_query  TEXT,
    category      TEXT,
    status        TEXT NOT NULL,          -- answered | not_found | out_of_scope | blocked | error
    answer        TEXT,
    sources       TEXT,                  -- JSON list of {source, page}
    retrieval_mode TEXT,
    latency_ms    INTEGER
);
CREATE TABLE IF NOT EXISTS feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL REFERENCES interactions(id),
    created_at     TEXT NOT NULL,
    rating         INTEGER NOT NULL,     -- 1 helpful, -1 not helpful
    comment        TEXT,
    UNIQUE(interaction_id)
);
CREATE INDEX IF NOT EXISTS idx_interactions_status ON interactions(status);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);
"""

ANALYTICS_QUERIES = {
    "headline": """
SELECT
  COUNT(*)                                                        AS total_questions,
  COUNT(DISTINCT session_id)                                      AS sessions,
  ROUND(100.0 * SUM(status = 'answered') / NULLIF(SUM(status IN ('answered','not_found')), 0), 1)
                                                                  AS answer_rate_pct,
  ROUND(100.0 * (SELECT SUM(rating = 1) FROM feedback) /
        NULLIF((SELECT COUNT(*) FROM feedback), 0), 1)            AS helpful_pct,
  (SELECT COUNT(*) FROM feedback)                                 AS feedback_count,
  CAST(AVG(latency_ms) AS INTEGER)                                AS avg_latency_ms
FROM interactions;""",
    "by_category": """
SELECT category,
       COUNT(*) AS questions,
       ROUND(100.0 * SUM(status = 'answered') / COUNT(*), 1) AS answered_pct
FROM interactions
WHERE status IN ('answered', 'not_found')
GROUP BY category
ORDER BY questions DESC;""",
    "content_gaps": """
SELECT question, category, COUNT(*) AS times_asked, MAX(created_at) AS last_asked
FROM interactions
WHERE status = 'not_found'
GROUP BY LOWER(TRIM(question))
ORDER BY times_asked DESC, last_asked DESC
LIMIT 25;""",
    "negative_feedback": """
SELECT i.created_at, i.question, i.answer, f.comment
FROM feedback f JOIN interactions i ON i.id = f.interaction_id
WHERE f.rating = -1
ORDER BY f.created_at DESC
LIMIT 25;""",
    "daily": """
SELECT DATE(created_at) AS day,
       COUNT(*) AS questions,
       SUM(status = 'answered') AS answered
FROM interactions
GROUP BY DATE(created_at)
ORDER BY day;""",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path = config.DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_interaction(
    session_id: str,
    question: str,
    search_query: str,
    status: str,
    answer: str,
    category: str,
    sources: list[dict],
    retrieval_mode: str,
    latency_ms: int,
    path: Path = config.DB_PATH,
) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO interactions
               (created_at, session_id, question, search_query, category, status, answer, sources,
                retrieval_mode, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), session_id, question, search_query, category, status, answer,
             json.dumps(sources), retrieval_mode, latency_ms),
        )
        return int(cur.lastrowid)


def log_feedback(interaction_id: int, rating: int, comment: str = "", path: Path = config.DB_PATH) -> None:
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO feedback (interaction_id, created_at, rating, comment) VALUES (?, ?, ?, ?)
               ON CONFLICT(interaction_id) DO UPDATE SET rating = excluded.rating,
               comment = excluded.comment, created_at = excluded.created_at""",
            (interaction_id, _now(), rating, comment),
        )


def run_query(name: str, path: Path = config.DB_PATH) -> list[dict]:
    with connect(path) as conn:
        return [dict(row) for row in conn.execute(ANALYTICS_QUERIES[name]).fetchall()]
