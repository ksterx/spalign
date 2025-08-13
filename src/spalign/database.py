"""Database management for conversation progress tracking."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any


def init_db(db_file: str) -> None:
    """Initialize SQLite database for progress tracking."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_hash TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def get_scenario_hash(data: dict[str, Any]) -> str:
    """Generate a unique hash for a scenario."""
    scenario_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(scenario_str.encode("utf-8")).hexdigest()


def insert_pending_scenarios(dataset: list[dict[str, Any]], db_file: str) -> None:
    """Insert all scenarios as pending if not already exists."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    for data in dataset:
        scenario_hash = get_scenario_hash(data)
        cursor.execute(
            """
            INSERT OR IGNORE INTO conversations (scenario_hash, status)
            VALUES (?, 'pending')
        """,
            (scenario_hash,),
        )

    conn.commit()
    conn.close()


def get_pending_scenarios(
    dataset: list[dict[str, Any]], db_file: str
) -> list[dict[str, Any]]:
    """Get scenarios that are still pending."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT scenario_hash FROM conversations WHERE status = 'pending'")
    pending_hashes = {row[0] for row in cursor.fetchall()}

    conn.close()

    return [data for data in dataset if get_scenario_hash(data) in pending_hashes]


def mark_completed(scenario_hash: str, result: dict[str, Any], db_file: str) -> None:
    """Mark a scenario as completed and store the result."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations
        SET status = 'completed', result = ?, updated_at = CURRENT_TIMESTAMP
        WHERE scenario_hash = ?
    """,
        (json.dumps(result, ensure_ascii=False), scenario_hash),
    )

    conn.commit()
    conn.close()


def mark_failed(scenario_hash: str, error: str, db_file: str) -> None:
    """Mark a scenario as failed."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations
        SET status = 'failed', result = ?, updated_at = CURRENT_TIMESTAMP
        WHERE scenario_hash = ?
    """,
        (json.dumps({"error": error}, ensure_ascii=False), scenario_hash),
    )

    conn.commit()
    conn.close()


def get_completed_results(db_file: str) -> list[dict[str, Any]]:
    """Get all completed conversation results."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT result FROM conversations WHERE status = 'completed' AND result IS NOT NULL"
    )
    results = []
    for row in cursor.fetchall():
        try:
            results.append(json.loads(row[0]))
        except json.JSONDecodeError:
            continue

    conn.close()
    return results


def get_progress_stats(db_file: str) -> dict[str, int]:
    """Get progress statistics."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT status, COUNT(*)
        FROM conversations
        GROUP BY status
    """)

    stats = dict(cursor.fetchall())
    conn.close()
    return stats


def get_failed_scenarios(
    dataset: list[dict[str, Any]], db_file: str
) -> list[dict[str, Any]]:
    """Get scenarios that failed previously."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT scenario_hash FROM conversations WHERE status = 'failed'")
    failed_hashes = {row[0] for row in cursor.fetchall()}

    conn.close()

    return [data for data in dataset if get_scenario_hash(data) in failed_hashes]


def reset_failed_to_pending(db_file: str) -> int:
    """Reset all failed scenarios back to pending status for retry."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE conversations
        SET status = 'pending', result = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE status = 'failed'
    """)

    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected
