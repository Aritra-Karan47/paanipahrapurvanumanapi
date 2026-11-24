"""Connection management and utilities for Neon Serverless Postgres."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    psycopg = None  # type: ignore
    dict_row = None  # type: ignore


class DatabaseNotConfiguredError(Exception):
    """Raised when an operation requires a database connection but no DATABASE_URL is configured."""
    pass


def get_database_url() -> Optional[str]:
    """Retrieve and normalize the PostgreSQL / Neon database connection URL.

    Checks:
    1. Direct URL env vars: DATABASE_URL, NEON_DATABASE_URL, POSTGRES_URL.
    2. Decomposed env vars: PGHOST, PGDATABASE, PGUSER, PGPASSWORD, PGPORT, PGSSLMODE.

    Normalizes 'postgres://' scheme to 'postgresql://' and ensures 'sslmode=require'
    for Neon Postgres endpoints.
    """
    raw_url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("NEON_DATABASE_URL")
        or os.environ.get("POSTGRES_URL")
    )

    if not raw_url and os.environ.get("PGHOST"):
        pghost = os.environ.get("PGHOST", "").strip()
        pgdatabase = os.environ.get("PGDATABASE", "neondb").strip()
        pguser = os.environ.get("PGUSER", "").strip()
        pgpassword = os.environ.get("PGPASSWORD", "").strip()
        pgport = os.environ.get("PGPORT", "5432").strip()
        pgsslmode = os.environ.get("PGSSLMODE", "require").strip()

        user_auth = f"{pguser}:{pgpassword}@" if (pguser or pgpassword) else ""
        raw_url = f"postgresql://{user_auth}{pghost}:{pgport}/{pgdatabase}?sslmode={pgsslmode}"

    if not raw_url or not raw_url.strip():
        return None

    url_str = raw_url.strip()
    if url_str.startswith("postgres://"):
        url_str = "postgresql://" + url_str[len("postgres://"):]

    # Parse and ensure sslmode=require for Neon serverless cloud
    try:
        parsed = urlparse(url_str)
        # Only inject sslmode for remote endpoints, not localhost/127.0.0.1
        hostname = (parsed.hostname or "").lower()
        if hostname not in ("localhost", "127.0.0.1"):
            query_params = parse_qs(parsed.query)
            if "sslmode" not in query_params:
                query_params["sslmode"] = ["require"]
                new_query = urlencode(query_params, doseq=True)
                url_str = urlunparse(parsed._replace(query=new_query))
    except Exception as exc:
        logger.debug("URL normalization skipped: %s", exc)

    return url_str


def is_db_configured() -> bool:
    """Return True if a database URL is present in the environment."""
    return bool(get_database_url())


def get_db_connection(timeout: float = 5.0) -> Any:
    """Open a new connection to PostgreSQL / Neon with dict row factory.

    Raises:
        DatabaseNotConfiguredError: If no DATABASE_URL is found in environment.
        RuntimeError: If psycopg is not installed.
        psycopg.Error: If the connection attempt fails.
    """
    if not PSYCOPG_AVAILABLE:
        raise RuntimeError("psycopg is not installed. Please install 'psycopg[binary]'.")

    url = get_database_url()
    if not url:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL or NEON_DATABASE_URL is not configured in the environment."
        )

    return psycopg.connect(
        url,
        row_factory=dict_row,
        connect_timeout=int(timeout),
        autocommit=True,
    )


def check_db_connection(timeout: float = 3.0) -> Tuple[bool, str]:
    """Check connectivity to the database by executing 'SELECT 1'.

    Returns:
        (True, "Connected") on success, or (False, error_message) on failure.
    """
    if not is_db_configured():
        return False, "DATABASE_URL is not configured"

    if not PSYCOPG_AVAILABLE:
        return False, "psycopg package not installed"

    try:
        with get_db_connection(timeout=timeout) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
                if row and row.get("ok") == 1:
                    return True, "Connected to Neon Postgres"
        return False, "No response from database"
    except Exception as exc:
        return False, str(exc)
