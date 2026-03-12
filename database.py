"""
database.py — GymBot Database Connection & Session Management

This file has one job: manage the connection to your database
and hand out sessions to anything that needs to read or write data.

SQLite  → for local development (zero setup, single file)
Postgres → for production (swap by changing DATABASE_URL)

Usage:
    from database import get_session, init_db

    # Create all tables (run once on startup)
    init_db()

    # Use a session in your code
    with get_session() as session:
        user = session.query(User).filter_by(username="thomas").first()
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from models import Base

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Database URL
#
# SQLite  (default for local dev — creates a file called gymbot.db):
#   DATABASE_URL = "sqlite:///./gymbot.db"
#
# PostgreSQL (for production or Supabase):
#   DATABASE_URL = "postgresql://user:password@localhost:5432/gymbot"
#
# Set the DATABASE_URL environment variable to switch between them.
# Everything else in this file works the same regardless.
# ─────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./gymbot.db"     # safe default — just creates gymbot.db in your project folder
)

# Detect which database we're using
IS_SQLITE   = DATABASE_URL.startswith("sqlite")
IS_POSTGRES = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")

# ─────────────────────────────────────────────────────────────
# Engine — the low-level connection to the database
# ─────────────────────────────────────────────────────────────

def _build_engine():
    """
    Build the SQLAlchemy engine with settings appropriate
    for the current database backend.
    """
    if IS_SQLITE:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},  # needed for SQLite in multi-threaded use
            echo=False          # set True to log all SQL to console (useful for debugging)
        )

        # Enable foreign key enforcement in SQLite
        # SQLite has foreign keys but doesn't enforce them by default — this fixes that
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    elif IS_POSTGRES:
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,            # number of persistent connections in the pool
            max_overflow=10,        # extra connections allowed when pool is full
            pool_pre_ping=True,     # test connections before use (handles dropped connections)
            echo=False
        )
        return engine

    else:
        raise ValueError(f"Unsupported database URL scheme: {DATABASE_URL}")


engine = _build_engine()

# ─────────────────────────────────────────────────────────────
# SessionFactory — creates Session objects
# ─────────────────────────────────────────────────────────────

SessionFactory = sessionmaker(
    bind=engine,
    autocommit=False,   # we control commits manually (safer)
    autoflush=False     # we control flushes manually (avoids surprise queries)
)


# ─────────────────────────────────────────────────────────────
# get_session — the main way to interact with the database
# ─────────────────────────────────────────────────────────────

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that provides a database session with automatic
    commit on success and rollback on error.

    Always use this with a 'with' statement:

        with get_session() as session:
            user = User(username="thomas", email="t@example.com")
            session.add(user)
            # session.commit() is called automatically when the block exits cleanly

    If an exception is raised inside the block, the transaction is
    rolled back automatically — the database is left unchanged.
    """
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error, rolling back: {e}")
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error, rolling back: {e}")
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────
# init_db — create all tables on first run
# ─────────────────────────────────────────────────────────────

def init_db():
    """
    Create all tables defined in models.py.

    Safe to call multiple times — SQLAlchemy uses CREATE TABLE IF NOT EXISTS
    so existing tables and data are never touched.

    Call this once when GymBot starts up.
    """
    logger.info(f"Initialising database: {_safe_url()}")
    Base.metadata.create_all(bind=engine)
    logger.info("All tables created (or already existed)")


def drop_all_tables():
    """
    ⚠️  DESTRUCTIVE — drops every table and all data.
    Only useful during development to reset the schema.
    Never call this in production.
    """
    logger.warning("Dropping all tables — all data will be lost!")
    Base.metadata.drop_all(bind=engine)
    logger.info("All tables dropped")


def check_connection() -> bool:
    """
    Verify the database is reachable.
    Returns True if the connection works, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _safe_url() -> str:
    """Return the DB URL with the password redacted for safe logging."""
    if "@" in DATABASE_URL:
        scheme, rest = DATABASE_URL.split("://", 1)
        credentials, host = rest.split("@", 1)
        user = credentials.split(":")[0]
        return f"{scheme}://{user}:****@{host}"
    return DATABASE_URL


def get_db_info() -> dict:
    """
    Return basic info about the current database configuration.
    Useful for the startup banner in start.py.
    """
    return {
        "url":     _safe_url(),
        "backend": "SQLite" if IS_SQLITE else "PostgreSQL",
        "file":    DATABASE_URL.replace("sqlite:///", "") if IS_SQLITE else None,
    }
