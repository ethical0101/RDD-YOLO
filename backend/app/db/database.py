"""SQLAlchemy engine/session management (SQLite by default, PostgreSQL-ready)."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..core.config import get_settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def init_engine(url: str | None = None) -> Engine:
    global _engine, _SessionLocal
    url = url or get_settings().database_url
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    _engine = create_engine(url, pool_pre_ping=True, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    from . import models  # noqa: F401  (register tables)

    Base.metadata.create_all(_engine)
    return _engine


def get_engine() -> Engine:
    return _engine or init_engine()


def session_factory() -> sessionmaker:
    if _SessionLocal is None:
        init_engine()
    return _SessionLocal


def get_db() -> Iterator[Session]:
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()
