from __future__ import annotations

import sqlite3
from collections.abc import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.settings import get_settings


class Base(DeclarativeBase):
    pass


def _create_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    connect_args = {"check_same_thread": False, "timeout": 10} if url.startswith("sqlite") else {}
    engine_kwargs: dict[str, object] = {"connect_args": connect_args, "pool_pre_ping": True}
    if url in {"sqlite://", "sqlite:///:memory:"}:
        engine_kwargs["poolclass"] = StaticPool
    created = create_engine(url, **engine_kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(created, "connect")
        def configure_sqlite(connection: sqlite3.Connection, _: object) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA busy_timeout = 10000")
            if ":memory:" not in url:
                cursor.execute("PRAGMA journal_mode = WAL")
            cursor.close()

    return created


engine = _create_engine()
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def configure_database(url: str) -> None:
    """Rebind the shared session factory to an explicitly selected database.

    This keeps ``create_app(custom_settings)`` deterministic in tests and deployment
    commands instead of leaving the database bound to whichever environment happened
    to be imported first.
    """
    global engine

    if str(engine.url) == url:
        return
    previous = engine
    engine = _create_engine(url)
    SessionLocal.configure(bind=engine)
    previous.dispose()


def init_db() -> None:
    from backend.app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as db:
        yield db
