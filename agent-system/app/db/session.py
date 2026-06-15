from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from app.core.config import settings


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    Plain generator for FastAPI Depends() injection.
    Usage: db: Session = Depends(get_db)

    Must NOT have @contextmanager — FastAPI drives the generator itself.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for manual usage (background tasks, scripts, loop_v3).
    Usage: with get_db_context() as db: ...

    Auto-commits on success, rolls back on exception.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Create all tables using the unified Base metadata."""
    from app.db.base import Base
    # Import all models so their metadata is registered on Base
    import app.models.task    # noqa: F401
    import app.models.memory  # noqa: F401

    Base.metadata.create_all(bind=engine)
