from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agent:agent_password@localhost:5432/agent_system")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    """Context manager for non-FastAPI usage"""
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
    """Initialize database tables (Week 1 + Week 2)"""
    from app.models.task import Base as TaskBase
    from app.models.memory import Base as MemoryBase    
    
    # Create Week 1 tables (tasks, steps)
    TaskBase.metadata.create_all(bind=engine)
    
    # Create Week 2 tables (memories, task_contexts)
    MemoryBase.metadata.create_all(bind=engine)