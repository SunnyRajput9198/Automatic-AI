from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base


class Memory(Base):
    """
    Stores learned patterns, successful strategies, and failure cases.

    The ConfidenceMemory agent uses this to:
    - Recall similar past tasks
    - Suggest proven approaches
    - Avoid repeated mistakes

    Column types are explicitly declared with Mapped[T] so static type
    checkers (Pyright / mypy) know the Python type of every attribute.
    """

    __tablename__ = "memories"

    # Primary key
    id: Mapped[str] = mapped_column(String, primary_key=True)

    # What was learned
    pattern_type: Mapped[str] = mapped_column(String, nullable=False)
    task_pattern: Mapped[str] = mapped_column(Text, nullable=False)

    # Context
    original_task_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    task_description: Mapped[str] = mapped_column(Text, nullable=False)

    # The learning
    strategy: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tools_used: Mapped[Optional[List[str]]] = mapped_column(JSONB, nullable=True)
    steps_taken: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, nullable=True)

    # Performance metrics
    # success_rate doubles as a confidence proxy (0.0–1.0).
    # Typed as Optional[float] because existing rows may be NULL.
    success_rate: Mapped[Optional[float]] = mapped_column(Float, default=1.0, nullable=True)
    avg_steps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Failure info (populated when pattern_type == "failure")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Usage tracking
    times_referenced: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )


class TaskContext(Base):
    """
    Extended context for tasks — stores intermediate results and metadata
    that persists across steps and is available to the orchestrator.
    """

    __tablename__ = "task_contexts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    # Shared state written by each step (step outputs, variables, etc.)
    context_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, default=dict, nullable=True
    )

    # Filenames created during the task execution
    created_files: Mapped[Optional[List[str]]] = mapped_column(
        JSONB, default=list, nullable=True
    )

    # IDs of Memory rows consulted during this task
    memories_used: Mapped[Optional[List[str]]] = mapped_column(
        JSONB, default=list, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )
