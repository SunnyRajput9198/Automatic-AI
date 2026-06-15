from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Unified SQLAlchemy 2.0 declarative base.

    All models (Task, Step, Memory, TaskContext) inherit from this single
    Base so that Base.metadata.create_all() creates every table in one call.
    """
    pass
