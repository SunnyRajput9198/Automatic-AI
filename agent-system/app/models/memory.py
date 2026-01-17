from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Memory(Base):
    """
    Stores learned patterns, successful strategies, and failure cases.
    
    The Memory Agent uses this to:
    - Recall similar past tasks
    - Suggest proven approaches
    - Avoid repeated mistakes
    """
    __tablename__ = "memories"
    
    id = Column(String, primary_key=True)
    
    # What was learned
    pattern_type = Column(String, nullable=False)  # 'success', 'failure', 'strategy'
    task_pattern = Column(Text, nullable=False)  # General pattern (e.g., "file creation")
    
    # Context
    original_task_id = Column(String, nullable=True)
    task_description = Column(Text, nullable=False)
    
    # The learning
    strategy = Column(Text, nullable=True)  # What worked
    tools_used = Column(JSON, nullable=True)  # List of tools
    steps_taken = Column(JSON, nullable=True)  # Successful step sequence
    
    # Performance metrics
    success_rate = Column(Float, default=1.0)  # How often this works
    avg_steps = Column(Float, nullable=True)
    avg_duration = Column(Float, nullable=True)
    
    # Failure info (if pattern_type='failure')
    error_message = Column(Text, nullable=True)
    failure_reason = Column(Text, nullable=True)
    
    # Usage tracking
    times_referenced = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TaskContext(Base):
    """
    Extended context for tasks - stores intermediate results and metadata
    """
    __tablename__ = "task_contexts"
    
    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, unique=True)
    
    # Shared state between steps
    context_data = Column(JSON, default={})  # {step_1_output: "...", variables: {...}}
    
    # Files created during task
    created_files = Column(JSON, default=[])  # ["file1.txt", "file2.py"]
    
    # Memory references used
    memories_used = Column(JSON, default=[])  # List of memory IDs consulted
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)