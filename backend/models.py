"""
SQLite job tracking via SQLAlchemy (no Alembic — CREATE TABLE IF NOT EXISTS on startup).
"""

import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, text
from sqlalchemy.orm import DeclarativeBase, Session

DB_PATH = os.getenv("DB_PATH", "jobs.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI threading
)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, nullable=False, default="queued")  # queued|processing|complete|error
    source_lang = Column(String, nullable=False, default="ta")
    target_lang = Column(String, nullable=False)
    video_path = Column(String, nullable=False)
    output_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


def init_db():
    """Create tables if they don't exist. Called at app startup."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)


def create_job(target_lang: str, video_path: str) -> Job:
    job = Job(
        id=str(uuid.uuid4()),
        status="queued",
        source_lang="ta",
        target_lang=target_lang,
        video_path=video_path,
        created_at=datetime.utcnow(),
    )
    with get_session() as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def update_job(job_id: str, **kwargs):
    with get_session() as session:
        job = session.get(Job, job_id)
        if job:
            for key, value in kwargs.items():
                setattr(job, key, value)
            session.commit()


def get_job(job_id: str) -> Job | None:
    with get_session() as session:
        return session.get(Job, job_id)
