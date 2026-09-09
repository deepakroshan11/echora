"""
models.py — SQLite job tracking via SQLAlchemy (no migrations needed for v1).
Table is created with CREATE TABLE IF NOT EXISTS on app startup.
"""
import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DB_PATH = os.environ.get("DB_PATH", "shortsdub.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI threading
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, nullable=False, default="queued")   # queued | processing | complete | error
    source_lang = Column(String, nullable=False, default="ta")
    target_lang = Column(String, nullable=False)
    video_path = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


def init_db():
    """Create tables if they don't exist. Called at app startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency-injection helper for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_job(db, target_lang: str, video_path: str) -> Job:
    job = Job(
        id=str(uuid.uuid4()),
        status="queued",
        source_lang="ta",
        target_lang=target_lang,
        video_path=video_path,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db, job_id: str) -> Job | None:
    return db.query(Job).filter(Job.id == job_id).first()


def update_job(db, job_id: str, **kwargs):
    db.query(Job).filter(Job.id == job_id).update(kwargs)
    db.commit()
