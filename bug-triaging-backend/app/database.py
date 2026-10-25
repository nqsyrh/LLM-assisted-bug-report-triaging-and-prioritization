"""
database.py
SQLAlchemy engine + session setup. Single SQLite file as your database,
matching FORM 01A's "PostgreSQL or SQLite" tool requirement.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./bug_triaging.db"

# check_same_thread=False is required for SQLite + FastAPI's threaded request handling
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a session per-request and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
