from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app import babotree_utils

SQLALCHEMY_DATABASE_URL = babotree_utils.get_secret('HETZNER_DATABASE_URL')

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_use_lifo=True,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependency
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_direct_db() -> Session:
    db = SessionLocal()
    return db