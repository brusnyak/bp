import logging
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, Session, declarative_base # Import declarative_base from sqlalchemy.orm
from fastapi import HTTPException

# Create a Base class for declarative models
Base = declarative_base()

# Define the User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True) # Added username field
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

def get_db_session_and_engine(database_url: str):
    """
    Configures and returns a database engine and a sessionmaker.
    This allows for easy switching between different databases (e.g., in-memory for tests).
    """
    _engine = create_engine(
        database_url, connect_args={"check_same_thread": False}
    )
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine, _SessionLocal

# Default database URL for development
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# Dependency to get the database session (now takes SessionLocal as an argument)
def init_db(engine):
    """Initializes the database by creating all defined tables."""
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created/checked.")

def get_db(db_session_local):
    """Dependency to get the database session."""
    db = db_session_local()
    try:
        yield db
    finally:
        db.close()
