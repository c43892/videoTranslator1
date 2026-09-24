import time
import uuid
from sqlalchemy import create_engine, String, Text, Float, Integer, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .config import settings

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)

class LoginSession(Base):
    __tablename__ = 'login_sessions'
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    expires: Mapped[float] = mapped_column(Float)

class Job(Base):
    __tablename__ = 'jobs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    filename: Mapped[str] = mapped_column(Text)
    input_key: Mapped[str] = mapped_column(Text)
    target_language: Mapped[str] = mapped_column(String(10))
    terminology: Mapped[str] = mapped_column(Text, default='')
    status: Mapped[str] = mapped_column(String(32), default='queued', index=True)
    stage: Mapped[str] = mapped_column(String(40), default='queued')
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default='')
    created: Mapped[float] = mapped_column(Float, default=time.time)
    updated: Mapped[float] = mapped_column(Float, default=time.time)
    heartbeat: Mapped[float] = mapped_column(Float, default=0)
    outputs: Mapped[dict] = mapped_column(JSON, default=dict)

engine = create_engine(settings().database_url, pool_pre_ping=True)
Session = sessionmaker(engine, expire_on_commit=False)

def init_db():
    # PostgreSQL lock prevents simultaneous first-start schema creation.
    if engine.dialect.name == 'postgresql':
        from sqlalchemy import text
        with engine.connect() as connection:
            connection.execute(text('SELECT pg_advisory_lock(7182951)'))
            connection.commit()
            try:
                Base.metadata.create_all(engine)
            finally:
                connection.execute(text('SELECT pg_advisory_unlock(7182951)'))
    else:
        Base.metadata.create_all(engine)
