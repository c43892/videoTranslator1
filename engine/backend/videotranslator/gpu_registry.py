"""Registered GPU identities, independent of hardware type and transport."""
import time

from sqlalchemy import Float, Integer, String, select, text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, engine


class GpuProvider(Base):
    __tablename__ = 'gpu_providers'
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    provider_type: Mapped[str] = mapped_column(String(16))
    transport: Mapped[str] = mapped_column(String(16))
    owner: Mapped[str] = mapped_column(String(64), default='')
    last_seen: Mapped[float] = mapped_column(Float, default=0)
    ready: Mapped[int] = mapped_column(Integer, default=0)


def migrate_registry():
    """Add routing fields without deleting existing workers or tasks."""
    if engine.dialect.name == 'postgresql':
        with engine.begin() as db:
            for table in ('gpu_workers', 'gpu_tasks'):
                db.execute(text(f"ALTER TABLE IF EXISTS {table} ADD COLUMN IF NOT EXISTS "
                                "provider_id VARCHAR(32) NOT NULL DEFAULT ''"))
            db.execute(text('CREATE INDEX IF NOT EXISTS ix_gpu_tasks_provider_id ON gpu_tasks (provider_id)'))


def registered(db):
    return list(db.scalars(select(GpuProvider).order_by(GpuProvider.id)))


def provider_status(db):
    from .cloud_worker import active_provider_count
    return [{'id': p.id, 'type': p.provider_type, 'transport': p.transport,
             'online': bool(p.ready and p.last_seen > time.time() - 30),
             'busy': active_provider_count(db, p.id) >= 1,
             'last_seen': p.last_seen, 'capacity': 1} for p in registered(db)]
