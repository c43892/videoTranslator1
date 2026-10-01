"""Configuration shared by ordered GPU provider schedulers.

Provider IDs are service identities, not hardware types. Registered identities
join the pool dynamically; type T4 is always ordered after every other type.
"""
import hashlib
import os
import re


LOCAL_PROVIDER = 'local'
AZURE_PROVIDER = 'azure_t4'
DEFAULT_PROVIDER_PRIORITY = (LOCAL_PROVIDER, AZURE_PROVIDER)
DEFAULT_PROVIDER_CAPACITY = 1
DEFAULT_POLL_SECONDS = 10
_NAME = re.compile(r'^[a-z][a-z0-9_]{0,31}$')


def provider_enabled(provider, db=None):
    kind = None
    if db is not None:
        from .gpu_registry import GpuProvider
        record = db.get(GpuProvider, provider)
        kind = record.provider_type if record else None
    if provider != AZURE_PROVIDER and kind != 't4':
        return True
    return os.getenv('GPU_AZURE_T4_ENABLED', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}


def provider_priority(db=None):
    raw = os.getenv('GPU_PROVIDER_PRIORITY', ','.join(DEFAULT_PROVIDER_PRIORITY))
    providers = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not providers or len(set(providers)) != len(providers) or any(not _NAME.fullmatch(p) for p in providers):
        raise RuntimeError('GPU_PROVIDER_PRIORITY must be a unique comma-separated provider list')
    # T4 is the final fallback even when another provider is appended to the list.
    kinds = {AZURE_PROVIDER: 't4'}
    if db is not None:
        from .gpu_registry import registered
        records = registered(db)
        providers += tuple(p.id for p in records if p.id not in providers)
        kinds.update({p.id: p.provider_type for p in records})
    return tuple(p for p in providers if kinds.get(p) != 't4') + tuple(
        p for p in providers if kinds.get(p) == 't4')


def provider_capacities(db=None):
    capacities = {provider: DEFAULT_PROVIDER_CAPACITY for provider in provider_priority(db)}
    raw = os.getenv('GPU_PROVIDER_CAPACITIES', '')
    for item in (value.strip() for value in raw.split(',') if value.strip()):
        try:
            provider, value = (part.strip() for part in item.split('=', 1))
            capacity = int(value)
        except (ValueError, TypeError):
            raise RuntimeError('GPU_PROVIDER_CAPACITIES must use provider=positive_integer entries')
        if provider not in capacities or capacity < 1:
            raise RuntimeError('GPU_PROVIDER_CAPACITIES contains an unknown provider or invalid capacity')
        capacities[provider] = capacity
    if db is not None:
        from .gpu_registry import registered
        for record in registered(db):
            capacities[record.id] = 1  # One service instance owns one serial GPU.
    return capacities


def always_available_providers():
    raw = os.getenv('GPU_PROVIDER_ALWAYS_AVAILABLE', AZURE_PROVIDER)
    providers = tuple(value.strip() for value in raw.split(',') if value.strip())
    unknown = set(providers) - set(provider_priority())
    if unknown:
        raise RuntimeError('GPU_PROVIDER_ALWAYS_AVAILABLE contains an unknown provider')
    return providers


def provider_lock_id(provider):
    """Return a stable signed bigint key for PostgreSQL advisory locking."""
    digest = hashlib.blake2b(('videotranslator-gpu:' + provider).encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder='big', signed=True)


def poll_seconds():
    value = int(os.getenv('GPU_SCHEDULER_POLL_SECONDS', str(DEFAULT_POLL_SECONDS)))
    if value < 1:
        raise RuntimeError('GPU_SCHEDULER_POLL_SECONDS must be positive')
    return value
