"""Configuration shared by ordered GPU provider schedulers.

Provider names are deployment identities, not hardware types. Adding a provider
means adding it to the ordered list and running a worker with the same name.
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


def provider_priority():
    raw = os.getenv('GPU_PROVIDER_PRIORITY', ','.join(DEFAULT_PROVIDER_PRIORITY))
    providers = tuple(value.strip() for value in raw.split(',') if value.strip())
    if not providers or len(set(providers)) != len(providers) or any(not _NAME.fullmatch(p) for p in providers):
        raise RuntimeError('GPU_PROVIDER_PRIORITY must be a unique comma-separated provider list')
    return providers


def provider_capacities():
    capacities = {provider: DEFAULT_PROVIDER_CAPACITY for provider in provider_priority()}
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
