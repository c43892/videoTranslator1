"""Private Blob objects with streaming I/O and short-lived delegation SAS."""
from __future__ import annotations

import datetime as dt
import mimetypes
import os
import threading
import uuid
from pathlib import Path, PurePosixPath

from ..domain.enums import BackendError


class AzureBlobStorage:
    def __init__(self, account_url, container='uploads', credential=None, *,
                 results_container='results', client=None):
        if client is None:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
            client = BlobServiceClient(account_url, credential=credential or DefaultAzureCredential())
        self._client = client
        self._container_name = container
        self._results_container = results_container
        self._delegation = None
        self._delegation_until = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls):
        url = os.environ.get('AZURE_STORAGE_ACCOUNT_URL')
        if not url:
            raise BackendError('AZURE_STORAGE_ACCOUNT_URL is required')
        return cls(url, os.environ.get('AZURE_STORAGE_CONTAINER', 'uploads'),
                   results_container=os.environ.get('AZURE_RESULTS_CONTAINER', 'results'))

    def _location(self, key):
        if (not key or '\\' in key or '?' in key or '#' in key or key.startswith('/')
                or any(p in ('', '.', '..') for p in key.split('/'))):
            raise ValueError('Invalid object key')
        if key.startswith('outputs/'):
            return self._results_container, key.removeprefix('outputs/')
        return self._container_name, key.removeprefix('inputs/')

    def _blob(self, key):
        return self._client.get_blob_client(*self._location(key))

    def _sas(self, key, seconds, *, upload):
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas
        if not 1 <= seconds <= 3600:
            raise ValueError('SAS lifetime must be between 1 and 3600 seconds')
        now = dt.datetime.now(dt.timezone.utc)
        expiry = now + dt.timedelta(seconds=seconds)
        with self._lock:
            if self._delegation_until is None or self._delegation_until <= expiry + dt.timedelta(minutes=1):
                until = now + dt.timedelta(hours=2)
                self._delegation = self._client.get_user_delegation_key(now - dt.timedelta(minutes=5), until)
                self._delegation_until = until
            delegation = self._delegation
        container, name = self._location(key)
        blob = self._blob(key)
        sas = generate_blob_sas(account_name=blob.account_name, container_name=container,
                                blob_name=name, user_delegation_key=delegation,
                                permission=BlobSasPermissions(write=True, create=True) if upload
                                else BlobSasPermissions(read=True),
                                start=now - dt.timedelta(minutes=5), expiry=expiry, protocol='https')
        return f'{blob.url}?{sas}'

    def create_upload_url(self, object_key, expires_in):
        if object_key.startswith('outputs/'):
            raise ValueError('Result objects cannot be uploaded by clients')
        return self._sas(object_key, expires_in, upload=True)

    def create_download_url(self, object_key, expires_in):
        return self._sas(object_key, expires_in, upload=False)

    def download(self, object_key, destination):
        from azure.core import MatchConditions
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
        blob = self._blob(object_key)
        metadata = blob.get_blob_properties()
        try:
            with temporary.open('wb') as stream:
                reader = blob.download_blob(etag=metadata.etag, match_condition=MatchConditions.IfNotModified,
                                            max_concurrency=1)
                for chunk in reader.chunks():
                    stream.write(chunk)
            if temporary.stat().st_size != metadata.size:
                raise BackendError('Downloaded object size mismatch')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def upload(self, source, object_key):
        from azure.storage.blob import ContentSettings
        name = PurePosixPath(object_key).name
        content_type = 'text/vtt; charset=utf-8' if name.endswith('.vtt') else (
            mimetypes.guess_type(name)[0] or 'application/octet-stream')
        with Path(source).open('rb') as stream:
            self._blob(object_key).upload_blob(stream, overwrite=True, max_concurrency=1,
                content_settings=ContentSettings(content_type=content_type))
        return object_key

    def exists(self, object_key, *, expected_size=None):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            properties = self._blob(object_key).get_blob_properties()
            return expected_size is None or properties.size == expected_size
        except ResourceNotFoundError:
            return False

    def local_path(self, object_key):
        return None

    def delete(self, object_key):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            self._blob(object_key).delete_blob()
        except ResourceNotFoundError:
            pass
