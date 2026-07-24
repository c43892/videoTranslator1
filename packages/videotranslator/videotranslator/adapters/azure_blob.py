"""Azure Blob storage adapter (§6.3); SDK imported lazily."""

from __future__ import annotations

import os
from pathlib import Path

from ..domain.enums import BackendError


class AzureBlobStorage:
    def __init__(self, account_url: str, container: str, credential=None):
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        self._container_name = container
        self._client = BlobServiceClient(account_url, credential=credential or DefaultAzureCredential())

    @classmethod
    def from_env(cls) -> "AzureBlobStorage":
        account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        container = os.environ.get("AZURE_STORAGE_CONTAINER", "media")
        if not account_url:
            raise BackendError("AZURE_STORAGE_ACCOUNT_URL is required")
        return cls(account_url, container)

    def create_upload_url(self, object_key: str, expires_in: int) -> str:
        import datetime

        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        blob = self._client.get_blob_client(self._container_name, object_key)
        sas = generate_blob_sas(
            account_name=blob.account_name,
            container_name=self._container_name,
            blob_name=object_key,
            credential=self._client.credential,
            permission=BlobSasPermissions(write=True, create=True),
            expiry=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expires_in),
        )
        return f"{blob.url}?{sas}"

    def create_download_url(self, object_key: str, expires_in: int) -> str:
        import datetime

        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        blob = self._client.get_blob_client(self._container_name, object_key)
        sas = generate_blob_sas(
            account_name=blob.account_name,
            container_name=self._container_name,
            blob_name=object_key,
            credential=self._client.credential,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expires_in),
        )
        return f"{blob.url}?{sas}"

    def download(self, object_key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = self._client.get_blob_client(self._container_name, object_key)
        with open(destination, "wb") as fh:
            fh.write(blob.download_blob().readall())
        return destination

    def upload(self, source: Path, object_key: str) -> str:
        blob = self._client.get_blob_client(self._container_name, object_key)
        with open(source, "rb") as fh:
            blob.upload_blob(fh, overwrite=True)
        return object_key

    def exists(self, object_key: str, *, expected_size: int | None = None) -> bool:
        blob = self._client.get_blob_client(self._container_name, object_key)
        if not blob.exists():
            return False
        if expected_size is None:
            return True
        return blob.get_blob_properties().size == expected_size

    def local_path(self, object_key: str) -> Path | None:
        return None

    def delete(self, object_key: str) -> None:
        blob = self._client.get_blob_client(self._container_name, object_key)
        try:
            blob.delete_blob()
        except Exception:
            pass
