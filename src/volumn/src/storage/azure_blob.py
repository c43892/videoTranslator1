"""Azure Blob Storage implementation for job input and output artifacts."""

import os
from pathlib import Path
from typing import Tuple
from urllib.parse import unquote, urlparse

from .base import ObjectStorage


class AzureBlobStorage(ObjectStorage):
    """Read and write ``az://container/blob`` or Azure Blob HTTPS URIs."""

    def __init__(self) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:
            raise ImportError(
                "Azure storage support requires azure-identity and azure-storage-blob"
            ) from exc

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            self.service = BlobServiceClient.from_connection_string(connection_string)
        else:
            account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            if not account_url:
                raise ValueError(
                    "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL"
                )
            self.service = BlobServiceClient(
                account_url=account_url,
                credential=DefaultAzureCredential(),
            )

    @staticmethod
    def split_uri(uri: str) -> Tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme == "az":
            container = parsed.netloc
            blob = parsed.path.lstrip("/")
        elif parsed.scheme == "https" and ".blob.core.windows.net" in parsed.netloc:
            parts = parsed.path.lstrip("/").split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Azure Blob URI must include container and blob: {uri}")
            container, blob = parts
        else:
            raise ValueError(f"Unsupported Azure Blob URI: {uri}")

        if not container or not blob:
            raise ValueError(f"Azure Blob URI must include container and blob: {uri}")
        return unquote(container), unquote(blob)

    def download(self, uri: str, destination: Path) -> Path:
        container, blob = self.split_uri(uri)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        client = self.service.get_blob_client(container=container, blob=blob)
        with destination.open("wb") as output:
            client.download_blob().readinto(output)
        return destination

    def upload(self, source: Path, uri: str) -> str:
        container, blob = self.split_uri(uri)
        source = Path(source)
        client = self.service.get_blob_client(container=container, blob=blob)
        with source.open("rb") as data:
            client.upload_blob(data, overwrite=True)
        return f"az://{container}/{blob}"
