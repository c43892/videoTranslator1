#!/bin/bash
set -euo pipefail
# Run as root on the validation VM. Only a new temporary restore database is dropped.
backup_dir=/srv/videotranslator/validation-backup
install -d -m 0700 "$backup_dir"
umask 077
docker exec videotranslator-cloud-postgres-1 pg_dump -U engineadmin -d videotranslator -Fc > "$backup_dir/engine.dump"
restore_db="vt_restore_$(date +%s)_$RANDOM"
docker exec videotranslator-cloud-postgres-1 createdb -U engineadmin "$restore_db"
trap 'docker exec videotranslator-cloud-postgres-1 dropdb -U engineadmin "$restore_db"' EXIT
docker exec -i videotranslator-cloud-postgres-1 pg_restore -U engineadmin -d "$restore_db" --no-owner --exit-on-error < "$backup_dir/engine.dump"
docker exec videotranslator-cloud-postgres-1 psql -U engineadmin -d "$restore_db" -v ON_ERROR_STOP=1 -c 'SELECT count(*) AS restored_executions FROM cloud_executions;'
docker exec -i videotranslator-cloud-web-1 python - <<'PY'
import os, sqlite3
from pathlib import Path
target=Path('/data/validation-backup')
target.mkdir(mode=0o700,exist_ok=True)
for name in ('store-stripe-sandbox.db','inspection-queue-sandbox.db'):
    with sqlite3.connect('file:/data/'+name+'?mode=ro',uri=True) as source, sqlite3.connect(target/name) as destination:
        source.backup(destination)
        assert destination.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert destination.execute('SELECT count(*) FROM sqlite_master WHERE type=\'table\'').fetchone()[0]>0
    os.chmod(target/name,0o600)
print('SQLite online snapshots reopened and passed integrity checks.')
PY
echo 'PostgreSQL restored successfully into a separate temporary database.'
install -o 10001 -g 10001 -m 0600 "$backup_dir/engine.dump" /srv/videotranslator/studio-data/validation-backup/engine.dump
docker exec -i videotranslator-cloud-web-1 python - <<'PY'
from datetime import datetime, timezone
import hashlib, os
from pathlib import Path
from azure.core.exceptions import ResourceExistsError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
service=BlobServiceClient(os.environ['AZURE_STORAGE_ACCOUNT_URL'],credential=ManagedIdentityCredential())
container=service.get_container_client('validation-backups')
try:
    container.create_container()
except ResourceExistsError:
    pass
assert container.get_container_properties().public_access is None
prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
for name in ('engine.dump','store-stripe-sandbox.db','inspection-queue-sandbox.db'):
    data=(Path('/data/validation-backup')/name).read_bytes()
    blob=container.get_blob_client(prefix+'/'+name)
    blob.upload_blob(data,overwrite=False)
    assert hashlib.sha256(blob.download_blob().readall()).digest()==hashlib.sha256(data).digest()
print('Private Blob backup upload/download hashes verified for PostgreSQL and both SQLite databases.')
PY
