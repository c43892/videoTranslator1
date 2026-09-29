#!/bin/bash
set -euo pipefail
# Run as root on the dedicated validation VM after copying the private bundle.
source_dir=/home/vtadmin/azure-deploy
deploy_dir=/srv/videotranslator
install -d -m 0755 "$deploy_dir" "$deploy_dir/secrets/postgres-tls" /mnt/videotranslator-data
install -d -m 0700 /etc/smbcredentials
for name in .env studio.env engine.env gpu-broker.env postgres.env; do
  install -m 0600 "$source_dir/$name" "$deploy_dir/$name"
done
for name in compose.cpu.yml Caddyfile pg_hba.conf registry-login.py; do
  install -m 0644 "$source_dir/$name" "$deploy_dir/$name"
done
install -o root -g 10001 -m 0440 "$source_dir/firebase-admin.json" "$deploy_dir/secrets/firebase-admin.json"
install -m 0600 "$source_dir/smb.credentials" /etc/smbcredentials/vtranslator.cred
mount_line='//vtranslatorjpe43892.file.core.windows.net/data /mnt/videotranslator-data cifs credentials=/etc/smbcredentials/vtranslator.cred,vers=3.1.1,uid=10001,gid=10001,file_mode=0660,dir_mode=0770,serverino,mfsymlinks,_netdev,nofail 0 0'
grep -qF '//vtranslatorjpe43892.file.core.windows.net/data ' /etc/fstab || printf '%s\n' "$mount_line" >> /etc/fstab
mountpoint -q /mnt/videotranslator-data || mount /mnt/videotranslator-data
install -d -o 10001 -g 10001 -m 0750 "$deploy_dir/studio-data" "$deploy_dir/studio-data/tmp"
if [ ! -s "$deploy_dir/secrets/postgres-tls/server.key" ]; then
  openssl req -x509 -newkey rsa:3072 -nodes -days 7 -subj /CN=postgres \
    -addext 'subjectAltName=DNS:postgres,IP:10.0.2.4' \
    -keyout "$deploy_dir/secrets/postgres-tls/server.key" \
    -out "$deploy_dir/secrets/postgres-tls/server.crt" 2>/dev/null
fi
chown 70:70 "$deploy_dir/secrets/postgres-tls/server.key"
chmod 0600 "$deploy_dir/secrets/postgres-tls/server.key"
chmod 0644 "$deploy_dir/secrets/postgres-tls/server.crt"
chmod 0600 "$source_dir"/*.env "$source_dir/.env" "$source_dir/firebase-admin.json" "$source_dir/smb.credentials"
cd "$deploy_dir"
python3 registry-login.py
docker compose -f compose.cpu.yml pull --quiet
docker compose -f compose.cpu.yml up -d
docker compose -f compose.cpu.yml ps
