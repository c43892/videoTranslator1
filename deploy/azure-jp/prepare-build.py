"""Explicit allowlist: never send local media, databases or secrets to Docker."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
target = root / 'vt-data' / 'azure-build'
for name in ('engine', 'packages/videotranslator', 'deploy/azure-jp'):
    shutil.copytree(root / name, target / name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info', '.pytest_cache', '.env*'))
(target / '.dockerignore').write_text('**/__pycache__\n**/*.pyc\n**/*.egg-info\n**/.env*\n')
print(target)
