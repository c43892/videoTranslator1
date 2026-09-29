"""Explicit allowlist: never send local media, databases or secrets to Docker."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
target = root / 'vt-data' / 'azure-build'
expected = (root / 'vt-data' / 'azure-build').absolute()
if target.exists():
    if target.resolve() != expected or not target.resolve().is_relative_to((root / 'vt-data').resolve()):
        raise RuntimeError('Refusing to replace an unexpected build context')
    shutil.rmtree(target)
for name in ('engine', 'packages/videotranslator', 'deploy/azure-jp'):
    shutil.copytree(root / name, target / name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info',
                        '.pytest_cache', '.venv', 'secrets', 'vt-data', 'models',
                        '.env*', '*.key', '*.pem', '*.db', '*.sqlite*',
                        '*.mp4', '*.mkv', '*.mov', '*.avi', '*.webm', '*.wav', '*.mp3'))
(target / '.dockerignore').write_text('**/__pycache__\n**/*.pyc\n**/*.egg-info\n**/.venv\n**/.env*\n')
print(target)
