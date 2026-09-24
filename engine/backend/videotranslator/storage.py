import json
import os
import uuid
from pathlib import Path

class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not key or not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('Invalid storage key')
        return path

    def exists(self, key: str):
        return bool(key) and self.path(key).is_file()

    def read_json(self, key: str):
        return json.loads(self.path(key).read_text(encoding='utf-8'))

    def write_json(self, key: str, value: dict):
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        with temp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
