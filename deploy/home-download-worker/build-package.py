"""Build a portable source ZIP from an explicit public-file allowlist."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parent
output = root.parents[1] / "vt-data/releases/videotranslator-home-worker.zip"
output.parent.mkdir(parents=True, exist_ok=True)
files = ("worker.py", "requirements.txt", "config.example.json", "README.md",
         "Install.cmd", "Install.ps1", "Start.cmd", "Stop.cmd", "Enable-Autostart.ps1", "Enable-Autostart.cmd")
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(root / name, "videotranslator-home-worker/" + name)
print(output)
