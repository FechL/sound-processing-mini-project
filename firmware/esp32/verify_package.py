"""PlatformIO pre-build guard; does not need NumPy in the PlatformIO environment."""
Import('env')
from pathlib import Path
import hashlib
import json

project = Path(env.subst('$PROJECT_DIR'))
root = project.parents[1]
manifest = json.loads((project / 'include/package_manifest.json').read_text())
for directory, key in [(root, 'input_sha256'), (project / 'include', 'generated_sha256')]:
    for name, expected in manifest[key].items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Stale ESP32 export: {name}; run tools/export_mp16_esp32.py again.')
