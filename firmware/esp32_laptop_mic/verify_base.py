Import('env')
from pathlib import Path
import hashlib
import json

project = Path(env.subst('$PROJECT_DIR'))
root = project.parents[1]
base = root / 'firmware/esp32/include'
manifest = json.loads((base / 'package_manifest.json').read_text())
for directory, key in [(root, 'input_sha256'), (base, 'generated_sha256')]:
    for name, expected in manifest[key].items():
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen base package mismatch: {name}')

