"""Shared serial gates and frozen-artifact audit for MP11--16."""

from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen():
    path = ROOT / "results/metrics/mp13_pretest_artifact_hashes.json"
    if not path.exists():
        raise RuntimeError("Audit hash MP13 belum tersedia.")
    hashes = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Artefak beku berubah: {name}")
    return hashes


def advance(status, number):
    status["last_completed_mini_project"] = number
    status["next_mini_project"] = number + 1 if number < 16 else None
    status["next_input"] = {
        11: "features/logmel/mp09",
        12: "models/mp12_tiny_cnn_state.pt",
        13: "config/evaluation_protocol.json",
        14: "config/realtime_architecture_config.json",
        15: "deployment/mp15_manifest.json",
        16: None,
    }[number]
    done = status.setdefault("completed_mini_projects", [])
    if number not in done:
        done.append(number)


def validate_split(data, split):
    ids = data["sample_ids"].astype(str)
    groups = data["groups"].astype(str)
    masks = [data[key].astype(bool) for key in ("train_mask", "val_mask", "test_mask")]
    if len(set(ids)) != len(ids) or not np.all(np.sum(masks, axis=0) == 1):
        raise RuntimeError(
            "Sample ID duplikat atau mask split tumpang tindih/tidak lengkap."
        )
    group_sets = []
    for mask, key in zip(masks, ("train_ids", "validation_ids", "test_ids")):
        if len(split[key]) != len(set(split[key])) or set(ids[mask]) != set(split[key]):
            raise RuntimeError(f"Split NPZ tidak cocok dengan MP9: {key}")
        group_sets.append(set(groups[mask]))
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Acquisition group tumpang tindih antar-split.")
