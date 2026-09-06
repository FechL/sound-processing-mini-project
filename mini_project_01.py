from pathlib import Path
import argparse
import json
import os
import warnings

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sound-processing-mini-project")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt

DIRS = [
    "config",
    "data/raw",
    "data/processed",
    "features",
    "models",
    "results/figures",
    "results/metrics",
    "firmware",
]

CONFIG = {
    "project_name": "embedded_keyword_spotting",
    "labels": ["nyala", "mati", "naik", "turun"],
    "sample_rate_hz": 16000,
    "channels": 1,
    "bit_depth": 16,
    "record_duration_s": 2.0,
    "reference_file": "data/raw/nyala_001.wav",
}


def ensure_project_files():
    for folder in DIRS:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "config/project_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)


def record_reference_audio(output_wav):
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Paket sounddevice belum terpasang. Jalankan: python -m pip install sounddevice"
        ) from exc

    fs = CONFIG["sample_rate_hz"]
    duration = CONFIG["record_duration_s"]

    print("Mini Project 1 - Keyword Spotting Embedded")
    print("Katakan kata: NYALA")
    input("Tekan ENTER ketika siap merekam...")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=CONFIG["channels"],
        dtype="float32",
    )
    sd.wait()

    x = recording[:, 0]
    x_pcm16 = np.int16(np.clip(x, -1.0, 1.0) * 32767)
    wavfile.write(output_wav, fs, x_pcm16)
    return fs, x


def read_reference_audio(input_wav):
    fs, audio = wavfile.read(input_wav)

    if audio.ndim == 2:
        audio = audio[:, 0]

    if audio.dtype == np.int16:
        x = audio.astype(np.float32) / 32768.0
        bit_depth = 16
    elif audio.dtype == np.int32:
        x = audio.astype(np.float32) / 2147483648.0
        bit_depth = 32
    elif np.issubdtype(audio.dtype, np.floating):
        x = audio.astype(np.float32)
        bit_depth = "float"
    else:
        max_value = np.iinfo(audio.dtype).max
        x = audio.astype(np.float32) / max_value
        bit_depth = int(np.iinfo(audio.dtype).bits)

    return fs, x, bit_depth


def write_metadata_and_waveform(input_wav, fs, x, detected_bit_depth):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    clipping_ratio = float(np.mean(np.abs(x) >= 0.99)) if len(x) else 0.0
    duration_s = float(len(x) / fs) if fs else 0.0

    expected_duration = CONFIG["record_duration_s"]
    duration_ok = abs(duration_s - expected_duration) <= 0.15
    format_ok = (
        fs == CONFIG["sample_rate_hz"]
        and detected_bit_depth == CONFIG["bit_depth"]
        and duration_ok
    )
    level_ok = peak > 0.02 and clipping_ratio == 0.0

    metadata = {
        "file": str(input_wav.relative_to(ROOT)),
        "sample_rate_hz": int(fs),
        "channels": 1,
        "bit_depth": detected_bit_depth,
        "duration_s": duration_s,
        "num_samples": int(len(x)),
        "peak_normalized": peak,
        "rms_normalized": rms,
        "clipping_ratio": clipping_ratio,
        "quality_check": "PASS" if format_ok and level_ok else "CHECK",
        "expected": {
            "sample_rate_hz": CONFIG["sample_rate_hz"],
            "channels": CONFIG["channels"],
            "bit_depth": CONFIG["bit_depth"],
            "duration_s": CONFIG["record_duration_s"],
        },
    }

    metadata_path = ROOT / "results/metrics/mp01_audio_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    axis_t = np.arange(len(x)) / fs
    plt.figure(figsize=(9, 3))
    plt.plot(axis_t, x, linewidth=0.8)
    plt.xlabel("Waktu (s)")
    plt.ylabel("Amplitudo ternormalisasi")
    plt.title("Mini Project 1 - Waveform Kata 'nyala'")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    figure_path = ROOT / "results/figures/mp01_waveform.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()

    status = {
        "last_completed_mini_project": 1 if metadata["quality_check"] == "PASS" else 0,
        "next_mini_project": 2 if metadata["quality_check"] == "PASS" else 1,
        "next_input": CONFIG["reference_file"],
        "config": "config/project_config.json",
        "status": (
            "ready_for_mp02"
            if metadata["quality_check"] == "PASS"
            else "mp01_needs_audio_check"
        ),
    }
    with open(ROOT / "project_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    return metadata, metadata_path, figure_path


def main():
    parser = argparse.ArgumentParser(
        description="Mini Project 1 - inisialisasi proyek keyword spotting."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Rekam data/raw/nyala_001.wav dari mikrofon.",
    )
    args = parser.parse_args()

    ensure_project_files()
    reference_wav = ROOT / CONFIG["reference_file"]

    if args.record:
        fs, x = record_reference_audio(reference_wav)
        detected_bit_depth = CONFIG["bit_depth"]
    else:
        if not reference_wav.exists():
            print("Struktur dan konfigurasi Mini Project 1 sudah siap.")
            print(f"Tambahkan rekaman WAV ke: {reference_wav}")
            print("Setelah itu jalankan: python mini_project_01.py")
            return
        fs, x, detected_bit_depth = read_reference_audio(reference_wav)

    metadata, metadata_path, figure_path = write_metadata_and_waveform(
        reference_wav, fs, x, detected_bit_depth
    )

    print("\nSelesai.")
    print("Audio    :", reference_wav)
    print("Metadata :", metadata_path)
    print("Waveform :", figure_path)
    print("Peak     :", round(metadata["peak_normalized"], 4))
    print("RMS      :", round(metadata["rms_normalized"], 4))
    print("Clipping :", metadata["clipping_ratio"])
    print("Check    :", metadata["quality_check"])


if __name__ == "__main__":
    main()
