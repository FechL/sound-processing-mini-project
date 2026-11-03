from pathlib import Path
import json, time, shutil, warnings
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn


from kws_common import advance, verify_frozen, sha256, validate_split
from kws_models import (
    TinyKWSNet as FloatTinyKWS,
    QuantTinyKWS,
    load_logmel as read_logmel,
)

ROOT = Path(__file__).resolve().parent
SEED = 2026
PERF_TOL = 0.03
TARGET_FRAMES = 96
BENCH_ITERS = 120
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_logmel(sample_id, mu, sigma):
    x, _ = read_logmel(ROOT, sample_id, TARGET_FRAMES)
    return (x - mu) / sigma


def bench(fn, warmup=20, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        vals.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "mean_ms": float(np.mean(vals)),
        "p95_ms": float(np.percentile(vals, 95)),
        "min_ms": float(np.min(vals)),
    }


# ---------- Gate ----------
status_path = ROOT / "project_status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
if status.get("status") != "ready_for_mp15":
    raise RuntimeError(
        f"Status harus ready_for_mp15, ditemukan {status.get('status')!r}"
    )

mp13 = json.loads(
    (ROOT / "results/metrics/mp13_final_test_metrics.json").read_text(encoding="utf-8")
)
rt = json.loads(
    (ROOT / "config/realtime_architecture_config.json").read_text(encoding="utf-8")
)
red = json.loads(
    (ROOT / "config/feature_reduction_config.json").read_text(encoding="utf-8")
)
clf_cfg = json.loads(
    (ROOT / "config/classifier_config.json").read_text(encoding="utf-8")
)
dl_cfg = json.loads(
    (ROOT / "config/deep_learning_config.json").read_text(encoding="utf-8")
)
split = json.loads((ROOT / "splits/mp09_split.json").read_text(encoding="utf-8"))

if mp13.get("post_test_tuning_performed"):
    raise RuntimeError("MP13 menunjukkan post-test tuning. Deployment dihentikan.")
if rt.get("final_model_selection") != "DEFERRED_TO_MP15":
    raise RuntimeError("MP14 tidak menyerahkan pemilihan model ke MP15.")

verify_frozen()
classes = list(clf_cfg["classes"])
CLASSICAL_NAME = (
    "classical_svm_int8"
    if clf_cfg["chosen_family"] == "linear_svm"
    else "classical_logreg_int8"
)
if clf_cfg["chosen_family"] not in ("linear_svm", "logreg"):
    raise RuntimeError(
        "Exporter INT8 membutuhkan baseline linear beku; model tidak boleh diganti setelah test."
    )

# ---------- A. Linear SVM float -> INT8 reference ----------
svm = joblib.load(ROOT / clf_cfg["model_file"])
D39 = np.load(ROOT / "features/dataset/mp09_features_39d.npz", allow_pickle=True)
validate_split(D39, split)
X39 = D39["X"].astype(np.float64)
y = D39["y"].astype(str)
train_mask = D39["train_mask"].astype(bool)
test_mask = D39["test_mask"].astype(bool)
sel = np.asarray(red["selected_indices"], dtype=np.int64)
mean39 = np.asarray(red["scaler_mean"], dtype=np.float32).astype(np.float64)
scale39 = np.asarray(red["scaler_scale"], dtype=np.float32).astype(np.float64)
if len(sel) != clf_cfg["input_dim"] or not np.all(scale39 > 0):
    raise RuntimeError(
        "Exporter memerlukan selected features dan scaler MP10 yang valid."
    )
Z8 = ((X39 - mean39) / scale39)[:, sel]
float_pred = svm.predict(Z8)

W = np.asarray(svm.coef_, dtype=np.float64)
b = np.asarray(svm.intercept_, dtype=np.float64)
# Input scale dan weight scale dihitung tanpa tuning test. Input calibration = training only.
train_max = float(np.max(np.abs(Z8[train_mask])))
x_scale = float(np.float32(max(train_max / 127.0, 1e-8)))
W_scales = (
    np.maximum(np.max(np.abs(W), axis=1) / 127.0, 1e-8)
    .astype(np.float32)
    .astype(np.float64)
)
b = b.astype(np.float32).astype(np.float64)
if W.shape != (len(classes), len(sel)):
    raise RuntimeError("Exporter ini membutuhkan skor linear multiclass empat kelas.")
Wq = np.clip(np.round(W / W_scales[:, None]), -127, 127).astype(np.int8)


def svm_int8_scores(z):
    zq = np.clip(np.round(z / x_scale), -127, 127).astype(np.int8)
    acc = Wq.astype(np.int32) @ zq.astype(np.int32)
    return acc.astype(np.float64) * (W_scales * x_scale) + b


q_pred = np.asarray([classes[int(np.argmax(svm_int8_scores(z)))] for z in Z8])
svm_agree_all = float(np.mean(q_pred == float_pred))
svm_agree_test = float(np.mean(q_pred[test_mask] == float_pred[test_mask]))

# Export package.
model_dir = ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)
firmware_dir = ROOT / "firmware"
firmware_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    model_dir / "mp15_classical_int8.npz",
    selected_indices=sel.astype(np.int32),
    selected_mean=mean39[sel].astype(np.float32),
    selected_scale=scale39[sel].astype(np.float32),
    input_quant_scale=np.asarray([x_scale], dtype=np.float32),
    weight_q=Wq,
    weight_scales=W_scales.astype(np.float32),
    bias=b.astype(np.float32),
    classes=np.asarray(classes, dtype=str),
    model_family=np.asarray([clf_cfg["chosen_family"]], dtype=str),
)


# C header for MCU reference implementation.
def c_float(value):
    return f"{float(value):.9e}f"


def c_arr(name, arr, ctype, fmt):

    flat = np.asarray(arr).reshape(-1)
    vals = ", ".join(fmt(v) for v in flat)
    return f"static const {ctype} {name}[{len(flat)}] = {{{vals}}};\n"


h = []
h.append(
    "// Auto-generated by Mini Project 15. Frozen deployment package.\n#pragma once\n#include <stdint.h>\n"
)
h.append(f"#define MP15_INPUT_DIM {len(sel)}\n#define MP15_N_CLASSES {len(classes)}\n")
h.append(c_arr("MP15_FEATURE_INDEX", sel, "int32_t", lambda v: str(int(v))))
h.append(c_arr("MP15_FEATURE_MEAN", mean39[sel], "float", c_float))
h.append(c_arr("MP15_FEATURE_SCALE", scale39[sel], "float", c_float))
h.append(f"static const float MP15_INPUT_Q_SCALE = {c_float(x_scale)};\n")
h.append(c_arr("MP15_WEIGHT_Q", Wq, "int8_t", lambda v: str(int(v))))
h.append(c_arr("MP15_WEIGHT_SCALE", W_scales, "float", c_float))
h.append(c_arr("MP15_BIAS", b, "float", c_float))
for i, c in enumerate(classes):
    h.append(f"// class {i}: {c}\n")
h.append("""
#include <math.h>
static inline int MP15_predict39(const float *features, float *scores) {
  int32_t xq[MP15_INPUT_DIM];
  for (int j = 0; j < MP15_INPUT_DIM; ++j) {
    double z = ((double)features[MP15_FEATURE_INDEX[j]] - MP15_FEATURE_MEAN[j])
               / MP15_FEATURE_SCALE[j];
    if (!isfinite(z)) return -1;
    double q = nearbyint(z / MP15_INPUT_Q_SCALE);
    xq[j] = q < -127 ? -127 : (q > 127 ? 127 : (int32_t)q);
  }
  int best = 0;
  double best_score = -INFINITY;
  for (int k = 0; k < MP15_N_CLASSES; ++k) {
    int32_t acc = 0;
    for (int j = 0; j < MP15_INPUT_DIM; ++j)
      acc += (int32_t)MP15_WEIGHT_Q[k * MP15_INPUT_DIM + j] * xq[j];
    double score = (double)acc * MP15_WEIGHT_SCALE[k] * MP15_INPUT_Q_SCALE + MP15_BIAS[k];
    if (scores) scores[k] = (float)score;
    if (score > best_score) { best_score = score; best = k; }
  }
  return best;
}
""")
(firmware_dir / "mp15_classical_model_int8.h").write_text("".join(h), encoding="utf-8")

# ---------- B. Tiny CNN float -> static INT8 PTQ ----------


ckpt = torch.load(ROOT / dl_cfg["model_file"], map_location="cpu", weights_only=True)
if list(ckpt["classes"]) != classes:
    raise RuntimeError("Urutan kelas checkpoint CNN berbeda dari baseline.")
mu = float(ckpt["normalization"]["mean"])
sigma = float(ckpt["normalization"]["std"])
float_net = FloatTinyKWS(len(classes))
float_net.load_state_dict(ckpt["state_dict"])
float_net.eval()
qnet = QuantTinyKWS(len(classes))
prefixes = {
    "features.0": "conv1",
    "features.1": "bn1",
    "features.4": "conv2",
    "features.5": "bn2",
    "classifier": "fc",
}
mp = {
    name: prefixes[name.rsplit(".", 1)[0]] + "." + name.rsplit(".", 1)[1]
    for name in ckpt["state_dict"]
}
qnet.load_state_dict({mp[k]: v for k, v in ckpt["state_dict"].items()}, strict=True)
qnet.eval()
torch.ao.quantization.fuse_modules(
    qnet, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]], inplace=True
)
engines = [
    e
    for e in ("fbgemm", "x86", "qnnpack")
    if e in torch.backends.quantized.supported_engines
]
if not engines:
    raise RuntimeError("Backend kuantisasi CPU tidak tersedia.")
engine = engines[0]
torch.backends.quantized.engine = engine
qnet.qconfig = torch.ao.quantization.get_default_qconfig(engine)
torch.ao.quantization.prepare(qnet, inplace=True)
# Calibration TRAINING ONLY.
with torch.no_grad():
    for sid in split["train_ids"]:
        x = load_logmel(sid, mu, sigma)
        qnet(torch.from_numpy(x[None, None]).float())
torch.ao.quantization.convert(qnet, inplace=True)
qnet.eval()

# Prediction agreement after quantization. This is equivalence checking, not retuning.
all_ids = (
    list(split["train_ids"]) + list(split["validation_ids"]) + list(split["test_ids"])
)
float_cnn = []
quant_cnn = []
with torch.no_grad():
    for sid in all_ids:
        x = load_logmel(sid, mu, sigma)
        xt = torch.from_numpy(x[None, None]).float()
        float_cnn.append(classes[int(float_net(xt).argmax(1).item())])
        quant_cnn.append(classes[int(qnet(xt).argmax(1).item())])
float_cnn = np.asarray(float_cnn)
quant_cnn = np.asarray(quant_cnn)
ntr = len(split["train_ids"])
nv = len(split["validation_ids"])
test_slice = slice(ntr + nv, len(all_ids))
cnn_agree_all = float(np.mean(float_cnn == quant_cnn))
cnn_agree_test = float(np.mean(float_cnn[test_slice] == quant_cnn[test_slice]))

# TorchScript host artifact. Note: this is not a TFLM flatbuffer.
example = torch.zeros(1, 1, *dl_cfg["input_shape_per_sample"])
scripted = torch.jit.trace(qnet, example)
qcnn_path = model_dir / "mp15_tiny_cnn_int8_torchscript.pt"
torch.jit.save(scripted, str(qcnn_path))

# ---------- Benchmark host reference ----------
ix0 = np.where(train_mask)[0][0]
z0 = Z8[ix0]
bench_svm_float = bench(lambda: svm.predict(z0.reshape(1, -1)), iters=300)
bench_svm_int8 = bench(lambda: svm_int8_scores(z0), iters=300)
x0 = load_logmel(split["train_ids"][0], mu, sigma)
xt0 = torch.from_numpy(x0[None, None]).float()
with torch.no_grad():
    bench_cnn_float = bench(lambda: float_net(xt0), iters=100)
    bench_cnn_int8 = bench(lambda: qnet(xt0), iters=100)

# Payload estimates, not runtime/library overhead.
classical_payload_bytes = int(
    Wq.size + 4 * len(W_scales) + 4 * len(b) + 4 * len(sel) * 3 + 4
)
dlmet = json.loads(
    (ROOT / "results/metrics/mp12_deep_learning_metrics.json").read_text(
        encoding="utf-8"
    )
)
cnn_int8_weight_bytes = int(dlmet["estimated_int8_parameter_bytes"])

# Follow the MP13 no-test-selection rule. Policy was fixed in MP11, before test opened.
mp11 = json.loads(
    (ROOT / "results/metrics/mp11_classifier_metrics.json").read_text(encoding="utf-8")
)
f1_class = float(mp13["classical"]["macro_f1"])
f1_cnn = float(mp13["tiny_cnn"]["macro_f1"])
val_class = float(mp11["chosen_validation_macro_f1"])
val_cnn = float(dlmet["validation_macro_f1"])
PERF_TOL = float(clf_cfg["deployment_performance_tolerance"])
best_val = max(val_class, val_cnn)
choices = []
if val_class >= best_val - PERF_TOL:
    choices.append((CLASSICAL_NAME, classical_payload_bytes, bench_svm_int8["p95_ms"]))
if val_cnn >= best_val - PERF_TOL:
    choices.append(("tiny_cnn_int8", cnn_int8_weight_bytes, bench_cnn_int8["p95_ms"]))
selected = min(choices, key=lambda item: (item[1], item[2]))[0]

config = {
    "source_status": "ready_for_mp15",
    "post_test_tuning_performed": False,
    "cnn_ptq_calibration_scope": "training_only",
    "svm_input_quant_calibration_scope": "training_only",
    "selection_uses_frozen_mp13_metrics": False,
    "selection_scope": "frozen_validation_then_payload_then_host_latency",
    "guide_deviation": "MP15 test-based ranking conflicts with MP13; validation scores are used instead.",
    "selection_policy": f"candidate within {PERF_TOL:.2f} validation macro-F1 of best -> smaller payload -> lower host p95 latency",
    "selected_deployment_model": selected,
    "primary_target_profile": (
        "MCU_first" if selected == CLASSICAL_NAME else "MPU_or_accelerated_MCU"
    ),
    "fallback_candidate": (
        "tiny_cnn_int8" if selected == CLASSICAL_NAME else CLASSICAL_NAME
    ),
    "realtime_contract": "config/realtime_architecture_config.json",
    "classical_package": "models/mp15_classical_int8.npz",
    "classical_c_header": "firmware/mp15_classical_model_int8.h",
    "tiny_cnn_host_quantized_package": "models/mp15_tiny_cnn_int8_torchscript.pt",
    "note": "TorchScript INT8 is used for host numerical/latency validation. MP16 ports the selected final model to the actual embedded runtime.",
}
save_json(ROOT / "config/deployment_config.json", config)

report = {
    "frozen_test_macro_f1": {"classical": f1_class, "tiny_cnn": f1_cnn},
    "svm_int8": {
        "agreement_all_24_vs_float": svm_agree_all,
        "agreement_test_vs_float": svm_agree_test,
        "input_quant_scale": x_scale,
        "weight_scales": W_scales.tolist(),
        "payload_bytes_estimate": classical_payload_bytes,
        "host_reference_latency": bench_svm_int8,
        "float_host_reference_latency": bench_svm_float,
    },
    "tiny_cnn_int8": {
        "calibration_scope": "training_only",
        "quantized_engine": engine,
        "agreement_all_24_vs_float": cnn_agree_all,
        "agreement_test_vs_float": cnn_agree_test,
        "weight_bytes_estimate": cnn_int8_weight_bytes,
        "torchscript_file_bytes": int(qcnn_path.stat().st_size),
        "host_reference_latency": bench_cnn_int8,
        "float_host_reference_latency": bench_cnn_float,
    },
    "selection": {
        "performance_tolerance": PERF_TOL,
        "selected_model": selected,
        "validation_macro_f1": {"classical": val_class, "tiny_cnn": val_cnn},
        "scope": "validation_only_then_payload_then_host_latency",
        "reason": "Validation within tolerance; smallest payload, then lower host p95 latency.",
    },
    "device_measurement_note": "Host latency is not target-device latency. Actual latency, peak RAM, flash and power are measured during MP16 integration.",
    "post_test_tuning_performed": False,
}
report["classical_int8"] = report["svm_int8"]
report["benchmark_scope"] = "host_reference"
report["device_benchmark_performed"] = False
save_json(ROOT / "results/metrics/mp15_deployment_metrics.json", report)

# Manifest for MP16.
manifest = {
    "selected_model": selected,
    "classes": classes,
    "sample_rate_hz": rt["sample_rate_hz"],
    "dma_block_samples": rt["dma_block_samples"],
    "feature_frame_samples": rt["feature_frame_samples"],
    "feature_hop_samples": rt["feature_hop_samples"],
    "inference_stride_frames": rt["inference_stride_frames"],
    "selected_feature_names": (
        red["selected_feature_names"] if selected == CLASSICAL_NAME else None
    ),
    "files_required": [
        "config/project_config.json",
        "config/filter_final_config.json",
        "config/preprocessing_config.json",
        "config/feature_config.json",
        "config/realtime_architecture_config.json",
        "config/deployment_config.json",
        "firmware/mp14_realtime_constants.h",
        "firmware/mp15_classical_model_int8.h",
        "models/mp15_classical_int8.npz",
        "models/mp15_tiny_cnn_int8_torchscript.pt",
        "models/mp12_tiny_cnn_state.pt",
        "config/deep_learning_config.json",
        "config/feature_reduction_config.json",
        "features/intermediate/mp06_filter_final.npz",
        "mini_project_16.py",
        "kws_common.py",
        "kws_models.py",
        "requirements.txt",
    ],
}
manifest["sha256"] = {p: sha256(ROOT / p) for p in manifest["files_required"]}
manifest["benchmark_scope"] = "host_reference"
manifest["device_benchmark_performed"] = False
save_json(ROOT / "deployment/mp15_manifest.json", manifest)

# Figures
figdir = ROOT / "results/figures"
figdir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7.2, 4.5))
names = [
    "SVM float\nserialized",
    "SVM INT8\npayload",
    "CNN FP32\nweights",
    "CNN INT8\nweights",
]
vals = [
    int(mp13["resource_proxy"]["classical"]["serialized_bytes"]),
    classical_payload_bytes,
    int(dlmet["float32_parameter_bytes"]),
    cnn_int8_weight_bytes,
]
plt.bar(names, vals)
plt.ylabel("Byte")
plt.title("MP15 - Ukuran model/payload deployment")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(figdir / "mp15_model_size.png", dpi=170)
plt.close()

plt.figure(figsize=(7.2, 4.5))
names = ["SVM float", "SVM INT8 ref", "CNN float", "CNN INT8"]
vals = [
    bench_svm_float["p95_ms"],
    bench_svm_int8["p95_ms"],
    bench_cnn_float["p95_ms"],
    bench_cnn_int8["p95_ms"],
]
plt.bar(names, vals)
plt.ylabel("p95 latency host (ms)")
plt.title("MP15 - Benchmark referensi host")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(figdir / "mp15_latency.png", dpi=170)
plt.close()

plt.figure(figsize=(7.2, 4.5))
plt.bar(
    ["Classical all", "Classical test", "CNN all", "CNN test"],
    [svm_agree_all, svm_agree_test, cnn_agree_all, cnn_agree_test],
)
plt.ylim(0, 1.05)
plt.ylabel("Prediction agreement")
plt.title("MP15 - Prediction agreement after quantization")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(figdir / "mp15_equivalence.png", dpi=170)
plt.close()

status["status"] = "ready_for_mp16"
status["last_completed_mini_project"] = 15
status["mp15"] = {
    "deployment_config": "config/deployment_config.json",
    "deployment_metrics": "results/metrics/mp15_deployment_metrics.json",
    "deployment_manifest": "deployment/mp15_manifest.json",
    "selected_model": selected,
    "classical_package": "models/mp15_classical_int8.npz",
    "classical_header": "firmware/mp15_classical_model_int8.h",
    "tiny_cnn_quantized_host_artifact": "models/mp15_tiny_cnn_int8_torchscript.pt",
    "post_test_tuning_performed": False,
    "device_benchmark_deferred_to_mp16": True,
}
verify_frozen()
advance(status, 15)
save_json(status_path, status)

print("Mini Project 15 selesai.")
print("SVM agreement all/test:", svm_agree_all, svm_agree_test)
print("CNN agreement all/test:", cnn_agree_all, cnn_agree_test)


print("Selected deployment model:", selected)
print("Status:", status["status"])
