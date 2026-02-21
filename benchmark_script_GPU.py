import os
import cv2
import numpy as np
from pathlib import Path
from time import perf_counter
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from EEFRecognizer_custom import EmotiEffLibRecognizerOnnxCustom


# ==============================
# CONFIG
# ==============================

MODEL_PATHS = {
    "fp32_afew": "models/enet_b0_8_best_afew.onnx",
    "fp32_mtl": "models/enet_b0_8_va_mtl.onnx",
    "fp16_afew": "models/fp16_enet_b0_8_best_afew.onnx",
    "fp16_mtl": "models/fp16_enet_b0_8_va_mtl.onnx"
}

POSITIVE_DIR = "../sample_positive"
NEGATIVE_DIR = "../sample_negative"
WARMUP_RUNS = 20


# ==============================
# UTILS
# ==============================

def load_images(folder):
    images = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"}:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


def is_happy(label: str):
    return label.lower() == "happiness"


# ==============================
# BENCHMARK
# ==============================

def benchmark_model(model_name, model_path):

    print(f"\n===== GPU BENCHMARK: {model_name} =====")

    recognizer = EmotiEffLibRecognizerOnnxCustom(
        model_name="enet_b0_8_best_afew",
        model_path=model_path
    )

    # Переключаем execution provider
    recognizer.ort_session.set_providers(
        ["CUDAExecutionProvider"])

    # Проверка
    print("Providers:", recognizer.ort_session.get_providers())

    pos = load_images(POSITIVE_DIR)
    neg = load_images(NEGATIVE_DIR)

    images = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)

    # --------------------------
    # Warm-up
    # --------------------------
    for _ in range(WARMUP_RUNS):
        recognizer.predict_emotions(images[0])

    predicted = []
    inference_times = []

    # --------------------------
    # Основной замер
    # --------------------------
    for img in images:

        input_tensor = recognizer._preprocess(img)

        start = perf_counter()
        logits = recognizer.ort_session.run(
            [recognizer.output_name],
            {recognizer.input_name: input_tensor}
        )[0]

        # синхронизация CUDA
        ort.cuda.synchronize() if hasattr(ort, "cuda") else None

        end = perf_counter()

        inference_times.append(end - start)

        emotion, _ = recognizer.classify_emotions(logits)
        predicted.append(1 if is_happy(emotion[0]) else 0)

    times = np.array(inference_times)

    acc = accuracy_score(labels, predicted)
    prec = precision_score(labels, predicted, zero_division=0)
    rec = recall_score(labels, predicted, zero_division=0)
    f1 = f1_score(labels, predicted, zero_division=0)

    mean_latency = times.mean()
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    fps = 1.0 / mean_latency

    start = perf_counter()
    for img in images:
        recognizer.predict_emotions(img)
    total_time = perf_counter() - start
    throughput = len(images) / total_time

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1        : {f1:.4f}")
    print("----- SPEED -----")
    print(f"Mean Latency : {mean_latency*1000:.2f} ms")
    print(f"P50 Latency  : {p50*1000:.2f} ms")
    print(f"P95 Latency  : {p95*1000:.2f} ms")
    print(f"FPS (latency): {fps:.2f}")
    print(f"Throughput   : {throughput:.2f} img/sec")


if __name__ == "__main__":
    for name, path in MODEL_PATHS.items():
        benchmark_model(name, path)
