"""Exporta el modelo entrenado (model/) a ONNX cuantizado para la web.

Genera web/ai/requirements-classifier/ con la estructura que espera Transformers.js
y comprueba la precisión del modelo ONNX con requirements_test.csv.
Ejecutar después de cada reentrenamiento.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from project_config import BASE_DIR, MAX_LENGTH, MODEL_DIR, TEST_DATA_PATH

WEB_MODEL_DIR = BASE_DIR / "web" / "ai" / "requirements-classifier"
ONNX_PATH = WEB_MODEL_DIR / "onnx" / "model_quantized.onnx"
FILES_TO_COPY = ["config.json", "tokenizer.json", "tokenizer_config.json"]
BATCH_SIZE = 32


class LogitsOnly(torch.nn.Module):
    """Envuelve el modelo para que la salida ONNX sea solo el tensor de logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model, tokenizer, fp32_path):
    sample = tokenizer(["The system shall export reports."], return_tensors="pt")
    torch.onnx.export(
        LogitsOnly(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def onnx_accuracy(onnx_path, tokenizer, label2id):
    df = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    predictions = []
    for start in range(0, len(df), BATCH_SIZE):
        batch = tokenizer(
            df["text"].iloc[start:start + BATCH_SIZE].tolist(),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        logits = session.run(["logits"], {
            "input_ids": batch["input_ids"].astype(np.int64),
            "attention_mask": batch["attention_mask"].astype(np.int64),
        })[0]
        predictions.extend(logits.argmax(axis=-1).tolist())
    true_ids = df["label"].str.strip().str.upper().map(label2id).to_numpy()
    return float((np.asarray(predictions) == true_ids).mean())


def main():
    print(f"Model: {MODEL_DIR}")
    print(f"Web model output: {WEB_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    # Con la atención SDPA por defecto, el trazado ONNX no respeta la máscara de atención
    # y los logits difieren de PyTorch; con "eager" coinciden.
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), attn_implementation="eager"
    )
    model.eval()
    label2id = {label: int(index) for label, index in model.config.label2id.items()}

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "model.onnx"
        print("\nExporting to ONNX...")
        export(model, tokenizer, fp32_path)
        print("Quantizing to int8...")
        quantize_dynamic(str(fp32_path), str(ONNX_PATH), weight_type=QuantType.QInt8)

        print("\nChecking accuracy on the test set...")
        fp32_acc = onnx_accuracy(fp32_path, tokenizer, label2id)
        q8_acc = onnx_accuracy(ONNX_PATH, tokenizer, label2id)

    for name in FILES_TO_COPY:
        shutil.copy2(MODEL_DIR / name, WEB_MODEL_DIR / name)

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"\nONNX fp32 accuracy:      {fp32_acc:.4f}")
    print(f"ONNX quantized accuracy: {q8_acc:.4f}")
    print(f"Quantized model size:    {size_mb:.1f} MB")
    if q8_acc < fp32_acc - 0.01:
        print("WARNING: quantization lost more than 1 point of accuracy.")
    if size_mb > 95:
        print("WARNING: the model is close to GitHub's 100 MB per-file limit.")
    print(f"\nWeb model ready in: {WEB_MODEL_DIR}")


if __name__ == "__main__":
    main()
