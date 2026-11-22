import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from project_config import (
    MAX_LENGTH,
    MODEL_DIR,
    OUTPUT_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    normalize_text,
)

BATCH_SIZE = 16


def main():
    print(f"Test dataset: {TEST_DATA_PATH}")
    print(f"Model: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # Las etiquetas salen del propio modelo, así nunca se desincronizan con train.py.
    id2label = {int(index): label for index, label in model.config.id2label.items()}
    label_names = [id2label[index] for index in range(len(id2label))]
    label2id = {label: index for index, label in enumerate(label_names)}


    # 1. LOAD AND VALIDATE TEST DATA

    df = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")

    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.upper()

    if df["text"].eq("").any():
        raise ValueError("The test dataset contains empty requirements.")

    invalid_labels = sorted(set(df["label"]) - set(label_names))
    if invalid_labels:
        raise ValueError(f"Invalid labels: {invalid_labels}. Model labels: {label_names}")

    normalized = df["text"].map(normalize_text)
    if normalized.duplicated().any():
        duplicates = df.loc[normalized.duplicated(keep=False), "text"].tolist()
        raise ValueError(f"Duplicate test texts found: {duplicates[:5]}")

    if TRAIN_DATA_PATH.exists():
        train_texts = set(pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8-sig")["text"].map(normalize_text))
        leaked = df.loc[normalized.isin(train_texts), "text"].tolist()
        if leaked:
            raise ValueError(f"{len(leaked)} test texts also appear in the training data: {leaked[:5]}")

    counts = df["label"].value_counts().reindex(label_names, fill_value=0)
    print("\nExamples per category:")
    print(counts)

    if counts.min() == 0:
        print("\nWARNING: some categories have no test examples; their metrics will be 0.")
    elif counts.max() > 2 * counts.min():
        print("\nWARNING: the test set is unbalanced; look at the macro averages, not only accuracy.")


    # 2. PREDICT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"\nRunning evaluation on: {device}")

    predicted_ids = []
    probability_rows = []

    for start in range(0, len(df), BATCH_SIZE):
        batch_texts = df["text"].iloc[start:start + BATCH_SIZE].tolist()
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)

        predicted_ids.extend(probabilities.argmax(dim=-1).cpu().tolist())
        probability_rows.extend(probabilities.cpu().tolist())


    # 3. METRICS

    true_ids = df["label"].map(label2id).to_numpy()
    predicted_ids = np.asarray(predicted_ids)
    label_indices = list(range(len(label_names)))

    print("\nAccuracy:", f"{accuracy_score(true_ids, predicted_ids):.4f}")
    print("\nDetailed classification report:")
    print(
        classification_report(
            true_ids,
            predicted_ids,
            labels=label_indices,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(true_ids, predicted_ids, labels=label_indices))


    # 4. SAVE PREDICTIONS

    probabilities = np.asarray(probability_rows)
    df["predicted_label"] = [label_names[index] for index in predicted_ids]
    df["correct"] = df["label"] == df["predicted_label"]
    for index, label in enumerate(label_names):
        df[f"prob_{label}"] = probabilities[:, index]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "test_predictions.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nPredictions saved to: {output_path}")
    print(f"Incorrect predictions: {(~df['correct']).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
