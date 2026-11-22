from collections import Counter

import numpy as np
import pandas as pd
from datasets import ClassLabel, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from project_config import (
    BASE_MODEL_NAME,
    CHECKPOINT_DIR,
    LABEL_NAMES,
    MAX_LENGTH,
    MODEL_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    normalize_text,
)


def main():
    # 1. PATHS

    print(f"Dataset: {TRAIN_DATA_PATH}")
    print(f"Model output: {MODEL_DIR}")


    # 2. LABELS

    label2id = {label: i for i, label in enumerate(LABEL_NAMES)}
    id2label = {i: label for i, label in enumerate(LABEL_NAMES)}

    print("\nLabels:")
    for label, idx in label2id.items():
        print(f"  {idx}: {label}")


    # 3. LOAD DATASET

    dataset = load_dataset(
        "csv",
        data_files=str(TRAIN_DATA_PATH),
        encoding="utf-8-sig",
    )["train"]

    print("\nDataset:")
    print(dataset)


    # 4. VALIDATE TEXTS

    texts = dataset["text"]
    empty_rows = [i for i, text in enumerate(texts) if text is None or not str(text).strip()]
    if empty_rows:
        raise ValueError(f"Empty requirement texts at rows: {empty_rows[:10]}")

    normalized = [normalize_text(text) for text in texts]
    duplicates = sorted(t for t, n in Counter(normalized).items() if n > 1)
    if duplicates:
        print(f"\nWARNING: {len(duplicates)} duplicated texts in the training data, e.g. {duplicates[:3]}")

    # Ninguna frase del test puede estar en el entrenamiento, o el test deja de ser fiable.
    if TEST_DATA_PATH.exists():
        test_texts = set(pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")["text"].map(normalize_text))
        leaked = sorted(set(normalized) & test_texts)
        if leaked:
            raise ValueError(
                f"{len(leaked)} training texts also appear in {TEST_DATA_PATH.name}: {leaked[:5]}"
            )


    # 5. CONVERT LABELS

    def convert_label(example):
        # Limpieza por si el CSV contiene espacios
        label = str(example["label"]).strip().upper()

        if label not in label2id:
            raise ValueError(
                f"Unknown label: '{label}'. "
                f"Expected one of: {LABEL_NAMES}"
            )

        return {"labels": label2id[label]}

    dataset = dataset.map(convert_label)
    dataset = dataset.cast_column("labels", ClassLabel(names=LABEL_NAMES))


    # 6. TRAIN / VALIDATION SPLIT
    # La validación sirve para seguir el entrenamiento por época;
    # el test real es requirements_test.csv (evaluate.py).

    dataset = dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="labels",
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    print("\nTrain examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))


    # 7. TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)


    # 8. REMOVE ORIGINAL COLUMNS

    columns_to_remove = [
        column
        for column in train_dataset.column_names
        if column not in ["input_ids", "attention_mask", "labels"]
    ]

    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)


    # 9. MODEL

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=id2label,
        label2id=label2id,
    )


    # 10. DATA COLLATOR

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # 11. METRICS

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        _, _, f1_weighted, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }


    # 12. TRAINING ARGUMENTS
    # Los checkpoints van a CHECKPOINT_DIR (no a model/) y solo se guarda el mejor,
    # que se recarga al final del entrenamiento.

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),

        num_train_epochs=3,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        learning_rate=2e-5,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",

        logging_steps=10,

        report_to="none",
        seed=42,
    )


    # 13. TRAINER

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    # 14. TRAIN

    print("\nStarting training...\n")

    trainer.train()


    # 15. EVALUATE ON VALIDATION SPLIT

    print("\nEvaluating model on the validation split...\n")
    metrics = trainer.evaluate()
    print("\nOverall metrics:")
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")

    print("\nDetailed classification report:")
    pred_output = trainer.predict(val_dataset)
    predictions = np.argmax(pred_output.predictions, axis=-1)
    true_labels = pred_output.label_ids

    print(
        classification_report(
            true_labels,
            predictions,
            labels=list(range(len(LABEL_NAMES))),
            target_names=LABEL_NAMES,
            zero_division=0,
            digits=4,
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            true_labels,
            predictions,
            labels=list(range(len(LABEL_NAMES))),
        )
    )


    # 16. SAVE MODEL

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\nModel saved to:")
    print(MODEL_DIR)


if __name__ == "__main__":
    main()
