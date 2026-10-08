from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


# 1. PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "requirements.csv"
MODEL_DIR = BASE_DIR / "model"

print(f"Dataset: {DATA_PATH}")
print(f"Model output: {MODEL_DIR}")


# 2. LABELS

label_names = [
    "FUNCTIONAL",
    "NON_FUNCTIONAL",
    "BUSINESS_RULE",
    "CONSTRAINT",
]

label2id = {
    label: i for i, label in enumerate(label_names)
}

id2label = {
    i: label for i, label in enumerate(label_names)
}

print("\nLabels:")
for label, idx in label2id.items():
    print(f"  {idx}: {label}")


# 3. LOAD DATASET

dataset = load_dataset(
    "csv",
    data_files=str(DATA_PATH)
)

dataset = dataset["train"]

print("\nDataset:")
print(dataset)


# 4. CONVERT LABELS

def convert_label(example):
    label = example["label"]

    # Limpieza por si el CSV contiene espacios
    label = str(label).strip().upper()

    if label not in label2id:
        raise ValueError(
            f"Unknown label: '{label}'. "
            f"Expected one of: {list(label2id.keys())}"
        )

    return {
        "labels": label2id[label]
    }


dataset = dataset.map(convert_label)


# 5. TRAIN / TEST SPLIT

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

print("\nTrain examples:", len(train_dataset))
print("Test examples:", len(test_dataset))


# 6. TOKENIZER

MODEL_NAME = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# 7. REMOVE ORIGINAL COLUMNS

columns_to_remove = [
    column
    for column in train_dataset.column_names
    if column not in ["input_ids", "attention_mask", "labels"]
]

train_dataset = train_dataset.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)


# 8. MODEL

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)


# 9. DATA COLLATOR

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# 10. METRICS

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = np.mean(predictions == labels)

    return {
        "accuracy": accuracy
    }


# 11. TRAINING ARGUMENTS

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),

    num_train_epochs=3,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=10,

    load_best_model_at_end=False,

    report_to="none",
)


# 12. TRAINER

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,


    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# 13. TRAIN

print("\nStarting training...\n")

trainer.train()


# 14. EVALUATE

print("\nEvaluating model...\n")

metrics = trainer.evaluate()

print(metrics)


# 15. SAVE MODEL

MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print(f"\nModel saved to:")
print(MODEL_DIR)