from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import load_dataset
import numpy as np
import evaluate
import csv

# Config
batch_size = 8
metric_name = "accuracy"
num_labels = 2
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
imdb_dataset = load_dataset("stanfordnlp/imdb")
# Split treino em treino/val
train_test_split = imdb_dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']
test_dataset = imdb_dataset['test']

# Modelo e tokenizer
models_config = [
    {"name": "google/electra-base-discriminator", "type": "electra"},
    {"name": "roberta-base", "type": "roberta"},
    {"name": "google-bert/bert-base-uncased", "type": "bert"}
]

# Funções utilitárias
def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=max_length)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metric_acc = evaluate.load("accuracy").compute(predictions=predictions, references=labels)["accuracy"]
    metric_f1 = evaluate.load("f1").compute(predictions=predictions, references=labels, average="micro")["f1"]
    return {"accuracy": metric_acc, "micro-f1": metric_f1}

def store_logits_embeddings(model, dataset, tokenizer, prefix):
    model.eval()
    all_logits, all_embeddings, all_labels = [], [], []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            logits = outputs.logits.cpu().numpy()
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # CLS token
            labels = batch["labels"].cpu().numpy()

        all_logits.append(logits)
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    np.savez_compressed(f"{prefix}_logits.npz", logits=np.concatenate(all_logits), labels=np.concatenate(all_labels))
    np.savez_compressed(f"{prefix}_embeddings.npz", embeddings=np.concatenate(all_embeddings), labels=np.concatenate(all_labels))
    print(f"Saved {prefix} logits and embeddings")

# Rodar modelos
for cfg in models_config:
    print(f"\n=== Treinando {cfg['name']} ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg['name'])
    model = AutoModelForSequenceClassification.from_pretrained(cfg['name'], num_labels=num_labels).to(device)

    # Preprocessar datasets
    train_ds = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    val_ds = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    test_ds = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    train_ds = train_ds.remove_columns(['text'])
    val_ds = val_ds.remove_columns(['text'])
    test_ds = test_ds.remove_columns(['text'])

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    args = TrainingArguments(
        output_dir=f"{cfg['name'].replace('/', '_')}_imdb_22",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate(test_ds)
    print(f"Test metrics for {cfg['name']}: {metrics}")

    # Salvar logits e embeddings
    store_logits_embeddings(model, test_ds, tokenizer, f"{cfg['name'].replace('/', '_')}_imdb_test_22")
    store_logits_embeddings(model, train_ds, tokenizer, f"{cfg['name'].replace('/', '_')}_imdb_train_2")
