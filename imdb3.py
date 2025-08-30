import torch
import numpy as np
import time
import csv
from datetime import datetime
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer, LongformerTokenizer
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

# ---------------- Configurações iniciais ----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 8

# ---------------- Função de preprocessamento ----------------
def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(examples[contentKey], truncation=True, padding="max_length", max_length=512)

# ---------------- Classe GenericEncoderModel ----------------
class GenericEncoderModel:
    def __init__(self, model_name, training_file_name, model_type, problem_type, num_labels):
        self.model_name = model_name
        self.training_file_name = training_file_name
        self.model_type = model_type
        self.problem_type = problem_type
        self.num_labels = num_labels
        self.tokenizer = self._load_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            problem_type=problem_type,
            num_labels=num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.trainer = None
        self.timing_log = {}

    def _load_tokenizer(self):
        if self.model_type == 'electra': return ElectraTokenizer.from_pretrained(self.model_name)
        if self.model_type == 'bert': return BertTokenizer.from_pretrained(self.model_name)
        if self.model_type == 'roberta': return RobertaTokenizer.from_pretrained(self.model_name)
        if self.model_type == 'longformer': return LongformerTokenizer.from_pretrained(self.model_name)
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, train_dataset, eval_dataset, dataset_name, batch_size=8, epochs=5):
        start_time = time.time()
        self.model.resize_token_embeddings(len(self.tokenizer))
        args = TrainingArguments(
            output_dir=f"{self.training_file_name}_{dataset_name}_2",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            seed=RANDOM_SEED,
            logging_steps=50,
            save_total_limit=1
        )
        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "micro-f1": f1_score(labels, preds, average="micro"),
                "macro-f1": f1_score(labels, preds, average="macro")
            }
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        self.trainer.train()
        self.timing_log['training_time'] = time.time() - start_time
        print(f"Training completed in {self.timing_log['training_time']:.2f}s")

    def evaluate_clean(self, dataset, batch_size=8):
        self.model.eval()
        all_preds, all_labels, all_logits = [], [], []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_logits.append(logits.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())
        all_logits = np.concatenate(all_logits)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = (all_preds == all_labels).mean()
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        print(f"Evaluation: Accuracy={accuracy:.4f}, Micro-F1={f1_micro:.4f}, Macro-F1={f1_macro:.4f}")
        return {"accuracy": accuracy, "micro-f1": f1_micro, "macro-f1": f1_macro,
                "logits": all_logits, "predictions": all_preds, "labels": all_labels}

    def store_logits_safe(self, logits, labels, dataset_name):
        if logits.ndim == 2:
            probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        else:
            probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
        clean_name = self.model_name.replace('/', '_')
        output_file = f"logits_{clean_name}_{dataset_name}.npz"
        np.savez_compressed(output_file, logits=logits, labels=labels, probabilities=probabilities)
        print(f"Saved logits to {output_file}. Shape={logits.shape}")

    def print_timing_summary(self):
        print(f"\n=== Timing Summary for {self.model_name} ===")
        total_time = sum(self.timing_log.values())
        for op, t in self.timing_log.items():
            print(f"{op}: {t:.2f}s ({t/60:.2f}m)")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print("="*50)

# ---------------- Carregamento dos datasets ----------------
amazon_dataset = load_dataset("fancyzhx/amazon_polarity")
imdb_dataset = load_dataset("stanfordnlp/imdb")
ag_news_dataset = load_dataset("fancyzhx/ag_news")
yelp_dataset = load_dataset("Yelp/yelp_review_full")
snli_dataset = load_dataset("stanfordnlp/snli")

datasets = [imdb_dataset]
datasetsNames = ['imdb']
numLabels = [2]

datasetStructure = {0: {'contentKey': 'text', 'labelKey': 'label'}}

# ---------------- Loop de experimentos ----------------
overall_start_time = time.time()
experiment_timing = {}

for dataset_idx, dataset in enumerate(datasets):
    dataset_name = datasetsNames[dataset_idx]
    structure = datasetStructure[dataset_idx]

    models = [
        GenericEncoderModel("google/electra-base-discriminator", "electra_training", "electra", "single_label_classification", numLabels[dataset_idx]),
        GenericEncoderModel("roberta-base", "roberta_training", "roberta", "single_label_classification", numLabels[dataset_idx]),
        GenericEncoderModel("google-bert/bert-base-uncased", "bert_training", "bert", "single_label_classification", numLabels[dataset_idx])
    ]

    for m in models:
        # Preprocess
        train_ds = dataset['train'].map(lambda x: preprocess_function(x, m.tokenizer, structure['contentKey']), batched=True)
        test_ds = dataset['test'].map(lambda x: preprocess_function(x, m.tokenizer, structure['contentKey']), batched=True)
        train_ds = train_ds.remove_columns([structure['contentKey']])
        test_ds = test_ds.remove_columns([structure['contentKey']])
        train_ds.set_format("torch")
        test_ds.set_format("torch")

        # Train
        m.train(train_ds, test_ds, dataset_name)

        # Evaluate
        result_test = m.evaluate_clean(test_ds)
        m.store_logits_safe(result_test['logits'], result_test['labels'], f"test_{dataset_name}")
        result_train = m.evaluate_clean(train_ds)
        m.store_logits_safe(result_train['logits'], result_train['labels'], f"train_{dataset_name}")

        # Save timing detalhado
        m.print_timing_summary()
        experiment_timing[f"{m.model_name}_{dataset_name}"] = sum(m.timing_log.values())

# ---------------- Resumo final ----------------
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"\nExperiment completed in {overall_time:.2f}s ({overall_time/60:.2f}m)")

with open('experiment_timing_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['experiment', 'time_seconds', 'time_minutes', 'time_hours'])
    for exp, t in experiment_timing.items():
        writer.writerow([exp, t, t/60, t/3600])
    writer.writerow(['total_experiment', overall_time, overall_time/60, overall_time/3600])

print("Timing results saved to 'experiment_timing_results.csv'")
