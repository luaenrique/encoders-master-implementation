from transformers import ElectraForSequenceClassification
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import ElectraTokenizer

from transformers import LongformerForSequenceClassification
from transformers import LongformerTokenizer

from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
import numpy as np
import evaluate
import csv
import time  # Added for wall time tracking
from datetime import datetime  # Added for timestamps

from transformers import TrainingArguments, Trainer
batch_size = 8
metric_name = "accuracy"

class GenericEncoderModel:
    def __init__(self, model_name, training_file_name, model_type, problem_type, num_labels):
        self.data = []
        self.model_name = model_name
        self.training_file_name = training_file_name
        self.model_type = model_type
        self.problem_type = problem_type
        self.tokenizer = self._load_tokenizer()
        self.trainer = None
        self.num_labels = num_labels
        self.model = self._load_model()
        
        # Added timing tracking
        self.timing_log = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def _load_tokenizer(self):
        if self.model_type == 'electra':
            tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
        elif self.model_type == 'longformer':
            tokenizer = LongformerTokenizer.from_pretrained(self.model_name)
        elif self.model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
        elif self.model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return tokenizer
    
    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                           problem_type=self.problem_type,  num_labels=self.num_labels)
        return model

    def compute_metrics(self, eval_preds, threshold = 0.5):
        logits, labels = eval_preds
        if self.problem_type == "single_label_classification" :
            ptype = None
            predictions = np.argmax(logits, axis=-1).reshape(-1,1)
            labels_ = labels
            metrics = ["accuracy", "micro-f1", "macro-f1"]
        elif self.problem_type ==  "multi_label_classification":
            ptype = "multilabel"
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(logits))
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs > threshold)] = 1
            predictions = predictions.astype('int32')
            labels_ = labels.astype('int32')
            metrics = ["micro-f1", "macro-f1"]
        else:
            raise ValueError("Wrong problem type")
            
        outputs = dict()
        if "accuracy" in metrics:
            metric = evaluate.load("accuracy")
            accuracy = metric.compute(predictions=predictions, references=labels_)
            outputs["accuracy"] = accuracy["accuracy"]
        if "micro-f1" in metrics:
            metric = evaluate.load("f1", ptype)
            f1_micro = metric.compute(predictions=predictions, references=labels_, average = 'micro')
            outputs["micro-f1"] = f1_micro["f1"]
        if "macro-f1" in metrics:
            metric = evaluate.load("f1",  ptype)
            f1_macro = metric.compute(predictions=predictions, references=labels_, average = 'macro')
            outputs["macro-f1"] = f1_macro["f1"]
        return outputs
    
    def train(self, train_dataset, test_dataset, dataset_name):
        print(f"Starting training for {self.model_name} on {dataset_name}")
        train_start_time = time.time()
        
        self.model.resize_token_embeddings(len(self._load_tokenizer()))

        args = TrainingArguments(
            f"{self.training_file_name}_{dataset_name}_2",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            seed=42,
            data_seed=42,
        )
        
        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        self.trainer = trainer
        
        train_end_time = time.time()
        train_wall_time = train_end_time - train_start_time
        self.timing_log['training_time'] = train_wall_time
        
        print(f"Training completed in {train_wall_time:.2f} seconds ({train_wall_time/60:.2f} minutes)")

    def store_logits(self, dataset, dataset_name):
        print(f"Storing logits for {dataset_name}")
        start_time = time.time()
        
        self.model.eval()
        all_logits = []
        all_labels = []
        all_texts = []

        dataloader = self.trainer.get_test_dataloader(dataset)
        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits.cpu().numpy()
                all_logits.append(logits)
                all_labels.append(batch["labels"].cpu().numpy())
                all_texts.append(batch["input_ids"].cpu().numpy())

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)

        # Improved filename to be more descriptive
        model_name_clean = self.model_name.replace('/', '_')
        output_file = f"logits_{model_name_clean}_{dataset_name}.npz"
        np.savez(output_file, logits=logits, labels=labels)
        
        end_time = time.time()
        wall_time = end_time - start_time
        self.timing_log[f'logits_storage_{dataset_name}'] = wall_time
        print(f"Logits saved to {output_file}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits storage completed in {wall_time:.2f} seconds")

    def store_predictions(self, dataset, predictions, output_csv_path):
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['prediction', 'label', 'text']) 
            for text, label, prediction in zip(dataset['text'], dataset['label'], predictions):
                writer.writerow([prediction, label, text])

    def evaluate(self, test_dataset, dataset_name):
        print(f"Starting evaluation for {self.model_name} on {dataset_name}")
        eval_start_time = time.time()
        
        metrics = self.trainer.evaluate()
        output_csv_path = f"metrics_{self.model_name}_{dataset_name}_2.csv"
        
        predictions = []
        for batch in self.trainer.get_test_dataloader(test_dataset):
            outputs = self.model(**batch)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1)
            predictions.extend(predicted_class.cpu().numpy())

        self.store_predictions(self.trainer.eval_dataset, predictions, output_csv_path=f"predictions_{self.model_name}_{dataset_name}_2.csv")
        
        eval_end_time = time.time()
        eval_wall_time = eval_end_time - eval_start_time
        self.timing_log['evaluation_time'] = eval_wall_time
        
        # Write metrics and timing to CSV file
        with open(output_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            file_is_empty = file.tell() == 0
            if file_is_empty:
                writer.writerow(['dataset', 'accuracy', 'micro-f1', 'macro-f1', 'training_time_seconds', 'evaluation_time_seconds'])
            
            writer.writerow([
                self.training_file_name, 
                metrics.get('eval_accuracy', 'N/A'),
                metrics.get('eval_micro-f1', 'N/A'), 
                metrics.get('eval_macro-f1', 'N/A'),
                self.timing_log.get('training_time', 'N/A'),
                eval_wall_time
            ])
        
        print(f"Evaluation completed in {eval_wall_time:.2f} seconds")
        return metrics
    
    def store_embeddings_only(self, dataset, dataset_name):
        print(f"Storing embeddings for {dataset_name}")
        start_time = time.time()
        
        self.model.eval()
        all_embeddings = []
        all_labels = []

        dataloader = self.trainer.get_test_dataloader(dataset)
        
        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True)
                
                last_hidden_states = outputs.hidden_states[-1]
                
                if self.model_type in ['bert', 'electra', 'roberta', 'longformer']:
                    embeddings = last_hidden_states[:, 0, :].cpu().numpy()
                else:
                    attention_mask = batch['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
                    sum_embeddings = torch.sum(last_hidden_states * attention_mask, 1)
                    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                all_embeddings.append(embeddings)
                all_labels.append(batch["labels"].cpu().numpy())

        embeddings = np.concatenate(all_embeddings)
        labels = np.concatenate(all_labels)

        output_file = f"embeddings_{self.model_name.replace('/', '_')}_{dataset_name}.npz"
        np.savez_compressed(output_file, 
                        embeddings=embeddings,
                        labels=labels)
        
        end_time = time.time()
        wall_time = end_time - start_time
        self.timing_log[f'embeddings_storage_{dataset_name}'] = wall_time
        
        print(f"Saved embeddings to {output_file}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings storage completed in {wall_time:.2f} seconds")
    
    def print_timing_summary(self):
        """Print a summary of all timing information"""
        print(f"\n=== Timing Summary for {self.model_name} ===")
        total_time = 0
        for operation, time_taken in self.timing_log.items():
            print(f"{operation}: {time_taken:.2f}s ({time_taken/60:.2f}m)")
            total_time += time_taken
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print("=" * 50)

from datasets import load_dataset

# Load datasets
amazon_dataset = load_dataset("fancyzhx/amazon_polarity")
imdb_dataset = load_dataset("stanfordnlp/imdb")
ag_news_dataset = load_dataset("fancyzhx/ag_news")
yelp_dataset = load_dataset("Yelp/yelp_review_full")

datasets = [ag_news_dataset]
datasetsNames = ['agnews']
numLabels = [4]

def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(examples[contentKey], truncation=True, padding="max_length", max_length=128)

datasetStructure = {
    0: {
        'contentKey': 'text',
        'labelKey': 'label'
    }
}

# Track overall execution time
overall_start_time = time.time()
print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create a timing log for all experiments
experiment_timing = {}

for countDataset in range(0, len(datasets)):
    dataset_start_time = time.time()
    
    models = [
        GenericEncoderModel(
            model_name='google/electra-base-discriminator', 
            training_file_name='electra_training', 
            model_type='electra', 
            problem_type='single_label_classification',
            num_labels=numLabels[countDataset],
        ),
        GenericEncoderModel(
            model_name='roberta-base', 
            training_file_name='roberta_training', 
            model_type='roberta', 
            problem_type='single_label_classification',
            num_labels=numLabels[countDataset],
        ),
        GenericEncoderModel(
            model_name='google-bert/bert-base-uncased', 
            training_file_name='bert_training', 
            model_type='bert', 
            problem_type='single_label_classification',
            num_labels=numLabels[countDataset],
        )
    ]
    
    for bertModel in models:
        model_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Processing {bertModel.model_name} on {datasetsNames[countDataset]}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        dataset = datasets[countDataset]
        structure = datasetStructure.get(countDataset, None)

        # Data preprocessing timing
        preprocess_start = time.time()
        
        contentList = dataset['train'][structure['contentKey']]
        labelList = dataset['train'][structure['labelKey']]
        contentTestList = dataset['test'][structure['contentKey']]
        labelTestList = dataset['test'][structure['labelKey']]

        print("Training labels:", set(labelList))
        print("Test labels:", set(labelTestList))

        train_dataset = dataset['train'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), batched=True)
        test_dataset = dataset['test'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), batched=True)
        train_dataset = train_dataset.remove_columns([structure['contentKey']])
        #test_dataset = test_dataset.remove_columns([structure['contentKey']])
        
        example = train_dataset[0]
        print(example.keys())
        print(bertModel.tokenizer.decode(example['input_ids']))

        train_dataset.set_format("torch")
        test_dataset.set_format("torch")
        
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        print(f"Data preprocessing completed in {preprocess_time:.2f} seconds")

        # Training
        bertModel.train(train_dataset=train_dataset, test_dataset=test_dataset, dataset_name=datasetsNames[countDataset])

        # Evaluation
        print(bertModel.evaluate(test_dataset, dataset_name=datasetsNames[countDataset]))
        
        print(f"\n{'='*40}")
        print("EXTRACTING LOGITS")
        print(f"{'='*40}")
        
        bertModel.store_logits(test_dataset, f"test_{datasetsNames[countDataset]}")
        bertModel.store_logits(train_dataset, f"train_{datasetsNames[countDataset]}")
        
        print("Logits extraction completed!")
        print(f"{'='*40}")

        bertModel.store_embeddings_only(test_dataset, f"agnews_test_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(train_dataset, f"agnews_train_{bertModel.model_name.split('/')[-1]}")
        
        bertModel.print_timing_summary()
        
        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        experiment_timing[f"{bertModel.model_name}_{datasetsNames[countDataset]}"] = model_total_time
        
        detailed_timing_file = f"detailed_timing_{bertModel.model_name.replace('/', '_')}_{datasetsNames[countDataset]}.csv"
        with open(detailed_timing_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['operation', 'time_seconds', 'time_minutes'])
            for operation, time_taken in bertModel.timing_log.items():
                writer.writerow([operation, time_taken, time_taken/60])
            writer.writerow(['model_total_time', model_total_time, model_total_time/60])
        
        print(f"Total time for {bertModel.model_name}: {model_total_time:.2f}s ({model_total_time/60:.2f}m)")
        print(f"Detailed timing saved to {detailed_timing_file}")
    
    dataset_end_time = time.time()
    dataset_total_time = dataset_end_time - dataset_start_time
    experiment_timing[f"dataset_{datasetsNames[countDataset]}_total"] = dataset_total_time

# Overall timing summary
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time

print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETED")
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {overall_time:.2f}s ({overall_time/60:.2f}m) ({overall_time/3600:.2f}h)")
print(f"{'='*80}")

# Save timing results to CSV
with open('experiment_timing_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['experiment', 'time_seconds', 'time_minutes', 'time_hours'])
    
    for experiment, time_taken in experiment_timing.items():
        writer.writerow([experiment, time_taken, time_taken/60, time_taken/3600])
    
    writer.writerow(['total_experiment', overall_time, overall_time/60, overall_time/3600])

print("Timing results saved to 'experiment_timing_results.csv'")

# Print summary of files generated
print(f"\n{'='*60}")
print("FILES GENERATED SUMMARY")
print(f"{'='*60}")
model_names = ['google_electra-base-discriminator', 'roberta-base', 'google-bert_bert-base-uncased']
for model_name in model_names:
    print(f"\n{model_name}:")
    print(f"  - logits_{model_name}_test_agnews.npz")
    print(f"  - logits_{model_name}_train_agnews.npz") 
    print(f"  - embeddings_{model_name}_agnews_test_{model_name.split('_')[-1]}.npz")
    print(f"  - embeddings_{model_name}_agnews_train_{model_name.split('_')[-1]}.npz")
    print(f"  - detailed_timing_{model_name}_agnews.csv")

print(f"\nGeneral files:")
print(f"  - experiment_timing_results.csv")
print(f"{'='*60}")