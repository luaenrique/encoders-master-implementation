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
import time
from datetime import datetime

from transformers import TrainingArguments, Trainer
from datasets import load_dataset

batch_size = 8
metric_name = "accuracy"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
        train_start_time = time.time()
        
        self.model.resize_token_embeddings(len(self._load_tokenizer()))

        args = TrainingArguments(
            f"{self.training_file_name}_{dataset_name}_2",
            eval_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            seed=42
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
        print(f"Training completed in {train_wall_time:.2f} seconds ({train_wall_time/60:.2f} minutes)")

    def store_logits(self, dataset, dataset_name):
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

        np.savez(f"logits_{self.model_name}_{dataset_name}.npz", logits=logits, labels=labels)

    def store_predictions(self, dataset, predictions, output_csv_path):
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['prediction', 'label', 'headline', 'short_description']) 
            for headline, description, label, prediction in zip(dataset['headline'], dataset['short_description'], dataset['label'], predictions):
                writer.writerow([prediction, label, headline, description])

    def evaluate(self, test_dataset, dataset_name):
        metrics = self.trainer.evaluate()
        output_csv_path=f"metrics_{self.model_name}_{dataset_name}_2.csv"

        predictions = []
        for batch in self.trainer.get_test_dataloader(test_dataset):
            outputs = self.model(**batch)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1)
            predictions.extend(predicted_class.cpu().numpy())

        self.store_predictions(self.trainer.eval_dataset, predictions, output_csv_path=f"predictions_{self.model_name}_{dataset_name}_2.csv")

        with open(output_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            file_is_empty = file.tell() == 0
            if file_is_empty:
                writer.writerow(['dataset', 'accuracy', 'micro-f1', 'macro-f1'])

            writer.writerow([self.training_file_name, metrics.get('eval_accuracy', 'N/A'),
                             metrics.get('eval_micro-f1', 'N/A'), metrics.get('eval_macro-f1', 'N/A')])

        return metrics

    def store_embeddings_only(self, dataset, dataset_name):
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
        
        print(f"Saved embeddings to {output_file}")
        print(f"Embeddings shape: {embeddings.shape}")


# ========== CARREGAR E PREPARAR DATASET ==========

huffpost_dataset = load_dataset("heegyu/news-category-dataset")

print("Dataset HuffPost carregado:")
print(huffpost_dataset)

full_dataset = huffpost_dataset['train']

# Verificar estrutura do dataset
print(f"\nColunas disponíveis: {full_dataset.column_names}")

# Criar mapeamento de categorias para números
print("\nCriando mapeamento de categorias...")
unique_categories = sorted(list(set(full_dataset['category'])))
category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
id_to_category = {idx: cat for cat, idx in category_to_id.items()}

print(f"\n{'='*60}")
print(f"Mapeamento de categorias:")
print(f"{'='*60}")
for cat, idx in sorted(category_to_id.items(), key=lambda x: x[1]):
    print(f"{idx:2d}: {cat}")
print(f"{'='*60}")

# Salvar mapeamento para referência futura
import json
with open('category_mapping.json', 'w') as f:
    json.dump({'category_to_id': category_to_id, 'id_to_category': id_to_category}, f, indent=2)
print("Mapeamento salvo em 'category_mapping.json'")

# Adicionar coluna 'label' com IDs numéricos
def map_category_to_id(example):
    example['label'] = category_to_id[example['category']]
    return example

print("\nMapeando categorias para IDs numéricos...")
full_dataset = full_dataset.map(map_category_to_id)

# CRITICAL: Converter a coluna label para ClassLabel para permitir estratificação
from datasets import ClassLabel
full_dataset = full_dataset.cast_column('label', ClassLabel(num_classes=len(unique_categories)))

print(f"Tipo da coluna label: {full_dataset.features['label']}")
print(f"Primeiras labels: {full_dataset['label'][:5]}")

# Agora podemos dividir com estratificação
print("\nDividindo dataset...")
train_test_split = full_dataset.train_test_split(
    test_size=0.3, 
    seed=RANDOM_SEED,
    stratify_by_column='label'
)
train_dataset_raw = train_test_split['train']
remaining_dataset = train_test_split['test']

test_val_split = remaining_dataset.train_test_split(
    test_size=0.333, 
    seed=RANDOM_SEED,
    stratify_by_column='label'
)
test_dataset_raw = test_val_split['train']  
validation_dataset_raw = test_val_split['test'] 

huffpost_split = {
    'train': train_dataset_raw,
    'validation': validation_dataset_raw,
    'test': test_dataset_raw
}

print(f"\nDivisão dos dados:")
print(f"Treino: {len(train_dataset_raw)} exemplos ({len(train_dataset_raw)/len(full_dataset)*100:.1f}%)")
print(f"Validação: {len(validation_dataset_raw)} exemplos ({len(validation_dataset_raw)/len(full_dataset)*100:.1f}%)")
print(f"Teste: {len(test_dataset_raw)} exemplos ({len(test_dataset_raw)/len(full_dataset)*100:.1f}%)")

print(f"\nPrimeiros exemplos do dataset:")
print(f"Headline: {huffpost_split['train']['headline'][:2]}")
print(f"Category: {huffpost_split['train']['category'][:2]}")
print(f"Label (numeric): {huffpost_split['train']['label'][:2]}")

num_categories = len(unique_categories)
print(f"\nNúmero de categorias: {num_categories}")

datasets = [huffpost_split]
datasetsNames = ['huffpost']
numLabels = [num_categories] 

def preprocess_function(examples, tokenizer, headline_key='headline', description_key='short_description'):
    inputs = tokenizer(
        examples[headline_key], 
        examples[description_key], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    return inputs

datasetStructure = {
    0: {
        'contentKey': ['headline', 'short_description'],
        'labelKey': 'label'
    }
}

# ========== TREINAMENTO ==========

overall_start_time = time.time()
print(f"\nStarting HuffPost experiment...")

for countDataset in range(0, len(datasets)):
    
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
        print(f"\n{'='*50}")
        print(f"Treinando modelo: {bertModel.model_name}")
        print(f"Dataset: {datasetsNames[countDataset]}")
        print(f"{'='*50}")
        
        dataset = datasets[countDataset]
        structure = datasetStructure.get(countDataset, None)

        # Preparando os datasets
        train_dataset = dataset['train'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey'][0], structure['contentKey'][1]), 
            batched=True
        )
        train_dataset = train_dataset.remove_columns(['short_description', 'headline', 'category'])

        validation_dataset = dataset['validation'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey'][0], structure['contentKey'][1]), 
            batched=True
        )
        
        test_dataset = dataset['test'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey'][0], structure['contentKey'][1]), 
            batched=True
        )
        
        example = train_dataset[0]
        print(f"Chaves do exemplo processado: {example.keys()}")
        print(f"Label: {example['label']}")
        print(f"Texto decodificado: {bertModel.tokenizer.decode(example['input_ids'])}")

        train_dataset.set_format("torch")
        validation_dataset.set_format("torch")
        test_dataset.set_format("torch")

        bertModel.train(
            train_dataset=train_dataset, 
            test_dataset=validation_dataset,
            dataset_name=datasetsNames[countDataset]
        )

        print(f"\nAvaliação do modelo {bertModel.model_name}:")
        metrics = bertModel.evaluate(test_dataset, dataset_name=datasetsNames[countDataset])
        print(metrics)

        print("Salvando logits e embeddings...")
        
        bertModel.store_logits(test_dataset, f"huffpost_test_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(test_dataset, f"huffpost_test_{bertModel.model_name.split('/')[-1]}")
        
        bertModel.store_logits(train_dataset, f"huffpost_train_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(train_dataset, f"huffpost_train_{bertModel.model_name.split('/')[-1]}")
        
        bertModel.store_logits(validation_dataset, f"huffpost_val_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(validation_dataset, f"huffpost_val_{bertModel.model_name.split('/')[-1]}")
        
        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        print(f"Modelo {bertModel.model_name} concluído em {model_total_time:.2f} segundos!")

overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time

print("\n" + "="*50)
print("EXPERIMENTO CONCLUÍDO!")
print(f"Tempo total: {overall_time:.2f} segundos ({overall_time/60:.2f} minutos)")
print("="*50)