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

import nltk
from nltk.tokenize import sent_tokenize

# ========== CONFIGURAÇÃO DO NLTK ==========
print("="*60)
print("CONFIGURANDO NLTK...")
print("="*60)

required_resources = [
    'punkt_tab',
    'punkt',
]

for resource in required_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
        print(f"✓ Recurso '{resource}' já disponível")
    except LookupError:
        print(f"✗ Baixando '{resource}'...")
        nltk.download(resource, quiet=False)
        print(f"✓ '{resource}' baixado!")

try:
    test_sentences = sent_tokenize("Test sentence. Another one.")
    print(f"✓ sent_tokenize funcionando! ({len(test_sentences)} sentenças)")
except Exception as e:
    print(f"✗ ERRO: {e}")
    print("Forçando download completo...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

print("="*60)

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
            evaluation_strategy = "epoch",
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

        dataloader = self.trainer.get_test_dataloader(dataset)
        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits.cpu().numpy()
                all_logits.append(logits)
                all_labels.append(batch["labels"].cpu().numpy())

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

    def store_sentence_based_embeddings(self, dataset, dataset_name, headline_field='headline', description_field='short_description'):
        """
        Gera embeddings médios de sentenças por documento.
        Combina headline + description, divide em sentenças, gera embedding de cada sentença,
        e retorna a média. Resultado: 1 embedding por documento.
        """
        print(f"\n{'='*60}")
        print(f"Gerando embeddings baseados em sentenças para {dataset_name}")
        print(f"{'='*60}")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        all_doc_embeddings = []
        all_labels = []
        all_sentence_counts = []
        
        for idx, example in enumerate(dataset):
            if idx % 100 == 0:
                print(f"Processando documento {idx}/{len(dataset)}...")

            headline = example.get(headline_field, "")
            description = example.get(description_field, "")
            
            if not headline:
                headline = " "
            if not description:
                description = " "
            
            # Combina headline e description
            combined_text = f"{headline} {description}"

            label = example.get('label', 0)
            
            # Divide o texto combinado em sentenças
            sentences = sent_tokenize(combined_text)
            if len(sentences) == 0:
                sentences = [combined_text]

            sentence_embeddings = []
            
            # Gera embedding para cada sentença
            for sentence in sentences:
                inputs = self.tokenizer(
                    sentence,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_hidden_states = outputs.hidden_states[-1]
                    
                    if self.model_type in ['bert', 'electra', 'roberta', 'longformer']:
                        sentence_emb = last_hidden_states[:, 0, :].cpu().numpy()
                    else:
                        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
                        sum_embeddings = torch.sum(last_hidden_states * attention_mask, 1)
                        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                        sentence_emb = (sum_embeddings / sum_mask).cpu().numpy()
                    
                    sentence_embeddings.append(sentence_emb[0])
            
            # Calcula a média dos embeddings das sentenças
            doc_embedding = np.mean(sentence_embeddings, axis=0) if sentence_embeddings else np.zeros(self.model.config.hidden_size)
            
            all_doc_embeddings.append(doc_embedding)
            all_labels.append(label)
            all_sentence_counts.append(len(sentences))
        
        embeddings = np.array(all_doc_embeddings)
        labels = np.array(all_labels)
        sentence_counts = np.array(all_sentence_counts)
        
        output_file = f"sentence_embeddings_{self.model_name.replace('/', '_')}_{dataset_name}.npz"
        np.savez_compressed(
            output_file,
            embeddings=embeddings,
            labels=labels,
            sentence_counts=sentence_counts
        )
        
        print(f"\n✓ Embeddings salvos em: {output_file}")
        print(f"✓ Shape dos embeddings: {embeddings.shape}")
        print(f"✓ Número médio de sentenças por documento: {np.mean(sentence_counts):.2f}")
        print(f"✓ Min sentenças: {np.min(sentence_counts)}, Max sentenças: {np.max(sentence_counts)}")
        print(f"{'='*60}\n")


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
        train_dataset_processed = train_dataset.remove_columns(['short_description', 'headline', 'category'])

        validation_dataset = dataset['validation'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey'][0], structure['contentKey'][1]), 
            batched=True
        )
        validation_dataset_processed = validation_dataset.remove_columns(['short_description', 'headline', 'category'])
        
        test_dataset = dataset['test'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey'][0], structure['contentKey'][1]), 
            batched=True
        )
        test_dataset_processed = test_dataset.remove_columns(['short_description', 'headline', 'category'])
        
        example = train_dataset_processed[0]
        print(f"Chaves do exemplo processado: {example.keys()}")
        print(f"Label: {example['label']}")
        print(f"Texto decodificado: {bertModel.tokenizer.decode(example['input_ids'])}")

        train_dataset_processed.set_format("torch")
        validation_dataset_processed.set_format("torch")
        test_dataset_processed.set_format("torch")

        bertModel.train(
            train_dataset=train_dataset_processed, 
            test_dataset=validation_dataset_processed,
            dataset_name=datasetsNames[countDataset]
        )

        print(f"\nAvaliação do modelo {bertModel.model_name}:")
        metrics = bertModel.evaluate(test_dataset_processed, dataset_name=datasetsNames[countDataset])
        print(metrics)

        print("\n" + "="*60)
        print("SALVANDO LOGITS E EMBEDDINGS TRADICIONAIS (texto completo)")
        print("="*60)
        
        bertModel.store_logits(test_dataset_processed, f"huffpost_test_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(test_dataset_processed, f"huffpost_test_{bertModel.model_name.split('/')[-1]}")
        
        bertModel.store_logits(train_dataset_processed, f"huffpost_train_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(train_dataset_processed, f"huffpost_train_{bertModel.model_name.split('/')[-1]}")
        
        bertModel.store_logits(validation_dataset_processed, f"huffpost_val_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(validation_dataset_processed, f"huffpost_val_{bertModel.model_name.split('/')[-1]}")
        
        # Salvando embeddings BASEADOS EM SENTENÇAS
        print("\n" + "="*60)
        print("SALVANDO EMBEDDINGS BASEADOS EM SENTENÇAS (média de sentenças)")
        print("="*60)
        
        # Usando os datasets originais que contêm os campos headline e short_description
        bertModel.store_sentence_based_embeddings(
            train_dataset, 
            f"huffpost_train_{bertModel.model_name.split('/')[-1]}",
            headline_field='headline',
            description_field='short_description'
        )
        bertModel.store_sentence_based_embeddings(
            validation_dataset, 
            f"huffpost_val_{bertModel.model_name.split('/')[-1]}",
            headline_field='headline',
            description_field='short_description'
        )
        bertModel.store_sentence_based_embeddings(
            test_dataset, 
            f"huffpost_test_{bertModel.model_name.split('/')[-1]}",
            headline_field='headline',
            description_field='short_description'
        )
        
        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        print(f"\nModelo {bertModel.model_name} concluído em {model_total_time:.2f} segundos!")

overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time

print("\n" + "="*60)
print("EXPERIMENTO CONCLUÍDO!")
print(f"Tempo total: {overall_time:.2f} segundos ({overall_time/60:.2f} minutos)")
print("="*60)
print("\nArquivos gerados por modelo:")
model_names = ['electra-base-discriminator', 'roberta-base', 'bert-base-uncased']
for model_name in model_names:
    print(f"\n{model_name}:")
    print(f"  - Embeddings tradicionais (texto completo):")
    print(f"    * embeddings_google_{model_name}_huffpost_test_{model_name.split('-')[-1]}.npz")
    print(f"    * embeddings_google_{model_name}_huffpost_train_{model_name.split('-')[-1]}.npz")
    print(f"    * embeddings_google_{model_name}_huffpost_val_{model_name.split('-')[-1]}.npz")
    print(f"  - Embeddings baseados em sentenças (média):")
    print(f"    * sentence_embeddings_google_{model_name}_huffpost_test_{model_name.split('-')[-1]}.npz")
    print(f"    * sentence_embeddings_google_{model_name}_huffpost_train_{model_name.split('-')[-1]}.npz")
    print(f"    * sentence_embeddings_google_{model_name}_huffpost_val_{model_name.split('-')[-1]}.npz")
print("="*60)