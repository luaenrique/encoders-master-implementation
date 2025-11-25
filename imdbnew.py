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

import nltk
from nltk.tokenize import sent_tokenize

from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# ========== CONFIGURAÇÃO DO NLTK ==========
print("="*60)
print("CONFIGURANDO NLTK...")
print("="*60)

required_resources = [
    'punkt_tab',  # Novo formato (NLTK 3.9+)
    'punkt',      # Formato antigo (fallback)
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
        self.model.resize_token_embeddings(len(self.tokenizer))

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
        """
        Store predictions along with true labels to a CSV file.
        """
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['prediction', 'label', 'text']) 
            
            for i in range(len(dataset)):
                example = dataset[i]
                label = example.get('label', None)
                
                # tenta pegar o texto original
                if 'text' in example:
                    text = example['text']
                else:
                    # reconstrói texto a partir de input_ids
                    input_ids = example.get('input_ids')
                    if input_ids is None:
                        text = ""
                    else:
                        try:
                            ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
                        except Exception:
                            ids = input_ids
                        text = self.tokenizer.decode(ids, skip_special_tokens=True)
                
                writer.writerow([predictions[i], label, text])

    def evaluate(self, test_dataset, dataset_name):
        metrics = self.trainer.evaluate()
        output_csv_path=f"metrics_{self.model_name}_{dataset_name}_2.csv"

        predictions = []
        for batch in self.trainer.get_test_dataloader(test_dataset):
            outputs = self.model(**batch)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1)
            predictions.extend(predicted_class.cpu().numpy())

        # Store predictions in CSV file
        self.store_predictions(test_dataset, predictions, output_csv_path=f"predictions_{self.model_name}_{dataset_name}_2.csv")

        # Write metrics to CSV file
        with open(output_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            file_is_empty = file.tell() == 0
            if file_is_empty:
                writer.writerow(['dataset', 'accuracy', 'micro-f1', 'macro-f1'])

            writer.writerow([self.training_file_name, metrics.get('eval_accuracy', 'N/A'),
                             metrics.get('eval_micro-f1', 'N/A'), metrics.get('eval_macro-f1', 'N/A')])

        return metrics

    def store_embeddings_only(self, dataset, dataset_name):
        """
        Store only embeddings (lighter version if you don't need logits).
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []

        dataloader = self.trainer.get_test_dataloader(dataset)
        
        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True)
                
                # Get embeddings from the last hidden state
                last_hidden_states = outputs.hidden_states[-1]
                
                # Extract embeddings based on model type
                if self.model_type in ['bert', 'electra', 'roberta', 'longformer']:
                    # Use [CLS] token (first token)
                    embeddings = last_hidden_states[:, 0, :].cpu().numpy()
                else:
                    # Mean pooling
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

    def store_sentence_based_embeddings(self, dataset, dataset_name, text_field='text'):
        """
        Gera embeddings médios de sentenças por documento.
        Para cada documento, divide em sentenças, gera embedding de cada sentença,
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

            text = example.get(text_field, "")
            if not text:
                text = " "  # fallback para texto vazio

            label = example.get('label', 0)
            
            # Divide o texto em sentenças
            sentences = sent_tokenize(text)
            if len(sentences) == 0:
                sentences = [text]

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


# ========== CONFIGURAÇÃO PARA O DATASET IMDB ==========

# Importing the datasets
imdb_dataset = load_dataset("stanfordnlp/imdb")

datasets = [imdb_dataset]
datasetsNames = ['imdb']
numLabels = [2]

def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(examples[contentKey], truncation=True, padding="max_length", max_length=128)

datasetStructure = {
    0: {
        'contentKey': 'text',
        'labelKey': 'label'
    },
}

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
        dataset = datasets[countDataset]
        structure = datasetStructure.get(countDataset, None)

        # Preprocessa os datasets
        train_dataset = dataset['train'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), batched=True)
        test_dataset = dataset['test'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), batched=True)
        
        # Remove coluna de texto apenas do dataset processado
        train_dataset_processed = train_dataset.remove_columns([structure['contentKey']])
        test_dataset_processed = test_dataset.remove_columns([structure['contentKey']])

        example = train_dataset_processed[0]
        print(example.keys())
        print(bertModel.tokenizer.decode(example['input_ids']))

        train_dataset_processed.set_format("torch")
        test_dataset_processed.set_format("torch")

        # Treina o modelo
        bertModel.train(train_dataset=train_dataset_processed, test_dataset=test_dataset_processed, dataset_name=datasetsNames[countDataset])

        # Avalia o modelo
        print(bertModel.evaluate(test_dataset_processed, dataset_name=datasetsNames[countDataset]))
        
        # Salva logits
        bertModel.store_logits(test_dataset_processed, "imdb_test")
        bertModel.store_logits(train_dataset_processed, "imdb_train")
        
        # Salva embeddings TRADICIONAIS (texto completo)
        print("\n" + "="*60)
        print("SALVANDO EMBEDDINGS TRADICIONAIS (texto completo)")
        print("="*60)
        bertModel.store_embeddings_only(test_dataset_processed, f"imdb_test_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(train_dataset_processed, f"imdb_train_{bertModel.model_name.split('/')[-1]}")
        
        # Salva embeddings BASEADOS EM SENTENÇAS (média de sentenças)
        print("\n" + "="*60)
        print("SALVANDO EMBEDDINGS BASEADOS EM SENTENÇAS (média de sentenças)")
        print("="*60)
        
        # Usa os datasets originais que ainda contêm o campo 'text'
        bertModel.store_sentence_based_embeddings(
            dataset['test'], 
            f"imdb_test_{bertModel.model_name.split('/')[-1]}",
            text_field='text'
        )
        bertModel.store_sentence_based_embeddings(
            dataset['train'], 
            f"imdb_train_{bertModel.model_name.split('/')[-1]}",
            text_field='text'
        )

print("\n" + "="*60)
print("EXPERIMENTO CONCLUÍDO!")
print("="*60)
print("\nArquivos gerados por modelo:")
model_names = ['electra-base-discriminator', 'roberta-base', 'bert-base-uncased']
for model_name in model_names:
    print(f"\n{model_name}:")
    print(f"  - Embeddings tradicionais (texto completo):")
    print(f"    * embeddings_google_{model_name}_imdb_test_{model_name.split('-')[-1]}.npz")
    print(f"    * embeddings_google_{model_name}_imdb_train_{model_name.split('-')[-1]}.npz")
    print(f"  - Embeddings baseados em sentenças (média):")
    print(f"    * sentence_embeddings_google_{model_name}_imdb_test_{model_name.split('-')[-1]}.npz")
    print(f"    * sentence_embeddings_google_{model_name}_imdb_train_{model_name.split('-')[-1]}.npz")
print("="*60)