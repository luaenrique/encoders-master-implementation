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
import time
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize

from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Download do recurso necessário do NLTK para segmentação de sentenças
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
            # single label classification
            ptype = None
            predictions = np.argmax(logits, axis=-1).reshape(-1,1)
            labels_ = labels
            metrics = ["accuracy", "micro-f1", "macro-f1"]
        elif self.problem_type ==  "multi_label_classification":
            # multi label classification
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
        
        # Compute the output
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
            for text, label, prediction in zip(dataset['text'], dataset['label'], predictions):
                writer.writerow([prediction, label, text])

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
        self.store_predictions(self.trainer.eval_dataset, predictions, output_csv_path=f"predictions_{self.model_name}_{dataset_name}_2.csv")

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
        NOVA FUNÇÃO: Quebra textos em sentenças, gera embeddings para cada sentença
        e calcula a média para representar o documento completo.
        
        Args:
            dataset: Dataset contendo os textos
            dataset_name: Nome para salvar o arquivo
            text_field: Nome do campo que contém o texto original
        """
        print(f"\n{'='*60}")
        print(f"Gerando embeddings baseados em sentenças para {dataset_name}")
        print(f"{'='*60}")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        all_doc_embeddings = []
        all_labels = []
        all_sentence_counts = []
        
        # Processar cada documento
        for idx, example in enumerate(dataset):
            if idx % 100 == 0:
                print(f"Processando documento {idx}/{len(dataset)}...")
            
            # Obter texto original e label
            text = example[text_field]
            label = example['label']
            
            # Quebrar texto em sentenças
            sentences = sent_tokenize(text)
            
            # Se não houver sentenças válidas, usar o texto completo
            if len(sentences) == 0:
                sentences = [text]
            
            sentence_embeddings = []
            
            # Processar cada sentença
            for sentence in sentences:
                # Tokenizar sentença
                inputs = self.tokenizer(
                    sentence,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                # Mover para o dispositivo correto
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Gerar embedding
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_hidden_states = outputs.hidden_states[-1]
                    
                    # Extrair embedding da sentença (usando [CLS] token)
                    if self.model_type in ['bert', 'electra', 'roberta', 'longformer']:
                        sentence_emb = last_hidden_states[:, 0, :].cpu().numpy()
                    else:
                        # Mean pooling
                        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
                        sum_embeddings = torch.sum(last_hidden_states * attention_mask, 1)
                        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                        sentence_emb = (sum_embeddings / sum_mask).cpu().numpy()
                    
                    sentence_embeddings.append(sentence_emb[0])
            
            # Calcular média dos embeddings das sentenças
            if len(sentence_embeddings) > 0:
                doc_embedding = np.mean(sentence_embeddings, axis=0)
            else:
                # Fallback: usar embedding zero se não houver sentenças
                doc_embedding = np.zeros(sentence_embeddings[0].shape)
            
            all_doc_embeddings.append(doc_embedding)
            all_labels.append(label)
            all_sentence_counts.append(len(sentences))
        
        # Converter para arrays numpy
        embeddings = np.array(all_doc_embeddings)
        labels = np.array(all_labels)
        sentence_counts = np.array(all_sentence_counts)
        
        # Salvar embeddings
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


# ========== CONFIGURAÇÃO PARA O DATASET DE EMOÇÕES ==========

# Carregando o dataset de emoções
emotion_dataset = load_dataset("dair-ai/emotion", "split")

print("Dataset de emoções carregado:")
print(emotion_dataset)

# Verificando as labels do dataset
print("\nPrimeiros exemplos do dataset:")
print("Texto:", emotion_dataset['train']['text'][:3])
print("Labels:", emotion_dataset['train']['label'][:3])

# Mapeamento das labels (0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise)
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
print(f"\nLabels do dataset: {label_names}")
print(f"Número de classes: {len(label_names)}")

# Configuração dos datasets
datasets = [emotion_dataset]
datasetsNames = ['emotion']
numLabels = [6]  # 6 classes de emoção

def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(examples[contentKey], truncation=True, padding="max_length", max_length=128)

# Estrutura do dataset de emoções
datasetStructure = {
    0: {
        'contentKey': 'text',
        'labelKey': 'label'
    }
}

# Track overall execution time
overall_start_time = time.time()
print(f"Starting emotion experiment...")

# Loop principal para treinar os modelos
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
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), 
            batched=True
        )
        validation_dataset = dataset['validation'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), 
            batched=True
        )
        test_dataset = dataset['test'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), 
            batched=True
        )
        
        # Removendo colunas desnecessárias do dataset de treino
        train_dataset_processed = train_dataset.remove_columns([structure['contentKey']])
        
        # Verificando exemplo processado
        example = train_dataset_processed[0]
        print(f"Chaves do exemplo processado: {example.keys()}")
        print(f"Texto decodificado: {bertModel.tokenizer.decode(example['input_ids'])}")

        # Configurando formato torch
        train_dataset_processed.set_format("torch")
        validation_dataset_processed = validation_dataset.remove_columns([structure['contentKey']])
        validation_dataset_processed.set_format("torch")
        test_dataset_processed = test_dataset.remove_columns([structure['contentKey']])
        test_dataset_processed.set_format("torch")

        # Treinamento
        bertModel.train(
            train_dataset=train_dataset_processed, 
            test_dataset=validation_dataset_processed,
            dataset_name=datasetsNames[countDataset]
        )

        # Avaliação no conjunto de teste
        print(f"\nAvaliação do modelo {bertModel.model_name}:")
        metrics = bertModel.evaluate(test_dataset_processed, dataset_name=datasetsNames[countDataset])
        print(metrics)

        # Salvando embeddings TRADICIONAIS (método antigo)
        print("\n" + "="*60)
        print("SALVANDO EMBEDDINGS TRADICIONAIS (texto completo)")
        print("="*60)
        
        bertModel.store_embeddings_only(test_dataset_processed, f"emotion_test_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(train_dataset_processed, f"emotion_train_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(validation_dataset_processed, f"emotion_val_{bertModel.model_name.split('/')[-1]}")
        
        # Salvando embeddings BASEADOS EM SENTENÇAS (método novo)
        print("\n" + "="*60)
        print("SALVANDO EMBEDDINGS BASEADOS EM SENTENÇAS (média de sentenças)")
        print("="*60)
        
        # Precisamos usar os datasets originais que ainda contém o campo 'text'
        bertModel.store_sentence_based_embeddings(
            dataset['test'], 
            f"emotion_test_{bertModel.model_name.split('/')[-1]}",
            text_field='text'
        )
        bertModel.store_sentence_based_embeddings(
            dataset['train'], 
            f"emotion_train_{bertModel.model_name.split('/')[-1]}",
            text_field='text'
        )
        bertModel.store_sentence_based_embeddings(
            dataset['validation'], 
            f"emotion_val_{bertModel.model_name.split('/')[-1]}",
            text_field='text'
        )
        
        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        print(f"Modelo {bertModel.model_name} concluído em {model_total_time:.2f} segundos!")

# Overall timing summary
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time

print("\n" + "="*50)
print("EXPERIMENTO CONCLUÍDO!")
print(f"Tempo total: {overall_time:.2f} segundos ({overall_time/60:.2f} minutos)")
print("="*50)