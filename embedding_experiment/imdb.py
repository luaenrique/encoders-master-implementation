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

from transformers import TrainingArguments, Trainer
batch_size = 8
metric_name = "accuracy"


class FrozenEmbeddingExtractor:
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # Congela todos os parâmetros do modelo
        for param in self.model.parameters():
            param.requires_grad = False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.device = device

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
        # Carrega o modelo pré-treinado sem modificar a cabeça de classificação
        if self.model_type == 'electra':
            model = ElectraForSequenceClassification.from_pretrained(self.model_name)
        elif self.model_type == 'longformer':
            model = LongformerForSequenceClassification.from_pretrained(self.model_name)
        elif self.model_type == 'bert':
            model = BertForSequenceClassification.from_pretrained(self.model_name)
        elif self.model_type == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained(self.model_name)
        else:
            # Fallback para AutoModel
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        return model

    def extract_embeddings_from_dataset(self, dataset, dataset_name, batch_size=16):
        """
        Extrai embeddings de um dataset completo de forma eficiente.
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_texts = []
        
        # Cria DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Extraindo embeddings para {dataset_name}...")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # Move batch para o device correto
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Forward pass com output_hidden_states=True para obter embeddings
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Extrai embeddings da última camada oculta
                last_hidden_states = outputs.hidden_states[-1]
                
                # Estratégias diferentes para extrair embeddings dependendo do modelo
                if self.model_type in ['bert', 'electra', 'roberta', 'longformer']:
                    # Usa o token [CLS] (primeiro token)
                    embeddings = last_hidden_states[:, 0, :].cpu().numpy()
                else:
                    # Mean pooling como fallback
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                    sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, 1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                all_embeddings.append(embeddings)
                all_labels.append(labels.numpy())
                
                # Opcional: armazenar textos decodificados
                for input_id_seq in input_ids:
                    text = self.tokenizer.decode(input_id_seq, skip_special_tokens=True)
                    all_texts.append(text)
                
                if i % 100 == 0:
                    print(f"Processados {i * batch_size} exemplos...")
        
        # Concatena todos os resultados
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return embeddings, labels, all_texts

    def save_embeddings(self, embeddings, labels, texts, dataset_name, include_texts=False):
        """
        Salva embeddings em arquivo .npz
        """
        output_file = f"frozen_embeddings_{self.model_name.replace('/', '_')}_{dataset_name}.npz"
        
        if include_texts:
            np.savez_compressed(
                output_file,
                embeddings=embeddings,
                labels=labels,
                texts=texts
            )
        else:
            np.savez_compressed(
                output_file,
                embeddings=embeddings,
                labels=labels
            )
        
        print(f"Embeddings salvos em: {output_file}")
        print(f"Shape dos embeddings: {embeddings.shape}")
        print(f"Número de labels: {len(labels)}")

    def extract_embeddings_from_texts(self, texts, labels=None, batch_size=16):
        """
        Extrai embeddings diretamente de uma lista de textos.
        Útil quando você não tem um dataset do HuggingFace.
        """
        self.model.eval()
        all_embeddings = []
        
        print(f"Extraindo embeddings de {len(texts)} textos...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokeniza o batch
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move para device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Extrai embeddings
                last_hidden_states = outputs.hidden_states[-1]
                
                if self.model_type in ['bert', 'electra', 'roberta', 'longformer']:
                    embeddings = last_hidden_states[:, 0, :].cpu().numpy()
                else:
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                    sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, 1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                all_embeddings.append(embeddings)
                
                if i % (batch_size * 10) == 0:
                    print(f"Processados {i} textos...")
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings


def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(examples[contentKey], truncation=True, padding="max_length", max_length=512)


from datasets import load_dataset

# Carrega datasets
imdb_dataset = load_dataset("stanfordnlp/imdb")
# amazon_dataset = load_dataset("fancyzhx/amazon_polarity")
# ag_news_dataset = load_dataset("fancyzhx/ag_news")

datasets = [imdb_dataset]
dataset_names = ['imdb']

# Define modelos para extração de embeddings
models_config = [
    {
        'model_name': 'google/electra-base-discriminator',
        'model_type': 'electra'
    },
    {
        'model_name': 'roberta-base',
        'model_type': 'roberta'
    },
    {
        'model_name': 'google-bert/bert-base-uncased',
        'model_type': 'bert'
    }
]

# Estrutura dos datasets
dataset_structure = {
    0: {
        'contentKey': 'text',
        'labelKey': 'label'
    }
}

# Loop principal
for count_dataset in range(len(datasets)):
    dataset = datasets[count_dataset]
    dataset_name = dataset_names[count_dataset]
    structure = dataset_structure[count_dataset]
    
    print(f"\n=== Processando dataset: {dataset_name} ===")
    
    for model_config in models_config:
        print(f"\n--- Modelo: {model_config['model_name']} ---")
        
        # Cria extrator de embeddings
        extractor = FrozenEmbeddingExtractor(
            model_name=model_config['model_name'],
            model_type=model_config['model_type']
        )
        
        # Preprocessa datasets
        train_dataset = dataset['train'].map(
            lambda x: preprocess_function(x, extractor.tokenizer, structure['contentKey']), 
            batched=True
        )
        test_dataset = dataset['test'].map(
            lambda x: preprocess_function(x, extractor.tokenizer, structure['contentKey']), 
            batched=True
        )
        
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])
        test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])
        
        # Define formato torch
        train_dataset.set_format("torch")
        test_dataset.set_format("torch")
        
        # Extrai embeddings do conjunto de treino
        train_embeddings, train_labels, train_texts = extractor.extract_embeddings_from_dataset(
            train_dataset, 
            f"{dataset_name}_train",
            batch_size=32
        )
        
        # Salva embeddings de treino
        extractor.save_embeddings(
            train_embeddings, 
            train_labels, 
            train_texts,
            f"{dataset_name}_train_{model_config['model_name'].split('/')[-1]}",
            include_texts=False
        )
        
        # Extrai embeddings do conjunto de teste
        test_embeddings, test_labels, test_texts = extractor.extract_embeddings_from_dataset(
            test_dataset, 
            f"{dataset_name}_test",
            batch_size=32
        )
        
        # Salva embeddings de teste
        extractor.save_embeddings(
            test_embeddings, 
            test_labels, 
            test_texts,
            f"{dataset_name}_test_{model_config['model_name'].split('/')[-1]}",
            include_texts=False
        )
        
        print(f"Embeddings extraídos e salvos para {model_config['model_name']}")
        
        # Limpa memória
        del extractor
        torch.cuda.empty_cache()

print("\n=== Extração de embeddings concluída! ===")