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
import gc

from transformers import TrainingArguments, Trainer

# OTIMIZAÇÕES DE MEMÓRIA
batch_size = 2  # Reduzido de 8 para 2
gradient_accumulation_steps = 4  # Para simular batch_size = 8
metric_name = "accuracy"

# Configurar mixed precision e otimizações
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
        
        # Limpeza de memória após carregar modelo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

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
        # Carregar modelo com otimizações de memória
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            problem_type=self.problem_type,  
            num_labels=self.num_labels,
            torch_dtype=torch.float16,  # Usar FP16 para economizar memória
            device_map="auto"  # Distribuição automática de camadas
        )
        
        # Ativar gradient checkpointing para economizar memória
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model

    def compute_metrics(self, eval_preds, threshold = 0.5):
        logits, labels = eval_preds
        
        # Limpeza de memória durante avaliação
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
        self.model.resize_token_embeddings(len(self._load_tokenizer()))

        args = TrainingArguments(
            f"{self.training_file_name}_{dataset_name}_2",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,  # NOVO
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            
            # OTIMIZAÇÕES DE MEMÓRIA
            fp16=True,  # Mixed precision
            dataloader_pin_memory=False,  # Economizar memória
            dataloader_num_workers=0,  # Reduzir workers se necessário
            save_total_limit=2,  # Limitar checkpoints salvos
            logging_steps=500,
            eval_steps=500,
            warmup_steps=100,
            
            # Otimizações adicionais
            group_by_length=True,  # Agrupar por tamanho para eficiência
            remove_unused_columns=True,
        )
        
        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Limpeza antes do treinamento
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        trainer.train(resume_from_checkpoint="./longformer_training_imdb_2/checkpoint-15625")
        self.trainer = trainer

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
        # Limpeza antes da avaliação
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        metrics = self.trainer.evaluate()
        output_csv_path=f"metrics_{self.model_name}_{dataset_name}_2.csv"
        
        predictions = []
        
        # Avaliar em batches menores para economizar memória
        test_dataloader = self.trainer.get_test_dataloader(test_dataset)
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                # Mover batch para GPU apenas quando necessário
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1)
                predictions.extend(predicted_class.cpu().numpy())
                
                # Limpeza após cada batch
                del outputs, logits, predicted_class
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

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

from datasets import load_dataset

# Importing the Amazon dataset recommended by Cristiano
amazon_dataset = load_dataset("fancyzhx/amazon_polarity")
imdb_dataset = load_dataset("stanfordnlp/imdb")
ag_news_dataset = load_dataset("fancyzhx/ag_news")
yelp_dataset = load_dataset("Yelp/yelp_review_full")
snli_dataset = load_dataset("stanfordnlp/snli")

datasets = [imdb_dataset, 
           # amazon_dataset,
            # ag_news_dataset, 
           # yelp_dataset, 
            #snli_dataset
            ]

datasetsNames = ['imdb', 
                 #'amazon', 
                 #'agnews', 
                 #'yelp', 
        #         'snli'
                 ]

numLabels = [
    2,
    #2,
    # 4,
   # 5,
#    3
]

# OTIMIZAÇÃO: Reduzir max_length para economizar memória
def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(
        examples[contentKey], 
        truncation=True, 
        padding="max_length", 
        max_length=512  # Reduzido de 128 para aproveitar melhor o Longformer, mas pode reduzir mais se necessário
    )

datasetStructure = {
    0: {
        'contentKey': 'text',
        'labelKey': 'label'
    },
}

# FUNÇÃO PARA LIMPEZA DE MEMÓRIA
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

for countDataset in range (0, len(datasets)):
    
    # Limpeza antes de cada dataset
    cleanup_memory()
    
    bertModel = GenericEncoderModel(
        model_name='allenai/longformer-base-4096', 
        training_file_name='longformer_training', 
        model_type='longformer', 
        problem_type='single_label_classification',
        num_labels=numLabels[countDataset],
    )

    dataset = datasets[countDataset]
    structure = datasetStructure.get(countDataset, None)

    # OTIMIZAÇÃO: Processar dataset em chunks se muito grande
    print(f"Dataset size - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    
    # Opcional: Limitar tamanho do dataset para testes
    # dataset['train'] = dataset['train'].select(range(min(1000, len(dataset['train']))))
    # dataset['test'] = dataset['test'].select(range(min(500, len(dataset['test']))))

    contentList = dataset['train'][structure['contentKey']]
    labelList = dataset['train'][structure['labelKey']]

    contentTestList = dataset['test'][structure['contentKey']]
    labelTestList = dataset['test'][structure['labelKey']]

    # Processar datasets com limpeza de memória
    train_dataset = dataset['train'].map(
        lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), 
        batched=True,
        remove_columns=[structure['contentKey']]  # Remover colunas desnecessárias
    )
    
    cleanup_memory()  # Limpeza após processamento
    
    test_dataset = dataset['test'].map(
        lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), 
        batched=True,
        remove_columns=[structure['contentKey']]  # Remover colunas desnecessárias
    )
    
    cleanup_memory()  # Limpeza após processamento

    example = train_dataset[0]
    print(example.keys())
    print(f"Sequence length: {len(example['input_ids'])}")
    print(bertModel.tokenizer.decode(example['input_ids']))

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Treinamento com limpeza de memória
    try:
        bertModel.train(train_dataset=train_dataset, test_dataset=test_dataset, dataset_name=datasetsNames[countDataset])
        print(bertModel.evaluate(test_dataset, dataset_name=datasetsNames[countDataset]))
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"CUDA out of memory error: {e}")
            print("Tentando limpeza de memória...")
            cleanup_memory()
            # Você pode tentar reduzir ainda mais o batch_size aqui
            batch_size = 1
            gradient_accumulation_steps = 8
            print(f"Reduzindo batch_size para {batch_size} e gradient_accumulation_steps para {gradient_accumulation_steps}")
        else:
            raise e
    
    # Limpeza após cada dataset
    del bertModel
    cleanup_memory()