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
import os

from transformers import TrainingArguments, Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

# OTIMIZAÇÕES DE MEMÓRIA - CONFIGURAÇÕES MAIS CONSERVADORAS
batch_size = 1  # Reduzido ainda mais para evitar OOM
gradient_accumulation_steps = 8  # Para simular batch_size = 8
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # Carregar modelo COM MAIS CAUTELA para FP16
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                problem_type=self.problem_type,  
                num_labels=self.num_labels,
                torch_dtype=torch.float16,  # Manter FP16 mas com mais cuidado
                device_map="auto",
                low_cpu_mem_usage=True  # NOVO: Reduzir uso de CPU durante carregamento
            )
        except Exception as e:
            print(f"Erro ao carregar modelo em FP16, tentando FP32: {e}")
            # Fallback para FP32 se FP16 falhar
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                problem_type=self.problem_type,  
                num_labels=self.num_labels,
                torch_dtype=torch.float32,  # Fallback para FP32
                device_map="auto",
                low_cpu_mem_usage=True
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

        # CONFIGURAÇÕES MAIS CONSERVADORAS PARA FP16
        args = TrainingArguments(
            f"{self.training_file_name}_{dataset_name}_2",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=1e-5,  # REDUZIDO: Learning rate menor para estabilidade
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=3,  # REDUZIDO: Menos épocas para teste inicial
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            
            # CONFIGURAÇÕES FP16 MAIS CONSERVADORAS
            fp16=True,
            fp16_opt_level="O1",  # NOVO: Nível de otimização mais conservador
            fp16_backend="auto",  # NOVO: Backend automático
            max_grad_norm=0.5,  # NOVO: Gradient clipping mais agressivo
            
            # OTIMIZAÇÕES DE MEMÓRIA
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            save_total_limit=1,  # REDUZIDO: Apenas 1 checkpoint
            logging_steps=500,
            eval_steps=500,
            warmup_steps=200,  # AUMENTADO: Mais warmup para estabilidade
            warmup_ratio=0.1,  # NOVO: Ratio de warmup
            
            # Otimizações adicionais
            group_by_length=True,
            remove_unused_columns=True,
            
            # NOVAS CONFIGURAÇÕES PARA ESTABILIDADE
            save_safetensors=True,  # Formato mais estável
            ignore_data_skip=True,  # Ignorar problemas de dados
            report_to=[],  # Desabilitar logging externo
        )
        
        # INICIALIZAR TRAINER COM TRATAMENTO DE ERRO
        try:
            trainer = Trainer(
                self.model,
                args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
        except Exception as e:
            print(f"Erro ao inicializar trainer: {e}")
            raise e
        
        # Limpeza antes do treinamento
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # TREINAMENTO COM TRATAMENTO DE ERRO ESPECÍFICO PARA FP16
        try:
            trainer.train()
            self.trainer = trainer
        except RuntimeError as e:
            if "F16P" in str(e) or "unscale" in str(e):
                print(f"Erro de FP16 detectado: {e}")
                print("Tentando novamente sem FP16...")
                
                # Recriar argumentos sem FP16
                args_fp32 = TrainingArguments(
                    f"{self.training_file_name}_{dataset_name}_2_fp32",
                    evaluation_strategy = "epoch",
                    save_strategy = "epoch",
                    learning_rate=2e-5,  # Learning rate normal para FP32
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    num_train_epochs=3,
                    weight_decay=0.01,
                    load_best_model_at_end=True,
                    metric_for_best_model=metric_name,
                    
                    # SEM FP16
                    fp16=False,
                    max_grad_norm=1.0,
                    
                    # Mesmas otimizações de memória
                    dataloader_pin_memory=False,
                    dataloader_num_workers=0,
                    save_total_limit=1,
                    logging_steps=500,
                    eval_steps=500,
                    warmup_steps=100,
                    
                    group_by_length=True,
                    remove_unused_columns=True,
                    save_safetensors=True,
                    report_to=[],
                )
                
                # Recriar modelo em FP32 se necessário
                if self.model.dtype == torch.float16:
                    print("Convertendo modelo para FP32...")
                    self.model = self.model.float()
                
                # Novo trainer
                trainer = Trainer(
                    self.model,
                    args_fp32,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics,
                )
                
                cleanup_memory()
                trainer.train()
                self.trainer = trainer
            else:
                raise e

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
                try:
                    # Mover batch para GPU apenas quando necessário
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    predicted_class = torch.argmax(logits, dim=-1)
                    predictions.extend(predicted_class.cpu().numpy())
                    
                    # Limpeza após cada batch
                    del outputs, logits, predicted_class
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM durante avaliação: {e}")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                        continue
                    else:
                        raise e

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

# OTIMIZAÇÃO: max_length menor para Longformer inicial
def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(
        examples[contentKey], 
        truncation=True, 
        padding="max_length", 
        max_length=256  # REDUZIDO: Começar com sequências menores
    )

datasetStructure = {
    0: {
        'contentKey': 'text',
        'labelKey': 'label'
    },
}

# FUNÇÃO PARA LIMPEZA DE MEMÓRIA APRIMORADA
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Forçar sincronização
        torch.cuda.synchronize()
    import time
    time.sleep(1)  # Pequena pausa para garantir limpeza

# VERIFICAR DISPONIBILIDADE DE CUDA E MEMÓRIA
if torch.cuda.is_available():
    print(f"CUDA disponível: {torch.cuda.get_device_name()}")
    print(f"Memória total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Memória livre: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
else:
    print("CUDA não disponível, usando CPU")

for countDataset in range (0, len(datasets)):
    
    # Limpeza antes de cada dataset
    cleanup_memory()
    
    print(f"\n=== Processando dataset {datasetsNames[countDataset]} ===")
    
    try:
        bertModel = GenericEncoderModel(
            model_name='allenai/longformer-base-4096', 
            training_file_name='longformer_training', 
            model_type='longformer', 
            problem_type='single_label_classification',
            num_labels=numLabels[countDataset],
        )

        dataset = datasets[countDataset]
        structure = datasetStructure.get(countDataset, None)

        # LIMITAR DATASET PARA TESTES (OPCIONAL)
        print(f"Dataset size original - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
        
        # Descomentar para limitar dataset durante testes
        dataset['train'] = dataset['train'].select(range(min(1000, len(dataset['train']))))
        dataset['test'] = dataset['test'].select(range(min(500, len(dataset['test']))))
        print(f"Dataset size limitado - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

        contentList = dataset['train'][structure['contentKey']]
        labelList = dataset['train'][structure['labelKey']]

        contentTestList = dataset['test'][structure['contentKey']]
        labelTestList = dataset['test'][structure['labelKey']]

        # Processar datasets com limpeza de memória
        print("Processando dataset de treino...")
        train_dataset = dataset['train'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), 
            batched=True,
            batch_size=100,  # NOVO: Processar em batches menores
            remove_columns=[structure['contentKey']]
        )
        
        cleanup_memory()
        
        print("Processando dataset de teste...")
        test_dataset = dataset['test'].map(
            lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), 
            batched=True,
            batch_size=100,  # NOVO: Processar em batches menores
            remove_columns=[structure['contentKey']]
        )
        
        cleanup_memory()

        example = train_dataset[0]
        print(f"Chaves do exemplo: {example.keys()}")
        print(f"Tamanho da sequência: {len(example['input_ids'])}")
        print(f"Exemplo tokenizado: {bertModel.tokenizer.decode(example['input_ids'][:50])}...")

        train_dataset.set_format("torch")
        test_dataset.set_format("torch")

        # Treinamento com tratamento de erro robusto
        print("Iniciando treinamento...")
        bertModel.train(train_dataset=train_dataset, test_dataset=test_dataset, dataset_name=datasetsNames[countDataset])
        print("Avaliando modelo...")
        print(bertModel.evaluate(test_dataset, dataset_name=datasetsNames[countDataset]))
        
        print(f"✅ Dataset {datasetsNames[countDataset]} processado com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no dataset {datasetsNames[countDataset]}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpeza após cada dataset (mesmo se houver erro)
        try:
            del bertModel
        except:
            pass
        cleanup_memory()
        print(f"Memória limpa após dataset {datasetsNames[countDataset]}")

print("\n🎉 Processamento concluído!")