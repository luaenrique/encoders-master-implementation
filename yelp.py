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
        # check num labels - use num from the dataset
        # check pretrained config class https://huggingface.co/transformers/v3.0.2/main_classes/configuration.html#transformers.PretrainedConfig
        
        # para cada execucao, guardar arquivo com as predicoes do teste
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
            # labels_ = labels
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
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            #push_to_hub=True,
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

    def store_predictions(self, dataset, predictions, output_csv_path):
        """
        Store predictions along with true labels to a CSV file.
        """
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['prediction', 'label', 'text']) 
            for text, label, prediction in zip(dataset['text'], dataset['label'], predictions):
                writer.writerow([prediction, label, text])
    

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
                all_texts.append(batch["input_ids"].cpu().numpy())  # ou a string original, se preferir

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)

        np.savez(f"logits_{self.model_name}_{dataset_name}.npz", logits=logits, labels=labels)

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
    
    


from datasets import load_dataset


# Importing the Amazon dataset recommended by Cristiano
amazon_dataset = load_dataset("fancyzhx/amazon_polarity")
imdb_dataset = load_dataset("stanfordnlp/imdb")
ag_news_dataset = load_dataset("fancyzhx/ag_news")
yelp_dataset = load_dataset("Yelp/yelp_review_full")
snli_dataset = load_dataset("stanfordnlp/snli")

datasets = [#imdb_dataset, 
           # amazon_dataset,
            # ag_news_dataset, 
            yelp_dataset, 
            #snli_dataset
            ]

datasetsNames = [#'imdb', 
                 #'amazon', 
                 #'agnews', 
                 'yelp', 
        #         'snli'
                 ]

numLabels = [
    #2,
    #2,
    # 4,
    5,
#    3
]


def preprocess_function(examples, tokenizer, contentKey):
    return tokenizer(examples[contentKey], truncation=True, padding="max_length", max_length=128)

datasetStructure = {
   # 0: {
   #     'contentKey': 'text',
   #     'labelKey': 'label'
   # },
   # 0: {
  #      'contentKey': 'content',
   #    'labelKey': 'label'
   # },
   # 0: {
   #     'contentKey': 'text',
   #     'labelKey': 'label'
  #  },
    0: {
        'contentKey': 'text',
        'labelKey': 'label'
    },
  # 1: {
  #      'contentKey': 'premise',
  #      'labelKey': 'label'
  #  }
}

for countDataset in range (0, len(datasets)):
    
    #bertModel = GenericEncoderModel(
    #    model_name='bert-base-uncased', 
    #    training_file_name='bert_training', 
    #    model_type='bert', 
    #    problem_type='single_label_classification',
    #    num_labels=numLabels[countDataset],
    #)


    #bertModel = GenericEncoderModel(
    #    model_name='google/electra-base-discriminator', 
    #    training_file_name='electra_training', 
    #    model_type='electra', 
    #    problem_type='single_label_classification',
    #    num_labels=numLabels[countDataset],
    #)


    #bertModel = GenericEncoderModel(
    #    model_name='roberta-base', 
    #    training_file_name='roberta_training', 
    #    model_type='roberta', 
    #    problem_type='single_label_classification',
    #    num_labels=numLabels[countDataset],
    #)

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

        contentList = dataset['train'][structure['contentKey']]
        labelList = dataset['train'][structure['labelKey']]

        contentTestList = dataset['test'][structure['contentKey']]
        labelTestList = dataset['test'][structure['labelKey']]

        train_dataset = dataset['train'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), batched=True)
        test_dataset = dataset['test'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey']), batched=True)
        train_dataset = train_dataset.map(remove_columns=[structure['contentKey']])

        example = train_dataset[0]
        print(example.keys())

        print(bertModel.tokenizer.decode(example['input_ids']))

        train_dataset.set_format("torch")
        test_dataset.set_format("torch")

        bertModel.train(train_dataset=train_dataset, test_dataset=test_dataset, dataset_name=datasetsNames[countDataset])

        print(bertModel.evaluate(test_dataset, dataset_name=datasetsNames[countDataset]))
        
        #bertModel.store_logits(test_dataset, "yelp_test")
        #bertModel.store_logits(train_dataset, "yelp_train")
        bertModel.store_embeddings_only(test_dataset, f"yelp_test_{bertModel.model_name.split('/')[-1]}")
        bertModel.store_embeddings_only(train_dataset, f"yelp_train_{bertModel.model_name.split('/')[-1]}")