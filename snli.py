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
            # yelp_dataset, 
            snli_dataset
            ]

datasetsNames = [#'imdb', 
                 #'amazon', 
                 #'agnews', 
                 #'yelp', 
                 'snli'
                 ]

numLabels = [
    #2,
    #2,
    # 4,
   # 5,
    3
]


def preprocess_function(examples, tokenizer, premise_key='premise', hypothesis_key='hypothesis'):
    # Concatenate premise and hypothesis with [SEP] tokens
    inputs = tokenizer(
        examples[premise_key], 
        examples[hypothesis_key], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    return inputs

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
 #   0: {
 #       'contentKey': 'text',
 #       'labelKey': 'label'
 #   },
   0: {
        'contentKey': ['premise', 'hypothesis'],  
        'labelKey': 'label'
    }
}

for countDataset in range (0, len(datasets)):
    
    bertModel = GenericEncoderModel(
        model_name='bert-base-uncased', 
        training_file_name='bert_training', 
        model_type='bert', 
        problem_type='single_label_classification',
        num_labels=numLabels[countDataset],
    )

    dataset = datasets[countDataset]

    structure = datasetStructure.get(countDataset, None)

    #contentList = dataset['train'][f"{structure['contentKey'][0]} [SEP] {structure['contentKey'][1]}"]
    labelList = dataset['train'][structure['labelKey']]

    #contentTestList = dataset['test'][structure['contentKey']]
    labelTestList = dataset['test'][structure['labelKey']]

    print(set(labelList))  # For the training dataset
    print(set(labelTestList))  # For the test dataset

    train_dataset = dataset['train'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey'][0], structure['contentKey'][1]), batched=True)
    test_dataset = dataset['test'].map(lambda x: preprocess_function(x, bertModel.tokenizer, structure['contentKey'][0], structure['contentKey'][1]), batched=True)
    train_dataset = train_dataset.map(remove_columns=[structure['contentKey'][0], structure['contentKey'][1]])
    train_dataset = train_dataset.filter(lambda x: x['label'] != -1)
    test_dataset = test_dataset.filter(lambda x: x['label'] != -1)
    
    example = train_dataset[0]
    print(example.keys())

    print(bertModel.tokenizer.decode(example['input_ids']))

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    bertModel.train(train_dataset=train_dataset, test_dataset=test_dataset, dataset_name=datasetsNames[countDataset])

    print(bertModel.evaluate(test_dataset, dataset_name=datasetsNames[countDataset]))
