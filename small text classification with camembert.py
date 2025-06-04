# Download dataset
import kagglehub

path = kagglehub.dataset_download("hbaflast/french-twitter-sentiment-analysis")
print("Path to dataset files:", path)

# convert the dataset to a pandas dataframe
import os
print("Files in dataset folder:")
print(os.listdir(path))

import pandas as pd
df = pd.read_csv(f"{path}/french_tweets.csv")
print(df.head())

# Sample 10 000 examples
from sklearn.model_selection import train_test_split
df_sample, _ = train_test_split(df,train_size=10000,stratify=df['label'],random_state=42)
df_sample = df_sample.reset_index(drop=True)

# import model
from transformers import CamembertTokenizer, CamembertForSequenceClassification

model_name = "camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare data and tokenization
from datasets import Dataset
dataset = Dataset.from_pandas(df_sample)

def tokenize_function(example):
  return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Training and evaluation
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = np.argmax(pred.predictions, axis=1)
  return {
    "accuracy": accuracy_score(labels, preds),
    "f1": f1_score(labels, preds),
  }

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to=[]
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
