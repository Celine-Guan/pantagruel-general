# -*- coding: utf-8 -*-

import os
import re
import xml.etree.ElementTree as ET
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import copy
import logging
import argparse

"""**set up seed**"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED = 42

"""**set up parameters**"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = None 
DATA_PARENT_DIR = r"/home/sgu/datasets/CLS/raw/cls-acl10-unprocessed/fr"
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATES = [1e-5, 5e-5, 1e-6, 5e-6]
DATASETS = ["books", "dvd", "music"]
MAX_SEQ_LENGTH = 512
VALID_RATIO = 0.2
EARLY_STOPPING_PATIENCE = 3

"""**set up argparse to change models**"""

def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification using pre-trained French model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/sgu/models/Text_Base_fr_OSCAR_camtok/HuggingFace",
        help="Pretrained model path"
    )
    return parser.parse_args()

"""**Function: extract texts and labels from datasets**"""

def parse_review_file(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # cut each review example by <item>
    items = re.findall(r"<item>(.*?)</item>", content, re.S)
    for item in items:
        root = ET.fromstring(f"<root>{item}</root>")
        rating = float(root.findtext("rating"))
        text = root.findtext("text").strip('"')
        # set label according to rating: 1=positive, 0=negative
        if rating == 3.0:
            continue  # skip 3 stars
        label = 1 if rating > 3 else 0
        texts.append(text)
        labels.append(label)
    return texts, labels

import os

def load_all_datasets(data_dir, tokenizer, max_len):
    """
    read train.review and test.review in three directories books, dvd and music
    return：
    {
      "books": {"train": ReviewDataset(...), "test": ReviewDataset(...)},
      "dvd": {"train": ReviewDataset(...), "test": ReviewDataset(...)},
      "music": {"train": ReviewDataset(...), "test": ReviewDataset(...)}
    }
    """
    datasets = {}
    for domain in ["books", "dvd", "music"]:
        domain_path = os.path.join(data_dir, domain)
        train_path = os.path.join(domain_path, "train.review")
        test_path = os.path.join(domain_path, "test.review")

        train_texts, train_labels = parse_review_file(train_path)
        test_texts, test_labels = parse_review_file(test_path)

        train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_len)
        test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, max_len)

        datasets[domain] = {
            "train": train_dataset,
            "test": test_dataset
        }
    return datasets

"""**Class: dataset**"""

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

"""**Class: pretrained model + classification head**"""

class TextClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        logits = self.classifier(cls_token)
        return logits

"""**Class: early stopping**"""

class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc is None or val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            if self.verbose:
                print(f"Validation accuracy improved to {val_acc:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation accuracy did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

"""**Function: train the model in one epoch**"""

def train_one_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

"""**Evaluation function**"""

def eval_model(model, dataloader):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(true_labels, preds)
    return acc

"""**Main**"""

def run_for_dataset(dataset_name):
    data_dir = os.path.join(DATA_PARENT_DIR, dataset_name)

    logging.info(f"Parsing train data for {dataset_name}...")
    train_texts, train_labels = parse_review_file(os.path.join(data_dir, "train.review"))
    logging.info(f"Parsing test data for {dataset_name}...")
    test_texts, test_labels = parse_review_file(os.path.join(data_dir, "test.review"))

    # split train and validation set
    n_valid = int(len(train_texts) * VALID_RATIO)
    n_train = len(train_texts) - n_valid

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH)
    train_set, valid_set = random_split(train_dataset, [n_train, n_valid])
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_valid_acc = 0
    best_lr = None
    best_model_state = None

    for lr in LEARNING_RATES:
        logging.info(f"\nTraining with learning rate {lr} on dataset {dataset_name}")

        model = TextClassificationModel(MODEL_NAME).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        num_training_steps = NUM_EPOCHS * len(train_loader)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

        for epoch in range(NUM_EPOCHS):
            logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
            logging.info(f"Train loss: {train_loss:.4f}")

            valid_acc = eval_model(model, valid_loader)
            logging.info(f"Validation accuracy: {valid_acc:.4f}")

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_lr = lr
                best_model_state = model.state_dict()

            early_stopping(valid_acc)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

    # choose the best performing model on validation set to test
    logging.info(f"\nLoading best model with LR={best_lr} for test evaluation on {dataset_name}...")
    model.load_state_dict(best_model_state)
    test_acc = eval_model(model, test_loader)
    logging.info(f"Test accuracy on {dataset_name}: {test_acc:.4f}")


def main():
    set_seed(SEED)
    args = parse_args()
    global MODEL_NAME
    MODEL_NAME = args.model_name

    short_model_id = MODEL_NAME.split("/models/")[-1].replace("/", "_")
    log_filename = f"training_textclassification_{short_model_id}.log"

    # initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    for ds in DATASETS:
        run_for_dataset(ds)

if __name__ == "__main__":
    main()
