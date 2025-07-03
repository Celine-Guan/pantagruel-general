# -*- coding: utf-8 -*-

import os
import re
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

"""**set up parameters**"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = None 
DATA_PARENT_DIR = r"/home/sgu/datasets/PAWS-X/fr"
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATES = [1e-5, 5e-5, 1e-6, 5e-6]
MAX_SEQ_LENGTH = 512
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

def parse_pawsx_tsv(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 4 or "NS" in cols[1:3]:
                continue
            _, sent1, sent2, label = cols
            texts.append((sent1, sent2))  # form a sentences pair 
            labels.append(int(label))
    return texts, labels



def load_pawsx_datasets(data_dir, tokenizer, max_len):
    """
    Load PAWS-X French datasets from TSV files:
    - translated_train.tsv (train)
    - dev_2k.tsv (dev)
    - test_2k.tsv (test)

    Return:
    {
      "train": ReviewDataset(...),
      "dev": ReviewDataset(...),
      "test": ReviewDataset(...)
    }
    """
    datasets = {}

    train_path = os.path.join(data_dir, "translated_train.tsv")
    dev_path = os.path.join(data_dir, "dev_2k.tsv")
    test_path = os.path.join(data_dir, "test_2k.tsv")

    train_texts, train_labels = parse_pawsx_tsv(train_path)
    dev_texts, dev_labels = parse_pawsx_tsv(dev_path)
    test_texts, test_labels = parse_pawsx_tsv(test_path)

    datasets["train"] = ReviewDataset(train_texts, train_labels, tokenizer, max_len)
    datasets["dev"] = ReviewDataset(dev_texts, dev_labels, tokenizer, max_len)
    datasets["test"] = ReviewDataset(test_texts, test_labels, tokenizer, max_len)

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
        sent1, sent2 = self.texts[idx]  # texts is a pair of sentences:tuple (sent1, sent2)
        label = self.labels[idx]
        encoded = self.tokenizer(
            sent1,
            sent2,
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

class ParaphrasingModel(nn.Module):
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

def run_pawsx():
    logging.info(f"Loading PAWS-X French dataset from {DATA_PARENT_DIR}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = load_pawsx_datasets(DATA_PARENT_DIR, tokenizer, MAX_SEQ_LENGTH)

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(datasets["dev"], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False)

    best_valid_acc = 0
    best_lr = None
    best_model_state = None

    for lr in LEARNING_RATES:
        logging.info(f"\nTraining with learning rate {lr}...")

        model = ParaphrasingModel(MODEL_NAME).to(DEVICE)
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

    logging.info(f"\nLoading best model with LR={best_lr} for final test evaluation...")
    model.load_state_dict(best_model_state)
    test_acc = eval_model(model, test_loader)
    logging.info(f"Test accuracy on PAWS-X French: {test_acc:.4f}")


def main():
    set_seed(42)
    args = parse_args()
    global MODEL_NAME
    MODEL_NAME = args.model_name

    short_model_id = MODEL_NAME.split("/models/")[-1].replace("/", "_")
    log_filename = f"training_paraphrasing_{short_model_id}.log"

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
    run_pawsx()

if __name__ == "__main__":
    main()
