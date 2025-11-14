import os
import torch
from torch.utils.data import Dataset

def parse_pawsx_tsv(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # skip header line
        for line in f:
            cols = line.strip().split("\t")
            # Skip incomplete or unclear examples
            if len(cols) < 4 or "NS" in cols[1:3]:
                continue
            _, sent1, sent2, label = cols
            texts.append((sent1, sent2))
            labels.append(int(label))
    return texts, labels

def load_pawsx_datasets(data_dir, tokenizer, max_len):
    """
    Load PAWS-X French datasets from TSV files:
    - translated_train.tsv (train)
    - dev_2k.tsv (dev)
    - test_2k.tsv (test)

    Returns a dict with keys "train", "dev", "test" and ParaphraseDataset objects.
    """
    datasets = {}

    train_path = os.path.join(data_dir, "translated_train.tsv")
    dev_path = os.path.join(data_dir, "dev_2k.tsv")
    test_path = os.path.join(data_dir, "test_2k.tsv")

    train_texts, train_labels = parse_pawsx_tsv(train_path)
    dev_texts, dev_labels = parse_pawsx_tsv(dev_path)
    test_texts, test_labels = parse_pawsx_tsv(test_path)

    datasets["train"] = ParaphraseDataset(train_texts, train_labels, tokenizer, max_len)
    datasets["dev"] = ParaphraseDataset(dev_texts, dev_labels, tokenizer, max_len)
    datasets["test"] = ParaphraseDataset(test_texts, test_labels, tokenizer, max_len)

    return datasets

class ParaphraseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts  # List of tuple pairs (sent1, sent2)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sent1, sent2 = self.texts[idx]
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
