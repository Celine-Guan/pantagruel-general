import os
import re
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis reviews.
    """
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


def parse_review_file(filepath):
    """
    Extracts review texts and labels from an XML-formatted file.
    Skips 3-star reviews. Ratings >3 are positive (1), ≤3 negative (0).
    """
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    items = re.findall(r"<item>(.*?)</item>", content, re.S)
    for item in items:
        root = ET.fromstring(f"<root>{item}</root>")
        rating = float(root.findtext("rating"))
        text = root.findtext("text").strip('"')
        if rating == 3.0:
            continue
        label = 1 if rating > 3 else 0
        texts.append(text)
        labels.append(label)
    return texts, labels


def load_all_datasets(data_dir, tokenizer, max_len):
    """
    Load sentiment review datasets for books, dvd, and music domains.

    Returns:
        dict: {
            "books": {"train": Dataset, "test": Dataset},
            "dvd": {"train": Dataset, "test": Dataset},
            "music": {"train": Dataset, "test": Dataset}
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
