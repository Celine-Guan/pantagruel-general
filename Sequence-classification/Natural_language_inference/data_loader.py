import os
import torch
from torch.utils.data import Dataset

def parse_xnli_tsv(filepath, target_lang="fr"):
    texts = []
    labels = []
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    with open(filepath, "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        lang_idx = header.index("language")
        label_idx = header.index("gold_label")
        sent1_idx = header.index("sentence1")
        sent2_idx = header.index("sentence2")

        for line in f:
            cols = line.strip().split("\t")
            if len(cols) <= max(sent1_idx, sent2_idx, label_idx, lang_idx):
                continue

            lang = cols[lang_idx]
            label = cols[label_idx]
            if lang != target_lang or label not in label_map:
                continue

            sent1 = cols[sent1_idx]
            sent2 = cols[sent2_idx]

            texts.append((sent1, sent2))
            labels.append(label_map[label])

    return texts, labels

def parse_multinli_train(filepath):
    texts = []
    labels = []
    label_map = {"entailment": 0, "neutral": 1, "contradictory": 2}

    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) != 3:
                continue
            premise, hypo, label = cols
            if label not in label_map:
                continue
            texts.append((premise, hypo))
            labels.append(label_map[label])
    return texts, labels

def load_xnli_datasets(data_dir, tokenizer, max_len):
    """
    Load XNLI French datasets from TSV files:
    - multinli.train.fr.tsv (train)
    - xnli.dev.tsv (dev)
    - xnli.test.tsv (test)

    Returns a dict with keys "train", "dev", "test" and InferenceDataset objects.
    """
    datasets = {}

    train_path = os.path.join(data_dir, "multinli.train.fr.tsv")
    dev_path = os.path.join(data_dir, "xnli.dev.tsv")
    test_path = os.path.join(data_dir, "xnli.test.tsv")

    train_texts, train_labels = parse_multinli_train(train_path)
    dev_texts, dev_labels = parse_xnli_tsv(dev_path, target_lang="fr")
    test_texts, test_labels = parse_xnli_tsv(test_path, target_lang="fr")

    datasets["train"] = InferenceDataset(train_texts, train_labels, tokenizer, max_len)
    datasets["dev"] = InferenceDataset(dev_texts, dev_labels, tokenizer, max_len)
    datasets["test"] = InferenceDataset(test_texts, test_labels, tokenizer, max_len)

    return datasets

class InferenceDataset(Dataset):
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
