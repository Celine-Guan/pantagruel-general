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
            sent1, sent2, label = cols
            if label not in label_map:
                continue
            texts.append((sent1, sent2))
            labels.append(label_map[label])
    return texts, labels

