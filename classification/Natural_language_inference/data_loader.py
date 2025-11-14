import torch
from torch.utils.data import Dataset
from datasets import load_dataset

def load_xnli_datasets(data_dir, tokenizer, max_len):
    """
    Load XNLI French datasets using Hugging Face datasets library.

    Returns a dict with keys "train", "dev", "test" and InferenceDataset objects.
    """
    # Load XNLI dataset from Hugging Face with French config
    # Note: data_dir parameter is kept for compatibility but not used with HF datasets
    dataset = load_dataset("xnli", "fr")

    datasets = {}
    datasets["train"] = InferenceDataset(dataset['train'], tokenizer, max_len)
    datasets["dev"] = InferenceDataset(dataset['validation'], tokenizer, max_len)
    datasets["test"] = InferenceDataset(dataset['test'], tokenizer, max_len)

    return datasets

class InferenceDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset  # Hugging Face dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        sent1 = example['premise']
        sent2 = example['hypothesis']
        label = example['label']

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
