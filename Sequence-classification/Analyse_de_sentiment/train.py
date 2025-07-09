import yaml
import os
import torch
import logging
from transformers import AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import DataLoader, random_split

from common import set_seed, init_logger, EarlyStopping, train_one_epoch, eval_model, SequenceClassificationModel
from .data_loader import parse_review_file, ReviewDataset


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training(cfg):
    set_seed(cfg["seed"])
    init_logger()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    results = {}

    for dataset_name in cfg["datasets"]:
        logging.info(f"\nRunning training on dataset: {dataset_name}")
        data_dir = os.path.join(cfg["data_dir"], dataset_name)
        train_texts, train_labels = parse_review_file(os.path.join(data_dir, "train.review"))
        test_texts, test_labels = parse_review_file(os.path.join(data_dir, "test.review"))

        # Train/valid split
        n_valid = int(len(train_texts) * cfg["valid_ratio"])
        n_train = len(train_texts) - n_valid
        full_dataset = ReviewDataset(train_texts, train_labels, tokenizer, cfg["max_seq_length"])
        train_dataset, valid_dataset = random_split(full_dataset, [n_train, n_valid])
        test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, cfg["max_seq_length"])

        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg["batch_size"])
        test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"])

        best_valid_acc = 0
        best_lr = None
        best_model_state = None

        for lr in cfg["learning_rates"]:
            lr = float(lr)
            logging.info(f"\nTraining with learning rate {lr}")
            model = SequenceClassificationModel(cfg["model_name"], num_classes=cfg["num_classes"], dropout_rate=cfg["dropout"]).to(device)
            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = cfg["num_epochs"] * len(train_loader)
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            criterion = torch.nn.CrossEntropyLoss()
            early_stopping = EarlyStopping(patience=cfg["early_stopping_patience"], verbose=True)

            for epoch in range(cfg["num_epochs"]):
                logging.info(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
                train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
                logging.info(f"Train Loss: {train_loss:.4f}")
                val_acc = eval_model(model, valid_loader, device)
                logging.info(f"Validation Accuracy: {val_acc:.4f}")

                if val_acc > best_valid_acc:
                    best_valid_acc = val_acc
                    best_lr = lr
                    best_model_state = model.state_dict()

                early_stopping(val_acc)
                if early_stopping.early_stop:
                    logging.info("Early stopping triggered.")
                    break

        # choose the best performing model on validation set to test
        logging.info(f"\n Loading best model (LR={best_lr}) for test evaluation on {dataset_name}")
        model.load_state_dict(best_model_state)
        test_acc = eval_model(model, test_loader, device)
        logging.info(f"Test Accuracy on {dataset_name}: {test_acc:.4f}")
        results[dataset_name] = {"valid_acc": best_valid_acc, "test_acc": test_acc, "best_lr": best_lr}

    return results
