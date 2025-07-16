import os
import yaml
import torch
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler

from common import (
    set_seed,
    init_logger,
    EarlyStopping,
    train_one_epoch,
    eval_model,
    NLIClassificationModel
)

from .data_loader import load_pawsx_datasets


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training(cfg):
    set_seed(cfg["seed"])
    init_logger()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # for some local models, the tokenizer and the backbone model may in different path
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    datasets = load_pawsx_datasets(cfg["data_dir"], tokenizer, cfg["max_seq_length"])

    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(datasets["dev"], batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False)

    best_valid_acc = 0
    best_lr = None
    best_model_state = None

    for lr in cfg["learning_rates"]:
        lr = float(lr)
        logging.info(f"\nTraining with learning rate {lr}...")
        model = NLIClassificationModel(
            cfg["model_name"],
            num_classes=cfg["num_classes"],
            dropout_rate=cfg["dropout"]
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = cfg["num_epochs"] * len(train_loader)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = torch.nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=cfg["early_stopping_patience"], verbose=True)

        for epoch in range(cfg["num_epochs"]):
            logging.info(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            logging.info(f"Train loss: {train_loss:.4f}")

            valid_acc = eval_model(model, valid_loader, device)
            logging.info(f"Validation accuracy: {valid_acc:.4f}")

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_lr = lr
                best_model_state = model.state_dict()

            early_stopping(valid_acc)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

    # Evaluate best model on test set
    logging.info(f"\nLoading best model (LR={best_lr}) for final test evaluation...")
    model.load_state_dict(best_model_state)
    test_acc = eval_model(model, test_loader, device)
    logging.info(f"Test accuracy on PAWS-X French: {test_acc:.4f}")

    return {
        "valid_acc": best_valid_acc,
        "test_acc": test_acc,
        "best_lr": best_lr
    }
