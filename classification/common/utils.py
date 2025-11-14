import os
import logging
import random
import numpy as np
import torch
import yaml

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience: int = 3, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def init_logger(log_file: str = None, log_level=logging.INFO):
    """
    Initialize logging configuration. Safe to call multiple times.
    """
    # Get root logger
    logger = logging.getLogger()
    
    # If logger already has handlers, remove them to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set logging level
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:  
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
