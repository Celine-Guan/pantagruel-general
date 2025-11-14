import yaml
import os
import torch
import logging
import numpy as np
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from common import set_seed, init_logger, train_one_epoch, eval_model, SequenceClassificationModel
from .data_loader import parse_review_file, ReviewDataset



def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training_single_seed(cfg, seed, mlflow_experiment_name="Sentiment_Analysis"):
    set_seed(seed)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    
    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    
    # Load tokenizer with token
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], token=hf_token, trust_remote_code=True)

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
        best_lr_run_id = None

        for lr in cfg["learning_rates"]:
            lr = float(lr)
            logging.info(f"\nTraining with learning rate {lr}")

            model = SequenceClassificationModel(cfg["model_name"], num_classes=cfg["num_classes"], dropout_rate=cfg["dropout"], token=hf_token, add_pooling_layer=cfg["add_pooling_layer"], use_pooler=cfg["use_pooler"]).to(device)
            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = cfg["num_epochs"] * len(train_loader)
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            criterion = torch.nn.CrossEntropyLoss()
            # early_stopping = EarlyStopping(patience=cfg["early_stopping_patience"], verbose=True)

            examples_seen = 0
            for epoch in range(cfg["num_epochs"]):
                logging.info(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
                
                # Training
                train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
                train_metrics = eval_model(model, train_loader, device, criterion)
                
                # Validation
                val_metrics = eval_model(model, valid_loader, device, criterion)
                
                # Log to console
                logging.info(f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1_macro']:.4f}")
                logging.info(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
                
                # Use cumulative examples seen as step
                examples_seen += len(train_loader.dataset)

                if val_metrics['accuracy'] > best_valid_acc:
                    best_valid_acc = val_metrics['accuracy']
                    best_lr = lr
                    best_model_state = model.state_dict()

            # Log final validation metrics
            logging.info(f"Best validation accuracy so far: {best_valid_acc:.4f}")

        # choose the best performing model on validation set to test
        logging.info(f"\n Loading best model (LR={best_lr}) for test evaluation on {dataset_name}")
        model.load_state_dict(best_model_state)
        test_metrics = eval_model(model, test_loader, device, criterion)
        
        # Print test metrics
        logging.info(f"Test Accuracy on {dataset_name}: {test_metrics['accuracy']:.4f}")
        logging.info(f"Test F1 (macro) on {dataset_name}: {test_metrics['f1_macro']:.4f}")
        logging.info(f"Test Precision (macro) on {dataset_name}: {test_metrics['precision_macro']:.4f}")
        logging.info(f"Test Recall (macro) on {dataset_name}: {test_metrics['recall_macro']:.4f}")
        
        results[dataset_name] = {
            "valid_acc": best_valid_acc, 
            "test_acc": test_metrics['accuracy'],
            "test_f1": test_metrics['f1_macro'],
            "test_precision": test_metrics['precision_macro'],
            "test_recall": test_metrics['recall_macro'],
            "best_lr": best_lr
        }
        logging.info(f"Test Accuracy on {dataset_name}: {test_metrics['accuracy']:.4f}")
        logging.info(f"Test F1 (macro) on {dataset_name}: {test_metrics['f1_macro']:.4f}")
        logging.info(f"Test Precision (macro) on {dataset_name}: {test_metrics['precision_macro']:.4f}")
        logging.info(f"Test Recall (macro) on {dataset_name}: {test_metrics['recall_macro']:.4f}")
        logging.info(f"Test F1 per class on {dataset_name}: {test_metrics['f1_per_class']}")
        logging.info(f"Test Precision per class on {dataset_name}: {test_metrics['precision_per_class']}")
        logging.info(f"Test Recall per class on {dataset_name}: {test_metrics['recall_per_class']}")
        logging.info(f"Test Support per class on {dataset_name}: {test_metrics['support_per_class']}")
    
    return results


def run_training(cfg):
    # Logging is already initialized in main.py
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    if not isinstance(seeds, list):
        seeds = [seeds]

    logging.info(f"Running training with {len(seeds)} seed(s): {seeds}")

    all_results = []
    for seed_idx, seed in enumerate(seeds):
        logging.info(f"\n{'='*80}")
        logging.info(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        logging.info(f"{'='*80}\n")
        results = run_training_single_seed(cfg, seed)
        all_results.append(results)

    if len(seeds) > 1:
        logging.info(f"\n{'='*80}")
        logging.info("AGGREGATED RESULTS ACROSS SEEDS")
        logging.info(f"{'='*80}\n")

        dataset_names = list(all_results[0].keys())
        aggregated_results = {}
        for dataset_name in dataset_names:
            logging.info(f"\nDataset: {dataset_name}")
            logging.info("-" * 60)

            test_accs = [r[dataset_name]['test_acc'] for r in all_results]
            test_f1s = [r[dataset_name]['test_f1'] for r in all_results]
            test_precisions = [r[dataset_name]['test_precision'] for r in all_results]
            test_recalls = [r[dataset_name]['test_recall'] for r in all_results]

            aggregated_results[dataset_name] = {
                'test_acc_mean': np.mean(test_accs),
                'test_acc_std': np.std(test_accs),
                'test_f1_mean': np.mean(test_f1s),
                'test_f1_std': np.std(test_f1s),
                'test_precision_mean': np.mean(test_precisions),
                'test_precision_std': np.std(test_precisions),
                'test_recall_mean': np.mean(test_recalls),
                'test_recall_std': np.std(test_recalls),
                'individual_results': [r[dataset_name] for r in all_results]
            }

            logging.info(f"Test Accuracy:  {aggregated_results[dataset_name]['test_acc_mean']:.4f} ± {aggregated_results[dataset_name]['test_acc_std']:.4f}")
            logging.info(f"Test F1:        {aggregated_results[dataset_name]['test_f1_mean']:.4f} ± {aggregated_results[dataset_name]['test_f1_std']:.4f}")
            logging.info(f"Test Precision: {aggregated_results[dataset_name]['test_precision_mean']:.4f} ± {aggregated_results[dataset_name]['test_precision_std']:.4f}")
            logging.info(f"Test Recall:    {aggregated_results[dataset_name]['test_recall_mean']:.4f} ± {aggregated_results[dataset_name]['test_recall_std']:.4f}")

        return aggregated_results
    else:
        return all_results[0]
