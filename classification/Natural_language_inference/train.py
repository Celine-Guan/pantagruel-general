import os
import yaml
import torch
import logging
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler


from common import (
    set_seed,
    init_logger,
    EarlyStopping,
    train_one_epoch,
    eval_model,
    NLIClassificationModel
)

from .data_loader import load_xnli_datasets




def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training_single_seed(cfg, seed, mlflow_experiment_name="NLI"):
    set_seed(seed)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    
    # for some local models, the tokenizer and the backbone model may in different path
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_name"], token=hf_token, trust_remote_code=True, use_fast=False
    )
    datasets = load_xnli_datasets(cfg["data_dir"], tokenizer, cfg["max_seq_length"])

    train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(datasets["dev"], batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False)

    best_valid_acc = 0
    best_lr = None
    best_model_state = None

    for lr in cfg["learning_rates"]:
        lr = float(lr)
        logging.info(f"\nTraining with learning rate {lr}...")
        
        # Start MLflow run for this learning rate
        with mlflow.start_run(run_name=f"XNLI_seed{seed}_lr{lr}", nested=True):
            # Log parameters
            mlflow.log_param("seed", seed)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", cfg["batch_size"])
            mlflow.log_param("num_epochs", cfg["num_epochs"])
            mlflow.log_param("max_seq_length", cfg["max_seq_length"])
            mlflow.log_param("num_classes", cfg["num_classes"])
            mlflow.log_param("dropout", cfg["dropout"])
            mlflow.log_param("early_stopping_patience", cfg["early_stopping_patience"])
            mlflow.log_param("model_name", cfg["model_name"])
            mlflow.log_param("n_train", len(datasets["train"]))
            mlflow.log_param("n_valid", len(datasets["dev"]))
            mlflow.log_param("n_test", len(datasets["test"]))
            
            model = NLIClassificationModel(
                cfg["model_name"],
                num_classes=cfg["num_classes"],
                dropout_rate=cfg["dropout"],
                token=hf_token
            ).to(device)

            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = cfg["num_epochs"] * len(train_loader)
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            criterion = torch.nn.CrossEntropyLoss()
            early_stopping = EarlyStopping(patience=cfg["early_stopping_patience"], verbose=True)

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
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_f1_macro", train_metrics['f1_macro'], step=epoch)
                mlflow.log_metric("train_accuracy", train_metrics['accuracy'], step=epoch)
                mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step=epoch)
                mlflow.log_metric("val_f1_macro", val_metrics['f1_macro'], step=epoch)
                mlflow.log_metric("val_precision_macro", val_metrics['precision_macro'], step=epoch)
                mlflow.log_metric("val_recall_macro", val_metrics['recall_macro'], step=epoch)

                if val_metrics['accuracy'] > best_valid_acc:
                    best_valid_acc = val_metrics['accuracy']
                    best_lr = lr
                    best_model_state = model.state_dict()

                # early_stopping(val_metrics['loss'])
                # if early_stopping.early_stop:
                #     logging.info("Early stopping triggered.")
                #     mlflow.log_param("early_stopped_epoch", epoch + 1)
                #     break
            
            # Log final validation metrics
            mlflow.log_metric("final_val_accuracy", best_valid_acc)

    # Evaluate best model on test set
    logging.info(f"\nLoading best model (LR={best_lr}) for final test evaluation...")
    model.load_state_dict(best_model_state)
    test_metrics = eval_model(model, test_loader, device, criterion)
    
    # Log test metrics in a separate run for the best model
    with mlflow.start_run(run_name=f"XNLI_seed{seed}_BEST", nested=True):
        mlflow.log_param("seed", seed)
        mlflow.log_param("best_learning_rate", best_lr)
        mlflow.log_param("model_name", cfg["model_name"])
        
        # Log all test metrics
        mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
        mlflow.log_metric("test_f1_macro", test_metrics['f1_macro'])
        mlflow.log_metric("test_precision_macro", test_metrics['precision_macro'])
        mlflow.log_metric("test_recall_macro", test_metrics['recall_macro'])
        mlflow.log_metric("valid_accuracy", best_valid_acc)
        
        # Log per-class metrics
        for i, (f1, prec, rec, supp) in enumerate(zip(
            test_metrics['f1_per_class'],
            test_metrics['precision_per_class'],
            test_metrics['recall_per_class'],
            test_metrics['support_per_class']
        )):
            mlflow.log_metric(f"test_f1_class_{i}", f1)
            mlflow.log_metric(f"test_precision_class_{i}", prec)
            mlflow.log_metric(f"test_recall_class_{i}", rec)
            mlflow.log_metric(f"test_support_class_{i}", supp)
        
        # Optionally log the model
        mlflow.pytorch.log_model(model, "model")
    
    logging.info(f"Test Accuracy on XNLI French: {test_metrics['accuracy']:.4f}")
    logging.info(f"Test F1 (macro) on XNLI French: {test_metrics['f1_macro']:.4f}")
    logging.info(f"Test Precision (macro) on XNLI French: {test_metrics['precision_macro']:.4f}")
    logging.info(f"Test Recall (macro) on XNLI French: {test_metrics['recall_macro']:.4f}")

    return {
        "valid_acc": best_valid_acc,
        "test_acc": test_metrics['accuracy'],
        "test_f1": test_metrics['f1_macro'],
        "test_precision": test_metrics['precision_macro'],
        "test_recall": test_metrics['recall_macro'],
        "best_lr": best_lr
    }


def run_training(cfg):
    # Logging is already initialized in main.py
    
    # Set up MLflow
    mlflow_tracking_uri = cfg.get("mlflow_tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    experiment_name = cfg.get("mlflow_experiment_name", "NLI")
    mlflow.set_experiment(experiment_name)
    
    logging.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    logging.info(f"MLflow experiment: {experiment_name}")
    
    # Handle seeds - support both single seed and list of seeds
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    if not isinstance(seeds, list):
        seeds = [seeds]
    
    logging.info(f"Running training with {len(seeds)} seed(s): {seeds}")
    
    # Collect results across all seeds
    all_results = []
    
    # Start parent MLflow run for all seeds
    with mlflow.start_run(run_name=f"MultiSeed_XNLI_{len(seeds)}seeds"):
        # Log experiment-level parameters
        mlflow.log_param("num_seeds", len(seeds))
        mlflow.log_param("seeds", str(seeds))
        mlflow.log_param("model_name", cfg["model_name"])
        mlflow.log_param("batch_size", cfg["batch_size"])
        mlflow.log_param("num_epochs", cfg["num_epochs"])
        
        for seed_idx, seed in enumerate(seeds):
            logging.info(f"\n{'='*80}")
            logging.info(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
            logging.info(f"{'='*80}\n")
            
            results = run_training_single_seed(cfg, seed, mlflow_experiment_name=experiment_name)
            all_results.append(results)
    
        # Calculate aggregated statistics across seeds
        if len(seeds) > 1:
            logging.info(f"\n{'='*80}")
            logging.info("AGGREGATED RESULTS ACROSS SEEDS")
            logging.info(f"{'='*80}\n")
            
            # For NLI, we only have one result (not per-dataset like sentiment)
            # Collect metrics across seeds
            test_accs = [r['test_acc'] for r in all_results]
            test_f1s = [r['test_f1'] for r in all_results]
            test_precisions = [r['test_precision'] for r in all_results]
            test_recalls = [r['test_recall'] for r in all_results]
            
            # Calculate mean and std
            aggregated_results = {
                'test_acc_mean': np.mean(test_accs),
                'test_acc_std': np.std(test_accs),
                'test_f1_mean': np.mean(test_f1s),
                'test_f1_std': np.std(test_f1s),
                'test_precision_mean': np.mean(test_precisions),
                'test_precision_std': np.std(test_precisions),
                'test_recall_mean': np.mean(test_recalls),
                'test_recall_std': np.std(test_recalls),
                'individual_results': all_results
            }
            
            # Log aggregated metrics to MLflow (in parent run)
            mlflow.log_metric("test_acc_mean", aggregated_results['test_acc_mean'])
            mlflow.log_metric("test_acc_std", aggregated_results['test_acc_std'])
            mlflow.log_metric("test_f1_mean", aggregated_results['test_f1_mean'])
            mlflow.log_metric("test_f1_std", aggregated_results['test_f1_std'])
            mlflow.log_metric("test_precision_mean", aggregated_results['test_precision_mean'])
            mlflow.log_metric("test_precision_std", aggregated_results['test_precision_std'])
            mlflow.log_metric("test_recall_mean", aggregated_results['test_recall_mean'])
            mlflow.log_metric("test_recall_std", aggregated_results['test_recall_std'])
            
            # Log aggregated results
            logging.info(f"Test Accuracy:  {aggregated_results['test_acc_mean']:.4f} ± {aggregated_results['test_acc_std']:.4f}")
            logging.info(f"Test F1:        {aggregated_results['test_f1_mean']:.4f} ± {aggregated_results['test_f1_std']:.4f}")
            logging.info(f"Test Precision: {aggregated_results['test_precision_mean']:.4f} ± {aggregated_results['test_precision_std']:.4f}")
            logging.info(f"Test Recall:    {aggregated_results['test_recall_mean']:.4f} ± {aggregated_results['test_recall_std']:.4f}")
            
            # Log individual seed results for reference
            logging.info(f"\nIndividual seed results:")
            for seed_idx, seed in enumerate(seeds):
                logging.info(f"  Seed {seed}: Acc={test_accs[seed_idx]:.4f}, F1={test_f1s[seed_idx]:.4f}, Prec={test_precisions[seed_idx]:.4f}, Rec={test_recalls[seed_idx]:.4f}")
            
            return aggregated_results
        else:
            # Single seed - return results directly
            return all_results[0]
