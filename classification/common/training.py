from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_model(model, dataloader, device, criterion=None):
    model.eval()
    preds = []
    true_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
    
    # Calculate metrics
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, preds, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, preds, average='macro', zero_division=0
    )
    
    avg_loss = total_loss / len(dataloader) if criterion is not None else None
    
    # Return metrics as dictionary
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1.tolist(),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'support_per_class': support.tolist()
    }
