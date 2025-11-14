import torch
import torch.nn as nn
from transformers import AutoModel

class SequenceClassificationModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, dropout_rate: float = 0.1, token: str = None, add_pooling_layer: bool = True, use_pooler: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            add_pooling_layer=add_pooling_layer,
        )
        self.use_pooler = use_pooler
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        logits = self.classifier(features)
        return logits

class NLIClassificationModel(nn.Module):
    """
    For natural language inference, using the FLUE classification head:
    Dropout → Linear
    """
    def __init__(self, model_name: str, num_classes: int = 3, dropout_rate: float = 0.1, token: str = None, add_pooling_layer: bool = True, use_pooler: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            add_pooling_layer=add_pooling_layer,
        )
        self.use_pooler = use_pooler
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(features))
        return logits
