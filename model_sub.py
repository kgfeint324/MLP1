import torch.nn as nn
from transformers import BertModel

# emotion code mapping ("e10" to 0, ..., "e69" to 59)
label_mapping = {
    f"e{10+i}": i for i in range(60)
}

# Inverse mapping: map label index back to original code (e.g., 0 → "E10")
inv_label_mapping = {v: k.upper() for k, v in label_mapping.items()}


class EmotionClassifier(nn.Module):
    """
    Emotion classification model based on a pretrained BERT.
    Output: 60 emotion categories.
    """
    def __init__(self, pretrained_model_name="klue/bert-base"):
        super(EmotionClassifier, self).__init__()

        # Load pretrained BERT model from Huggingface
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # Dropout and activation for regularization and non-linearity
        self.dropout = nn.Sequential(
            nn.Dropout(0.3),   # First dropout (30%)
            nn.ReLU(),         # ReLU activation
            nn.Dropout(0.1),   # Second dropout (10%)
        )

        # Final classification layer (hidden_size → 60 emotion classes)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 60)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids: tokenized input sentences
            attention_mask: masks to ignore padding tokens

        Returns:
            logits: raw output scores for each class (before softmax)
        """
        # Pass inputs through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get [CLS] token representation (pooled output)
        pooled_output = outputs.pooler_output

        # Apply dropout and ReLU
        pooled_output = self.dropout(pooled_output)

        # Compute class logits
        logits = self.classifier(pooled_output)

        return logits
