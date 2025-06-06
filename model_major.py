import torch.nn as nn
from transformers import BertModel

# Mapping fine-grained emotion labels (E##) to coarse-grained categories (A##)
# For example: e10~e19 → a01 (e.g., anger group)
fine_to_coarse = {
    **dict.fromkeys([f"e{10+i:02d}" for i in range(10)], "a01"),
    **dict.fromkeys([f"e{20+i:02d}" for i in range(10)], "a02"),
    **dict.fromkeys([f"e{30+i:02d}" for i in range(10)], "a03"),
    **dict.fromkeys([f"e{40+i:02d}" for i in range(10)], "a04"),
    **dict.fromkeys([f"e{50+i:02d}" for i in range(10)], "a05"),
    **dict.fromkeys([f"e{60+i:02d}" for i in range(10)], "a06")
}

# Convert fine labels (e.g., "e12") to integer class index (0~5)
label_mapping = {k: int(v[1:]) - 1 for k, v in fine_to_coarse.items()}

# Inverse mapping: map index back to coarse category label
inv_label_mapping = {
    0: "A01",  # 분노
    1: "A02",  # 슬픔
    2: "A03",  # 불안
    3: "A04",  # 상처
    4: "A05",  # 당황
    5: "A06",  # 기쁨
}


class EmotionClassifier(nn.Module):
    """
    BERT-based emotion classifier for 6 coarse emotion categories (A01~A06).
    """
    def __init__(self, pretrained_model_name="klue/bert-base"):
        super(EmotionClassifier, self).__init__()

        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # Regularization and activation
        self.dropout = nn.Sequential(
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Final classifier: output size is 6 (number of coarse categories)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            len(set(label_mapping.values()))  # = 6
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: tokenized sentence inputs
            attention_mask: attention mask to ignore padding tokens

        Returns:
            logits: raw scores for 6 emotion categories
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
