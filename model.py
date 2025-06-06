
import torch.nn as nn
from transformers import BertModel

# 감정 코드 매핑 (E10 ~ E69 → 0 ~ 59)
label_mapping = {
    f"e{10+i}": i for i in range(60)
}
inv_label_mapping = {v: k.upper() for k, v in label_mapping.items()}

class EmotionClassifier(nn.Module):
    def __init__(self, pretrained_model_name="klue/bert-base"):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Sequential(
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Dropout(0.1),
    )
        self.classifier = nn.Linear(self.bert.config.hidden_size, 60)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
