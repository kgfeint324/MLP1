
import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer
from model import EmotionClassifier as EmotionClassifier60, inv_label_mapping as inv_label_mapping60, label_mapping
from model2 import EmotionClassifier as EmotionClassifier6, fine_to_coarse, inv_label_mapping as inv_label_mapping6
import matplotlib.pyplot as plt
import seaborn as sns
import kss  # ✅ 문장 분리용 추가

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
project_path = "/content/drive/MyDrive/emotion_project"

def load_validation_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        emotion_code = item.get("profile", {}).get("emotion", {}).get("type", "").lower()
        if emotion_code not in label_mapping:
            continue
        talk = item.get("talk", {}).get("content", {})
        for key, sentence in talk.items():
            if key.startswith("HS") and sentence.strip():
                for s in kss.split_sentences(sentence.strip()):  # ✅ 문장 분리
                    s = s.strip()
                    if s:
                        rows.append((s, emotion_code))
    return rows

def evaluate():
    val_path = os.path.join(project_path, "감성대화말뭉치(최종데이터)_Validation.json")
    data = load_validation_data(val_path)

    model60 = EmotionClassifier60("klue/bert-base").to(device)
    model60.load_state_dict(torch.load(f"{project_path}/best_model0.pt", map_location=device))
    model60.eval()

    model6 = EmotionClassifier6("klue/bert-base").to(device)
    model6.load_state_dict(torch.load(f"{project_path}/best_model.pt", map_location=device))
    model6.eval()

    true_fine, pred_fine = [], []
    true_coarse, pred_coarse = [], []

    for sentence, true_code in data:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits60 = model60(inputs["input_ids"], inputs["attention_mask"])
            logits6 = model6(inputs["input_ids"], inputs["attention_mask"])
        pred_label60 = torch.argmax(F.softmax(logits60, dim=1), dim=1).item()
        pred_label6 = torch.argmax(F.softmax(logits6, dim=1), dim=1).item()

        pred_code60 = inv_label_mapping60[pred_label60]
        pred_code6 = inv_label_mapping6[pred_label6]

        true_fine.append(label_mapping[true_code])
        pred_fine.append(pred_label60)

        true_group = fine_to_coarse[true_code]
        true_coarse.append(int(true_group[1:]) - 1)
        pred_coarse.append(pred_label6)

    print("📊 소분류(60 클래스) 평가")
    print(f"Accuracy: {accuracy_score(true_fine, pred_fine):.4f}")
    print(f"Micro F1 : {f1_score(true_fine, pred_fine, average='micro'):.4f}")
    print(f"Macro F1 : {f1_score(true_fine, pred_fine, average='macro'):.4f}")

    print("\n📊 대분류(6 클래스) 평가")
    print(f"Accuracy: {accuracy_score(true_coarse, pred_coarse):.4f}")
    print(f"Micro F1 : {f1_score(true_coarse, pred_coarse, average='micro'):.4f}")
    print(f"Macro F1 : {f1_score(true_coarse, pred_coarse, average='macro'):.4f}")
    print("\n[Classification Report - Coarse]")
    print(classification_report(true_coarse, pred_coarse, digits=4))

    cm = confusion_matrix(true_coarse, pred_coarse)
    plt.figure(figsize=(6, 5))
    labels = [inv_label_mapping6[i] for i in range(6)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Coarse)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
