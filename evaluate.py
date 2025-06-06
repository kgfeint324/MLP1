import os
import kss 
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer
from model_sub import EmotionClassifier as EmotionClassifier_sub, label_mapping
from model_major import EmotionClassifier as EmotionClassifier_major, fine_to_coarse, inv_label_mapping as inv_label_mapping_major

# Set device and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
project_path = "/content/drive/MyDrive/emotion_project"

# Load validation data from JSON and preprocess
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
                for s in kss.split_sentences(sentence.strip()): 
                    s = s.strip()
                    if s:
                        rows.append((s, emotion_code))
    return rows

# Evaluation function
def evaluate():
    val_path = os.path.join(project_path, "감성대화말뭉치(최종데이터)_Validation.json")
    data = load_validation_data(val_path)

    # Load sub (fine-grained) model
    model_sub = EmotionClassifier_sub("klue/bert-base").to(device)
    model_sub.load_state_dict(torch.load(f"{project_path}/best_model_sub.pt", map_location=device))
    model_sub.eval()

    # Load major (coarse-grained) model
    model_major = EmotionClassifier_major("klue/bert-base").to(device)
    model_major.load_state_dict(torch.load(f"{project_path}/best_model_major.pt", map_location=device))
    model_major.eval()

    true_sub, pred_sub = [], []
    true_major, pred_major = [], []

    # Perform predictions
    for sentence, true_code in data:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits_sub = model_sub(inputs["input_ids"], inputs["attention_mask"])
            logits_major = model_major(inputs["input_ids"], inputs["attention_mask"])

        pred_label_sub = torch.argmax(F.softmax(logits_sub, dim=1), dim=1).item()
        pred_label_major = torch.argmax(F.softmax(logits_major, dim=1), dim=1).item()

        true_sub.append(label_mapping[true_code])
        pred_sub.append(pred_label_sub)

        major_label = fine_to_coarse[true_code]  # e.g., "C3"
        true_major.append(int(major_label[1:]) - 1)  # "C3" → 2
        pred_major.append(pred_label_major)

    # Evaluation for sub (fine-grained) classification
    print("📊 Sub Classification (60 classes)")
    print(f"Accuracy: {accuracy_score(true_sub, pred_sub):.4f}")
    print(f"Micro F1 : {f1_score(true_sub, pred_sub, average='micro'):.4f}")
    print(f"Macro F1 : {f1_score(true_sub, pred_sub, average='macro'):.4f}")

    # Evaluation for major (coarse-grained) classification
    print("\n📊 Major Classification (6 classes)")
    print(f"Accuracy: {accuracy_score(true_major, pred_major):.4f}")
    print(f"Micro F1 : {f1_score(true_major, pred_major, average='micro'):.4f}")
    print(f"Macro F1 : {f1_score(true_major, pred_major, average='macro'):.4f}")
    print("\n[Classification Report - Major]")
    print(classification_report(true_major, pred_major, digits=4))

    # Confusion matrix for major classification
    cm = confusion_matrix(true_major, pred_major)
    plt.figure(figsize=(6, 5))
    labels = [inv_label_mapping_major[i] for i in range(6)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Major)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()