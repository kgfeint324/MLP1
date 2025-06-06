import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler
from model2 import EmotionClassifier, label_mapping
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import kss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_path = "/content/drive/MyDrive/emotion_project"
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

class EmotionDataset(Dataset):
    def __init__(self, data):
        self.sentences = data["Sentence"].tolist()
        self.labels = data["Label"].tolist()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        inputs = tokenizer(self.sentences[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

def parse_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        emotion_code = item.get("profile", {}).get("emotion", {}).get("type", "").lower()
        if emotion_code not in label_mapping:
            continue
        label = label_mapping[emotion_code]
        talk = item.get("talk", {}).get("content", {})
        for key, sentence in talk.items():
            if key.startswith("HS"):
                sentence = sentence.strip()
                if not sentence:
                    continue
                for s in kss.split_sentences(sentence):
                    s = s.strip()
                    if len(s) >= 5:
                        rows.append({"Sentence": s, "Label": label})
    return pd.DataFrame(rows)

def train2(start_epoch=0, total_epochs=10, resume_from=None):
    file_path = f"{project_path}/감성대화말뭉치(최종데이터)_Training.json"
    df = parse_json_dataset(file_path)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["Label"])
    train_dataset = EmotionDataset(train_df)
    val_dataset = EmotionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = EmotionClassifier("klue/bert-base").to(device)
    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"🔁 Resumed model from {resume_from}")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=total_epochs * len(train_loader)
    )
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = 0
    patience = 3
    counter = 0

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"✅ Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        print(f"📊 Val Accuracy: {acc:.4f}, Micro F1: {f1_micro:.4f}, Macro F1: {f1_macro:.4f}")

        torch.save(model.state_dict(), f"{project_path}/model_epoch_{epoch+1}.pt")

        if f1_micro > best_f1:
            best_f1 = f1_micro
            torch.save(model.state_dict(), f"{project_path}/best_model.pt")
            print("✅ Best model updated!")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered!")
                break

if __name__ == "__main__":
    train()


