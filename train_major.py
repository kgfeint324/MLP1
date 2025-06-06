import os
import kss
import json
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler
from model2 import EmotionClassifier, label_mapping
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set project root path and tokenizer
project_path = "/content/drive/MyDrive/emotion_project"
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# Custom Dataset class for emotion classification
class EmotionDataset(Dataset):
    def __init__(self, data):
        self.sentences = data["Sentence"].tolist()  # List of sentences
        self.labels = data["Label"].tolist()        # Corresponding label indices

    def __len__(self):
        return len(self.sentences)  # Total number of samples

    def __getitem__(self, idx):
        # Tokenize each sentence
        inputs = tokenizer(self.sentences[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),          # Remove extra batch dimension
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])              # Convert label to tensor
        }

# Parse the original JSON dataset and return a cleaned DataFrame
def parse_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        # Extract the emotion type (e.g., 'anger', 'sadness', etc.)
        emotion_code = item.get("profile", {}).get("emotion", {}).get("type", "").lower()
        if emotion_code not in label_mapping:
            continue  # Skip unknown labels

        label = label_mapping[emotion_code]
        talk = item.get("talk", {}).get("content", {})

        for key, sentence in talk.items():
            if key.startswith("HS"):  # Only consider speaker's sentence
                sentence = sentence.strip()
                if not sentence:
                    continue
                # Split into sub-sentences
                for s in kss.split_sentences(sentence):
                    s = s.strip()
                    if len(s) >= 5:
                        rows.append({"Sentence": s, "Label": label})
    return pd.DataFrame(rows)

# Main training function
def train_major(start_epoch=0, total_epochs=10, resume_from=None):
    # Load and preprocess dataset
    file_path = f"{project_path}/감성대화말뭉치(최종데이터)_Training.json"
    df = parse_json_dataset(file_path)

    # Split into training and validation sets (stratified by label)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["Label"])
    train_dataset = EmotionDataset(train_df)
    val_dataset = EmotionDataset(val_df)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize the model
    model = EmotionClassifier("klue/bert-base").to(device)

    # Optionally resume from checkpoint
    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"🔁 Resumed model from {resume_from}")

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=total_epochs * len(train_loader)
    )
    loss_fn = nn.CrossEntropyLoss()  # Classification loss

    best_f1 = 0        # Best F1 score to track model improvement
    patience = 3       # Early stopping patience
    counter = 0        # Counts epochs without improvement

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")  # Progress bar

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
            loop.set_postfix(loss=loss.item())  # Display current loss in tqdm

        print(f"✅ Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

        # Evaluate on validation set
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

        # Compute evaluation metrics
        acc = accuracy_score(all_labels, all_preds)
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        print(f"📊 Val Accuracy: {acc:.4f}, Micro F1: {f1_micro:.4f}, Macro F1: {f1_macro:.4f}")

        # Save checkpoint after every epoch
        torch.save(model.state_dict(), f"{project_path}/model_epoch_{epoch+1}.pt")

        # Save best model based on micro F1
        if f1_micro > best_f1:
            best_f1 = f1_micro
            torch.save(model.state_dict(), f"{project_path}/best_model_major.pt")
            print("✅ Best model updated!")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered!")
                break

# Run training when script is executed directly
if __name__ == "__main__":
    train_major()
