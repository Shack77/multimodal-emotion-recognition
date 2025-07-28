import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from src.model import LSTMEmotionClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class AudioDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = []
        self.label_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5
        }

        for emotion in os.listdir(feature_dir):
            emotion_path = os.path.join(feature_dir, emotion)
            if not os.path.isdir(emotion_path):
                continue
            for file in os.listdir(emotion_path):
                if file.endswith(".npy"):
                    path = os.path.join(emotion_path, file)
                    label = self.label_map[emotion]
                    self.files.append((path, label))

        print(f"Loaded {len(self.files)} audio feature files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            mfcc = np.load(path)  # Shape: (time, n_mfcc)
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-6) # Normalize

        # --- Pad or truncate to fixed length ---
            fixed_len = 100
            if mfcc.shape[0] < fixed_len:
                pad_width = fixed_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:fixed_len, :]

        except Exception as e:
            print(f"Error loading {path}: {e}")
            mfcc = np.zeros((100, 40), dtype=np.float32)  # fallback

        return torch.tensor(mfcc, dtype=torch.float32), label

def train_model(
    feature_dir="data/features/audio_mfcc",
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-5,
    num_epochs=40,
    patience=5,
    model_save_path="best_emotion_model.pt"
):
    # Load full dataset
    full_dataset = AudioDataset(feature_dir)

    # Split: 60% train, 20% val, 20% test
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LSTMEmotionClassifier(input_dim=120)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1)
            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

        train_acc = correct / len(train_dataset)
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_val = x_val.squeeze(1)
                out_val = model(x_val)
                loss = loss_fn(out_val, y_val)
                val_loss += loss.item()
                val_correct += (out_val.argmax(1) == y_val).sum().item()

        val_acc = val_correct / len(val_dataset)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(model_save_path))
    test_acc, all_preds = test_model(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.2f}")

    df = pd.DataFrame({"predicted_label": all_preds})
    df.to_csv("test_predictions.csv", index=False)
    print("Test predictions saved to test_predictions.csv")


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_acc = correct / total if total > 0 else 0.0

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["angry", "disgust", "fear", "happy", "neutral", "sad"]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["angry", "disgust", "fear", "happy", "neutral", "sad"], yticklabels=["angry", "disgust", "fear", "happy", "neutral", "sad"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    return test_acc, all_preds
