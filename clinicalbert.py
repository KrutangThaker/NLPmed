import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score

from model import ClinicalBERTICD10Model
from data_preprocessing import ClinicalPreprocessor

class ICD10Dataset(Dataset):
    """
    Dataset class for ICD-10 classification.
    Expects a DataFrame with columns: 'text', and multiple columns for each ICD-10 code label (binary).
    """
    def __init__(self, df, tokenizer, vectorizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.max_length = max_length
        self.texts = df['text'].tolist()
        # Assuming labels are multi-label binary columns
        self.label_cols = [col for col in df.columns if col.startswith('ICD_')]
        self.labels = df[self.label_cols].values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length, 
            padding='max_length'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # TF-IDF features
        tfidf_vec = self.vectorizer.transform([text])
        tfidf_features = torch.tensor(tfidf_vec.toarray(), dtype=torch.float32)
        #print(tfidf_features.shape)
        #print("debug 16, torch,tensor isnt working")

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, tfidf_features, labels

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, tfidf_features, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        tfidf_features = tfidf_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, tfidf_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, tfidf_features, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tfidf_features = tfidf_features.to(device)
            labels = labels.numpy()

            logits = model(input_ids, attention_mask, tfidf_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute multi-label metrics
    # For multi-label, you might consider 'micro' averages:
    f1 = f1_score(all_labels, all_preds, average='micro')
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    return f1, precision, recall

def find_optimal_thresholds(model, dataloader, device):
    """
    Simple approach to find optimal threshold per label:
    We try a small set of thresholds and pick the best one for each label 
    based on F1-score. This could be expensive for large label sets.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, tfidf_features, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tfidf_features = tfidf_features.to(device)
            labels = labels.numpy()
            
            logits = model(input_ids, attention_mask, tfidf_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # We'll assume all_probs: [num_samples, num_labels]
    num_labels = all_probs.shape[1]
    thresholds = np.linspace(0.1, 0.9, 9)
    best_thresholds = []
    for label_idx in range(num_labels):
        best_f1 = 0
        best_thr = 0.5
        for thr in thresholds:
            preds = (all_probs[:, label_idx] > thr).astype(int)
            f1 = f1_score(all_labels[:, label_idx], preds, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        best_thresholds.append(best_thr)
    return best_thresholds
    #print("debug: 42, threshold is erroring"); URGENT

if __name__ == "__main__":
    # Hyperparameters
    bert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    num_labels = 100  # adjust based on your dataset
    epochs = 3
    batch_size = 8
    learning_rate = 2e-5
    max_length = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    #hypothetically, this is where we woud load our data from a CSV or other source
    train_df = pd.read_csv("data/train_data.csv")  #ive coded so columns: text, ICD_code columns
    val_df = pd.read_csv("data/val_data.csv")

    # Preprocess & TF-IDF
    preprocessor = ClinicalPreprocessor()
    # Fit TF-IDF on training texts
    train_texts = train_df['text'].tolist()
    preprocessor.fit_tfidf(train_texts)
    vectorizer = preprocessor.vectorizer

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # Create Datasets and Dataloaders
    train_dataset = ICD10Dataset(train_df, tokenizer, vectorizer, max_length=max_length)
    val_dataset = ICD10Dataset(val_df, tokenizer, vectorizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    tfidf_feature_size = len(vectorizer.vocabulary_)
    model = ClinicalBERTICD10Model(bert_model_name, num_labels, tfidf_feature_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop with validation step
    best_f1 = 0.0
    best_model_path = "icd10_model_best.pt"

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        f1, precision, recall = validate(model, val_loader, device, threshold=0.5)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)

    #URGENT: NOW DONE TRAIING, ret load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Optional: Threshold tuning on validation set
    best_thresholds = find_optimal_thresholds(model, val_loader, device)
    with open("best_thresholds.pkl", "wb") as f:
        pickle.dump(best_thresholds, f)

    print("Training complete. Best F1:", best_f1)

 
    #print("Debug29, finished training but best_thresholds.pkl not found")
    print("Optimal thresholds per label saved to best_thresholds.pkl")
