import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import numpy as np

class ClinicalBERTICD10Model(nn.Module):
    def __init__(self, bert_model_name: str, num_labels: int, tfidf_feature_size: int):
        super(ClinicalBERTICD10Model, self).__init__()
        # Load Clinical BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        # Combine TF-IDF + BERT embeddings
        combined_input_size = hidden_size + tfidf_feature_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, tfidf_features):
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token embedding or pooler_output
        # Depending on the model, use pooler_output or mean pooling
        # For simplicity:
        pooled_output = outputs.pooler_output  # shape [batch_size, hidden_size]
        
        # Concatenate TF-IDF features
        combined = torch.cat((pooled_output, tfidf_features), dim=1)
        
        logits = self.classifier(combined)
        return logits

class ICD10Predictor:
    def __init__(self, 
                 bert_model_name="emilyalsentzer/Bio_ClinicalBERT", 
                 num_labels=100, 
                 tfidf_feature_size=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model = ClinicalBERTICD10Model(bert_model_name, num_labels, tfidf_feature_size).to(self.device)
        self.model.eval()

    def predict(self, text, tfidf_vector):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        tfidf_tensor = torch.tensor(tfidf_vector.toarray(), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, tfidf_tensor)
        
        # For multi-label ICD-10 codes: sigmoid and thresholding
        probs = torch.sigmoid(logits)
        # Here we assume a threshold. 
            #URGENT: LOOK INTO  tune or apply top-k selection.
        threshold = 0.5
        predicted_labels = (probs > threshold).nonzero(as_tuple=False)[:, 1].cpu().numpy()
        return predicted_labels
