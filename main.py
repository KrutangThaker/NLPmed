from preprocessData import ClinicalPreprocessor
from modelfin import ICD10Predictor
import pickle

# Load from preprocessData.py here
# Assume my trained model and vectorizer js working

if __name__ == "__main__":
    # Example usage using alex's examples
    # Load pre-fitted TF-IDF vectorizer and model weights
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    predictor = ICD10Predictor(num_labels=100, tfidf_feature_size=len(vectorizer.vocabulary_))
    predictor.model.load_state_dict(torch.load("icd10_model.pt"))
    
    preprocessor = ClinicalPreprocessor()
    user_note = "Patient presented with chest pain and shortness of breath..."
    cleaned_note, entities = preprocessor.preprocess_text(user_note)
    tfidf_vec = vectorizer.transform([cleaned_note])
    predicted_codes = predictor.predict(cleaned_note, tfidf_vec)
    print("Predicted ICD-10 codes:", predicted_codes)
