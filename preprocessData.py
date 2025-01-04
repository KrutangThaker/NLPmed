import spacy
import medspacy
from sklearn.feature_extraction.text import TfidfVectorizer

class ClinicalPreprocessor:
    def __init__(self, model_name="en_core_web_sm"):
        # Load a spaCy model. For better results, consider a clinical model.
        self.nlp = spacy.load(model_name)
        self.vectorizer = None

    def preprocess_text(self, text):
        doc = self.nlp(text)
        # Basic cleaning: remove stopwords, punctuation
        # You can add section splitting, negation detection from medspaCy pipeline if configured.
        cleaned_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
        return cleaned_text, [(ent.text, ent.label_) for ent in doc.ents]

    def fit_tfidf(self, texts):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1,2), 
            min_df=2, 
            max_df=0.95
        )
        self.vectorizer.fit(texts)

    def transform_tfidf(self, texts):
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted.")
        return self.vectorizer.transform(texts)
