# -*- coding: utf-8 -*-
#working on streamlit to be a basic UI for the model

import streamlit as st
from data_preprocessing import ClinicalPreprocessor
from model import ICD10Predictor
import pickle
import torch

st.title("Clinical Note to ICD-10 Code Converter")

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

predictor = ICD10Predictor(num_labels=100, tfidf_feature_size=len(vectorizer.vocabulary_))
predictor.model.load_state_dict(torch.load("icd10_model.pt", map_location=torch.device('cpu')))

preprocessor = ClinicalPreprocessor()

user_input = st.text_area("Enter the clinical note here:")

if st.button("Predict ICD-10 Codes"):
    cleaned_note, entities = preprocessor.preprocess_text(user_input)
    tfidf_vec = vectorizer.transform([cleaned_note])
    predicted_codes = predictor.predict(cleaned_note, tfidf_vec)
    # In a real scenario, map indices back to ICD-10 code strings
    st.write("Predicted ICD-10 codes:", predicted_codes)

#PIPA - check regulationsupto 50k based on ppl, submit a report to attorney general