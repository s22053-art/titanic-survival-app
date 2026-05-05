import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'titanic_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        payload = pickle.load(f)
    model           = payload["model"]
    feature_columns = payload["feature_columns"]
    model_loaded    = True
except Exception as e:
    model_loaded    = False
    feature_columns = []

sex_d      = {0: 'Kobieta', 1: 'Mężczyzna'}
pclass_d   = {1: 'Pierwsza', 2: 'Druga', 3: 'Trzecia'}
embarked_d = {'S': 'Southampton (S)', 'C': 'Cherbourg (C)', 'Q': 'Queenstown (Q)'}

def set_background(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.62);
            border-radius: 16px;
            padding: 2rem 2.5rem;
        }}
        h1, h2, h3, h4, label, p, div {{
            color: #f0f0f0 !important;
        }}
        </style>""", unsafe_allow_html=True)
    except Exception:
        pass

def prepare_input(pclass, sex_val, age, sibsp, parch, fare, embarked, feature_columns):
    sex_str = "male" if sex_val == 1 else "female"
    raw = pd.DataFrame([{
        "Pclass": pclass, "Age": fl
