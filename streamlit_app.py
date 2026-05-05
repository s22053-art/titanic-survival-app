import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# ── Wczytanie modelu ──────────────────────────────────────────────────────────
try:
    payload = pickle.load(open('titanic_model.pkl', 'rb'))
    model           = payload["model"]
    feature_columns = payload["feature_columns"]
    model_loaded    = True
except FileNotFoundError:
    model_loaded = False

# ── Słowniki etykiet ──────────────────────────────────────────────────────────
sex_d      = {0: 'Kobieta', 1: 'Mężczyzna'}
pclass_d   = {1: 'Pierwsza', 2: 'Druga', 3: 'Trzecia'}
embarked_d = {'S': 'Southampton (S)', 'C': 'Cherbourg (C)', 'Q': 'Queenstown (Q)'}

# ── Tło z lokalnego pliku JPG ─────────────────────────────────────────────────
def set_background(image_path: str):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
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
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── Przygotowanie danych wejściowych ─────────────────────────────────────────
def prepare_input(pclass, sex_val, age, sibsp, parch, fare, embarked, feature_columns):
    sex_str = "male" if sex_val == 1 else "female"
    raw = pd.DataFrame([{
        "Pclass": pclass, "Age": float(age), "SibSp": sibsp,
        "Parch": parch, "Fare": float(fare), "Sex": sex_str, "Embarked": embarked,
    }])
    raw["Fare"] = raw["Fare"].clip(upper=65.0)
    raw["Age"]  = raw["Age"].clip(upper=64.0)
    raw = pd.get_dummies(raw, columns=["Sex", "Embarked"], dtype="int", drop_first=True)
    for col in feature_columns:
        if col not in raw.columns:
            raw[col] = 0
    return raw[feature_columns]

# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title='Czy przeżyłbyś katastrofę "Titanica"?',
        page_icon="🚢",
        layout="centered",
    )

    try:
        set_background("tlo_suml.JPG")
    except FileNotFoundError:
        pass

    overview    = st.container()
    left, right = st.columns(2)
    prediction  = st.container()

    with overview:
        st.title('🚢 Czy przeżyłbyś katastrofę "Titanica"?')
        st.markdown(
            "Wypełnij dane pasażera i sprawdź, czy model regresji logistycznej "
            "przewidziałby Twoje przeżycie."
        )
        if not model_loaded:
            st.error("⚠️ Brak pliku `titanic_model.pkl`. Uruchom najpierw `train_model.py`.")

    with left:
        age = st.slider("🎂 Wiek", min_value=1, max_value=100, value=30, step=1)
        sex = st.radio("👤 Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass = st.radio("🎫 Klasa kabiny", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])

    with right:
        sibsp    = st.slider("👫 Rodzeństwo / małżonkowie (SibSp)", 0, 8, 0)
        parch    = st.slider("👨‍👩‍👧 Rodzice / dzieci (Parch)", 0, 6, 0)
        fare     = st.slider("💰 Cena biletu (Fare)", 0.0, 300.0, 32.0, step=0.5)
        embarked = st.radio("⚓ Port zaokrętowania", list(embarked_d.keys()),
                            format_func=lambda x: embarked_d[x])

    with prediction:
        st.divider()
        st.markdown(
            f"**Podsumowanie:** Wiek: `{age}` | Płeć: `{sex_d[sex]}` | "
            f"Klasa: `{pclass_d[pclass]}` | SibSp: `{sibsp}` | "
            f"Parch: `{parch}` | Fare: `{fare}` | Port: `{embarked_d[embarked]}`"
        )

        predict_btn = st.button(
            "🔍 Sprawdź szanse przeżycia",
            type="primary",
            use_container_width=True,
            disabled=not model_loaded,
        )

        if predict_btn and model_loaded:
            input_df     = prepare_input(pclass, sex, age, sibsp, parch, fare, embarked, feature_columns)
            result       = model.predict(input_df)[0]
            proba        = model.predict_proba(input_df)[0]
            prob_survive = proba[1] * 100
            prob_die     = proba[0] * 100

            st.divider()
            if result == 1:
                st.success("✅ Pasażer **PRZEŻYŁBY** katastrofę Titanica!")
            else:
                st.error("❌ Pasażer **NIE PRZEŻYŁBY** katastrofy Titanica.")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🟢 Prawdopodobieństwo przeżycia", f"{prob_survive:.1f} %")
            with col_b:
                st.metric("🔴 Prawdopodobieństwo śmierci",   f"{prob_die:.1f} %")

            st.progress(int(prob_survive), text=f"Szansa przeżycia: {prob_survive:.1f} %")

            with st.expander("ℹ️ Dane wejściowe modelu (po transformacji)"):
                st.dataframe(input_df, use_container_width=True)

        st.divider()
        st.caption("Model: Regresja logistyczna (scikit-learn) · Dane: Titanic – Kaggle · s22053")

if __name__ == '__main__':
    main()
