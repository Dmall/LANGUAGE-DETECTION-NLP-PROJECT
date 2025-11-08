import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Load and Train Model
# -----------------------------
@st.cache_resource  # Cache the model so it doesn't retrain every time
def load_model():
    data = pd.read_csv("language.csv")

    x = np.array(data['Text'])
    y = np.array(data['language'])

    cv = CountVectorizer()
    X = cv.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, cv, accuracy

model, cv, accuracy = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Language Detector", page_icon="🌍", layout="centered")

st.title("🌍 Language Detection App")
st.markdown("This app uses **Naive Bayes** to detect the language of your text.")

st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

user_input = st.text_area("✍️ Enter a text sample to identify the language:", "")

if st.button("Detect Language"):
    # The line `if user_input.strip() == "":` is checking if the user input text is empty or consists
    # only of whitespace characters.
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        data = cv.transform([user_input]).toarray()
        prediction = model.predict(data)
        st.success(f"✅ Predicted Language: **{prediction[0]}**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
