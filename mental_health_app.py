
import streamlit as st
import joblib

# ---------- Load model and vectorizer ----------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("classifier_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        mlb = joblib.load("multilabel_binarizer.pkl")
        return model, vectorizer, mlb
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model/vectorizer: {e}")
        st.stop()

model, vectorizer, mlb = load_model()

# ---------- Streamlit App Layout ----------
st.set_page_config(page_title="Mental Health Journal Analyzer", page_icon="🧠")
st.title("🧠 Mental Health Journal Analyzer")
st.write("Enter a journal entry to detect mental health-related tags:")

user_input = st.text_area("✍️ Journal Entry", height=200)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a journal entry.")
    else:
        try:
            with st.spinner("Analyzing your input..."):
                X_input = vectorizer.transform([user_input])
                y_pred = model.predict(X_input)
                tags = mlb.inverse_transform(y_pred)[0]

            st.subheader("📋 Predicted Mental Health Tags:")
            if tags:
                st.success(", ".join(tags))
            else:
                st.info("No mental health conditions detected.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
