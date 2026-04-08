import streamlit as st
import pickle
import re
import pdfplumber

# Load trained model files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# Streamlit UI
st.title("AI Resume Analyzer and Job Role Predictor")
st.write("Upload a resume PDF to predict the job role.")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text.strip() != "":
            cleaned_resume = clean_text(resume_text)
            resume_vector = vectorizer.transform([cleaned_resume]).toarray()
            prediction = model.predict(resume_vector)
            predicted_role = label_encoder.inverse_transform(prediction)

            st.success(f"Predicted Job Role: {predicted_role[0]}")

            with st.expander("Show extracted text"):
                st.write(resume_text)
        else:
            st.error("Could not extract text from the uploaded PDF.")

    except Exception as e:
        st.error(f"Error: {e}")