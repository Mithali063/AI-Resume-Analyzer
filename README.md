📄 AI Resume Analyzer & Job Role Predictor

An AI-powered web application that analyzes resumes and predicts suitable job roles using Machine Learning and NLP.

🚀 Live Demo
👉 https://ai-resume-analyzer-jsw6gndkxcduzkanvnxetx.streamlit.app/

📌 Features
- 🎯 Job Role Prediction
- 🏆 Top 3 Role Predictions with Confidence Scores
- 📊 Resume Score (out of 100)
- 🛠 Skill Detection
- 📌 Missing Skill Suggestions
- 📈 Role-based Recommendations (Courses, Projects, Tools)
- 📄 Resume Text Extraction (PDF)

🧠 Tech Stack
- Python
- Machine Learning (Scikit-learn)
- NLP (TF-IDF)
- Streamlit (Web App)
- PDF Processing (pdfplumber)

⚙️ How it Works
1. Upload a resume (PDF)
2. Text is extracted using pdfplumber
3. Text is cleaned using NLP techniques
4. TF-IDF vectorization is applied
5. ML model predicts job role
6. App displays:
   - Predicted Role
   - Top 3 roles with confidence
   - Resume score
   - Skills + missing skills
   - Recommendations



📁 Project Structure
AI-Resume-Analyzer/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md

💡 Future Improvements
Add chatbot for resume feedback
Add resume download report (PDF)
Improve UI design
Add login system
