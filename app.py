import streamlit as st
import pickle
import re
import pdfplumber

# Page config
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

# Custom UI styling
st.markdown("""
<style>
.title {
    font-size: 36px;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    margin-bottom: 20px;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 Features")
st.sidebar.write("✔ Job Role Prediction")
st.sidebar.write("✔ Top 3 Role Prediction")
st.sidebar.write("✔ Resume Score")
st.sidebar.write("✔ Skill Detection")
st.sidebar.write("✔ Missing Skills")
st.sidebar.write("✔ Role Recommendations")

# Keywords
skills_keywords = [
    "python", "java", "sql", "machine learning", "deep learning",
    "html", "css", "javascript", "react", "excel", "power bi",
    "tensorflow", "nlp", "pandas", "numpy", "c", "c++"
]

education_keywords = ["bachelor", "btech", "mtech", "degree", "university", "college"]
project_keywords = ["project", "developed", "implemented", "designed", "built"]
experience_keywords = ["internship", "experience", "worked", "company", "intern"]

# Role skills
role_skills = {
    "Data Science": ["python", "machine learning", "deep learning", "pandas", "numpy", "sql"],
    "Web Designing": ["html", "css", "javascript", "react"],
    "Java Developer": ["java", "spring", "sql"],
    "Python Developer": ["python", "django", "flask", "sql"],
    "DevOps Engineer": ["docker", "kubernetes", "linux", "aws"],
    "Business Analyst": ["excel", "sql", "power bi", "analysis"]
}

# Role recommendations
role_recommendations = {
    "Data Science": {
        "courses": ["Machine Learning", "Deep Learning", "Data Science Bootcamp"],
        "projects": ["Recommendation system", "Stock prediction", "Chatbot"],
        "tools": ["Python", "TensorFlow", "Pandas", "NumPy"]
    },
    "Web Designing": {
        "courses": ["Frontend Development", "React JS"],
        "projects": ["Portfolio website", "E-commerce UI"],
        "tools": ["HTML", "CSS", "JavaScript", "React"]
    },
    "Java Developer": {
        "courses": ["Java Programming", "Spring Boot"],
        "projects": ["REST API", "Banking system"],
        "tools": ["Java", "Spring Boot", "MySQL"]
    },
    "Python Developer": {
        "courses": ["Python", "Django"],
        "projects": ["Blog app", "Automation scripts"],
        "tools": ["Python", "Flask", "Django"]
    },
    "DevOps Engineer": {
        "courses": ["DevOps", "AWS"],
        "projects": ["CI/CD pipeline", "Docker deployment"],
        "tools": ["Docker", "Kubernetes", "Linux"]
    },
    "Business Analyst": {
        "courses": ["Business Analysis", "Power BI"],
        "projects": ["Dashboard", "Market analysis"],
        "tools": ["Excel", "Power BI", "SQL"]
    }
}

# Load ML files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Extract PDF text
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Score function
def calculate_score(text):
    text = text.lower()
    s = min(sum(1 for x in skills_keywords if x in text) * 5, 40)
    e = min(sum(1 for x in education_keywords if x in text) * 5, 20)
    p = min(sum(1 for x in project_keywords if x in text) * 5, 20)
    ex = min(sum(1 for x in experience_keywords if x in text) * 5, 20)
    return s + e + p + ex, s, e, p, ex

# Detect skills
def detect_skills(text):
    text = text.lower()
    return list(set([s for s in skills_keywords if s in text]))

# Title
st.markdown('<div class="title">📄 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a resume and get smart AI insights</div>', unsafe_allow_html=True)

# Upload
file = st.file_uploader("📤 Upload Resume PDF", type=["pdf"])

if file is not None:
    try:
        text = extract_text_from_pdf(file)

        if text.strip():

            clean = clean_text(text)
            vec = vectorizer.transform([clean]).toarray()

            probs = model.predict_proba(vec)[0]
            idx = probs.argsort()[-3:][::-1]

            roles = label_encoder.inverse_transform(idx)
            scores = probs[idx] * 100

            main_role = roles[0]

            detected = detect_skills(text)
            required = role_skills.get(main_role, [])
            missing = [r for r in required if r.lower() not in [d.lower() for d in detected]]

            rec = role_recommendations.get(main_role, None)

            total, s, e, p, ex = calculate_score(text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🎯 Predicted Role")
                st.success(main_role)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Resume Score")
                st.progress(total / 100)
                st.write(f"**{total}/100**")
                st.markdown('</div>', unsafe_allow_html=True)

            # Top 3 Roles
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🏆 Top 3 Roles")
            for r, sc in zip(roles, scores):
                st.write(f"**{r}** — {sc:.2f}%")
                st.progress(min(sc/100, 1.0))
            st.markdown('</div>', unsafe_allow_html=True)

            # Skills
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🛠 Detected Skills")
            st.write(", ".join(detected) if detected else "No skills detected")
            st.markdown('</div>', unsafe_allow_html=True)

            # Missing Skills
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📌 Missing Skills")
            if missing:
                st.write(", ".join(missing))
            else:
                st.success("Great! You have most required skills")
            st.markdown('</div>', unsafe_allow_html=True)

            # Recommendations
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Role Recommendations")

            if rec:
                st.write("📚 Courses:", ", ".join(rec["courses"]))
                st.write("💡 Projects:", ", ".join(rec["projects"]))
                st.write("🛠 Tools:", ", ".join(rec["tools"]))
            else:
                st.write("No recommendations available.")

            st.markdown('</div>', unsafe_allow_html=True)

            # Breakdown
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📋 Score Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Skills", f"{s}/40")
            c2.metric("Education", f"{e}/20")
            c3.metric("Projects", f"{p}/20")
            c4.metric("Experience", f"{ex}/20")
            st.markdown('</div>', unsafe_allow_html=True)

            # Quality
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if total > 80:
                st.success("🚀 Excellent Resume")
            elif total > 60:
                st.info("👍 Good Resume")
            else:
                st.warning("⚠ Needs Improvement")
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📄 Show Resume Text"):
                st.write(text)

        else:
            st.error("Could not extract text")

    except Exception as e:
        st.error(f"Error: {e}")
