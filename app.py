import streamlit as st
import pickle
import re
import pdfplumber

st.set_page_config(page_title="Fresher Resume Analyzer", page_icon="🎓", layout="wide")

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

st.sidebar.title("📌 Features")
st.sidebar.write("✔ Specific Fresher Job Roles")
st.sidebar.write("✔ Resume Score")
st.sidebar.write("✔ Skill Detection")
st.sidebar.write("✔ Branch Detection")
st.sidebar.write("✔ Missing Skills")
st.sidebar.write("✔ Role Recommendations")

skills_keywords = [
    "python", "java", "sql", "machine learning", "deep learning", "artificial intelligence",
    "ai", "ml", "data science", "data analysis", "excel", "power bi", "tableau",
    "tensorflow", "pytorch", "scikit-learn", "nlp", "pandas", "numpy",
    "html", "css", "javascript", "react", "bootstrap",
    "c", "c++", "embedded", "embedded c", "iot", "arduino", "nodemcu",
    "microcontroller", "sensors", "vlsi", "matlab", "pcb",
    "cybersecurity", "network security", "linux", "networking", "firewall",
    "digital marketing", "seo", "social media", "content marketing", "canva",
    "testing", "selenium", "manual testing", "automation testing",
    "business analysis", "communication", "analytics"
]

education_keywords = [
    "bachelor", "btech", "b.e", "degree", "university", "college", "cgpa",
    "electronics", "communication", "electrical", "computer science"
]

project_keywords = [
    "project", "developed", "implemented", "designed", "built", "created", "deployed"
]

experience_keywords = [
    "internship", "experience", "worked", "company", "intern", "training"
]

role_skills = {
    "Data Analyst": ["python", "sql", "excel", "power bi", "data analysis", "statistics", "tableau"],
    "Data Science Intern": ["python", "data science", "machine learning", "pandas", "numpy", "scikit-learn"],
    "Machine Learning Intern": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn"],
    "AI Engineer Intern": ["artificial intelligence", "ai", "ml", "python", "nlp", "deep learning"],
    "Python Developer": ["python", "django", "flask", "sql", "oops"],
    "Frontend Developer": ["html", "css", "javascript", "react", "bootstrap"],
    "Software Developer": ["c", "c++", "java", "python", "oops", "data structures"],
    "Embedded Engineer": ["c", "embedded", "embedded c", "microcontroller", "arduino", "iot"],
    "IoT Developer": ["iot", "arduino", "nodemcu", "sensors", "embedded", "mqtt"],
    "VLSI / Electronics Engineer": ["vlsi", "electronics", "communication", "verilog", "matlab", "pcb"],
    "Electrical Engineer": ["electrical", "power systems", "matlab", "circuits", "electronics"],
    "Cybersecurity Analyst": ["cybersecurity", "network security", "linux", "networking", "firewall"],
    "Business Analyst": ["excel", "sql", "power bi", "business analysis", "communication", "analytics"],
    "Digital Marketing Executive": ["digital marketing", "seo", "social media", "content marketing", "canva"],
    "QA Tester": ["testing", "manual testing", "automation testing", "selenium", "bug tracking"]
}

branch_keywords = {
    "ECE": ["electronics", "communication", "ece", "embedded", "iot", "vlsi", "microcontroller"],
    "EEE": ["electrical", "electronics", "power systems", "matlab"],
    "CSE": ["computer science", "cse", "software", "programming", "data structures"],
    "ISE": ["information science", "ise", "software", "database", "web"],
    "Mechanical": ["mechanical", "cad", "solidworks", "manufacturing"],
    "Civil": ["civil", "construction", "autocad", "surveying"]
}

role_recommendations = {
    "Data Analyst": {
        "courses": ["SQL for Data Analysis", "Excel Dashboarding", "Power BI"],
        "projects": ["Sales dashboard", "COVID data analysis", "Customer analysis"],
        "tools": ["Excel", "SQL", "Power BI", "Python"]
    },
    "Data Science Intern": {
        "courses": ["Data Science with Python", "Statistics for ML", "Scikit-learn"],
        "projects": ["Loan prediction", "Customer churn prediction", "Recommendation system"],
        "tools": ["Python", "Pandas", "NumPy", "Scikit-learn"]
    },
    "Machine Learning Intern": {
        "courses": ["Machine Learning", "Deep Learning", "Model Deployment"],
        "projects": ["Resume analyzer", "Image classification", "Prediction model"],
        "tools": ["Python", "Scikit-learn", "TensorFlow", "Streamlit"]
    },
    "AI Engineer Intern": {
        "courses": ["Artificial Intelligence Basics", "NLP", "Generative AI"],
        "projects": ["AI chatbot", "Resume analyzer", "Text classifier"],
        "tools": ["Python", "NLP", "TensorFlow", "Streamlit"]
    },
    "Embedded Engineer": {
        "courses": ["Embedded C", "Microcontrollers", "Linux Basics"],
        "projects": ["Sensor monitoring system", "IoT device", "Arduino project"],
        "tools": ["C", "Arduino", "NodeMCU", "Linux"]
    },
    "IoT Developer": {
        "courses": ["IoT Basics", "MQTT", "Embedded Systems"],
        "projects": ["Smart home system", "Sensor dashboard", "IoT automation"],
        "tools": ["NodeMCU", "Arduino", "Sensors", "MQTT"]
    },
    "VLSI / Electronics Engineer": {
        "courses": ["VLSI Design", "Digital Electronics", "Verilog"],
        "projects": ["Digital circuit design", "FPGA mini project", "PCB design"],
        "tools": ["Verilog", "MATLAB", "Xilinx", "PCB tools"]
    },
    "Electrical Engineer": {
        "courses": ["Power Systems", "Electrical Machines", "MATLAB"],
        "projects": ["Power monitoring system", "Circuit simulation", "Energy meter"],
        "tools": ["MATLAB", "Simulink", "Proteus"]
    },
    "Cybersecurity Analyst": {
        "courses": ["Cybersecurity Basics", "Network Security", "Linux"],
        "projects": ["Phishing detection", "Network scanner", "Log analysis"],
        "tools": ["Linux", "Wireshark", "Nmap", "Firewall"]
    },
    "Business Analyst": {
        "courses": ["Business Analysis", "Excel", "Power BI"],
        "projects": ["Market analysis", "Sales dashboard", "Requirement analysis"],
        "tools": ["Excel", "Power BI", "SQL"]
    },
    "Digital Marketing Executive": {
        "courses": ["SEO", "Social Media Marketing", "Google Analytics"],
        "projects": ["Instagram campaign analysis", "SEO audit", "Content calendar"],
        "tools": ["Canva", "Google Analytics", "Meta Business Suite"]
    },
    "Frontend Developer": {
        "courses": ["HTML CSS JavaScript", "React JS", "UI Design"],
        "projects": ["Portfolio website", "Responsive website", "Landing page"],
        "tools": ["HTML", "CSS", "JavaScript", "React", "Figma"]
    },
    "QA Tester": {
        "courses": ["Manual Testing", "Automation Testing", "Selenium"],
        "projects": ["Test case design", "Bug tracking project", "Selenium automation"],
        "tools": ["Selenium", "Jira", "Excel"]
    }
}

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def calculate_score(text):
    text = text.lower()
    s = min(sum(1 for x in skills_keywords if x in text) * 5, 40)
    e = min(sum(1 for x in education_keywords if x in text) * 5, 20)
    p = min(sum(1 for x in project_keywords if x in text) * 5, 20)
    ex = min(sum(1 for x in experience_keywords if x in text) * 5, 20)
    return s + e + p + ex, s, e, p, ex

def detect_skills(text):
    text = text.lower()
    detected = [s for s in skills_keywords if s in text]
    return sorted(list(set(detected)))

def detect_branch(text):
    text = text.lower()
    for branch, keywords in branch_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return branch
    return "Not clearly detected"

def specific_role_recommendation(text):
    text = text.lower()
    role_scores = {}

    for role, skills in role_skills.items():
        score = 0
        for skill in skills:
            if skill.lower() in text:
                score += 1
        role_scores[role] = score

    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_roles[:5]

st.markdown(
    '<div class="title">🎓 Fresher Resume Analyzer & Job Role Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload your resume to get specific job role suggestions based on skills, projects, certifications, and branch</div>',
    unsafe_allow_html=True
)

file = st.file_uploader("📤 Upload Resume PDF", type=["pdf"])

if file is not None:
    try:
        text = extract_text_from_pdf(file)

        if text.strip():
            clean = clean_text(text)

            vec = vectorizer.transform([clean]).toarray()
            probs = model.predict_proba(vec)[0]
            idx = probs.argsort()[-3:][::-1]
            broad_roles = label_encoder.inverse_transform(idx)
            broad_scores = probs[idx] * 100

            detected = detect_skills(text)
            detected_branch = detect_branch(text)
            specific_roles = specific_role_recommendation(text)

            main_specific_role = specific_roles[0][0]
            required = role_skills.get(main_specific_role, [])
            missing = [
                r for r in required
                if r.lower() not in [d.lower() for d in detected]
            ]

            rec = role_recommendations.get(main_specific_role, None)
            total, s, e, p, ex = calculate_score(text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🎯 Recommended Fresher Role")
                st.success(main_specific_role)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Resume Score")
                st.progress(total / 100)
                st.write(f"**{total}/100**")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💼 Top Specific Job Role Suggestions")
            for role, score in specific_roles:
                if score > 0:
                    percentage = min((score / len(role_skills[role])) * 100, 100)
                    st.write(f"**{role}** — Match Score: {percentage:.2f}%")
                    st.progress(percentage / 100)
                else:
                    st.write(f"**{role}** — Low match")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📌 Broad ML Category Prediction")
            for r, sc in zip(broad_roles, broad_scores):
                st.write(f"**{r}** — {sc:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🛠 Detected Skills")
            st.write(", ".join(detected) if detected else "No skills detected")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎓 Detected Branch")
            st.write(detected_branch)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📌 Missing Skills for Recommended Role")
            if missing:
                st.write(", ".join(missing))
            else:
                st.success("Great! You have most required skills for this role")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Role Recommendations")
            if rec:
                st.write("📚 Courses:", ", ".join(rec["courses"]))
                st.write("💡 Projects:", ", ".join(rec["projects"]))
                st.write("🛠 Tools:", ", ".join(rec["tools"]))
            else:
                st.write("No recommendations available.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📋 Score Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Skills", f"{s}/40")
            c2.metric("Education", f"{e}/20")
            c3.metric("Projects", f"{p}/20")
            c4.metric("Experience", f"{ex}/20")
            st.markdown('</div>', unsafe_allow_html=True)

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
            st.error("Could not extract text from the uploaded PDF.")

    except Exception as e:
        st.error(f"Error: {e}")
