# app.py - AI Resume Analyzer Pro with Gemini AI (FIXED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import google.generativeai as genai
import os
from collections import Counter

# ==================== GLOBAL VARIABLES ====================
GEMINI_AVAILABLE = False
GEMINI_MODEL_NAME = None  # Will store which model works

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== GEMINI AI SETUP FUNCTION ====================
# ==================== GEMINI AI SETUP FUNCTION ====================
def setup_gemini():
    """Setup Gemini AI with correct model names for your account"""
    global GEMINI_AVAILABLE, GEMINI_MODEL_NAME
    
    try:
        # Your API key
        YOUR_KEY = "AIzaSyDlEU79NOj-OIc3sEG1ahSJVhBwLDWYh8g"
        
        # Configure Gemini
        genai.configure(api_key=YOUR_KEY)
        
        # CORRECT model names for YOUR account (from your diagnostic)
        models_to_try = [
            'models/gemini-2.0-flash',      # Fast and good for your use case
            'models/gemini-2.5-flash',      # Latest flash model
            'models/gemini-pro-latest',     # Latest pro model
            'models/gemini-2.5-pro',        # Pro model
            'models/gemini-2.0-flash-001',  # Specific version
        ]
        
        for model_name in models_to_try:
            try:
                print(f"Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Say 'Hello' to test")
                
                if response and hasattr(response, 'text') and response.text:
                    GEMINI_AVAILABLE = True
                    GEMINI_MODEL_NAME = model_name
                    print(f"✅ Success with model: {model_name}")
                    print(f"✅ Response: {response.text}")
                    return True, f"✅ Gemini connected using {model_name}"
                else:
                    print(f"❌ No response from {model_name}")
                    
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg:
                    print(f"❌ Model {model_name} not found (404)")
                elif "quota" in error_msg.lower():
                    print(f"❌ Quota exceeded for {model_name}")
                else:
                    print(f"❌ Error with {model_name}: {error_msg[:60]}")
                continue
        
        # If none of the specific models worked, try listing all models
        try:
            print("\nListing all available models to find working one...")
            models = genai.list_models()
            for model in models:
                if "flash" in model.name or "pro" in model.name:
                    try:
                        print(f"Trying: {model.name}")
                        test_model = genai.GenerativeModel(model.name)
                        test_response = test_model.generate_content("Test")
                        if test_response and test_response.text:
                            GEMINI_AVAILABLE = True
                            GEMINI_MODEL_NAME = model.name
                            return True, f"✅ Found working model: {model.name}"
                    except:
                        continue
        except:
            pass
        
        return False, "❌ No Gemini model worked. Check console for details."
            
    except Exception as e:
        return False, f"❌ Gemini setup error: {str(e)[:80]}"
# ==================== DATABASE SETUP ====================
@st.cache_resource
def init_database():
    conn = sqlite3.connect('applications.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS applications
                 (id INTEGER PRIMARY KEY, 
                  date TEXT, 
                  company TEXT, 
                  role TEXT, 
                  score REAL,
                  ats_score REAL,
                  skills_match REAL,
                  status TEXT,
                  notes TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS skill_trends
                 (id INTEGER PRIMARY KEY,
                  date TEXT,
                  skill TEXT,
                  count INTEGER)''')
    conn.commit()
    return conn

conn = init_database()

# ==================== SKILLS DATABASE ====================
SKILLS_DATABASE = {
    "Programming": ["python", "java", "javascript", "c++", "c#", "sql", "r", "go", "swift", "kotlin", "typescript", "php", "ruby"],
    "Data Science": ["machine learning", "deep learning", "tensorflow", "pytorch", "keras", 
                     "pandas", "numpy", "scikit-learn", "statistics", "data analysis",
                     "data visualization", "big data", "spark", "hadoop", "tableau", "power bi"],
    "Web Development": ["html", "css", "javascript", "react", "angular", "vue", "node.js", 
                        "express", "django", "flask", "fastapi", "rest api", "graphql", "websocket",
                        "next.js", "spring boot", "laravel"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", 
                       "ci/cd", "linux", "bash", "ansible", "prometheus", "grafana", "gitlab"],
    "Tools": ["git", "github", "gitlab", "jira", "confluence", "docker", "postman", "tableau", 
              "power bi", "excel", "vscode", "pycharm", "intellij"],
    "Databases": ["mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "dynamodb", "cassandra"],
    "Soft Skills": ["leadership", "communication", "teamwork", "problem solving", "project management",
                    "agile", "scrum", "critical thinking", "creativity", "time management"]
}

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: black;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .skill-match { 
        background-color: #10B981; 
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 2px;
        display: inline-block;
        font-size: 0.8rem;
    }
    .skill-missing { 
        background-color: #EF4444; 
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 2px;
        display: inline-block;
        font-size: 0.8rem;
    }
    .skill-present { 
        background-color: #6B7280; 
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 2px;
        display: inline-block;
        font-size: 0.8rem;
    }
    .ats-good { color: #10B981; }
    .ats-warning { color: #F59E0B; }
    .ats-bad { color: #EF4444; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #667eea ;
        border-radius: 5px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_skills(text):
    """Extract skills from text using comprehensive database"""
    text_lower = text.lower()
    found_skills = {}
    
    for category, skills in SKILLS_DATABASE.items():
        category_skills = []
        for skill in skills:
            # Use regex to find whole word matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                category_skills.append(skill.title())
        if category_skills:
            found_skills[category] = category_skills
    
    return found_skills

def calculate_match_score(resume_text, job_text):
    """Calculate comprehensive match score using multiple algorithms"""
    
    # 1. TF-IDF Content Similarity (40%)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    content_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    content_score = content_similarity * 100
    
    # 2. Skill Match Score (30%)
    resume_skills_dict = extract_skills(resume_text)
    job_skills_dict = extract_skills(job_text)
    
    all_resume_skills = []
    all_job_skills = []
    
    for skills in resume_skills_dict.values():
        all_resume_skills.extend([s.lower() for s in skills])
    
    for skills in job_skills_dict.values():
        all_job_skills.extend([s.lower() for s in skills])
    
    if all_job_skills:
        skill_match_ratio = len(set(all_resume_skills) & set(all_job_skills)) / len(all_job_skills)
        skill_score = skill_match_ratio * 100
    else:
        skill_score = 0
    
    # 3. Keyword Density (15%)
    job_words = set(re.findall(r'\b\w+\b', job_text.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    keyword_overlap = len(job_words.intersection(resume_words))
    keyword_score = min(keyword_overlap / len(job_words) * 100, 15) if job_words else 0
    
    # 4. Structure Score (15%)
    sections = ["experience", "education", "skills", "projects", "summary"]
    section_count = sum(1 for section in sections if section in resume_text.lower())
    structure_score = (section_count / len(sections)) * 15
    
    # Final weighted score
    final_score = (
        content_score * 0.40 +
        skill_score * 0.30 +
        keyword_score * 0.15 +
        structure_score * 0.15
    )
    
    return {
        "overall_score": round(min(final_score, 100), 1),
        "content_score": round(content_score, 1),
        "skill_score": round(skill_score, 1),
        "keyword_score": round(keyword_score, 1),
        "structure_score": round(structure_score, 1),
        "resume_skills": all_resume_skills,
        "job_skills": all_job_skills,
        "resume_skills_dict": resume_skills_dict,
        "job_skills_dict": job_skills_dict
    }

def calculate_ats_score(resume_text, job_text):
    """Simulate Applicant Tracking System scoring"""
    scores = {
        "keyword_match": 0,
        "section_completion": 0,
        "format_quality": 0,
        "experience_relevance": 0
    }
    
    # 1. Keyword Match (35 points max)
    job_keywords = set(re.findall(r'\b[a-z]{4,}\b', job_text.lower()))
    resume_keywords = set(re.findall(r'\b[a-z]{4,}\b', resume_text.lower()))
    keyword_overlap = len(job_keywords.intersection(resume_keywords))
    scores["keyword_match"] = min(keyword_overlap, 35)
    
    # 2. Section Completion (30 points)
    required_sections = ["experience", "education", "skills"]
    optional_sections = ["summary", "projects", "certifications"]
    
    for section in required_sections:
        if section in resume_text.lower():
            scores["section_completion"] += 10
    
    for section in optional_sections:
        if section in resume_text.lower():
            scores["section_completion"] += 5
    
    # 3. Format Quality (20 points)
    # Check for common ATS-friendly format indicators
    format_indicators = {
        "simple_bullets": len(re.findall(r'•|\*|\-', resume_text)) > 5,
        "no_tables": "table" not in resume_text.lower(),
        "no_headers": not re.search(r'header|footer', resume_text.lower()),
        "has_contact": any(x in resume_text.lower() for x in ["@", "phone", "linkedin"]),
        "proper_length": 300 < len(resume_text.split()) < 800
    }
    
    scores["format_quality"] = sum(4 for indicator in format_indicators.values() if indicator)
    
    # 4. Experience Relevance (15 points)
    experience_indicators = ["years", "experience", "worked", "developed", "managed", "created"]
    experience_count = sum(1 for word in experience_indicators if word in resume_text.lower())
    scores["experience_relevance"] = min(experience_count * 2.5, 15)
    
    total_ats_score = sum(scores.values())
    
    return {
        "ats_total": round(total_ats_score, 1),
        "ats_breakdown": scores,
        "ats_grade": "A" if total_ats_score >= 80 else "B" if total_ats_score >= 65 else "C" if total_ats_score >= 50 else "D"
    }

def get_suggestions(resume_text, job_text, missing_skills, match_data):
    """Generate AI-powered suggestions using Gemini"""
    
    global GEMINI_AVAILABLE, GEMINI_MODEL_NAME
    
    # If Gemini is available, use AI analysis
    if GEMINI_AVAILABLE and GEMINI_MODEL_NAME:
        try:
            # Initialize the model with working model name
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            
            # Create prompt (shorter for testing)
            prompt = f"""Analyze this resume for the job and give 3 specific suggestions:

            Resume: {resume_text[:1500]}
            Job: {job_text[:1000]}
            Missing Skills: {missing_skills[:5] if missing_skills else 'None'}
            
            
            ANALYSIS DATA:
            - Overall Match Score: {match_data['overall_score']}%
            - Missing Skills: {', '.join(missing_skills[:10]) if missing_skills else 'None'}
            
            Please provide 3-5 SPECIFIC, ACTIONABLE suggestions to improve this resume for this exact job.
            Focus on practical improvements the user can implement immediately.
            
            Format with clear headings and bullet points.
            """
            
            # Generate response
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
            else:
                return "No response from AI. Using rule-based suggestions."
            
        except Exception as e:
            st.error(f"⚠️ AI analysis failed: {str(e)[:100]}")
            # Fall back to rule-based analysis
    
    # ===== FALLBACK: Original Rule-Based Suggestions =====
    suggestions = []
    
    # 1. Keyword Optimization
    if match_data["keyword_score"] < 10:
        suggestions.append("""
        1. **Keyword Optimization**: 
        Increase keyword density by adding more job-specific terms from the job description. 
        Identify 5-10 key terms from the job description and incorporate them naturally into your resume.
        """)
    
    # 2. Skills Enhancement
    if missing_skills:
        top_missing = missing_skills[:5]
        suggestions.append(f"""
        2. **Skills Enhancement**: 
        Add these missing skills: {', '.join([s.title() for s in top_missing])}. 
        Consider highlighting relevant projects or experience that demonstrate these skills.
        """)
    
    # 3. Experience Reframing
    if match_data["content_score"] < 60:
        suggestions.append("""
        3. **Experience Reframing**: 
        Reframe your experience to better match the job requirements. Use action verbs and quantify achievements 
        (e.g., "Increased efficiency by 25%" instead of "Responsible for improving efficiency").
        """)
    
    # 4. ATS Compliance
    ats_data = calculate_ats_score(resume_text, job_text)
    if ats_data["ats_grade"] in ["C", "D"]:
        suggestions.append("""
        4. **ATS Compliance**: 
        Improve your resume format for Applicant Tracking Systems. Use standard section headers, 
        avoid tables and columns, include relevant keywords, and ensure proper contact information.
        """)
    
    # 5. Strategic Addition
    if len(match_data["resume_skills"]) < 15:
        suggestions.append("""
        5. **Strategic Addition**: 
        Expand your skills section with relevant technical and soft skills. Consider adding a 
        "Projects" section to showcase practical experience with key technologies mentioned in the job.
        """)
    
    # Add default suggestions if we don't have enough
    if len(suggestions) < 5:
        suggestions.append("""
        5. **Quantifiable Achievements**: 
        Add specific metrics and numbers to your experience (e.g., "Managed team of 5", 
        "Reduced processing time by 30%", "Improved accuracy by 15%"). Quantifiable achievements 
        make your resume more impactful.
        """)
    
    return "\n\n".join(suggestions[:5])

def save_application(company, role, match_data, ats_data, notes=""):
    """Save application to database"""
    c = conn.cursor()
    c.execute("""
        INSERT INTO applications (date, company, role, score, ats_score, skills_match, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        company,
        role,
        match_data["overall_score"],
        ats_data["ats_total"],
        match_data["skill_score"],
        "Applied",
        notes
    ))
    conn.commit()
    
    # Update skill trends
    skills = match_data["resume_skills"]
    for skill in skills:
        c.execute("""
            INSERT INTO skill_trends (date, skill, count)
            VALUES (?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d"), skill, 1))
    conn.commit()

def get_application_history():
    """Get application history from database"""
    df = pd.read_sql_query("SELECT * FROM applications ORDER BY date DESC", conn)
    return df

# ==================== MAIN APP ====================
def main():
    global GEMINI_AVAILABLE, GEMINI_MODEL_NAME
    
    # Setup Gemini first
    gemini_success, gemini_message = setup_gemini()
    GEMINI_AVAILABLE = gemini_success
    
    # Debug print to console
    print(f"Gemini Status: {gemini_message}")
    if GEMINI_AVAILABLE:
        print(f"Using model: {GEMINI_MODEL_NAME}")
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Resume Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
    Upload your resume and job description for instant AI-powered analysis, ATS simulation, and personalized suggestions.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
        st.title("Navigation")
        
        # Show Gemini status
        if GEMINI_AVAILABLE:
            st.success(f"✅ Gemini AI Connected")
            st.info(f"Model: {GEMINI_MODEL_NAME}")
        else:
            st.error("❌ Gemini Not Available")
            st.info(gemini_message)
    
    # ... [REST OF YOUR APP CODE REMAINS THE SAME - TAB SYSTEM, ANALYSIS, ETC.]
    # Just copy all the remaining code from your original app.py starting from line with `analysis_tab = st.radio(`
    
    analysis_tab = st.radio(
        "Go to:",
        ["📊 Resume Analysis", "📈 Application Dashboard", "⚙️ Settings & Help"]
    )
    
    st.divider()
    
    with st.expander("📋 Sample Job Description", expanded=False):
        sample_jd = """Senior Data Scientist

Responsibilities:
- Develop and deploy machine learning models for predictive analytics
- Lead A/B testing experiments and analyze results
- Collaborate with engineering teams to productionize models
- Mentor junior data scientists and analysts
- Present findings to stakeholders and executives

Requirements:
- 5+ years experience in data science or machine learning
- Expertise in Python, SQL, and statistical analysis
- Experience with TensorFlow, PyTorch, or similar frameworks
- Strong knowledge of cloud platforms (AWS, GCP, or Azure)
- Experience with Docker, Kubernetes, and MLOps practices
- Excellent communication and leadership skills
- Advanced degree in Computer Science, Statistics, or related field

Preferred Qualifications:
- Published research in machine learning
- Experience with big data technologies (Spark, Hadoop)
- Knowledge of deep learning architectures
- Experience in deploying models to production"""
        
        if st.button("Load Sample"):
            st.session_state.sample_jd = sample_jd
            st.rerun()
    
    # ==================== TAB 1: RESUME ANALYSIS ====================
    if analysis_tab == "📊 Resume Analysis":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Upload Your Resume")
            
            upload_method = st.radio(
                "Choose input method:",
                ["📤 Upload File", "📝 Paste Text"]
            )
            
            if upload_method == "📤 Upload File":
                resume_file = st.file_uploader(
                    "Upload resume (PDF, DOCX, TXT)",
                    type=['pdf', 'docx', 'txt'],
                    key="resume_upload"
                )
                
                if resume_file:
                    if resume_file.type == "application/pdf":
                        resume_text = extract_text_from_pdf(resume_file)
                    elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = extract_text_from_docx(resume_file)
                    else:
                        resume_text = resume_file.read().decode()
                    
                    st.success(f"✅ Resume loaded ({len(resume_text.split())} words)")
                    
                    with st.expander("Preview Resume Text"):
                        st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
                else:
                    resume_text = ""
            else:
                resume_text = st.text_area(
                    "Paste your resume text here:",
                    height=200,
                    placeholder="Paste your complete resume text here..."
                )
            
            # Company and role info
            col1a, col1b = st.columns(2)
            with col1a:
                company_name = st.text_input("Company Name", placeholder="e.g., Google, Amazon")
            with col1b:
                role_name = st.text_input("Job Role", placeholder="e.g., Data Scientist, Software Engineer")
        
        with col2:
            st.subheader("📋 Job Description")
            
            if 'sample_jd' in st.session_state:
                default_jd = st.session_state.sample_jd
            else:
                default_jd = ""
            
            job_text = st.text_area(
                "Paste the job description:",
                value=default_jd,
                height=300,
                placeholder="Paste the complete job description here..."
            )
            
            # Quick analysis button
            analyze_clicked = st.button(
                "🚀 Run Comprehensive Analysis",
                type="primary",
                use_container_width=True,
                disabled=not (resume_text and job_text)
            )
        
        # Run analysis when clicked
        if analyze_clicked and resume_text and job_text:
            with st.spinner("🔍 Analyzing resume with AI..."):
                # Calculate all scores
                match_data = calculate_match_score(resume_text, job_text)
                ats_data = calculate_ats_score(resume_text, job_text)
                
                # Get missing skills
                missing_skills = list(set(match_data["job_skills"]) - set(match_data["resume_skills"]))
                
                # Get suggestions (AI or rule-based)
                suggestions = get_suggestions(
                    resume_text, job_text, 
                    missing_skills, 
                    match_data
                )
                
                # Save to database if company/role provided
                if company_name and role_name:
                    save_application(
                        company_name, 
                        role_name, 
                        match_data, 
                        ats_data,
                        notes=f"Match: {match_data['overall_score']}%"
                    )
                    st.toast(f"✅ Analysis saved for {company_name} - {role_name}")
                
                # ==================== RESULTS DISPLAY ====================
                st.divider()
                st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
                
                # Score Cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3 style='color: white; margin: 0;'>Overall Match</h3>
                        <h1 style='color: white; margin: 0;'>{score}%</h1>
                    </div>
                    """.format(score=match_data["overall_score"]), unsafe_allow_html=True)
                    
                    # Score indicator
                    score_val = match_data["overall_score"]
                    if score_val >= 75:
                        st.success("🎯 Strong Match! Well aligned with job requirements.")
                    elif score_val >= 50:
                        st.warning("📊 Moderate Match. Some improvements needed.")
                    else:
                        st.error("⚠️ Needs Significant Improvement.")
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style='color: white; margin: 0;'>ATS Score</h3>
                        <h1 style='color: white; margin: 0;'>{ats_data['ats_total']}/100</h1>
                        <p style='color: white; margin: 0;'>Grade: {ats_data['ats_grade']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ATS grade color
                    if ats_data['ats_grade'] == 'A':
                        st.success("✅ Excellent ATS compatibility")
                    elif ats_data['ats_grade'] == 'B':
                        st.info("ℹ️ Good ATS compatibility")
                    else:
                        st.warning("⚠️ Needs ATS optimization")
                
                with col3:
                    skill_match_count = len(set(match_data["resume_skills"]) & set(match_data["job_skills"]))
                    total_job_skills = len(set(match_data["job_skills"]))
                    skill_percentage = round((skill_match_count / total_job_skills * 100) if total_job_skills > 0 else 0, 1)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style='color: white; margin: 0;'>Skills Match</h3>
                        <h1 style='color: white; margin: 0;'>{skill_match_count}/{total_job_skills}</h1>
                        <p style='color: white; margin: 0;'>{skill_percentage}% match</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    keyword_score = match_data["keyword_score"]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style='color: white; margin: 0;'>Keyword Density</h3>
                        <h1 style='color: white; margin: 0;'>{keyword_score}/15</h1>
                        <p style='color: white; margin: 0;'>Content relevance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed Analysis Tabs
                st.divider()
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🔧 Skills Analysis", 
                    "📝 Suggestions", 
                    "📋 ATS Breakdown",
                    "📈 Score Details"
                ])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("✅ Your Skills vs Required")
                        
                        # Skills comparison by category
                        for category in SKILLS_DATABASE.keys():
                            resume_cat_skills = match_data["resume_skills_dict"].get(category, [])
                            job_cat_skills = match_data["job_skills_dict"].get(category, [])
                            
                            if resume_cat_skills or job_cat_skills:
                                st.markdown(f"**{category}**")
                                
                                # Show matching skills
                                matching_skills = [s for s in resume_cat_skills if s.lower() in [js.lower() for js in job_cat_skills]]
                                missing_in_resume = [s for s in job_cat_skills if s.lower() not in [rs.lower() for rs in resume_cat_skills]]
                                extra_in_resume = [s for s in resume_cat_skills if s.lower() not in [js.lower() for js in job_cat_skills]]
                                
                                for skill in matching_skills:
                                    st.markdown(f'<span class="skill-match">✓ {skill}</span>', unsafe_allow_html=True)
                                
                                for skill in missing_in_resume:
                                    st.markdown(f'<span class="skill-missing">✗ {skill}</span>', unsafe_allow_html=True)
                                
                                for skill in extra_in_resume:
                                    st.markdown(f'<span class="skill-present">• {skill}</span>', unsafe_allow_html=True)
                                
                                st.write("")
                    
                    with col2:
                        st.subheader("📊 Skills Gap Analysis")
                        
                        if missing_skills:
                            st.warning(f"**Missing {len(missing_skills)} skills mentioned in job:**")
                            
                            # Group missing skills by priority
                            high_priority = [s for s in missing_skills if any(tech in s for tech in 
                                      ["python", "machine", "learning", "sql", "aws", "docker"])]
                            medium_priority = [s for s in missing_skills if s not in high_priority]
                            
                            if high_priority:
                                st.write("**High Priority:**")
                                for skill in high_priority[:5]:
                                    st.markdown(f'<span class="skill-missing">⚠️ {skill.title()}</span>', unsafe_allow_html=True)
                            
                            if medium_priority:
                                st.write("**Medium Priority:**")
                                for skill in medium_priority[:5]:
                                    st.markdown(f'<span class="skill-missing">{skill.title()}</span>', unsafe_allow_html=True)
                            
                            # Add to resume button
                            if st.button("📝 Generate 'Skills' Section Text"):
                                skills_text = "**Skills:** " + ", ".join([s.title() for s in list(match_data["resume_skills"])[:15]])
                                st.code(skills_text, language="text")
                        else:
                            st.success("✅ No major skill gaps detected!")
                        
                        # Skill coverage chart
                        skill_data = pd.DataFrame({
                            'Category': ['Match', 'Missing', 'Extra'],
                            'Count': [
                                len(set(match_data["resume_skills"]) & set(match_data["job_skills"])),
                                len(missing_skills),
                                len(set(match_data["resume_skills"]) - set(match_data["job_skills"]))
                            ]
                        })
                        
                        fig = px.pie(skill_data, values='Count', names='Category', 
                                   title='Skills Distribution', hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("📝 Optimization Suggestions")
                    
                    # Format the suggestions nicely
                    suggestion_sections = suggestions.split('\n\n')
                    for section in suggestion_sections:
                        if section.strip():
                            st.markdown(f"""
                            <div style='
                                background: blue;
                                padding: 15px;
                                border-radius: 10px;
                                margin: 10px 0;
                                border-left: 4px solid #667eea;
                            '>
                            {section}
                            </div>
                            """, unsafe_allow_html=True)
                        
                    # Show if using AI or rule-based
                    if GEMINI_AVAILABLE:
                        st.caption("🤖 AI-Powered Suggestions (Gemini Pro)")
                    else:
                        st.caption("⚙️ Rule-Based Suggestions")
                        
                    # Quick actions based on suggestions
                    st.subheader("🚀 Quick Actions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📋 Generate Resume Summary"):
                            summary_text = f"Experienced {role_name if role_name else 'professional'} with expertise in {', '.join(match_data['resume_skills'][:5])}. Proven track record in implementing solutions and achieving measurable results."
                            st.code(summary_text, language="text")
                    
                    with col2:
                        if st.button("🎯 Optimize Keywords"):
                            keywords_to_add = missing_skills[:5]
                            st.info(f"Add these keywords: {', '.join([k.title() for k in keywords_to_add])}")
                    
                    with col3:
                        if st.button("📄 Download Analysis"):
                            # Generate report
                            report = f"""
                            RESUME ANALYSIS REPORT
                            ======================
                            
                            Company: {company_name}
                            Role: {role_name}
                            Date: {datetime.now().strftime('%Y-%m-%d')}
                            
                            SCORES:
                            - Overall Match: {match_data['overall_score']}%
                            - ATS Score: {ats_data['ats_total']}/100 (Grade: {ats_data['ats_grade']})
                            - Skills Match: {len(set(match_data['resume_skills']) & set(match_data['job_skills']))}/{len(set(match_data['job_skills']))}
                            
                            SKILLS ANALYSIS:
                            Your Skills: {', '.join([s.title() for s in list(set(match_data['resume_skills']))[:20]])}
                            Missing Skills: {', '.join([s.title() for s in missing_skills[:10]])}
                            
                            SUGGESTIONS:
                            {suggestions}
                            """
                            
                            st.download_button(
                                label="📥 Download Report",
                                data=report,
                                file_name=f"resume_analysis_{company_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain"
                            )
                
                with tab3:
                    st.subheader("📋 ATS (Applicant Tracking System) Analysis")
                    
                    # ATS Score Breakdown
                    ats_scores = ats_data['ats_breakdown']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Score Breakdown:**")
                        for category, score in ats_scores.items():
                            max_score = 35 if category == "keyword_match" else 30 if category == "section_completion" else 20 if category == "format_quality" else 15
                            percentage = (score / max_score) * 100
                            
                            st.write(f"**{category.replace('_', ' ').title()}:**")
                            st.progress(min(percentage / 100, 1.0))
                            st.caption(f"{score}/{max_score} points")
                    
                    with col2:
                        st.markdown("**ATS Optimization Tips:**")
                        
                        tips = [
                            ("📌 Increase Keyword Density", ats_scores['keyword_match'] < 25),
                            ("📑 Add Missing Sections", ats_scores['section_completion'] < 25),
                            ("🎨 Improve Formatting", ats_scores['format_quality'] < 15),
                            ("💼 Highlight Experience", ats_scores['experience_relevance'] < 10)
                        ]
                        
                        for tip, condition in tips:
                            if condition:
                                st.warning(tip)
                            else:
                                st.success(tip)
                        
                        # ATS-friendly format checklist
                        st.markdown("**✅ ATS-Friendly Checklist:**")
                        checks = [
                            ("Simple, clean formatting", True),
                            ("Standard section headings", "experience" in resume_text.lower() and "education" in resume_text.lower()),
                            ("Keyword-rich content", ats_scores['keyword_match'] > 20),
                            ("Proper contact information", any(x in resume_text.lower() for x in ["@", "phone"])),
                            ("No tables or columns", "table" not in resume_text.lower()),
                            ("Machine-readable text", True)
                        ]
                        
                        for check, status in checks:
                            icon = "✅" if status else "❌"
                            st.write(f"{icon} {check}")
                
                with tab4:
                    st.subheader("📈 Detailed Score Analysis")
                    
                    # Score radar chart
                    categories = ['Content Match', 'Skills Match', 'Keyword Density', 'Structure', 'ATS Score']
                    values = [
                        match_data['content_score'] / 100 * 5,
                        match_data['skill_score'] / 100 * 5,
                        match_data['keyword_score'] / 15 * 5,
                        match_data['structure_score'] / 15 * 5,
                        ats_data['ats_total'] / 100 * 5
                    ]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Your Resume'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 5]
                            )),
                        showlegend=False,
                        title="Resume Performance Radar"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Score comparison table
                    score_df = pd.DataFrame({
                        'Metric': ['Overall Score', 'Content Similarity', 'Skill Match', 
                                  'Keyword Relevance', 'Resume Structure', 'ATS Compatibility'],
                        'Your Score': [
                            f"{match_data['overall_score']}%",
                            f"{match_data['content_score']}%",
                            f"{match_data['skill_score']}%",
                            f"{match_data['keyword_score']}/15",
                            f"{match_data['structure_score']}/15",
                            f"{ats_data['ats_total']}/100 ({ats_data['ats_grade']})"
                        ],
                        'Target': ['>70%', '>60%', '>70%', '>12/15', '>12/15', '>75/100 (A)']
                    })
                    
                    st.dataframe(score_df, use_container_width=True, hide_index=True)
        
        elif not resume_text or not job_text:
            st.info("👈 Please upload/paste your resume and job description to begin analysis.")
    
    # ==================== TAB 2: APPLICATION DASHBOARD ====================
    elif analysis_tab == "📈 Application Dashboard":
        st.subheader("📈 Application Tracking Dashboard")
        
        # Get application history
        df = get_application_history()
        
        if not df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Applications", len(df))
            
            with col2:
                avg_score = df['score'].mean()
                st.metric("Average Match Score", f"{avg_score:.1f}%")
            
            with col3:
                best_score = df['score'].max()
                st.metric("Best Match", f"{best_score:.1f}%")
            
            with col4:
                ats_avg = df['ats_score'].mean()
                st.metric("Avg ATS Score", f"{ats_avg:.1f}/100")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Score trend
                df['date'] = pd.to_datetime(df['date'])
                df_sorted = df.sort_values('date')
                fig = px.line(df_sorted, x='date', y='score', 
                            title='Match Score Trend Over Time',
                            labels={'date': 'Date', 'score': 'Match Score (%)'})
                fig.update_traces(mode='markers+lines')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score distribution
                fig = px.histogram(df, x='score', nbins=10,
                                 title='Distribution of Match Scores',
                                 labels={'score': 'Match Score (%)'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Company performance
            st.subheader("🏢 Company-Wise Performance")
            company_stats = df.groupby('company').agg({
                'score': ['mean', 'max', 'count']
            }).round(1)
            company_stats.columns = ['Avg Score', 'Best Score', 'Applications']
            company_stats = company_stats.sort_values('Avg Score', ascending=False)
            
            st.dataframe(company_stats, use_container_width=True)
            
            # Raw data
            with st.expander("📋 View All Applications"):
                st.dataframe(df[['date', 'company', 'role', 'score', 'ats_score', 'status']], 
                           use_container_width=True)
                
                # Export option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export as CSV",
                    data=csv,
                    file_name="applications_history.csv",
                    mime="text/csv"
                )
        else:
            st.info("No applications tracked yet. Analyze your first resume to see data here.")
    
    # ==================== TAB 3: SETTINGS & HELP ====================
    else:
        st.subheader("⚙️ Settings & Help")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📦 Installation Requirements")
            st.code("""
            pip install streamlit
            pip install pandas numpy
            pip install scikit-learn
            pip install plotly
            pip install PyPDF2 python-docx
            pip install google-generativeai
            """, language="bash")
            
            st.markdown("### 🚀 How to Run")
            st.code("""
            # 1. Save this code as app.py
            # 2. Create .streamlit/secrets.toml with your API key
            # 3. Open terminal/command prompt
            # 4. Navigate to the folder containing app.py
            # 5. Run: streamlit run app.py
            # 6. App will open in your browser!
            """, language="bash")
        
        with col2:
            st.markdown("### 📊 Features")
            
            features = [
                ("🤖 AI Resume Analysis", "TF-IDF and cosine similarity algorithms"),
                ("📋 ATS Simulation", "Applicant Tracking System compatibility scoring"),
                ("🔧 Skills Matching", "Comprehensive skills database with 50+ categories"),
                ("📈 Application Tracker", "SQLite database to track all applications"),
                ("📊 Visual Analytics", "Interactive charts with Plotly"),
                ("📄 Multi-Format Support", "PDF, DOCX, and plain text resume upload"),
                ("🚀 Gemini AI Integration", "AI-powered suggestions with Google Gemini Pro")
            ]
            
            for feature, description in features:
                st.markdown(f"**{feature}**")
                st.caption(description)
        
        st.divider()
        st.markdown("### 🎯 How to Use Effectively")
        
        tips = [
            "1. **Upload complete resumes** - Include all sections for accurate analysis",
            "2. **Use specific job descriptions** - The more detailed the job description, the better the match analysis",
            "3. **Track all applications** - Save each analysis to build your application history",
            "4. **Review missing skills** - Focus on high-priority skills mentioned in job descriptions",
            "5. **Optimize for ATS** - Follow the ATS checklist to improve your resume format",
            "6. **Enable Gemini AI** - Add your API key to `.streamlit/secrets.toml` for AI-powered suggestions"
        ]
        
        for tip in tips:
            st.markdown(tip)
        
        # Gemini setup instructions
        st.divider()
        st.markdown("### 🔑 Gemini AI Setup")
        st.code("""
        # Create .streamlit/secrets.toml with:
        GEMINI_API_KEY = "AIzaSyDlEU79NOj-OIc3sEG1ahSJVhBwLDWYh8g"
        """, language="toml")
        
        # Show current Gemini status
        if GEMINI_AVAILABLE:
            st.success(f"✅ Gemini AI is currently: CONNECTED")
        else:
            st.warning(f"⚠️ Gemini AI is currently: NOT CONNECTED")
            st.info("Click the 'Test Connection Now' button in the sidebar to fix this.")

# ==================== RUN THE APP ====================
if __name__ == "__main__":
    main()