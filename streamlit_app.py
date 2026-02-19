import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import parse_resume, clean_text, load_data, get_category_stats, get_top_keywords
import re

# Page config
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

# Title
st.title("AI Resume Screening & Ranking System")

# Load Dataset
DATA_PATH = "data/Resume.csv"
df_dataset = load_data(DATA_PATH)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Resume Analysis", "🤖 JD Helper", "🏆 Rank Resumes"])

# --- TAB 1: Analysis ---
with tab1:
    st.header("Dataset Overview")
    if not df_dataset.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Distribution")
            category_stats = get_category_stats(df_dataset)
            if not category_stats.empty:
                fig = px.pie(values=category_stats.values, names=category_stats.index, title="Resume Categories")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available.")
            
        with col2:
            st.subheader("Category Insights")
            if not category_stats.empty:
                selected_category_stats = st.selectbox("Select Category for Insights", category_stats.index)
                if selected_category_stats:
                    top_k = get_top_keywords(df_dataset, selected_category_stats, top_n=15)
                    st.write(f"**Top Skills/Keywords for {selected_category_stats}:**")
                    st.write(", ".join(top_k))
                    
                    st.info(f"Total Resumes in this category: {category_stats[selected_category_stats]}")
            else:
                st.info("No category data available.")
    else:
        st.warning("Resume dataset not found. Please ensure 'data/Resume.csv' exists for analysis features.")

# --- TAB 2: JD Helper ---
with tab2:
    st.header("Job Description Helper")
    st.markdown("Select a category to generate a recommended list of skills and requirements for your JD.")
    
    if not df_dataset.empty and 'Category' in df_dataset.columns:
        categories = df_dataset['Category'].unique()
        target_category = st.selectbox("Select Target Role", categories)
        
        if target_category:
            recommended_skills = get_top_keywords(df_dataset, target_category, top_n=20)
            
            st.subheader(f"Recommended Skills for {target_category}")
            st.success(", ".join(recommended_skills))
            
            st.subheader("Sample JD Template")
            jd_template = f"""
**Job Title**: {target_category}

**Role Overview**:
We are looking for a skilled {target_category} to join our team. The ideal candidate should have strong experience in the following areas:

**Key Requirements**:
- Proficiency in: {', '.join(recommended_skills[:10])}
- Experience with: {', '.join(recommended_skills[10:15])}
- Strong problem-solving skills and ability to work in a team.

**Responsibilities**:
- Develop and maintain solutions related to {target_category}.
- Collaborate with cross-functional teams.
            """
            st.text_area("Copy this Template:", value=jd_template, height=300)
    else:
        st.warning("Dataset not available or missing 'Category' column.")

# --- TAB 3: Ranking ---
with tab3:
    st.header("Resume Ranking")
    
    col_rank_1, col_rank_2 = st.columns([1, 2])
    
    with col_rank_1:
        st.subheader("Job Details")
        job_description = st.text_area("Job Description", height=200, 
                                       placeholder="Paste JD here or use the Helper tab...")
        required_skills_input = st.text_input("Required Skills (comma-separated)", 
                                              placeholder="e.g., Python, SQL")
        rank_button = st.button("Rank Resumes")

    with col_rank_2:
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT", 
                                          type=["pdf", "docx", "txt"], 
                                          accept_multiple_files=True)

    def extract_skills_from_text(text, skills_list):
        found_skills = []
        text_lower = text.lower()
        for skill in skills_list:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_skills.append(skill)
        return found_skills

    if rank_button:
        if not job_description:
            st.warning("Please enter a Job Description.")
        elif not uploaded_files:
            st.warning("Please upload at least one resume.")
        else:
            with st.spinner("Ranking resumes..."):
                clean_job_desc = clean_text(job_description)
                required_skills = [s.strip().lower() for s in required_skills_input.split(',') if s.strip()]

                resume_data = []
                for file in uploaded_files:
                    text = parse_resume(file)
                    cleaned_text = clean_text(text)
                    matched_skills = extract_skills_from_text(text, required_skills)
                    missing_skills = list(set(required_skills) - set(matched_skills))
                    skill_score = len(matched_skills) / len(required_skills) if required_skills else 0

                    resume_data.append({
                        "Filename": file.name,
                        "Cleaned Text": cleaned_text,
                        "Matched Skills": matched_skills,
                        "Missing Skills": missing_skills,
                        "Skill Score": skill_score
                    })

                df = pd.DataFrame(resume_data)

                if not df.empty:
                    # Dynamic TF-IDF Training
                    # Mix dataset samples into training for better vocabulary?
                    # For simplicity and relevance, we stick to training on the CURRENT corpus (JD + Uploads)
                    # This is the most robust "zero-shot" way without maintaining a large model.
                    
                    all_docs = [clean_job_desc] + df["Cleaned Text"].tolist()
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(all_docs)
                    
                    job_vector = tfidf_matrix[0]
                    resume_vectors = tfidf_matrix[1:]
                    
                    cosine_sim = cosine_similarity(resume_vectors, job_vector).flatten()
                    df["Content Match"] = cosine_sim
                    
                    # Final Score: 70% Semantic, 30% Keyword
                    df["Final Score"] = (df["Content Match"] * 0.7) + (df["Skill Score"] * 0.3)
                    
                    df = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
                    df["Rank"] = df.index + 1

                    st.success("Ranking Complete!")
                    
                    # Display Results in a more visual card format
                    for index, row in df.iterrows():
                        with st.expander(f"#{row['Rank']} {row['Filename']} - Score: {row['Final Score']:.1%}", expanded=(index==0)):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**✅ Matched Skills**")
                                if row['Matched Skills']:
                                    st.write(f", ".join([f"**{skill}**" for skill in row['Matched Skills']]))
                                else:
                                    st.write("No direct keyword matches.")
                            
                            with col2:
                                st.write("**❌ Missing Skills**")
                                if row['Missing Skills']:
                                    # Use red text or similar for missing
                                    st.markdown(f"<span style='color:red'>{', '.join(row['Missing Skills'])}</span>", unsafe_allow_html=True)
                                else:
                                    st.success("All required skills found!")
                            
                            st.divider()
                            st.write(f"**Semantic Match:** {row['Content Match']:.1%} | **Keyword Match:** {row['Skill Score']:.1%}")
                            
                            if index == 0:
                                st.caption("🏆 This candidate is the top match based on the job description.")
                    
                    # Summary Table
                    st.divider()
                    st.subheader("Comparison Table")
                    st.dataframe(df[["Rank", "Filename", "Final Score", "Matched Skills", "Missing Skills", "Content Match"]])
