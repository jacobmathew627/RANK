import PyPDF2
import docx
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st

# Configure Gemini API
genai.configure(api_key="AIzaSyCQM4lEwFRf5N6XBs21Of4FyMmouo8g00A")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def upload_and_parse_resume(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        return None
    return text

def calculate_match_score(resume_text, job_description):
    # Calculate semantic similarity
    resume_embedding = embedding_model.encode([resume_text])
    job_embedding = embedding_model.encode([job_description])
    similarity_score = np.dot(resume_embedding, job_embedding.T)[0][0]
    
    # Normalize similarity score to [0, 1]
    similarity_score = (similarity_score + 1) / 2
    
    # Keyword matching
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
    keyword_overlap = len(job_keywords.intersection(resume_keywords))
    keyword_score = keyword_overlap / len(job_keywords) if job_keywords else 0
    
    # Weighted scoring
    semantic_weight = 0.7
    keyword_weight = 0.3
    combined_score = (semantic_weight * similarity_score) + (keyword_weight * keyword_score)
    
    # Scale score to 1-10
    scaled_score = combined_score * 9 + 1
    return scaled_score

def optimize_resume(resume_text, job_description):
    prompt = f"""
    You are an AI resume optimization expert. Your task is to optimize the following resume to better align with the job description provided.
    Ensure that no relevant content is eliminated, and focus on enhancing keyword usage, experience refinement, and clarity.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Please provide:
    
    1. An optimized version of the resume with improved keyword usage, experience refinement, and clarity.
    2. Specific, actionable suggestions for further improvement.Also make sure the optimization should be tailored to the job description and resume and a common optimization should not be repeated.
    """
    response = model.generate_content(prompt)
    return response.text

def analyze_low_matching(resume_text, job_description):
    prompt = f"""
    As an AI expert, analyze the alignment of the following resume with the job description provided. Your analysis should focus on the following aspects:
    
    1. Keyword Usage: Identify any missing or underutilized keywords that are critical for the job description.
    2. Experience Alignment: Evaluate how well the candidate's experience matches the job requirements.
    3. Content Relevance: Assess the overall relevance of the resume content to the job description.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Provide:
    - Detailed reasons for the match score, highlighting specific areas of strength and weakness.
    - Specific, actionable suggestions for improvement, including keyword additions, experience expansion, and sentence refinement for clarity and impact.
    """
    try:
        response = model.generate_content(prompt)
        reasons_and_suggestions = response.text.split('\n')
        reasons = [line for line in reasons_and_suggestions if 'Reason:' in line]
        suggestions = [line for line in reasons_and_suggestions if 'Suggestion:' in line]
        
        # Log the response for debugging
        print("API Response:", response.text)
        
        # Fallback content if API response is empty
        if not reasons:
            reasons = ["The resume lacks specific keywords from the job description.", 
                       "The experience section does not align well with the job requirements."]
        if not suggestions:
            suggestions = ["Consider adding more relevant keywords from the job description.", 
                           "Expand on your experience to better match the job requirements.",
                           "Refine sentences for clarity and impact."]
        
    except Exception as e:
        print("Error during API call:", e)
        reasons = ["Error in generating reasons."]
        suggestions = ["Error in generating suggestions."]
    
    return reasons, suggestions

def batch_processing_ui():
    st.header("Batch Processing")
    uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx"], accept_multiple_files=True)
    job_description = st.text_area("Paste the job description here")
    
    if uploaded_files and job_description:
        all_resumes = []
        for uploaded_file in uploaded_files:
            resume_text = upload_and_parse_resume(uploaded_file)
            if resume_text:
                all_resumes.append({
                    "name": uploaded_file.name,
                    "text": resume_text
                })
        
        if all_resumes:
            # Calculate scores and sort resumes
            for resume in all_resumes:
                resume['score'] = calculate_match_score(resume['text'], job_description)
            
            # Sort resumes by score
            all_resumes.sort(key=lambda x: x['score'], reverse=True)
            
            st.write("## Resume Analysis")
            for idx, resume in enumerate(all_resumes, start=1):
                if resume['score'] > 8:
                    classification = "Highly Matching"
                elif resume['score'] > 5:
                    classification = "Medium Matching"
                else:
                    classification = "Low Matching"
                
                st.markdown(f"### {idx}. {resume['name']}")
                st.markdown(f"**Match Score:** {resume['score']:.2f} - **{classification}**")
                
                if classification in ["Low Matching", "Medium Matching"]:
                    st.markdown("**Reasons for Match Score:**")
                    reasons, suggestions = analyze_low_matching(resume['text'], job_description)
                    for reason in reasons:
                        st.markdown(f"{reason}")
                    st.markdown("**Suggestions for Improvement:**")
                    for suggestion in suggestions:
                        st.markdown(f"{suggestion}")
                    optimized_resume = optimize_resume(resume['text'], job_description)
                    st.markdown("**Optimized Resume:**")
                    st.markdown(optimized_resume)

def single_resume_optimization_ui():
    st.header("Single Resume Optimization")
    resume_file = st.file_uploader("Upload a single resume", type=["pdf", "docx"], key="single_resume")
    single_job_description = st.text_area("Paste the job description here", key="single_job")
    
    if resume_file and single_job_description:
        resume_text = upload_and_parse_resume(resume_file)
        if resume_text:
            match_score = calculate_match_score(resume_text, single_job_description)
            optimized_resume = optimize_resume(resume_text, single_job_description)
            
            st.markdown(f"**Match Score:** {match_score:.2f}")
            st.markdown("## Optimized Resume")
            st.markdown(optimized_resume)

def main_ui():
    st.title("Resume Optimization and Classification with RAG using Gemini API")
    
    tab1, tab2 = st.tabs(["Batch Processing", "Single Resume Optimization"])
    
    with tab1:
        batch_processing_ui()
    
    with tab2:
        single_resume_optimization_ui()

if __name__ == "__main__":
    main_ui()
