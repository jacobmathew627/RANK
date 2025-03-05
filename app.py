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
        # Store the full API response for display in the UI
        full_response = response.text
        
        # Extract reasons and suggestions for structured display
        reasons_and_suggestions = response.text.split('\n')
        reasons = [line for line in reasons_and_suggestions if 'Reason:' in line]
        suggestions = [line for line in reasons_and_suggestions if 'Suggestion:' in line]
        
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
        full_response = "Error in generating analysis."
        reasons = ["Error in generating reasons."]
        suggestions = ["Error in generating suggestions."]
    
    return reasons, suggestions, full_response

def batch_processing_ui():
    st.header("Batch Processing")
    
    # Add super styling to the page
    st.markdown("""
    <style>
        /* Main styling */
        .main {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        h1, h2, h3, h4 {
            color: #2c3e50;
            font-weight: 600;
        }
        
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        
        /* Resume card styling */
        .resume-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .resume-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .resume-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        /* Score badge styling */
        .score-badge {
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-left: 10px;
            font-size: 14px;
        }
        
        .high-match {
            background-color: #27ae60;
            color: white;
        }
        
        .medium-match {
            background-color: #f39c12;
            color: white;
        }
        
        .low-match {
            background-color: #e74c3c;
            color: white;
        }
        
        .classification {
            font-style: italic;
            margin-top: 8px;
            color: #7f8c8d;
        }
        
        /* Analysis section styling */
        .analysis-section {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
        }
        
        .analysis-section strong {
            color: #3498db;
            font-size: 16px;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
        }
        
        /* File uploader styling */
        .stFileUploader>div>div {
            border: 2px dashed #3498db;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        /* Text area styling */
        .stTextArea>div>div>textarea {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f5f7fa;
            border-radius: 4px;
            padding: 10px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .streamlit-expanderContent {
            background-color: white;
            border: 1px solid #ecf0f1;
            border-radius: 0 0 4px 4px;
            padding: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    
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
                    badge_class = "high-match"
                elif resume['score'] > 5:
                    classification = "Medium Matching"
                    badge_class = "medium-match"
                else:
                    classification = "Low Matching"
                    badge_class = "low-match"
                
                # Enhanced display with score
                st.markdown(f"""
                <div class="resume-card">
                    <div class="resume-title">
                        {idx}. {resume['name']} 
                        <span class="score-badge {badge_class}">Score: {resume['score']:.2f}</span>
                    </div>
                    <div class="classification">{classification}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Show Analysis"):
                    if classification in ["Low Matching", "Medium Matching"]:
                        st.markdown("<div class='analysis-section'><strong>Reasons for Match Score:</strong></div>", unsafe_allow_html=True)
                        reasons, suggestions, full_response = analyze_low_matching(resume['text'], job_description)
                        for reason in reasons:
                            st.markdown(f"{reason}")
                        st.markdown("<div class='analysis-section'><strong>Suggestions for Improvement:</strong></div>", unsafe_allow_html=True)
                        for suggestion in suggestions:
                            st.markdown(f"{suggestion}")
                        optimized_resume = optimize_resume(resume['text'], job_description)
                        st.markdown("<div class='analysis-section'><strong>Optimized Resume:</strong></div>", unsafe_allow_html=True)
                        st.markdown(optimized_resume)
                        st.markdown("<div class='analysis-section'><strong>Full API Response:</strong></div>", unsafe_allow_html=True)
                        st.markdown(f"{full_response}")
                
                # Log to terminal for debugging
                print(f"Resume: {resume['name']}, Score: {resume['score']}, Classification: {classification}")

def single_resume_optimization_ui():
    st.header("Single Resume Optimization")
    
    # Add super styling to the page (same as batch processing)
    st.markdown("""
    <style>
        /* Main styling */
        .main {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        h1, h2, h3, h4 {
            color: #2c3e50;
            font-weight: 600;
        }
        
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        
        /* Resume card styling */
        .resume-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .resume-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        .resume-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        /* Score badge styling */
        .score-badge {
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-left: 10px;
            font-size: 14px;
        }
        
        .high-match {
            background-color: #27ae60;
            color: white;
        }
        
        .medium-match {
            background-color: #f39c12;
            color: white;
        }
        
        .low-match {
            background-color: #e74c3c;
            color: white;
        }
        
        .classification {
            font-style: italic;
            margin-top: 8px;
            color: #7f8c8d;
        }
        
        /* Analysis section styling */
        .analysis-section {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
        }
        
        .analysis-section strong {
            color: #3498db;
            font-size: 16px;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
        }
        
        /* File uploader styling */
        .stFileUploader>div>div {
            border: 2px dashed #3498db;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        /* Text area styling */
        .stTextArea>div>div>textarea {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f5f7fa;
            border-radius: 4px;
            padding: 10px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .streamlit-expanderContent {
            background-color: white;
            border: 1px solid #ecf0f1;
            border-radius: 0 0 4px 4px;
            padding: 15px;
        }
        
        /* Results container styling */
        .results-container {
            background-color: white;
            border-radius: 8px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        /* Optimization section styling */
        .optimization-section {
            background-color: #f1f9ff;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a resume", type=["pdf", "docx"])
    job_description = st.text_area("Paste the job description here")
    
    if uploaded_file and job_description:
        resume_text = upload_and_parse_resume(uploaded_file)
        if resume_text:
            score = calculate_match_score(resume_text, job_description)
            
            # Determine classification based on score
            if score > 8:
                classification = "Highly Matching"
                badge_class = "high-match"
            elif score > 5:
                classification = "Medium Matching"
                badge_class = "medium-match"
            else:
                classification = "Low Matching"
                badge_class = "low-match"
            
            # Display results in an enhanced UI
            st.markdown("""
            <div class="results-container">
                <h2>Resume Analysis Results</h2>
            """, unsafe_allow_html=True)
            
            # Display score with badge
            st.markdown(f"""
                <div class="resume-title">
                    {uploaded_file.name} 
                    <span class="score-badge {badge_class}">Score: {score:.2f}</span>
                </div>
                <div class="classification">{classification}</div>
            """, unsafe_allow_html=True)
            
            # If medium or low matching, analyze and provide suggestions
            if classification in ["Low Matching", "Medium Matching"]:
                reasons, suggestions, full_response = analyze_low_matching(resume_text, job_description)
                
                st.markdown("<div class='analysis-section'><strong>Reasons for Match Score:</strong></div>", unsafe_allow_html=True)
                for reason in reasons:
                    st.markdown(f"<li>{reason}</li>", unsafe_allow_html=True)
                
                st.markdown("<div class='analysis-section'><strong>Suggestions for Improvement:</strong></div>", unsafe_allow_html=True)
                for suggestion in suggestions:
                    st.markdown(f"<li>{suggestion}</li>", unsafe_allow_html=True)
                
                # Display full API response
                with st.expander("View Full Analysis"):
                    st.markdown(f"<pre style='white-space: pre-wrap; word-break: break-word;'>{full_response}</pre>", unsafe_allow_html=True)
            
            # Display optimized resume
            optimized_resume = optimize_resume(resume_text, job_description)
            st.markdown("<div class='analysis-section'><strong>Optimized Resume:</strong></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="optimization-section">
                {optimized_resume}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close results container
            
            # Log to terminal for debugging
            print(f"Resume: {uploaded_file.name}, Score: {score}, Classification: {classification}")

def main_ui():
    # Add app title and description with enhanced styling
    st.markdown("""
    <style>
        /* App title and header styling */
        .app-header {
            text-align: center;
            padding: 30px 0;
            background: linear-gradient(135deg, #2980b9, #3498db);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        
        .app-title {
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        }
        
        .app-description {
            font-size: 18px;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.5;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 6px 6px 0 0;
            padding: 10px 20px;
            font-weight: 600;
            color: #2c3e50;
            border: 1px solid #e0e0e0;
            border-bottom: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white !important;
            color: #3498db !important;
            border-top: 3px solid #3498db !important;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 20px 0;
            margin-top: 50px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
    
    <div class="app-header">
        <div class="app-title">Resume Rank & Optimization</div>
        <div class="app-description">
            Upload resumes, compare them with job descriptions, and get AI-powered optimization suggestions to improve your chances of landing your dream job.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Batch Processing", "Single Resume Optimization"])
    
    with tab1:
        batch_processing_ui()
    
    with tab2:
        single_resume_optimization_ui()
    
    # Add footer
    st.markdown("""
    <div class="footer">
        Resume Rank & Optimization Tool © 2023 | Powered by AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_ui()
