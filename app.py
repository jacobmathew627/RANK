import PyPDF2
import docx
import re
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
import os

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
    # Calculate semantic similarity as baseline
    resume_embedding = embedding_model.encode([resume_text])
    job_embedding = embedding_model.encode([job_description])
    similarity_score = np.dot(resume_embedding, job_embedding.T)[0][0]
    
    # Normalize similarity score to [0, 1]
    similarity_score = (similarity_score + 1) / 2
    
    # Use AI to intelligently analyze skills, experience, and education
    prompt = f"""
    You are a precise resume analysis system that provides objective, factual scoring. 
    Analyze the following resume and job description for match scoring purposes.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Please provide a detailed analysis with the following components:
    
    1. Technical Skills Analysis:
       - Identify specific technical skills mentioned in the job description (like programming languages, tools, frameworks)
       - Identify specific technical skills mentioned in the resume
       - Calculate a match score (0-1) based on exact presence and semantic similarity between skills
       - Consider only actual technical skills, not general terms or concepts
       - Base your scoring on objective evidence in the texts

    2. Soft Skills Analysis:
       - Identify specific soft skills mentioned in the job description
       - Identify specific soft skills mentioned in the resume
       - Calculate a match score (0-1) based on exact presence and semantic similarity
       - Consider only actual soft skills, not general terms or concepts
       - Base your scoring on objective evidence in the texts

    3. Experience Analysis:
       - Identify specific years of experience required in the job description
       - Identify specific years of experience mentioned in the resume
       - Calculate a match score (0-1) using these clear metrics:
         * If job requires X years and resume shows X or more years: 1.0
         * If job requires X years and resume shows Y years (Y < X): Y/X
         * If no specific experience requirement: 0.8 if resume shows relevant experience, 0.5 if unclear

    4. Education Analysis:
       - Identify specific education requirements in the job description (degrees, fields)
       - Identify specific education mentioned in the resume
       - Calculate a match score (0-1) that reflects exact fulfillment of requirements
       - If no education requirements specified, score 1.0 if any formal education is present, 0.5 if unclear

    5. Keyword Match Analysis:
       - Identify important domain-specific keywords in the job description 
       - Calculate a match score (0-1) for keyword overlap with the resume
       - Consider only meaningful keywords, not common words
       - Base score on percentage of important keywords present in resume
    
    Format your response as a JSON object with these exact keys:
    {{
        "technical_skills_match": <float between 0-1>,
        "soft_skills_match": <float between 0-1>,
        "experience_match": <float between 0-1>,
        "education_match": <float between 0-1>,
        "keyword_score": <float between 0-1>,
        "technical_skills_job": [list of strings],
        "technical_skills_resume": [list of strings],
        "soft_skills_job": [list of strings],
        "soft_skills_resume": [list of strings],
        "explanation": "<brief explanation of overall match>"
    }}
    
    Important: 
    1. Be conservative in your scoring - if something is unclear, score it lower rather than assuming a match
    2. All scores MUST be between 0.0 and 1.0
    3. Base all scoring on objective evidence from the text, not assumptions
    4. For every score component, document the exact evidence you found in the explanation
    5. If a requested skill/requirement isn't mentioned at all in the resume, its match score should be 0.0 
    6. Provide precise lists of actual skills found, not general categories
    """
    
    try:
        response = model.generate_content(prompt)
        analysis_text = response.text
        
        # Remove any markdown code block markers that might be in the response
        analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        analysis = json.loads(analysis_text)
        
        # Validate scores to ensure they're in the valid range [0,1]
        tech_skills_match = max(0.0, min(1.0, analysis.get("technical_skills_match", 0.5)))
        soft_skills_match = max(0.0, min(1.0, analysis.get("soft_skills_match", 0.5)))
        experience_match = max(0.0, min(1.0, analysis.get("experience_match", 0.5)))
        education_match = max(0.0, min(1.0, analysis.get("education_match", 0.5)))
        keyword_score = max(0.0, min(1.0, analysis.get("keyword_score", 0.5)))
        
        # Log the extracted skills for debugging
        print(f"Technical Skills in Job: {analysis.get('technical_skills_job', [])}")
        print(f"Technical Skills in Resume: {analysis.get('technical_skills_resume', [])}")
        print(f"Soft Skills in Job: {analysis.get('soft_skills_job', [])}")
        print(f"Soft Skills in Resume: {analysis.get('soft_skills_resume', [])}")
        print(f"Explanation: {analysis.get('explanation', 'No explanation provided')}")
        
        # Sanity check: ensure tech_skills_match and soft_skills_match align with the skills lists
        tech_skills_job = analysis.get('technical_skills_job', [])
        tech_skills_resume = analysis.get('technical_skills_resume', [])
        soft_skills_job = analysis.get('soft_skills_job', [])
        soft_skills_resume = analysis.get('soft_skills_resume', [])
        
        # If no skills found in job, these components should have lower weight
        if len(tech_skills_job) == 0:
            tech_skills_match = similarity_score # fallback to semantic similarity
        if len(soft_skills_job) == 0:
            soft_skills_match = similarity_score # fallback to semantic similarity
            
        # Verify the match scores make sense given the overlap in skills
        if len(tech_skills_job) > 0 and len(tech_skills_resume) == 0:
            tech_skills_match = 0.0  # No technical skills in resume
        if len(soft_skills_job) > 0 and len(soft_skills_resume) == 0:
            soft_skills_match = 0.0  # No soft skills in resume
            
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        # Fallback to simpler analysis using sentence embeddings only
        tech_skills_match = similarity_score
        soft_skills_match = similarity_score
        experience_match = 0.5  # Default middle value
        education_match = 0.5   # Default middle value
        keyword_score = similarity_score
        
    # Weighted scoring with dynamic weights based on job requirements
    # Default weights
    semantic_weight = 0.3
    keyword_weight = 0.15
    tech_skills_weight = 0.25
    soft_skills_weight = 0.1
    experience_weight = 0.15
    education_weight = 0.05
    
    # Get flags for whether each component was meaningfully assessed
    has_tech_requirements = len(analysis.get('technical_skills_job', [])) > 0 if 'analysis' in locals() else True
    has_soft_requirements = len(analysis.get('soft_skills_job', [])) > 0 if 'analysis' in locals() else True
    
    # Adjust weights if certain components aren't relevant
    if not has_tech_requirements:
        tech_skills_weight = 0.05
        # Redistribute weight
        semantic_weight += 0.1
        keyword_weight += 0.05
        soft_skills_weight += 0.05
        
    if not has_soft_requirements:
        soft_skills_weight = 0.05
        # Redistribute weight
        semantic_weight += 0.05
        
    # Normalize weights to ensure they sum to 1.0
    total_weight = (semantic_weight + keyword_weight + tech_skills_weight + 
                   soft_skills_weight + experience_weight + education_weight)
    
    semantic_weight /= total_weight
    keyword_weight /= total_weight
    tech_skills_weight /= total_weight
    soft_skills_weight /= total_weight
    experience_weight /= total_weight
    education_weight /= total_weight
    
    combined_score = (
        (semantic_weight * similarity_score) +
        (keyword_weight * keyword_score) +
        (tech_skills_weight * tech_skills_match) +
        (soft_skills_weight * soft_skills_match) +
        (experience_weight * experience_match) +
        (education_weight * education_match)
    )
    
    # Scale score to 1-10
    scaled_score = combined_score * 9 + 1
    
    # Calculate a percentage match for the UI (0-100%)
    percentage_match = combined_score * 100
    
    # Print scores for debugging (can be commented out in production)
    print(f"Semantic Score: {similarity_score:.2f}")
    print(f"AI Keyword Score: {keyword_score:.2f}")
    print(f"AI Tech Skills Match: {tech_skills_match:.2f}")
    print(f"AI Soft Skills Match: {soft_skills_match:.2f}")
    print(f"AI Experience Match: {experience_match:.2f}")
    print(f"AI Education Match: {education_match:.2f}")
    print(f"Weights: Semantic={semantic_weight:.2f}, Keyword={keyword_weight:.2f}, "
          f"Tech={tech_skills_weight:.2f}, Soft={soft_skills_weight:.2f}, "
          f"Exp={experience_weight:.2f}, Edu={education_weight:.2f}")
    print(f"Final Score: {scaled_score:.2f}")
    print(f"Percentage Match: {percentage_match:.2f}%")
    
    # Create enhanced scores dictionary for UI display
    enhanced_scores = {
        "technical_skills": tech_skills_match * 100,
        "soft_skills": soft_skills_match * 100,
        "experience": experience_match * 100,
        "education": education_match * 100,
        "keyword_match": keyword_score * 100,
        "overall_relevance": similarity_score * 100
    }
    
    # Keywords found and missing (from the analysis if available)
    keywords_found = []
    keywords_missing = []
    
    try:
        if 'analysis' in locals():
            # Collect all skills (technical and soft) found in both resume and job
            all_job_keywords = set()
            all_job_keywords.update(analysis.get('technical_skills_job', []))
            all_job_keywords.update(analysis.get('soft_skills_job', []))
            
            all_resume_keywords = set()
            all_resume_keywords.update(analysis.get('technical_skills_resume', []))
            all_resume_keywords.update(analysis.get('soft_skills_resume', []))
            
            # Find matching and missing keywords
            keywords_found = list(all_job_keywords.intersection(all_resume_keywords))
            keywords_missing = list(all_job_keywords - all_resume_keywords)
    except Exception as e:
        print(f"Error extracting keywords: {e}")
    
    return percentage_match, enhanced_scores, keywords_found, keywords_missing

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
    As an expert resume analyst, provide a comprehensive analysis of how this specific resume aligns with the job description.
    You must thoroughly understand both the resume content and the job requirements before providing feedback.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Your analysis should include:
    
    1. Resume Structure and Content Analysis:
       - Evaluate the overall format, organization, and clarity of the resume
       - Assess the strength of achievement statements (quantifiable results vs. vague descriptions)
       - Identify any content gaps or sections that need improvement
    
    2. Skills Gap Analysis:
       - Identify technical skills required in the job description that are missing or underdeveloped in the resume
       - Identify soft skills required in the job description that are missing or underdeveloped in the resume
       - Evaluate how effectively existing skills are presented and contextualized
    
    3. Experience Relevance Analysis:
       - Evaluate how well the candidate's experience aligns with job requirements
       - Identify specific experience gaps that could be addressed
       - Assess whether experience descriptions emphasize the right achievements for this role
    
    4. Education and Credentials Analysis:
       - Evaluate if the education and certifications meet the job requirements
       - Identify any missing credentials that would strengthen the application

    5. ATS Optimization Analysis:
       - Identify missing keywords that could improve ATS scoring
       - Assess formatting issues that might impair ATS readability
    
    Provide your response in the following structured format:
    
    RESUME ASSESSMENT:
    - Overall Quality: Provide a brief assessment of the resume's overall quality
    - Key Strengths: List 2-3 strongest elements of the resume relative to this job
    - Critical Gaps: List 2-3 most significant gaps or weaknesses relative to this job
    
    DETAILED REASONS:
    - Reason 1: (detailed explanation of a key alignment issue)
    - Reason 2: (detailed explanation of another key alignment issue)
    - Reason 3: (detailed explanation of another key alignment issue if applicable)
    
    ACTIONABLE RECOMMENDATIONS:
    - Recommendation 1: (specific, actionable advice for improvement)
    - Recommendation 2: (specific, actionable advice for improvement)
    - Recommendation 3: (specific, actionable advice for improvement)
    - Recommendation 4: (specific, actionable advice for improvement if applicable)
    
    TAILORED KEYWORDS TO ADD:
    - List 5-8 specific keywords or phrases from the job description that should be incorporated
    
    Your analysis must be specific to this resume and job - avoid generic feedback that could apply to any resume.
    Focus on providing insightful, personalized analysis that shows you understand this specific resume and job.
    """
    try:
        response = model.generate_content(prompt)
        # Store the full API response for display in the UI
        full_response = response.text
        
        # Extract sections for structured display
        reasons = []
        suggestions = []
        
        # Extract different sections based on headers
        assessment_section = False
        reasons_section = False
        recommendations_section = False
        keywords_section = False
        
        for line in response.text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if "RESUME ASSESSMENT:" in line.upper():
                assessment_section = True
                reasons_section = False
                recommendations_section = False
                keywords_section = False
                continue
                
            if "DETAILED REASONS:" in line.upper() or "REASONS:" in line.upper():
                assessment_section = False
                reasons_section = True
                recommendations_section = False
                keywords_section = False
                continue
            
            if "ACTIONABLE RECOMMENDATIONS:" in line.upper() or "SUGGESTIONS:" in line.upper() or "RECOMMENDATIONS:" in line.upper():
                assessment_section = False
                reasons_section = False
                recommendations_section = True
                keywords_section = False
                continue
                
            if "TAILORED KEYWORDS" in line.upper():
                assessment_section = False
                reasons_section = False
                recommendations_section = False
                keywords_section = True
                continue
            
            # Extract content from each section
            if reasons_section and line.startswith('-'):
                reasons.append(line.lstrip('- '))
            
            if recommendations_section and line.startswith('-'):
                suggestions.append(line.lstrip('- '))
        
        # Fallback content if API response is empty or parsing failed
        if not reasons:
            reasons = ["The resume lacks specific keywords from the job description.", 
                       "The experience section does not align well with the job requirements.",
                       "The resume structure may not be optimized for ATS systems."]
        if not suggestions:
            suggestions = ["Consider adding more relevant keywords from the job description.", 
                           "Expand on your experience to better match the job requirements.",
                           "Quantify your achievements with specific metrics and outcomes.",
                           "Refine formatting for better ATS compatibility."]
        
    except Exception as e:
        print("Error during API call:", e)
        full_response = "Error in generating analysis."
        reasons = ["Error in generating detailed resume analysis."]
        suggestions = ["Please try again or contact support for assistance."]
    
    return reasons, suggestions, full_response

def format_analysis(text):
    """Format the analysis text for better readability in HTML"""
    # Replace section headers with styled headers
    text = text.replace("RESUME ASSESSMENT:", "<h3>Resume Assessment</h3>")
    text = text.replace("DETAILED REASONS:", "<h3>Alignment Issues</h3>")
    text = text.replace("REASONS:", "<h3>Alignment Issues</h3>")
    text = text.replace("ACTIONABLE RECOMMENDATIONS:", "<h3>Actionable Recommendations</h3>")
    text = text.replace("RECOMMENDATIONS:", "<h3>Actionable Recommendations</h3>")
    text = text.replace("SUGGESTIONS:", "<h3>Actionable Recommendations</h3>")
    text = text.replace("TAILORED KEYWORDS TO ADD:", "<h3>Recommended Keywords</h3>")
    
    # Format bullet points
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append("<br>")
        elif line.startswith('-'):
            # Convert bullet points to styled paragraphs
            content = line[1:].strip()
            formatted_lines.append(f"<p>• {content}</p>")
        elif line.startswith('•'):
            # Already has bullet point
            content = line[1:].strip()
            formatted_lines.append(f"<p>• {content}</p>")
        elif not any(header in line for header in ["<h3>", "<br>"]):
            # Regular text that's not a header or bullet
            formatted_lines.append(f"<p>{line}</p>")
        else:
            # Headers or breaks
            formatted_lines.append(line)
    
    return "".join(formatted_lines)

def load_css():
    # Load external CSS file
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def batch_processing_ui():
    st.header("Batch Processing")
    
    uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx"], accept_multiple_files=True)
    job_description = st.text_area("Paste the job description here", key="batch_job_description", height=200)
    
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
                percentage_match, enhanced_scores, keywords_found, keywords_missing = calculate_match_score(resume['text'], job_description)
                resume['percentage_match'] = percentage_match
                resume['enhanced_scores'] = enhanced_scores
                resume['keywords_found'] = keywords_found
                resume['keywords_missing'] = keywords_missing
            
            # Sort resumes by score
            all_resumes.sort(key=lambda x: x['percentage_match'], reverse=True)
            
            st.write("## Resume Analysis")

            for idx, resume in enumerate(all_resumes, start=1):
                if resume['percentage_match'] > 80:
                    classification = "Highly Matching"
                    badge_class = "high-match"
                elif resume['percentage_match'] > 50:
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
                        <span class="score-badge {badge_class}">Score: {resume['percentage_match']:.2f}%</span>
                    </div>
                    <div class="classification">{classification}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Show Analysis"):
                    if classification in ["Low Matching", "Medium Matching"]:
                        # Get the analysis but don't display reasons and suggestions separately
                        reasons, suggestions, full_response = analyze_low_matching(resume['text'], job_description)
                        
                        # Display optimized resume
                        optimized_resume = optimize_resume(resume['text'], job_description)
                        st.markdown("<div class='analysis-section'><strong>Optimized Resume:</strong></div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="background-color: #f8f8f8; border-left: 4px solid #1a5276; padding: 20px; 
                        margin: 20px 0; border-radius: 0 8px 8px 0; color: #000000; line-height: 1.6; font-size: 16px;">
                            {optimized_resume}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display full API response with enhanced formatting
                        st.markdown("<div class='analysis-section'><strong>Full Analysis:</strong></div>", unsafe_allow_html=True)
                        
                        # Process the full response to improve its structure
                        formatted_response = format_analysis(full_response)
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f8f8; border: 1px solid #999999; border-radius: 8px; 
                        padding: 20px; margin: 15px 0; color: #000000; font-size: 15px; line-height: 1.6;">
                            {formatted_response}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Log to terminal for debugging
                print(f"Resume: {resume['name']}, Score: {resume['percentage_match']}, Classification: {classification}")

def single_resume_optimization_ui():
    st.header("Single Resume Optimization")
    
    uploaded_file = st.file_uploader("Upload a resume", type=["pdf", "docx"], key="single_resume")
    job_description = st.text_area("Paste the job description here", key="single_job_description", height=200)
    
    if uploaded_file and job_description:
        resume_text = upload_and_parse_resume(uploaded_file)
        if resume_text:
            percentage_match, enhanced_scores, keywords_found, keywords_missing = calculate_match_score(resume_text, job_description)
            
            # Determine classification based on score
            if percentage_match > 80:
                classification = "Highly Matching"
                badge_class = "high-match"
            elif percentage_match > 50:
                classification = "Medium Matching"
                badge_class = "medium-match"
            else:
                classification = "Low Matching"
                badge_class = "low-match"
            
            # Display results in an enhanced UI
            st.markdown("""
            <div class="results-container">
                <h2 style="font-size: 24px; color: #000000; margin-bottom: 20px;">Resume Analysis Results</h2>
            """, unsafe_allow_html=True)
            
            # Display score with badge
            st.markdown(f"""
                <div class="resume-title">
                    {uploaded_file.name} 
                    <span class="score-badge {badge_class}">Score: {percentage_match:.2f}%</span>
                </div>
                <div class="classification">{classification}</div>
            """, unsafe_allow_html=True)
            
            # If medium or low matching, analyze but don't display reasons and suggestions separately
            if classification in ["Low Matching", "Medium Matching"]:
                reasons, suggestions, full_response = analyze_low_matching(resume_text, job_description)
                
                # Display full analysis with improved formatting
                with st.expander("View Full Analysis"):
                    # Process the full response to improve its structure
                    formatted_response = format_analysis(full_response)
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f8f8; border: 1px solid #999999; border-radius: 8px; 
                    padding: 20px; margin: 15px 0; color: #000000; font-size: 15px; line-height: 1.6;">
                        {formatted_response}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display optimized resume
            optimized_resume = optimize_resume(resume_text, job_description)
            st.markdown("<div class='analysis-section'><strong>Optimized Resume:</strong></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: #f8f8f8; border-left: 4px solid #1a5276; padding: 20px; 
            margin: 20px 0; border-radius: 0 8px 8px 0; color: #000000; line-height: 1.6; font-size: 16px;">
                {optimized_resume}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close results container
            
            # Log to terminal for debugging
            print(f"Resume: {uploaded_file.name}, Score: {percentage_match}, Classification: {classification}")

def main_ui():
    # Load external CSS
    load_css()
    
    # Add app header with title and description
    st.markdown("""
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
